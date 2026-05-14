from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
import h5py
import numpy as np

if TYPE_CHECKING:
    from openfcd.io.project import ProjectModel

FORMAT_VERSION = 0
OPENFCD_VERSION = "0.0.1"


class HDF5ResultStore:
    """h5py SWMR result container with atomic-rename writes."""

    def __init__(self, path: str | Path, mode: str = "r"):
        self._path = Path(path)
        self._mode = mode
        self._file: h5py.File | None = None
        self._tmp_path: Path | None = None

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> "HDF5ResultStore":
        store = cls(path, mode)
        if mode == "w":
            store._tmp_path = Path(str(path) + ".tmp")
            store._file = h5py.File(store._tmp_path, "w")
            store._file.attrs["format_version"] = FORMAT_VERSION
            store._file.attrs["openfcd_version"] = OPENFCD_VERSION
        else:
            store._tmp_path = None
            store._file = h5py.File(path, mode)
        return store

    @classmethod
    def open_existing_for_append(cls, h5_path: str | Path) -> "HDF5ResultStore":
        """Open an existing h5 file for appending new groups/datasets.

        Unlike ``open(mode='w')`` (which truncates), this preserves all
        existing content and allows writing additional groups.

        The returned store uses ``_mode='append'`` so that ``close()`` only
        closes the file handle — it does **not** trigger the atomic os.replace
        that 'w' mode uses.  The caller (e.g. ``cmd_postprocess``) is
        responsible for the final ``os.replace(tmp, target)``.

        Note: if this call is interrupted before ``os.replace`` completes,
        a ``results.h5.tmp`` residue may remain; delete it manually to recover.
        """
        store = cls(h5_path, "append")
        store._tmp_path = None
        store._file = h5py.File(Path(h5_path), "a")
        return store

    def list_batches(self) -> list[str]:
        grp = self._file.get("batches", None)
        if grp is None:
            return []
        return list(grp.keys())

    def read_batch_meta(self, batch: str) -> dict:
        grp = self._file[f"batches/{batch}"]
        return dict(grp.attrs)

    def read_summary(self, batch: str, kind: str) -> np.ndarray:
        return self._file[f"batches/{batch}/{kind}"][:]

    def list_frames(self, batch: str) -> list[int]:
        grp = self._file.get(f"batches/{batch}/frames", None)
        if grp is None:
            return []
        return sorted(int(k) for k in grp.keys())

    def read_frame(self, batch: str, frame_id: int) -> np.ndarray:
        return self._file[f"batches/{batch}/frames/{frame_id}"][:]

    def read_frame_attrs(self, batch: str, frame_id: int) -> dict:
        return dict(self._file[f"batches/{batch}/frames/{frame_id}"].attrs)

    def read_stack(self, batch: str, frame_ids: list[int] | None = None) -> np.ndarray:
        ds = self._file[f"batches/{batch}/eta_stack"]
        if frame_ids is None:
            return ds[:]
        return ds[frame_ids]

    def batch_status(self, batch: str) -> dict:
        grp = self._file.get(f"batches/{batch}", None)
        if grp is None:
            return {"total": 0, "done": 0, "error": 0, "skipped": 0}
        frames = grp.get("frames", None)
        if frames is None:
            return {"total": 0, "done": 0, "error": 0, "skipped": 0}
        total = len(frames)
        done = sum(1 for k in frames if frames[k].attrs.get("status") == "ok")
        error = sum(1 for k in frames if frames[k].attrs.get("status") == "error")
        skipped = sum(1 for k in frames if frames[k].attrs.get("status") == "skipped")
        return {"total": total, "done": done, "error": error, "skipped": skipped}

    def fingerprint(self) -> str:
        return self._file.attrs.get("annotation_fingerprint", "")

    def is_stale(self, project: "ProjectModel") -> bool:
        fp = self.fingerprint()
        current = _project_fingerprint(project)
        return fp != current

    def write_batch(self, batch: str, data: dict, meta: dict) -> None:
        grp = self._file.require_group(f"batches/{batch}")
        for k, v in meta.items():
            grp.attrs[k] = v
        for key in ("eta_mean", "eta_median", "eta_rms"):
            if key in data:
                arr = np.asarray(data[key], dtype=np.float64)
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=arr)
        if "eta_stack" in data:
            stack = np.asarray(data["eta_stack"], dtype=np.float64)
            if "eta_stack" in grp:
                del grp["eta_stack"]
            n, h, w = stack.shape
            grp.create_dataset("eta_stack", data=stack, chunks=(1, h, w))

    def read_frame_qc(self, batch: str, frame_id: int, name: str) -> np.ndarray:
        return self._file[f"batches/{batch}/qc/{frame_id}/{name}"][:]

    def list_frame_qc(self, batch: str, frame_id: int) -> list[str]:
        grp = self._file.get(f"batches/{batch}/qc/{frame_id}", None)
        if grp is None:
            return []
        return sorted(grp.keys())

    def write_frame(
        self,
        batch: str,
        frame_id: int,
        eta: np.ndarray,
        attrs: dict,
        *,
        qc_datasets: dict[str, np.ndarray] | None = None,
    ) -> None:
        grp = self._file.require_group(f"batches/{batch}/frames")
        if str(frame_id) in grp:
            del grp[str(frame_id)]
        ds = grp.create_dataset(str(frame_id), data=np.asarray(eta, dtype=np.float64))
        for k, v in attrs.items():
            ds.attrs[k] = v
        qc_root = self._file.require_group(f"batches/{batch}/qc")
        if str(frame_id) in qc_root:
            del qc_root[str(frame_id)]
        if qc_datasets is not None:
            qc_grp = qc_root.create_group(str(frame_id))
            for name, arr in qc_datasets.items():
                qc_grp.create_dataset(name, data=np.asarray(arr))

    def write_wave_stats(
        self,
        batch: str,
        segments: list[dict] | None,
        attrs: dict,
    ) -> None:
        """Write wave statistics (v2) into ``batches/{batch}/wave_stats/``.

        Layout::

            batches/{batch}/wave_stats/
              attrs: <attrs dict, written in sorted-key order>
              segments/
                {idx:04d}/
                  attrs: s_lo_mm, s_hi_mm, label, color, visible (uint8)
                  wavelength_mm     (n_frames,) float64
                  wavenumber_per_mm (n_frames,) float64
                  peaks/{frame_id}      (n, 2) float64
                  troughs/{frame_id}    (n, 2) float64
                  heights/{frame_id}    (n,)   float64

        Parameters
        ----------
        batch :
            Batch name (e.g. ``"default"``).
        segments :
            Ordered list of per-segment dicts.  ``None`` skips the
            ``segments`` group entirely.  Each dict must contain:

            * ``segment_idx``  (int) — used to form the ``{idx:04d}`` key
            * ``s_lo_mm``, ``s_hi_mm``  (float) — written as attrs
            * ``label``, ``color``  (str) — written as attrs
            * ``visible``  (bool) — stored as uint8
            * ``wavelength_mm``      (n_frames,)
            * ``wavenumber_per_mm``  (n_frames,)
            * ``peaks``    ``{frame_id_int: (n, 2) float64}``
            * ``troughs``  ``{frame_id_int: (n, 2) float64}``
            * ``heights``  ``{frame_id_int: (n,) float64}``
        attrs :
            Group-level attributes (e.g. ``prominence_k``, ``ds_mm``,
            ``n_frames``, ``n_segments``, ``schema_version``).  Keys are
            traversed in sorted order for byte-equal output.
        """
        ws_grp = self._file.require_group(f"batches/{batch}/wave_stats")
        for k in sorted(attrs.keys()):
            ws_grp.attrs[k] = attrs[k]

        if segments is None:
            return

        segs_grp = ws_grp.require_group("segments")
        # Sort by segment_idx for byte-equal output
        sorted_segs = sorted(segments, key=lambda d: int(d["segment_idx"]))
        for seg in sorted_segs:
            idx = int(seg["segment_idx"])
            seg_key = f"{idx:04d}"
            if seg_key in segs_grp:
                del segs_grp[seg_key]
            seg_grp = segs_grp.create_group(seg_key)

            # Per-segment attrs (sorted key order)
            seg_attrs = {
                "s_lo_mm": float(seg["s_lo_mm"]),
                "s_hi_mm": float(seg["s_hi_mm"]),
                "label": str(seg.get("label", "")),
                "color": str(seg.get("color", "#1f77b4")),
                "visible": np.uint8(1 if bool(seg.get("visible", True)) else 0),
            }
            for k in sorted(seg_attrs.keys()):
                seg_grp.attrs[k] = seg_attrs[k]

            # Scalar-per-frame arrays
            wl = np.asarray(seg["wavelength_mm"], dtype=np.float64)
            wn = np.asarray(seg["wavenumber_per_mm"], dtype=np.float64)
            seg_grp.create_dataset("wavelength_mm", data=wl)
            seg_grp.create_dataset("wavenumber_per_mm", data=wn)

            # Per-frame nested groups
            peaks_grp = seg_grp.create_group("peaks")
            troughs_grp = seg_grp.create_group("troughs")
            heights_grp = seg_grp.create_group("heights")
            for fid in sorted(seg.get("peaks", {}).keys()):
                peaks_grp.create_dataset(
                    str(fid),
                    data=np.asarray(seg["peaks"][fid], dtype=np.float64),
                )
            for fid in sorted(seg.get("troughs", {}).keys()):
                troughs_grp.create_dataset(
                    str(fid),
                    data=np.asarray(seg["troughs"][fid], dtype=np.float64),
                )
            for fid in sorted(seg.get("heights", {}).keys()):
                heights_grp.create_dataset(
                    str(fid),
                    data=np.asarray(seg["heights"][fid], dtype=np.float64),
                )

        self._file.flush()

    def write_wave_stats_per_frame(
        self,
        batch: str,
        per_frame_payloads: dict[int, dict],
        attrs: dict,
    ) -> None:
        """Write per-frame wave statistics (v3) into
        ``batches/{batch}/wave_stats/``.

        Layout::

            batches/{batch}/wave_stats/
              attrs: <attrs dict, sorted-key order>
              frame_{fid}/
                attrs: frame_id, n_segments, segments_meta
                segments/{idx:04d}/
                  attrs: s_lo_mm, s_hi_mm, label, color, visible (uint8)
                  wavelength_mm     scalar float64
                  wavenumber_per_mm scalar float64
                  S_gamma_N_per_m   scalar float64
                  S_g_N_per_m       scalar float64
                  S_cg_N_per_m      scalar float64
                  peaks   (n, 2) float64
                  troughs (n, 2) float64
                  heights (n,)   float64

        Frames not in ``per_frame_payloads`` are NOT written.
        """
        ws_grp = self._file.require_group(f"batches/{batch}/wave_stats")
        for k in sorted(attrs.keys()):
            ws_grp.attrs[k] = attrs[k]

        for fid in sorted(per_frame_payloads.keys()):
            payload = per_frame_payloads[fid]
            frame_key = f"frame_{fid}"
            if frame_key in ws_grp:
                del ws_grp[frame_key]
            frame_grp = ws_grp.create_group(frame_key)

            seg_payloads = payload["segments"]
            frame_attrs = {
                "frame_id": int(fid),
                "n_segments": int(len(seg_payloads)),
                "segments_meta": str(payload["segments_meta_json"]),
            }
            for k in sorted(frame_attrs.keys()):
                frame_grp.attrs[k] = frame_attrs[k]

            segs_grp = frame_grp.create_group("segments")
            sorted_segs = sorted(seg_payloads, key=lambda d: int(d["segment_idx"]))
            for seg in sorted_segs:
                idx = int(seg["segment_idx"])
                seg_grp = segs_grp.create_group(f"{idx:04d}")
                seg_attrs = {
                    "s_lo_mm": float(seg["s_lo_mm"]),
                    "s_hi_mm": float(seg["s_hi_mm"]),
                    "label": str(seg.get("label", "")),
                    "color": str(seg.get("color", "#1f77b4")),
                    "visible": np.uint8(1 if bool(seg.get("visible", True)) else 0),
                }
                for k in sorted(seg_attrs.keys()):
                    seg_grp.attrs[k] = seg_attrs[k]
                seg_grp.create_dataset(
                    "wavelength_mm",
                    data=np.float64(seg["wavelength_mm"]),
                )
                seg_grp.create_dataset(
                    "wavenumber_per_mm",
                    data=np.float64(seg["wavenumber_per_mm"]),
                )
                seg_grp.create_dataset(
                    "S_gamma_N_per_m",
                    data=np.float64(seg.get("S_gamma_N_per_m", float("nan"))),
                )
                seg_grp.create_dataset(
                    "S_g_N_per_m",
                    data=np.float64(seg.get("S_g_N_per_m", float("nan"))),
                )
                seg_grp.create_dataset(
                    "S_cg_N_per_m",
                    data=np.float64(seg.get("S_cg_N_per_m", float("nan"))),
                )
                seg_grp.create_dataset(
                    "peaks",
                    data=np.asarray(seg["peaks"], dtype=np.float64),
                )
                seg_grp.create_dataset(
                    "troughs",
                    data=np.asarray(seg["troughs"], dtype=np.float64),
                )
                seg_grp.create_dataset(
                    "heights",
                    data=np.asarray(seg["heights"], dtype=np.float64),
                )

        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._mode == "w" and self._tmp_path and self._tmp_path.exists():
            os.replace(self._tmp_path, self._path)

    def __enter__(self) -> "HDF5ResultStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _project_fingerprint(project: "ProjectModel") -> str:
    payload = (
        json.dumps(project.geometry.model_dump(), sort_keys=True)
        + json.dumps(project.process.model_dump(), sort_keys=True)
    )
    return hashlib.sha256(payload.encode()).hexdigest()
