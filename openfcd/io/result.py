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

    def write_frame(self, batch: str, frame_id: int, eta: np.ndarray, attrs: dict) -> None:
        grp = self._file.require_group(f"batches/{batch}/frames")
        if str(frame_id) in grp:
            del grp[str(frame_id)]
        ds = grp.create_dataset(str(frame_id), data=np.asarray(eta, dtype=np.float64))
        for k, v in attrs.items():
            ds.attrs[k] = v

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
