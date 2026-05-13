"""AC-7 — wave_stats subtree determinism (v2 schema)."""

from __future__ import annotations

import hashlib
import io

import h5py
import numpy as np
import pytest

from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveSegment,
    WaveStatsConfig,
)
from openfcd.io.result import HDF5ResultStore
from openfcd.pipeline.wave_stats_pipeline import compute_and_write_wave_stats


# ── Test constants ───────────────────────────────────────────────────────────

H, W = 80, 300
PX_PER_MM = 10.0
LAMBDA_MM = 10.0
A0 = 1.0
ALPHA = 0.03
N_FRAMES = 5
L_MM = (W - 1) / PX_PER_MM
BATCH = "default"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_eta(frame_idx: int) -> np.ndarray:
    cols = np.arange(W, dtype=np.float64)
    s_mm = cols / PX_PER_MM
    shift_mm = frame_idx * 0.3
    eta_row = A0 * np.exp(-ALPHA * s_mm) * np.sin(
        2.0 * np.pi * (s_mm - shift_mm) / LAMBDA_MM
    )
    return np.tile(eta_row, (H, 1))


def _make_annotation() -> AnnotationSchema:
    # s_mm is centered on the line midpoint (s=0): fore = left half, aft = right half.
    half = L_MM / 2.0
    ws = WaveStatsConfig(
        segments=[
            WaveSegment(s_lo_mm=-half, s_hi_mm=0.0, label="fore"),
            WaveSegment(s_lo_mm=0.0, s_hi_mm=half, label="aft"),
        ],
        peak_prominence_k=0.15,
    )
    pl = ProfileLineData(
        start=(float(H // 2), 0.0),
        end=(float(H // 2), float(W - 1)),
    )
    return AnnotationSchema(wave_stats=ws, profile_line=pl)


def _build_and_run(h5_path) -> None:
    store = HDF5ResultStore.open(h5_path, "w")
    for i in range(N_FRAMES):
        store.write_frame(BATCH, i, _make_eta(i), {"status": "ok"})
    store.write_batch(
        BATCH, data={}, meta={"pixel_per_mm_median": PX_PER_MM}
    )
    compute_and_write_wave_stats(
        annotation=_make_annotation(),
        store=store,
        batches=[BATCH],
        px_per_mm_by_batch={BATCH: PX_PER_MM},
        body_polygons_by_batch_frame={BATCH: {}},
        eta_loader=lambda b, fid: store._file[f"batches/{b}/frames/{fid}"][:],
    )
    store.close()


def _sha256_wave_stats(h5_path, batch: str = BATCH) -> str:
    buf = io.BytesIO()
    with h5py.File(h5_path, "r") as f:
        grp = f.get(f"batches/{batch}/wave_stats")
        if grp is None:
            return ""

        items: list[tuple[str, bytes]] = []

        def _collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                raw = arr.tobytes() if hasattr(arr, "tobytes") else str(arr).encode()
                items.append((name, raw))

        grp.visititems(_collect)
        items.sort(key=lambda x: x[0])

        for name, raw in items:
            buf.write(name.encode())
            buf.write(raw)

    return hashlib.sha256(buf.getvalue()).hexdigest()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_wave_stats_deterministic_double_run(tmp_path):
    _build_and_run(tmp_path / "run_a.h5")
    _build_and_run(tmp_path / "run_b.h5")

    digest_a = _sha256_wave_stats(tmp_path / "run_a.h5")
    digest_b = _sha256_wave_stats(tmp_path / "run_b.h5")

    assert digest_a != ""
    assert digest_b != ""
    assert digest_a == digest_b


def test_wave_stats_subtree_structure(tmp_path):
    _build_and_run(tmp_path / "struct.h5")

    with h5py.File(tmp_path / "struct.h5", "r") as f:
        for idx in (0, 1):
            seg = f[f"batches/{BATCH}/wave_stats/segments/{idx:04d}"]
            for dataset in ("wavelength_mm", "wavenumber_per_mm"):
                assert dataset in seg
            for subgrp in ("peaks", "troughs", "heights"):
                assert subgrp in seg


def test_wave_stats_wavelength_shape(tmp_path):
    _build_and_run(tmp_path / "shape.h5")

    with h5py.File(tmp_path / "shape.h5", "r") as f:
        for idx in (0, 1):
            wl = f[f"batches/{BATCH}/wave_stats/segments/{idx:04d}/wavelength_mm"][:]
            assert wl.shape == (N_FRAMES,)


def test_wave_stats_attrs_present(tmp_path):
    _build_and_run(tmp_path / "attrs.h5")
    required = {
        "prominence_k", "ds_mm", "n_frames", "n_segments",
        "schema_version", "segments_meta",
    }
    with h5py.File(tmp_path / "attrs.h5", "r") as f:
        ws_attrs = dict(f[f"batches/{BATCH}/wave_stats"].attrs)
        for key in required:
            assert key in ws_attrs, f"missing attr: {key!r}"
        assert ws_attrs["ds_mm"] == pytest.approx(1.0 / PX_PER_MM)
        assert ws_attrs["n_frames"] == N_FRAMES
        assert int(ws_attrs["schema_version"]) == 2
