"""Tests for openfcd/pipeline/wave_stats_pipeline.py (v3 — per-frame layout)."""

from __future__ import annotations

import hashlib
import json

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


# ── Fixtures ─────────────────────────────────────────────────────────────────


H, W = 100, 300
PX_PER_MM = 10.0
LAMBDA_MM = 10.0
A = 1.0
L_MM = (W - 1) / PX_PER_MM


def _make_eta(shift: float = 0.0) -> np.ndarray:
    cols = np.arange(W, dtype=np.float64)
    s_mm = cols / PX_PER_MM
    alpha = 0.03
    eta_row = A * np.exp(-alpha * s_mm) * np.sin(2.0 * np.pi * (s_mm - shift) / LAMBDA_MM)
    return np.tile(eta_row, (H, 1))


def _two_seg_per_frame(frame_ids: list[int]) -> dict[str, list[WaveSegment]]:
    mid = L_MM / 2.0
    segs = [
        WaveSegment(s_lo_mm=0.0, s_hi_mm=mid, label="fore", color="#1f77b4"),
        WaveSegment(s_lo_mm=mid, s_hi_mm=L_MM, label="aft", color="#d62728"),
    ]
    return {str(fid): list(segs) for fid in frame_ids}


@pytest.fixture()
def annotation() -> AnnotationSchema:
    ws_cfg = WaveStatsConfig(
        segments_by_frame=_two_seg_per_frame([0, 1]),
        peak_prominence_k=0.15,
    )
    pl = ProfileLineData(start=(50.0, 0.0), end=(50.0, float(W - 1)))
    return AnnotationSchema(wave_stats=ws_cfg, profile_line=pl)


@pytest.fixture()
def store_with_frames(tmp_path):
    h5_path = tmp_path / "results.h5"
    store = HDF5ResultStore.open(h5_path, "w")
    store.write_frame("test", 0, _make_eta(0.0), {"status": "ok"})
    store.write_frame("test", 1, _make_eta(0.1), {"status": "ok"})
    store.write_batch("test", data={}, meta={"pixel_per_mm_median": PX_PER_MM})
    yield store
    store.close()


def _eta_loader(store: HDF5ResultStore):
    def _load(batch: str, fid_str: str) -> np.ndarray:
        return store._file[f"batches/{batch}/frames/{fid_str}"][:]
    return _load


# ── Test 1: basic write + h5 structure ───────────────────────────────────────


def test_basic_wave_stats_written(annotation, store_with_frames):
    store = store_with_frames

    ok = compute_and_write_wave_stats(
        annotation=annotation,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )

    assert ok is True
    assert "batches/test/wave_stats" in store._file

    ws_grp = store._file["batches/test/wave_stats"]
    assert ws_grp.attrs["prominence_k"] == pytest.approx(0.15)
    assert ws_grp.attrs["ds_mm"] == pytest.approx(1.0 / PX_PER_MM)
    assert ws_grp.attrs["n_frames"] == 2
    assert int(ws_grp.attrs["schema_version"]) == 3

    # per-frame groups
    for fid in (0, 1):
        frame_grp = ws_grp[f"frame_{fid}"]
        assert frame_grp.attrs["n_segments"] == 2
        for idx in (0, 1):
            seg = frame_grp[f"segments/{idx:04d}"]
            assert "wavelength_mm" in seg
            assert "wavenumber_per_mm" in seg
            assert "peaks" in seg
            assert "troughs" in seg
            assert "heights" in seg


# ── Test 2: wavelength approximately correct ─────────────────────────────────


def test_wavelength_approximately_correct(annotation, store_with_frames):
    store = store_with_frames
    compute_and_write_wave_stats(
        annotation=annotation,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )

    wl = float(store._file["batches/test/wave_stats/frame_0/segments/0000/wavelength_mm"][()])
    if np.isfinite(wl):
        assert abs(wl - LAMBDA_MM) / LAMBDA_MM < 0.20


# ── Test 3: no-op when wave_stats is None ────────────────────────────────────


def test_no_op_when_wave_stats_none(store_with_frames):
    store = store_with_frames
    ann = AnnotationSchema()

    ok = compute_and_write_wave_stats(
        annotation=ann,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )
    assert ok is False
    assert "batches/test/wave_stats" not in store._file


def test_no_op_when_profile_line_none(store_with_frames):
    store = store_with_frames
    seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=L_MM / 2.0)
    ann = AnnotationSchema(
        wave_stats=WaveStatsConfig(segments_by_frame={"0": [seg]}),
        profile_line=None,
    )

    ok = compute_and_write_wave_stats(
        annotation=ann,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )
    assert ok is False
    assert "batches/test/wave_stats" not in store._file


def test_no_op_when_no_visible_segments(store_with_frames):
    """All segments invisible → no-op."""
    store = store_with_frames
    seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=L_MM / 2.0, visible=False)
    pl = ProfileLineData(start=(50.0, 0.0), end=(50.0, float(W - 1)))
    ann = AnnotationSchema(
        wave_stats=WaveStatsConfig(segments_by_frame={"0": [seg], "1": [seg]}),
        profile_line=pl,
    )

    ok = compute_and_write_wave_stats(
        annotation=ann,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )
    assert ok is False


# ── Test 4: existing frames untouched ────────────────────────────────────────


def test_frames_unaffected(annotation, tmp_path):
    h5_path = tmp_path / "coexist.h5"
    store = HDF5ResultStore.open(h5_path, "w")
    for i in range(3):
        store.write_frame("test", i, _make_eta(i * 0.05), {"status": "ok"})
    store.write_batch("test", data={}, meta={"pixel_per_mm_median": PX_PER_MM})

    # Annotation only covers frames 0/1; frame 2 should be skipped.
    ok = compute_and_write_wave_stats(
        annotation=annotation,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=lambda b, fid: store._file[f"batches/{b}/frames/{fid}"][:],
    )
    assert ok is True
    assert store.list_frames("test") == [0, 1, 2]
    # Only frames 0 and 1 should have wave_stats groups
    ws_grp = store._file["batches/test/wave_stats"]
    assert "frame_0" in ws_grp
    assert "frame_1" in ws_grp
    assert "frame_2" not in ws_grp
    store.close()


# ── Test 5: segments_meta is JSON-decodable ──────────────────────────────────


def test_segments_meta_is_json(annotation, store_with_frames):
    store = store_with_frames
    compute_and_write_wave_stats(
        annotation=annotation,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )
    meta_str = store._file["batches/test/wave_stats/frame_0"].attrs["segments_meta"]
    parsed = json.loads(meta_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    labels = sorted(item["label"] for item in parsed)
    assert labels == ["aft", "fore"]


# ── Test 6: visibility filtering — invisible segments excluded ───────────────


def test_invisible_segment_excluded(store_with_frames):
    store = store_with_frames
    segs = [
        WaveSegment(s_lo_mm=0.0, s_hi_mm=10.0, label="A", visible=True),
        WaveSegment(s_lo_mm=10.0, s_hi_mm=20.0, label="B", visible=False),
        WaveSegment(s_lo_mm=20.0, s_hi_mm=L_MM, label="C", visible=True),
    ]
    pl = ProfileLineData(start=(50.0, 0.0), end=(50.0, float(W - 1)))
    ann = AnnotationSchema(
        wave_stats=WaveStatsConfig(
            segments_by_frame={"0": segs, "1": segs},
            peak_prominence_k=0.15,
        ),
        profile_line=pl,
    )

    compute_and_write_wave_stats(
        annotation=ann,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )

    frame_grp = store._file["batches/test/wave_stats/frame_0"]
    keys = list(frame_grp["segments"].keys())
    # idx 0 (A) and 2 (C) should exist, idx 1 (B, invisible) should not
    assert "0000" in keys
    assert "0002" in keys
    assert "0001" not in keys
    assert frame_grp.attrs["n_segments"] == 2


# ── Test 7: empty-frame skip ─────────────────────────────────────────────────


def test_frame_without_segments_is_skipped(store_with_frames):
    """Frame 0 has segments; frame 1 has no segments → no frame_1 group."""
    store = store_with_frames
    mid = L_MM / 2.0
    seg = WaveSegment(s_lo_mm=0.0, s_hi_mm=mid, label="fore")
    pl = ProfileLineData(start=(50.0, 0.0), end=(50.0, float(W - 1)))
    ann = AnnotationSchema(
        wave_stats=WaveStatsConfig(segments_by_frame={"0": [seg]}),
        profile_line=pl,
    )
    ok = compute_and_write_wave_stats(
        annotation=ann,
        store=store,
        batches=["test"],
        px_per_mm_by_batch={"test": PX_PER_MM},
        body_polygons_by_batch_frame={"test": {}},
        eta_loader=_eta_loader(store),
    )
    assert ok is True
    ws_grp = store._file["batches/test/wave_stats"]
    assert "frame_0" in ws_grp
    assert "frame_1" not in ws_grp
    assert ws_grp.attrs["n_frames"] == 1


# ── Test 8: sorted-order reproducibility ─────────────────────────────────────


def _sha256_wave_stats_group(store: HDF5ResultStore, batch: str) -> str:
    import io as _io
    buf = _io.BytesIO()
    grp = store._file.get(f"batches/{batch}/wave_stats")
    if grp is None:
        return ""

    items: list[tuple[str, bytes]] = []

    def _visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset):
            items.append((name, obj[()].tobytes()))

    grp.visititems(_visit)
    items.sort(key=lambda x: x[0])
    for name, raw in items:
        buf.write(name.encode())
        buf.write(raw)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def test_sorted_order_reproducibility(annotation, tmp_path):
    def _make_store(path, batch_order: list[str]) -> str:
        store = HDF5ResultStore.open(path, "w")
        for b in batch_order:
            store.write_frame(b, 0, _make_eta(0.0), {"status": "ok"})
            store.write_frame(b, 1, _make_eta(0.1), {"status": "ok"})
            store.write_batch(b, data={}, meta={"pixel_per_mm_median": PX_PER_MM})
        compute_and_write_wave_stats(
            annotation=annotation,
            store=store,
            batches=batch_order,
            px_per_mm_by_batch={b: PX_PER_MM for b in batch_order},
            body_polygons_by_batch_frame={b: {} for b in batch_order},
            eta_loader=lambda batch, fid: store._file[f"batches/{batch}/frames/{fid}"][:],
        )
        digest = "".join(
            _sha256_wave_stats_group(store, b) for b in sorted(batch_order)
        )
        store.close()
        return digest

    h1 = _make_store(tmp_path / "order1.h5", ["b1", "b2"])
    h2 = _make_store(tmp_path / "order2.h5", ["b2", "b1"])
    assert h1 == h2
