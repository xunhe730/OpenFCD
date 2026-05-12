"""Tests for HDF5ResultStore.write_wave_stats (v2 layout)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from openfcd.io.result import HDF5ResultStore


# ── helpers ──────────────────────────────────────────────────────────────────


def _open_write(tmp_path: Path, name: str = "result.h5") -> tuple[HDF5ResultStore, Path]:
    p = tmp_path / name
    return HDF5ResultStore.open(p, "w"), p


def _make_segment(
    idx: int,
    s_lo: float = 0.0,
    s_hi: float = 10.0,
    label: str = "",
    color: str = "#1f77b4",
    visible: bool = True,
    wavelengths=None,
    wavenumbers=None,
    peaks=None,
    troughs=None,
    heights=None,
) -> dict:
    return {
        "segment_idx": idx,
        "s_lo_mm": s_lo,
        "s_hi_mm": s_hi,
        "label": label,
        "color": color,
        "visible": visible,
        "wavelength_mm": np.asarray(wavelengths if wavelengths is not None else [1.0]),
        "wavenumber_per_mm": np.asarray(wavenumbers if wavenumbers is not None else [1.0]),
        "peaks": peaks or {},
        "troughs": troughs or {},
        "heights": heights or {},
    }


# ── basic round-trip ─────────────────────────────────────────────────────────


class TestWriteWaveStatsBasic:
    def test_single_segment_groups_exist(self, tmp_path):
        store, path = _open_write(tmp_path)
        seg = _make_segment(
            0, s_lo=1.0, s_hi=10.0, label="A", color="#ff0000",
            wavelengths=[5.0, 5.1],
            wavenumbers=[0.2, 0.196],
            peaks={0: np.array([[1.0, 0.5]]), 1: np.array([[2.0, 0.6]])},
            troughs={0: np.array([[1.5, -0.5]])},
            heights={0: np.array([1.0]), 1: np.array([1.1, 1.2])},
        )
        store.write_wave_stats("b0", [seg], {"schema_version": 2})
        store.close()

        with h5py.File(path, "r") as f:
            assert "batches/b0/wave_stats/segments/0000" in f
            assert "batches/b0/wave_stats/segments/0000/wavelength_mm" in f
            assert "batches/b0/wave_stats/segments/0000/wavenumber_per_mm" in f
            assert "batches/b0/wave_stats/segments/0000/peaks/0" in f
            assert "batches/b0/wave_stats/segments/0000/peaks/1" in f
            assert "batches/b0/wave_stats/segments/0000/troughs/0" in f
            assert "batches/b0/wave_stats/segments/0000/heights/0" in f
            assert "batches/b0/wave_stats/segments/0000/heights/1" in f

    def test_segment_attrs(self, tmp_path):
        store, path = _open_write(tmp_path)
        seg = _make_segment(
            3, s_lo=2.5, s_hi=18.0, label="alpha", color="#abcdef", visible=False
        )
        store.write_wave_stats("b0", [seg], {})
        store.close()

        with h5py.File(path, "r") as f:
            attrs = dict(f["batches/b0/wave_stats/segments/0003"].attrs)
        assert attrs["s_lo_mm"] == pytest.approx(2.5)
        assert attrs["s_hi_mm"] == pytest.approx(18.0)
        assert attrs["label"] == "alpha"
        assert attrs["color"] == "#abcdef"
        assert int(attrs["visible"]) == 0

    def test_top_level_attrs_written(self, tmp_path):
        store, path = _open_write(tmp_path)
        attrs = {
            "schema_version": 2,
            "prominence_k": 0.3,
            "ds_mm": 0.1,
            "n_frames": 5,
            "n_segments": 2,
        }
        store.write_wave_stats("b0", [], attrs)
        store.close()

        with h5py.File(path, "r") as f:
            stored = dict(f["batches/b0/wave_stats"].attrs)
        assert int(stored["schema_version"]) == 2
        assert stored["prominence_k"] == pytest.approx(0.3)
        assert int(stored["n_frames"]) == 5
        assert int(stored["n_segments"]) == 2

    def test_segments_none_skips_group(self, tmp_path):
        store, path = _open_write(tmp_path)
        store.write_wave_stats("b0", None, {"marker": "noseg"})
        store.close()

        with h5py.File(path, "r") as f:
            ws = f["batches/b0/wave_stats"]
            assert "segments" not in ws
            assert ws.attrs["marker"] == "noseg"

    def test_three_segments_sorted_by_idx(self, tmp_path):
        store, path = _open_write(tmp_path)
        segs = [
            _make_segment(2, label="b"),
            _make_segment(0, label="a"),
            _make_segment(1, label="c"),
        ]
        store.write_wave_stats("b0", segs, {})
        store.close()

        with h5py.File(path, "r") as f:
            keys = list(f["batches/b0/wave_stats/segments"].keys())
        assert keys == ["0000", "0001", "0002"]


# ── byte-equal regression: insertion order irrelevant ────────────────────────


class TestWriteWaveStatsSortedOrder:
    def test_insertion_order_irrelevant(self, tmp_path):
        rng = np.random.default_rng(0)
        peaks_data = {i: rng.random((2, 2)) for i in range(4)}

        seg_a = _make_segment(
            0, wavelengths=rng.random(4), wavenumbers=rng.random(4),
            peaks=peaks_data,
        )
        # reverse insertion order
        peaks_rev = {k: peaks_data[k] for k in reversed(list(peaks_data.keys()))}
        seg_b = _make_segment(
            0, wavelengths=seg_a["wavelength_mm"], wavenumbers=seg_a["wavenumber_per_mm"],
            peaks=peaks_rev,
        )

        path_a = tmp_path / "a.h5"
        path_b = tmp_path / "b.h5"
        with HDF5ResultStore.open(path_a, "w") as s:
            s.write_wave_stats("b0", [seg_a], {"schema_version": 2})
        with HDF5ResultStore.open(path_b, "w") as s:
            s.write_wave_stats("b0", [seg_b], {"schema_version": 2})

        def _sha(p):
            buf = hashlib.sha256()
            with h5py.File(p, "r") as f:
                grp = f["batches/b0/wave_stats/segments/0000/peaks"]
                for k in sorted(grp.keys()):
                    buf.update(grp[k][:].tobytes())
            return buf.hexdigest()

        assert _sha(path_a) == _sha(path_b)


# ── dtype coercion ───────────────────────────────────────────────────────────


class TestWriteWaveStatsDtype:
    def test_integer_input_coerced_to_float64(self, tmp_path):
        store, path = _open_write(tmp_path)
        seg = _make_segment(
            0,
            wavelengths=np.array([1, 2, 3], dtype=np.int32),
            wavenumbers=np.array([1, 2, 3], dtype=np.int32),
            heights={0: np.array([1, 2, 3], dtype=np.int32)},
        )
        store.write_wave_stats("b0", [seg], {})
        store.close()

        with h5py.File(path, "r") as f:
            assert f["batches/b0/wave_stats/segments/0000/wavelength_mm"].dtype == np.float64
            assert f["batches/b0/wave_stats/segments/0000/heights/0"].dtype == np.float64


# ── isolation ────────────────────────────────────────────────────────────────


class TestWriteWaveStatsIsolation:
    def test_does_not_corrupt_write_frame(self, tmp_path):
        store, path = _open_write(tmp_path)
        eta = np.ones((4, 4))
        store.write_frame("b0", 0, eta, {"status": "ok"})
        seg = _make_segment(0, peaks={0: np.array([[1.0, 2.0]])})
        store.write_wave_stats("b0", [seg], {})
        store.close()

        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["batches/b0/frames/0"][:], eta)
            assert f["batches/b0/frames/0"].attrs["status"] == "ok"

    def test_does_not_touch_eta_mean(self, tmp_path):
        store, path = _open_write(tmp_path)
        eta_mean = np.full((4, 4), 3.14)
        store.write_batch("b0", {"eta_mean": eta_mean}, {})
        seg = _make_segment(0)
        store.write_wave_stats("b0", [seg], {})
        store.close()

        with h5py.File(path, "r") as f:
            np.testing.assert_array_almost_equal(f["batches/b0/eta_mean"][:], eta_mean)
