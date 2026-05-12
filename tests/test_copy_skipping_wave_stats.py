"""Tests for _copy_skipping_wave_stats in openfcd/cli/cmd_postprocess.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from openfcd.cli.cmd_postprocess import _copy_skipping_wave_stats


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _build_synthetic_h5(path: Path) -> None:
    """Write a realistic synthetic h5 with two batches and wave_stats."""
    with h5py.File(path, "w") as f:
        # File-level attrs
        f.attrs["format_version"] = 0
        f.attrs["openfcd_version"] = "0.0.1"

        batches = f.create_group("batches")

        for bname in ("B0", "B1"):
            b = batches.create_group(bname)
            b.attrs["pixel_per_mm_median"] = 10.0
            b.attrs["batch_note"] = f"note-{bname}"

            # frames sub-group
            frames = b.create_group("frames")
            for fid in (0, 1):
                ds = frames.create_dataset(str(fid), data=np.ones((4, 4)))
                ds.attrs["status"] = "ok"

            # summary datasets
            b.create_dataset("eta_mean", data=np.full((4, 4), 1.0))
            b.create_dataset("eta_rms", data=np.full((4, 4), 0.1))

            # wave_stats — must be skipped (v2 layout)
            ws = b.create_group("wave_stats")
            ws.attrs["prominence_k"] = 0.3
            ws.attrs["schema_version"] = 2
            seg = ws.require_group("segments/0000")
            seg.attrs["label"] = "seg0"
            seg.create_dataset("wavelength_mm", data=np.array([10.0, 10.0]))
            seg.create_dataset("wavenumber_per_mm", data=np.array([0.1, 0.1]))
            seg.require_group("peaks").create_dataset("0", data=np.array([[1.0, 2.0]]))

        # Top-level non-batches group (should be copied verbatim)
        meta = f.create_group("meta")
        meta.attrs["extra"] = "info"
        meta.create_dataset("run_ids", data=np.array([1, 2, 3]))


@pytest.fixture()
def synthetic_src(tmp_path) -> Path:
    p = tmp_path / "src.h5"
    _build_synthetic_h5(p)
    return p


@pytest.fixture()
def dst_path(tmp_path) -> Path:
    return tmp_path / "dst.h5"


# ── Tests ────────────────────────────────────────────────────────────────────

class TestCopySkippingWaveStats:
    def test_frame_datasets_present_in_dst(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "batches/B0/frames/0" in f
            assert "batches/B0/frames/1" in f
            assert "batches/B1/frames/0" in f

    def test_eta_mean_and_rms_present(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "batches/B0/eta_mean" in f
            assert "batches/B0/eta_rms" in f
            assert "batches/B1/eta_mean" in f

    def test_wave_stats_absent_in_dst(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "batches/B0/wave_stats" not in f
            assert "batches/B1/wave_stats" not in f

    def test_batch_attrs_preserved(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert f["batches/B0"].attrs["pixel_per_mm_median"] == pytest.approx(10.0)
            assert f["batches/B0"].attrs["batch_note"] == "note-B0"
            assert f["batches/B1"].attrs["batch_note"] == "note-B1"

    def test_file_attrs_preserved(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert f.attrs["format_version"] == 0
            assert f.attrs["openfcd_version"] == "0.0.1"

    def test_non_batches_group_copied_verbatim(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "meta" in f
            assert f["meta"].attrs["extra"] == "info"
            np.testing.assert_array_equal(f["meta/run_ids"][:], [1, 2, 3])

    def test_frame_data_values_unchanged(self, synthetic_src, dst_path):
        with h5py.File(synthetic_src, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            np.testing.assert_array_equal(
                f["batches/B0/frames/0"][:], np.ones((4, 4))
            )
            assert f["batches/B0/frames/0"].attrs["status"] == "ok"

    def test_empty_batches_group(self, tmp_path, dst_path):
        """src with empty batches/ → dst has empty batches group, no crash."""
        src_path = tmp_path / "empty.h5"
        with h5py.File(src_path, "w") as f:
            f.create_group("batches")
            f.attrs["format_version"] = 0
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "batches" in f
            assert list(f["batches"].keys()) == []

    def test_no_batches_key(self, tmp_path, dst_path):
        """src without batches/ → dst also has no batches, no crash."""
        src_path = tmp_path / "nobatches.h5"
        with h5py.File(src_path, "w") as f:
            f.attrs["format_version"] = 0
            other = f.create_group("other")
            other.create_dataset("x", data=np.array([1.0]))
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "batches" not in f
            assert "other" in f
            np.testing.assert_array_equal(f["other/x"][:], [1.0])

    def test_batch_without_wave_stats(self, tmp_path, dst_path):
        """Batch that never had wave_stats is copied cleanly."""
        src_path = tmp_path / "nowavest.h5"
        with h5py.File(src_path, "w") as f:
            b = f.require_group("batches/B0")
            b.create_dataset("eta_mean", data=np.zeros((3, 3)))
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_skipping_wave_stats(src, dst)
        with h5py.File(dst_path, "r") as f:
            assert "batches/B0/eta_mean" in f
            assert "batches/B0/wave_stats" not in f
