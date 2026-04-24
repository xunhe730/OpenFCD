"""
Golden regression tests for openfcd.core (T2 gate).

All tests are skipped until openfcd.core.fcd and openfcd.core.polygon_mask
are importable (i.e. T2 is complete).

Key facts about the golden data:
- Batch dir: BOS/output/20hz_s2/batch_2000606-2000746/
- Per-frame golden: frames/Img*.npy  (contain NaN — use equal_nan=True)
- Aggregates: eta_mean.npy, eta_rms.npy  (also contain NaN)
- Reference image: BOS/DATA/_ref_20hz_s2.jpg  (from condition_batch.json)
- Raw frames: BOS/DATA/20Hz/s2/Img*.jpg
"""
import sys
import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T2 gate: skip the entire module if openfcd.core is not yet implemented
# ---------------------------------------------------------------------------

def _can_import_core() -> bool:
    try:
        from openfcd.core import fcd, polygon_mask  # noqa: F401
        return True
    except ImportError:
        return False


_CORE_READY = _can_import_core()
_SKIP_REASON = "openfcd.core not implemented (T2 pending)"

# Paths — set OPENFCD_BOS_ROOT to enable golden tests; skipped otherwise.
import os as _os
_BOS_ROOT_STR = _os.environ.get("OPENFCD_BOS_ROOT")
if _BOS_ROOT_STR is None:
    import pytest as _pt
    _pt.skip("OPENFCD_BOS_ROOT not set", allow_module_level=True)
_BOS_ROOT = pathlib.Path(_BOS_ROOT_STR)
_BATCH_DIR = _BOS_ROOT / "output" / "20hz_s2" / "batch_2000606-2000746"
_FRAMES_DIR = _BATCH_DIR / "frames"
_DATA_DIR = _BOS_ROOT / "DATA" / "20Hz" / "s2"
_REF_IMAGE = _BOS_ROOT / "DATA" / "_ref_20hz_s2.jpg"

# Fixed probe frame
_PROBE_FRAME = "Img2000606"


# ---------------------------------------------------------------------------
# Helper: run the legacy BOS pipeline for one frame
# ---------------------------------------------------------------------------

def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _run_legacy_single_frame(ref_path: pathlib.Path, frame_path: pathlib.Path) -> np.ndarray:
    """
    Dynamically import the legacy BOS fcd_core / fcd_run and compute eta for
    one frame.  Returns an ndarray with the same shape as the golden .npy.

    This function is only called when _CORE_READY is True (T2 complete),
    so we can assume the legacy BOS path is available at that point.
    """
    bos_root = str(_BOS_ROOT)
    bos_pkg = str(_BOS_ROOT / "bos")
    for p in (bos_root, bos_pkg):
        if p not in sys.path:
            sys.path.insert(0, p)

    # The exact import depends on the legacy BOS structure; adjust when T2
    # lands and the real API is known.
    import fcd_core  # type: ignore
    import cv2  # type: ignore

    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE).astype(float)
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE).astype(float)
    eta = fcd_core.compute_eta(ref, frame)
    return eta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.skipif(not _CORE_READY, reason=_SKIP_REASON)
def test_single_frame_byte_equal(data_dir, golden_dir):
    """New openfcd.core result matches legacy BOS output for one probe frame."""
    from openfcd.core import fcd  # noqa: F401

    frame_jpg = data_dir / f"{_PROBE_FRAME}.jpg"
    if not frame_jpg.exists():
        pytest.skip(f"Probe frame not found: {frame_jpg}")

    golden_npy = _FRAMES_DIR / f"{_PROBE_FRAME}.npy"
    if not golden_npy.exists():
        pytest.skip(f"Golden frame not found: {golden_npy}")

    eta_golden = np.load(golden_npy)

    # Run new pipeline
    eta_new = fcd.compute_eta_from_images(_REF_IMAGE, frame_jpg)

    np.testing.assert_allclose(
        eta_new, eta_golden, atol=1e-6, equal_nan=True,
        err_msg=f"Single-frame mismatch for {_PROBE_FRAME}",
    )


@pytest.mark.golden
@pytest.mark.skipif(not _CORE_READY, reason=_SKIP_REASON)
@pytest.mark.skipif(not _has_cv2(), reason="cv2 not installed — skip legacy comparison")
def test_single_frame_vs_legacy(data_dir, golden_dir):
    """openfcd.core output is byte-equal to legacy BOS output for one frame."""
    from openfcd.core import fcd  # noqa: F401

    frame_jpg = data_dir / f"{_PROBE_FRAME}.jpg"
    if not frame_jpg.exists():
        pytest.skip(f"Probe frame not found: {frame_jpg}")

    eta_old = _run_legacy_single_frame(_REF_IMAGE, frame_jpg)
    eta_new = fcd.compute_eta_from_images(_REF_IMAGE, frame_jpg)

    assert np.allclose(eta_old, eta_new, atol=1e-6, equal_nan=True), (
        "openfcd.core diverges from legacy BOS for single frame"
    )


@pytest.mark.golden
@pytest.mark.skipif(not _CORE_READY, reason=_SKIP_REASON)
def test_all_frames_vs_golden(data_dir, golden_dir):
    """Every frame in batch_2000606-2000746/frames/ matches the golden .npy."""
    from openfcd.core import fcd  # noqa: F401

    frame_npys = sorted(_FRAMES_DIR.glob("Img*.npy"))
    if not frame_npys:
        pytest.skip(f"No golden frame .npy files found in {_FRAMES_DIR}")

    mismatches = []
    for golden_npy in frame_npys:
        stem = golden_npy.stem          # e.g. "Img2000606"
        frame_jpg = data_dir / f"{stem}.jpg"
        if not frame_jpg.exists():
            continue                    # raw frame absent → skip this frame

        eta_golden = np.load(golden_npy)
        eta_new = fcd.compute_eta_from_images(_REF_IMAGE, frame_jpg)

        if not np.allclose(eta_new, eta_golden, atol=1e-6, equal_nan=True):
            mismatches.append(stem)

    assert not mismatches, (
        f"{len(mismatches)} frame(s) differ from golden: {mismatches[:5]}..."
    )


@pytest.mark.golden
@pytest.mark.skipif(not _CORE_READY, reason=_SKIP_REASON)
def test_aggregate_eta_mean(golden_dir):
    """eta_mean.npy produced by openfcd.core matches the golden aggregate."""
    from openfcd.core import fcd  # noqa: F401

    golden_mean = np.load(_BATCH_DIR / "eta_mean.npy")

    # The new pipeline must expose a way to compute the per-batch mean.
    # Adjust the call signature when T2 is implemented.
    eta_mean_new = fcd.compute_batch_mean(_FRAMES_DIR)

    np.testing.assert_allclose(
        eta_mean_new, golden_mean, atol=1e-6, equal_nan=True,
        err_msg="eta_mean mismatch vs golden",
    )


@pytest.mark.golden
@pytest.mark.skipif(not _CORE_READY, reason=_SKIP_REASON)
def test_aggregate_eta_rms(golden_dir):
    """eta_rms.npy produced by openfcd.core matches the golden aggregate."""
    from openfcd.core import fcd  # noqa: F401

    golden_rms = np.load(_BATCH_DIR / "eta_rms.npy")

    eta_rms_new = fcd.compute_batch_rms(_FRAMES_DIR)

    np.testing.assert_allclose(
        eta_rms_new, golden_rms, atol=1e-6, equal_nan=True,
        err_msg="eta_rms mismatch vs golden",
    )
