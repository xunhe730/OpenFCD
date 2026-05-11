"""AC-B7 — manual golden regression for the Run = N × Compute refactor.

Compares the η frames written by ``ComputeStage`` to a reference
``results.h5`` snapshot from ``main`` for the ``20Hz/s2`` dataset. Prints,
for each frame, whether the eta_mm dataset is byte-equal (NaN-aware) and
whether the ``_valid_crop`` path was active for that frame (inferred from
the per-frame ``scale_correction`` attribute when present).

Usage::

    python scripts/golden_run_compare.py \
        --baseline /path/to/main/runs/<run-id>/results.h5 \
        --candidate /path/to/refactor/runs/<run-id>/results.h5 \
        [--frames 0,7,42]

Pick at least one frame where ``_valid_crop is None`` (no shape-mismatch
between ref and def, no auto_scale_ref reflow) AND one where it is not.
Both must match byte-for-byte; if they differ, document the diff in the PR
description as the parity-bug fix this plan intends.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openfcd.io.result import HDF5ResultStore


def _nan_aware_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    return bool(np.array_equal(a[~nan_a], b[~nan_b]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--frames", default="", help="comma-separated frame ids (default: all common)")
    ap.add_argument("--batch", default="default")
    args = ap.parse_args()

    base = HDF5ResultStore.open(args.baseline, "r")
    cand = HDF5ResultStore.open(args.candidate, "r")
    try:
        base_ids = set(base.list_frames(args.batch))
        cand_ids = set(cand.list_frames(args.batch))
        common = sorted(base_ids & cand_ids)
        if args.frames.strip():
            wanted = {int(x) for x in args.frames.split(",") if x.strip()}
            common = [i for i in common if i in wanted]
        if not common:
            print("no common frames found")
            return 1

        n_equal = 0
        n_diff = 0
        for fid in common:
            a = base.read_frame(args.batch, fid)
            b = cand.read_frame(args.batch, fid)
            attrs_b = cand.read_frame_attrs(args.batch, fid)
            crop_marker = attrs_b.get("scale_correction", "n/a")
            equal = _nan_aware_equal(np.asarray(a), np.asarray(b))
            print(f"frame {fid:>5}: equal={equal} scale_correction={crop_marker}")
            n_equal += int(equal)
            n_diff += int(not equal)

        # Compare summary mean too
        try:
            ma = base.read_summary(args.batch, "eta_mean")
            mb = cand.read_summary(args.batch, "eta_mean")
            print(f"eta_mean: equal={_nan_aware_equal(np.asarray(ma), np.asarray(mb))}")
        except Exception as exc:  # noqa: BLE001
            print(f"eta_mean compare skipped: {exc}")

        print(f"summary: {n_equal} equal, {n_diff} differ")
        return 0 if n_diff == 0 else 2
    finally:
        base.close()
        cand.close()


if __name__ == "__main__":
    raise SystemExit(main())
