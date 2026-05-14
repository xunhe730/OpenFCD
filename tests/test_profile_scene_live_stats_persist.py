"""Regression: ProfileSceneView.load() must synchronously recompute live
wave-stats on the non-annotation-mode branch.

Without this, the wave-stats table columns (λ / k / Heights / N peaks /
S_γ / S_g / S_cg) appear empty when the user clicks away from Profile and
returns — `_recompute_live_stats` only fires inside the debounced render
path, and on re-entry can bind None when `_compute_live_stats` falls
through any missing-input branch.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

PyQt6 = pytest.importorskip("PyQt6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _make_view(qapp):
    from openfcd.gui.scenes.profile_scene import ProfileSceneView
    return ProfileSceneView()


def test_load_else_branch_invokes_synchronous_recompute(qapp, monkeypatch):
    """The else-branch (no missing profile lines) MUST call
    `_recompute_live_stats` synchronously inside `load()` so the table
    reflects fresh stats before the debounced render fires."""
    view = _make_view(qapp)

    spec = SimpleNamespace(
        frame_indices=[100],
        profile_lines={},
        viz_params={},
        run_id=None,
        name="test",
    )

    # Stub out file I/O and rendering so load() exercises only the
    # control flow we care about.
    monkeypatch.setattr(view, "_load_eta_frames", lambda: None)
    monkeypatch.setattr(view, "_first_missing_line_pos", lambda: None)
    monkeypatch.setattr(view, "_request_render", lambda pos: None)
    monkeypatch.setattr(view, "_sync_table_current_frame", lambda: None)
    # Stub table-panel methods load() invokes so we don't depend on
    # annotation/h5 bindings.
    monkeypatch.setattr(view._table_panel, "set_annotation", lambda ann: None)
    monkeypatch.setattr(view._table_panel, "set_h5_path", lambda p, batch="default": None)

    calls: list[int] = []
    monkeypatch.setattr(view, "_recompute_live_stats", lambda: calls.append(1))

    view.load(spec, Path("/nonexistent.ofcd"))

    assert len(calls) == 1, (
        f"expected exactly one synchronous _recompute_live_stats call in the "
        f"else-branch, got {len(calls)}"
    )


def test_load_annotation_mode_branch_skips_synchronous_recompute(qapp, monkeypatch):
    """The if-branch (some frame missing a profile line) MUST NOT call
    `_recompute_live_stats` synchronously — annotation mode has no line so
    `_compute_live_stats` would return None anyway; the call is wasted work."""
    view = _make_view(qapp)

    spec = SimpleNamespace(
        frame_indices=[100, 101],
        profile_lines={},
        viz_params={},
        run_id=None,
        name="test",
    )

    monkeypatch.setattr(view, "_load_eta_frames", lambda: None)
    # Force the annotation-mode branch by returning a frame position.
    monkeypatch.setattr(view, "_first_missing_line_pos", lambda: 0)
    monkeypatch.setattr(view, "_request_render", lambda pos: None)
    monkeypatch.setattr(view, "_sync_table_current_frame", lambda: None)
    monkeypatch.setattr(view, "_enter_annotation", lambda: None)
    monkeypatch.setattr(view._table_panel, "set_annotation", lambda ann: None)
    monkeypatch.setattr(view._table_panel, "set_h5_path", lambda p, batch="default": None)

    calls: list[int] = []
    monkeypatch.setattr(view, "_recompute_live_stats", lambda: calls.append(1))

    view.load(spec, Path("/nonexistent.ofcd"))

    assert calls == [], (
        f"expected no synchronous _recompute_live_stats call when load() "
        f"enters annotation mode, got {len(calls)} call(s)"
    )
