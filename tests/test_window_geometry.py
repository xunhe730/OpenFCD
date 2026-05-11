"""Tests for MainWindow window geometry memory — AC-1 through AC-8.

All Qt-widget tests rely on ``QT_QPA_PLATFORM=offscreen`` set in conftest.py
and the session-scoped ``qapp`` fixture.  Pure-UserPrefs tests need no Qt at all.
"""
from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from openfcd.gui.preferences import UserPrefs, reset_prefs_singleton


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_window(prefs: UserPrefs, monkeypatch):
    """Instantiate MainWindow with injected prefs (patches module-level get_prefs)."""
    import openfcd.gui.mainwindow as mw_mod

    monkeypatch.setattr(mw_mod, "get_prefs", lambda: prefs)
    reset_prefs_singleton()
    from openfcd.gui.mainwindow import MainWindow  # pulls from cached module

    return MainWindow()


def _available(win) -> "QRect":  # noqa: F821
    from PyQt6.QtGui import QGuiApplication

    return (win.screen() or QGuiApplication.primaryScreen()).availableGeometry()


# ── Pure UserPrefs tests (no Qt widgets required) ─────────────────────────────

def test_userprefs_geometry_round_trip() -> None:
    """Write arbitrary bytes, read them back unchanged — pure dict backend."""
    prefs = UserPrefs(backend={})
    data = b"\x01\x02\x03\xfe\xff"
    prefs.window_geometry = data
    assert prefs.window_geometry == data


def test_userprefs_reset_clears_geometry() -> None:
    """AC-8: reset() must restore all three geometry fields to empty bytes."""
    prefs = UserPrefs(backend={})
    prefs.window_geometry = b"some bytes"
    prefs.window_state = b"more bytes"
    prefs.splitter_state = b"splitter"

    prefs.reset()

    assert prefs.window_geometry == b""
    assert prefs.window_state == b""
    assert prefs.splitter_state == b""


# ── MainWindow geometry tests (require offscreen QApplication) ─────────────────

def test_first_launch_fits_screen(qapp, monkeypatch) -> None:
    """AC-1: Fresh prefs → window ≤ 90 % of available screen AND ≤ 1400×900."""
    prefs = UserPrefs(backend={})  # window_geometry == b""
    win = _make_window(prefs, monkeypatch)
    try:
        avail = _available(win)
        g = win.geometry()

        assert g.width() <= 1400
        assert g.height() <= 900

        # Skip proportional check only when offscreen platform reports a
        # degenerate (zero) screen — happens on some CI environments.
        if avail.width() > 0 and avail.height() > 0:
            assert g.width() <= int(avail.width() * 0.9) + 1, (
                f"width {g.width()} > 90% of avail.width {avail.width()}"
            )
            assert g.height() <= int(avail.height() * 0.9) + 1, (
                f"height {g.height()} > 90% of avail.height {avail.height()}"
            )
    finally:
        win.close()


def test_first_launch_centered(qapp, monkeypatch) -> None:
    """AC-2: First launch centers the window within availableGeometry (±2 px)."""
    prefs = UserPrefs(backend={})
    win = _make_window(prefs, monkeypatch)
    try:
        avail = _available(win)
        if avail.width() <= 0 or avail.height() <= 0:
            pytest.skip("Offscreen platform reports degenerate screen size — skip centering check")

        g = win.geometry()
        win_cx = g.x() + g.width() // 2
        win_cy = g.y() + g.height() // 2
        avail_cx = avail.x() + avail.width() // 2
        avail_cy = avail.y() + avail.height() // 2

        assert abs(win_cx - avail_cx) <= 2, (
            f"Center X mismatch: window={win_cx}, screen={avail_cx}"
        )
        assert abs(win_cy - avail_cy) <= 2, (
            f"Center Y mismatch: window={win_cy}, screen={avail_cy}"
        )
    finally:
        win.close()


def test_geometry_persisted_on_close(qapp, monkeypatch) -> None:
    """AC-3: After close, prefs.window_geometry / window_state / splitter_state are non-empty."""
    backend: dict = {}
    prefs = UserPrefs(backend=backend)
    win = _make_window(prefs, monkeypatch)
    try:
        QApplication.processEvents()
        # Simulate non-dirty close (no session open, so event.accept() → _save_window_state)
        win._save_window_state()
    finally:
        win.close()

    assert prefs.window_geometry != b"", "window_geometry should be non-empty after save"
    assert prefs.window_state != b"", "window_state should be non-empty after save"
    assert prefs.splitter_state != b"", "splitter_state should be non-empty after save"


def test_geometry_round_trip(qapp, monkeypatch) -> None:
    """AC-4: Saved geometry is faithfully restored in a fresh window (±2 px).

    We pick target dimensions as 60 % of the available screen so the clamp
    in _clamp_to_available_screen never fires and the round-trip is clean.
    """
    backend: dict = {}

    # First window — query available screen, derive safe target size, save
    prefs1 = UserPrefs(backend=backend)
    win1 = _make_window(prefs1, monkeypatch)
    try:
        avail = _available(win1)
        if avail.width() <= 0 or avail.height() <= 0:
            pytest.skip("Offscreen platform has degenerate screen size")
        target_w = int(avail.width() * 0.6)
        target_h = int(avail.height() * 0.6)
        win1.resize(target_w, target_h)
        win1.move(avail.x() + 20, avail.y() + 20)
        QApplication.processEvents()
        win1._save_window_state()
    finally:
        win1.close()

    assert prefs1.window_geometry != b"", "geometry bytes must be non-empty after save"

    # Second window — should restore the exact saved size (no clamping expected)
    prefs2 = UserPrefs(backend=backend)
    win2 = _make_window(prefs2, monkeypatch)
    try:
        QApplication.processEvents()
        g = win2.geometry()
        assert abs(g.width() - target_w) <= 2, f"Width {g.width()} != {target_w} (±2)"
        assert abs(g.height() - target_h) <= 2, f"Height {g.height()} != {target_h} (±2)"
    finally:
        win2.close()


def test_clamp_offscreen_geometry(qapp, monkeypatch) -> None:
    """AC-5: Geometry saved at (5000, 5000) is clamped back into availableGeometry on restore."""
    backend: dict = {}

    # Build a valid geometry blob by positioning a throwaway window offscreen
    prefs_src = UserPrefs(backend=backend)
    win_src = _make_window(prefs_src, monkeypatch)
    try:
        win_src.move(5000, 5000)
        win_src.resize(800, 600)
        QApplication.processEvents()
        # Write the offscreen geometry directly to the shared backend
        prefs_src.window_geometry = bytes(win_src.saveGeometry())
    finally:
        win_src.close()

    # Restore into a fresh window — _apply_initial_geometry + _clamp_to_available_screen
    prefs_dest = UserPrefs(backend=backend)
    win_dest = _make_window(prefs_dest, monkeypatch)
    try:
        QApplication.processEvents()
        avail = _available(win_dest)

        if avail.width() <= 0 or avail.height() <= 0:
            pytest.skip("Offscreen platform has degenerate screen dimensions")

        g = win_dest.geometry()
        assert g.x() >= avail.x(), f"Left edge {g.x()} < avail left {avail.x()}"
        assert g.y() >= avail.y(), f"Top edge {g.y()} < avail top {avail.y()}"
        assert g.right() <= avail.right() + 1, (
            f"Right edge {g.right()} > avail right {avail.right()}"
        )
        assert g.bottom() <= avail.bottom() + 1, (
            f"Bottom edge {g.bottom()} > avail bottom {avail.bottom()}"
        )
    finally:
        win_dest.close()


def test_corrupt_bytes_fallback(qapp, monkeypatch) -> None:
    """AC-6: Corrupt geometry bytes → no crash, window falls back to first-launch sizing."""
    prefs = UserPrefs(backend={})
    prefs.window_geometry = b"garbage_not_a_valid_qbytearray"

    win = _make_window(prefs, monkeypatch)
    try:
        QApplication.processEvents()
        g = win.geometry()
        # Must not exceed first-launch caps
        assert g.width() <= 1400, f"Width {g.width()} exceeds first-launch cap 1400"
        assert g.height() <= 900, f"Height {g.height()} exceeds first-launch cap 900"
    finally:
        win.close()


def test_no_resize_after_stack_switch(qapp, monkeypatch) -> None:
    """AC-7: Switching _center_stack to SceneContainer (index 3) must not change window geometry."""
    prefs = UserPrefs(backend={})
    win = _make_window(prefs, monkeypatch)
    try:
        QApplication.processEvents()
        before = win.geometry()

        # Switch to SceneContainer (index 3) — was the resize offender before CompactCanvas
        win._center_stack.setCurrentIndex(3)
        QApplication.processEvents()

        after = win.geometry()
        assert abs(after.width() - before.width()) <= 1, (
            f"Width changed after stack switch: {before.width()} → {after.width()}"
        )
        assert abs(after.height() - before.height()) <= 1, (
            f"Height changed after stack switch: {before.height()} → {after.height()}"
        )
    finally:
        win.close()
