"""Unit tests for ProfileSceneView render state machine (Tier 1c).

Covers:
- Burst coalescing: 10 rapid _request_render calls → single __show_chart
- Dirty-during-render reschedules correctly
- Page-1 (annotator) resize does NOT trigger chart render
- Dirty rearm cap: after 3 rearms, final consume + logger.error + idle reset
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_view(qapp):
    from openfcd.gui.scenes.profile_scene import ProfileSceneView
    view = ProfileSceneView()
    return view


def _flush_render(view):
    """Stop the debounce timer and execute _do_render synchronously."""
    view._render_timer.stop()
    view._do_render()


# ── Test 1: burst coalesces to a single __show_chart call ───────────────────

def test_request_render_coalesces_burst(qapp):
    """10 rapid _request_render() calls → __show_chart called exactly once."""
    view = _make_view(qapp)
    call_count = [0]

    original = view._ProfileSceneView__show_chart

    def counting_show_chart(pos):
        call_count[0] += 1
        # Don't actually render — just record and set state to idle
        view._render_state = "idle"
        view._render_dirty_rearm_count = 0
        import collections, time
        view._render_state_log.append(
            (time.monotonic(), "rendering", "idle", "mock_render_complete")
        )

    view._ProfileSceneView__show_chart = counting_show_chart

    # Burst 10 requests
    for i in range(10):
        view._request_render(i)

    # _pending_idx should have the last value
    assert view._pending_idx == 9
    # Still in scheduled state (timer hasn't fired)
    assert view._render_state == "scheduled"
    assert call_count[0] == 0

    # Now flush: simulate timer firing
    _flush_render(view)

    assert call_count[0] == 1, f"Expected 1 __show_chart call, got {call_count[0]}"
    assert view._pending_idx == 9

    view.close()


# ── Test 2: dirty-during-render reschedules ──────────────────────────────────

def test_dirty_during_render_reschedules(qapp):
    """A _request_render() called mid-render marks dirty; finally reschedules."""
    view = _make_view(qapp)

    original_show_chart = view._ProfileSceneView__show_chart

    call_count = [0]

    def intercepted_show_chart(pos):
        call_count[0] += 1
        if call_count[0] == 1:
            # Simulate a new request arriving during rendering
            # At this point _render_state is "rendering"
            view._request_render(99)
        # Now call the original to let finally block run
        original_show_chart(pos)

    view._ProfileSceneView__show_chart = intercepted_show_chart

    # Prime to scheduled state
    view._request_render(0)
    # Fire timer
    _flush_render(view)

    # After the first __show_chart call, dirty was set → finally should have
    # rearmed the timer (state = scheduled) OR (if rearm cap hit) idle.
    # Since rearm_count starts at 0, first dirty → scheduled
    log_entries = list(view._render_state_log)
    transitions_from_rendering = [
        (f, t, trigger)
        for (_, f, t, trigger) in log_entries
        if f == "rendering"
    ]

    # There should be a rendering→rendering (dirty branch log) OR
    # rendering→scheduled (dirty_rearm branch)
    dirty_branch_logged = any(
        trigger == "request_render_dirty"
        for (_, _, trigger) in transitions_from_rendering
        if _ == "rendering"
    )
    # The log entry for dirty request
    all_triggers = [trigger for (_, _, _, trigger) in log_entries]
    assert "request_render_dirty" in all_triggers, (
        f"Expected 'request_render_dirty' in state log; got: {all_triggers}"
    )

    # Final state should be scheduled (rearmed) or idle (cap hit)
    assert view._render_state in {"scheduled", "idle"}, (
        f"Expected scheduled or idle after dirty rearm; got {view._render_state!r}"
    )

    view.close()


# ── Test 3: resize on page-1 (annotator) does NOT call __show_chart ─────────

def test_resize_during_annotator_does_not_render_chart(qapp):
    """While on page-1 (annotator), resizeEvent must NOT invoke __show_chart."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QResizeEvent

    view = _make_view(qapp)
    call_count = [0]

    def counting_show_chart(pos):
        call_count[0] += 1

    view._ProfileSceneView__show_chart = counting_show_chart

    # Force page-1 (annotator)
    view._mode_stack.setCurrentIndex(1)
    assert view._mode_stack.currentIndex() == 1

    # Emit resize event
    old_size = view.size()
    new_size = QSize(old_size.width() + 20, old_size.height() + 20)
    ev = QResizeEvent(new_size, old_size)
    view.resizeEvent(ev)

    # Even if a render was somehow scheduled, flush it
    if view._render_state == "scheduled":
        _flush_render(view)

    assert call_count[0] == 0, (
        f"__show_chart should NOT be called on page-1 resize; called {call_count[0]} time(s)"
    )
    assert view._render_state in {"idle", "scheduled"}, (
        f"Expected idle or scheduled state; got {view._render_state!r}"
    )

    view.close()


# ── Test 4: dirty rearm cap ──────────────────────────────────────────────────

def test_dirty_rearm_cap(qapp, caplog):
    """When _render_dirty_rearm_count reaches 3: final consume, logger.error, idle, reset."""
    view = _make_view(qapp)

    call_count = [0]
    original_show_chart = view._ProfileSceneView__show_chart

    def intercepted_show_chart(pos):
        call_count[0] += 1
        # Keep marking dirty until the cap hits to force recursion
        # The rearm cap is triggered in the finally block when count >= 3
        # We simulate: every call (except the cap-triggered final one) sets dirty
        current_rearm = view._render_dirty_rearm_count
        if current_rearm < 3:
            # Re-inject a dirty request mid-render
            # Since state is "rendering", _request_render sets dirty=True
            view._pending_render_dirty = True
            view._pending_idx = pos + 1
        # Call original to let finally block run
        original_show_chart.__wrapped__(pos) if hasattr(original_show_chart, '__wrapped__') else None

    # Instead of wrapping, we force the state directly:
    # Set rearm count to 3 so the NEXT render immediately hits the cap.
    # The cap path: sees dirty=True + count>=3 → calls __show_chart(final_pos) recursively,
    # then logs error, sets idle, resets count.

    # Reset state
    view._render_state = "idle"
    view._render_dirty_rearm_count = 3  # pre-set to cap
    view._pending_render_dirty = True
    view._pending_idx = 42
    # state must be "rendering" for __show_chart's assert to hold
    view._render_state = "rendering"

    with caplog.at_level(logging.ERROR, logger="openfcd.gui.scenes.profile_scene"):
        # Directly call __show_chart — cap path will fire immediately in finally
        try:
            view._ProfileSceneView__show_chart(42)
        except Exception:
            pass  # We expect no exception but guard anyway

    # (a) Verify the extra __show_chart call happened (recursive final consume)
    # The cap code: calls self.__show_chart(final_pos) recursively
    # The recursive call increments nothing — it sees dirty=False + rearm reset
    # We check state log for "rearm_cap_final_consume"
    log_triggers = [t for (_, _, _, t) in view._render_state_log]
    assert "rearm_cap_final_consume" in log_triggers, (
        f"Expected 'rearm_cap_final_consume' in state log; got: {log_triggers}"
    )

    # (b) logger.error called with "render rearm cap reached"
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("render rearm cap reached" in str(m) for m in error_messages), (
        f"Expected 'render rearm cap reached' in error log; got: {error_messages}"
    )

    # (c) Final state = idle, rearm reset to 0
    assert view._render_state == "idle", (
        f"Expected idle after cap; got {view._render_state!r}"
    )
    assert view._render_dirty_rearm_count == 0, (
        f"Expected rearm count reset to 0; got {view._render_dirty_rearm_count}"
    )

    view.close()
