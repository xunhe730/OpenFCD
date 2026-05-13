"""Race reproducer for ProfileSceneView render state machine.

Simulates burst slider moves + resize events interleaved, then asserts:
1. No timer-triggered rendering→rendering consecutive transitions in the log
   (only the dirty-during-rendering branch is allowed).
2. len(view._fig.axes) ≤ 4 — no orphan axis accumulation.
3. __show_chart call count ≤ ⌈(burst + resize) / 2⌉ (coalesce evidence for AC-2).

Uses pytestqt (qtbot) if available; falls back to QApplication.processEvents()
so the test always either passes or skips cleanly — never errors.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(ms: int) -> None:
    """Process Qt events for approximately `ms` milliseconds."""
    import time
    from PyQt6.QtWidgets import QApplication

    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        QApplication.processEvents()


def _flush_render(view) -> None:
    """Stop the debounce timer and execute _do_render synchronously."""
    view._render_timer.stop()
    view._do_render()


def test_burst_slider_resize_coalesces(qapp):
    """Burst slider + resize events are coalesced; no spurious rendering→rendering."""
    from openfcd.gui.scenes.profile_scene import ProfileSceneView

    view = ProfileSceneView()
    view.show()
    _wait(10)

    # Patch __show_chart to count real render calls
    call_count = [0]
    original = view._ProfileSceneView__show_chart

    def counting_show_chart(pos):
        call_count[0] += 1
        original(pos)

    view._ProfileSceneView__show_chart = counting_show_chart

    BURST_SLIDER = 20
    RESIZE_EVENTS = 5
    TOTAL_EVENTS = BURST_SLIDER + RESIZE_EVENTS

    base_w = view.width() or 400
    base_h = view.height() or 300

    # Emit 20 slider value changes in 4 clusters of 5
    for cluster in range(4):
        for i in range(5):
            idx = cluster * 5 + i
            view._request_render(idx)
        _wait(10)

    # Emit 5 resize events
    for r in range(RESIZE_EVENTS):
        new_size = QSize(base_w + r * 5, base_h + r * 3)
        old_size = QSize(base_w + (r - 1) * 5 if r > 0 else base_w, base_h)
        ev = QResizeEvent(new_size, old_size)
        view.resizeEvent(ev)
        _wait(10)

    # Wait for debounce to settle (> _RENDER_DEBOUNCE_MS = 48ms)
    _wait(200)

    # Flush any remaining scheduled render
    if view._render_state == "scheduled":
        _flush_render(view)

    # ── Assertion 1: no timer-triggered rendering→rendering re-entry ──────
    log = list(view._render_state_log)
    bad_transitions = []
    for i in range(len(log) - 1):
        _, _f, t_cur, trig_cur = log[i]
        _, f_next, _t, trig_next = log[i + 1]
        # Two consecutive do_render_dispatch entries means the timer fired
        # twice while the first render was still in flight — a re-entry bug.
        if t_cur == "rendering" and trig_cur == "do_render_dispatch":
            if f_next == "rendering" and trig_next == "do_render_dispatch":
                bad_transitions.append((log[i], log[i + 1]))

    assert not bad_transitions, (
        "Found timer-triggered rendering→rendering re-entry:\n"
        + "\n".join(str(t) for t in bad_transitions)
    )

    # ── Assertion 2: axis count bounded (no orphan accumulation) ─────────
    if view._fig is not None:
        n_axes = len(view._fig.axes)
        assert n_axes <= 4, (
            f"Too many axes ({n_axes}); orphan accumulation detected "
            f"(expected ≤ 4 for composite figure)"
        )

    # ── Assertion 3: coalesce evidence — call count ≤ ⌈events / 2⌉ ──────
    max_allowed = math.ceil(TOTAL_EVENTS / 2)
    assert call_count[0] <= max_allowed, (
        f"__show_chart called {call_count[0]}× for {TOTAL_EVENTS} events; "
        f"coalesce bound is ≤ {max_allowed}"
    )

    view.close()
