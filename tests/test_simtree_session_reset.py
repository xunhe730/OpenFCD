"""SimTree must drop stale disabled-frame state when a new session loads."""
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from openfcd.gui.panels.sim_tree import SimTree


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_populate_from_session_resets_disabled_indices(qapp):
    tree = SimTree()
    frames_a = [Path(f"/tmp/A{i:04d}.png") for i in range(4)]
    tree.populate_from_session("A", frames_a, ref_index=-1)
    tree.set_frame_disabled(1, True)
    tree.set_frame_disabled(2, True)
    assert tree.disabled_indices == frozenset({1, 2})

    # Simulate "click New" → re-populate with a fresh frame list.
    frames_b = [Path(f"/tmp/B{i:04d}.png") for i in range(3)]
    tree.populate_from_session("B", frames_b, ref_index=-1)
    assert tree.disabled_indices == frozenset()


def test_populate_skeleton_resets_disabled_indices(qapp):
    tree = SimTree()
    tree.populate_from_session("A", [Path("/tmp/x.png")], ref_index=-1)
    tree.set_frame_disabled(0, True)
    assert tree.disabled_indices == frozenset({0})
    tree.populate("blank")
    assert tree.disabled_indices == frozenset()
