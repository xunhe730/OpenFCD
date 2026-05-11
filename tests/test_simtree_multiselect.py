"""Multi-select + batch enable/disable/compute on SimTree IMAGE_FRAME rows."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QTreeWidget

from openfcd.gui.panels.sim_tree import SimTree, NodeType, ROLE_NODE_TYPE


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_tree(qapp, n_frames: int = 5) -> SimTree:
    tree = SimTree()
    frames = [Path(f"/tmp/Img{i:04d}.png") for i in range(n_frames)]
    tree.populate_from_session("proj", frames, ref_index=-1)
    return tree


def _frame_items(tree: SimTree):
    out = []
    root = tree.topLevelItem(0)
    for i in range(root.childCount()):
        section = root.child(i)
        if section.data(0, ROLE_NODE_TYPE) == NodeType.IMAGES:
            for j in range(section.childCount()):
                out.append(section.child(j))
    return out


def test_extended_selection_mode_enabled(qapp):
    tree = _make_tree(qapp)
    assert tree.selectionMode() == QTreeWidget.SelectionMode.ExtendedSelection


def test_selected_frame_indices_filters_non_image_frame(qapp):
    tree = _make_tree(qapp, n_frames=4)
    items = _frame_items(tree)
    assert len(items) == 4
    # Select frames 1 and 3 plus the Images parent (non-IMAGE_FRAME).
    items[1].setSelected(True)
    items[3].setSelected(True)
    root = tree.topLevelItem(0)
    images_parent = next(
        root.child(i) for i in range(root.childCount())
        if root.child(i).data(0, ROLE_NODE_TYPE) == NodeType.IMAGES
    )
    images_parent.setSelected(True)
    assert tree.selected_frame_indices() == [1, 3]


def test_batch_signals_emit_list(qapp):
    tree = _make_tree(qapp)
    captured: dict[str, list] = {"d": [], "e": [], "c": []}
    tree.frames_disabled_requested.connect(lambda xs: captured["d"].extend(xs))
    tree.frames_enabled_requested.connect(lambda xs: captured["e"].extend(xs))
    tree.frames_compute_requested.connect(lambda xs: captured["c"].extend(xs))

    tree.frames_disabled_requested.emit([0, 2, 4])
    tree.frames_enabled_requested.emit([1])
    tree.frames_compute_requested.emit([0, 1, 2])

    assert captured["d"] == [0, 2, 4]
    assert captured["e"] == [1]
    assert captured["c"] == [0, 1, 2]
