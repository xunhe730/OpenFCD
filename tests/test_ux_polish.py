"""Tests for US-404 UX polish: tooltip, shortcut, reveal signal."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_scene_tooltip_includes_type_and_frames(qapp):
    from openfcd.gui.panels.sim_tree import SimTree
    from openfcd.io.scene import SceneSpec, SceneType
    tree = SimTree()
    spec = SceneSpec(
        id="t1", name="Wake", type=SceneType.ETA_MAP,
        frame_indices=list(range(42)), created_at="2026-04-27T00:00:00Z"
    )
    tree.populate_scenes([spec])
    parent = tree._find_or_create_parent("Scenes")
    item = parent.child(0)
    tip = item.toolTip(0)
    assert "eta_map" in tip.lower() or "ETA_MAP" in tip
    assert "42" in tip


def test_reveal_scene_signal_exists(qapp):
    from openfcd.gui.panels.sim_tree import SimTree
    tree = SimTree()
    assert hasattr(tree, "reveal_scene_requested")


def test_reveal_label_platform_specific():
    from openfcd.gui.panels.sim_tree import SimTree
    label = SimTree._reveal_label()
    assert label in ("Reveal in Finder", "Show in Explorer", "Open Folder")


def test_shortcut_new_eta_exists():
    """Verify _shortcut_new_eta is wired in MainWindow._connect_signals."""
    import inspect
    from openfcd.gui.mainwindow import MainWindow
    source = inspect.getsource(MainWindow._connect_signals)
    assert "_shortcut_new_eta" in source


def test_viz_settings_tooltips_set(qapp):
    from openfcd.gui.panels.properties import VizSettingsPanel
    p = VizSettingsPanel()
    assert p._cmap.toolTip() != ""
    assert p._dpi.toolTip() != ""
