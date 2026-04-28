import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from openfcd.gui.panels.sim_tree import SimTree
from openfcd.io.scene import SceneSpec, SceneType

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])

def test_populate_scenes_adds_nodes(qapp):
    tree = SimTree()
    specs = [
        SceneSpec(id="s1", name="Wake η", type=SceneType.ETA_MAP, frame_indices=[0, 1]),
        SceneSpec(id="s2", name="RMS map", type=SceneType.RMS, frame_indices=[0, 1, 2]),
    ]
    tree.populate_scenes(specs)
    scenes_parent = tree._find_or_create_parent("Scenes")
    assert scenes_parent.childCount() == 2

def test_new_scene_signal_emitted(qapp):
    tree = SimTree()
    received = []
    tree.new_scene_requested.connect(received.append)
    tree.new_scene_requested.emit("eta_map")
    assert received == ["eta_map"]

def test_delete_signal_emitted(qapp):
    tree = SimTree()
    received = []
    tree.delete_scene_requested.connect(received.append)
    tree.delete_scene_requested.emit("scene-id-123")
    assert received == ["scene-id-123"]
