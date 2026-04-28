import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from openfcd.gui.dialogs.frame_picker import FramePickerDialog

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])

def _make(qapp, n=10, preselected=None):
    frames = [f"frame_{i:04d}.jpg" for i in range(n)]
    return FramePickerDialog(run_exists=True, frames=frames, preselected=preselected)

def test_select_all(qapp):
    d = _make(qapp)
    d._select_all()
    assert d.selected_indices() == list(range(10))

def test_invert(qapp):
    d = _make(qapp, preselected=[0, 1, 2])
    d._invert()
    assert 0 not in d.selected_indices()
    assert 3 in d.selected_indices()

def test_stride(qapp, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **kw: (3, True)))
    d = _make(qapp, n=12)
    d._stride()
    assert d.selected_indices() == [0, 3, 6, 9]

def test_range_select(qapp, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **kw: ("2-5", True)))
    d = _make(qapp, n=10)
    d._range_select()
    assert d.selected_indices() == [2, 3, 4, 5]

def test_count_label(qapp):
    d = _make(qapp, preselected=[0, 1])
    assert "2" in d._count_lbl.text()
    assert "10" in d._count_lbl.text()
