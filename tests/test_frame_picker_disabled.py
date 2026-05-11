"""FramePickerDialog must hide / lock disabled frames out of selection."""
from __future__ import annotations
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from openfcd.gui.dialogs.frame_picker import FramePickerDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_disabled_frames_excluded_from_preselect(qapp):
    dlg = FramePickerDialog(
        frames=[f"Img{i:04d}.jpg" for i in range(6)],
        preselected=list(range(6)),
        disabled=[1, 4],
    )
    assert sorted(dlg.selected_indices()) == [0, 2, 3, 5]


def test_disabled_rows_not_user_checkable(qapp):
    dlg = FramePickerDialog(
        frames=[f"Img{i:04d}.jpg" for i in range(4)],
        preselected=[0, 1, 2, 3],
        disabled=[2],
    )
    item = dlg._list.item(2)
    assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    # Even calling _set_state cannot toggle it on
    dlg._set_state(2, Qt.CheckState.Checked)
    assert item.checkState() == Qt.CheckState.Unchecked


def test_select_all_skips_disabled(qapp):
    dlg = FramePickerDialog(
        frames=[f"Img{i:04d}.jpg" for i in range(5)],
        preselected=[],
        disabled=[2, 3],
    )
    dlg._select_all()
    assert sorted(dlg.selected_indices()) == [0, 1, 4]


def test_stride_and_range_skip_disabled(qapp):
    dlg = FramePickerDialog(
        frames=[f"Img{i:04d}.jpg" for i in range(10)],
        preselected=[],
        disabled=[0, 4],
    )
    # Stride 2 would normally select 0,2,4,6,8 — 0 and 4 are filtered out.
    for i in range(dlg._list.count()):
        dlg._set_state(i, Qt.CheckState.Checked if i % 2 == 0 else Qt.CheckState.Unchecked)
    assert sorted(dlg.selected_indices()) == [2, 6, 8]
