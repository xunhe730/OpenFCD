"""Tests for ImagePickerDialog."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication
from openfcd.gui.dialogs.image_picker import ImagePickerDialog


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_image_picker_dialog_construction(tmp_path: Path) -> None:
    """Dialog can be constructed with a list of temporary paths."""
    app = _app()
    frames = [tmp_path / f"img{i:04d}.jpg" for i in range(5)]

    dialog = ImagePickerDialog(frames=frames)

    assert dialog is not None
    assert len(dialog.selected_frames) == 5
    dialog.close()


def test_select_all_checks_every_item(tmp_path: Path) -> None:
    """Select All makes all items checked."""
    app = _app()
    frames = [tmp_path / f"img{i:04d}.jpg" for i in range(8)]

    dialog = ImagePickerDialog(frames=frames)
    # Start with a clean slate: deselect all first
    dialog._deselect_all()
    assert len(dialog.selected_indices()) == 0

    dialog._select_all()
    assert len(dialog.selected_indices()) == 8
    dialog.close()


def test_deselect_all_unchecks_every_item(tmp_path: Path) -> None:
    """Deselect All makes all items unchecked."""
    app = _app()
    frames = [tmp_path / f"img{i:04d}.jpg" for i in range(6)]

    dialog = ImagePickerDialog(frames=frames)
    # All should be checked by default
    assert len(dialog.selected_indices()) == 6

    dialog._deselect_all()
    assert len(dialog.selected_indices()) == 0
    dialog.close()
