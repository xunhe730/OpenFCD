"""Unit tests for ImagePickerDialog auto-ref UI controls."""
from __future__ import annotations

from pathlib import Path

import pytest

from openfcd.gui.dialogs.image_picker import ImagePickerDialog

_FAKE_FRAMES = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]


@pytest.fixture()
def dialog(qapp):
    """Construct ImagePickerDialog with three fake paths.

    The thumbnail loader handles missing files gracefully (emits None),
    so fake paths are fine here.
    """
    dlg = ImagePickerDialog(parent=None, frames=_FAKE_FRAMES)
    yield dlg
    dlg.close()


def test_auto_ref_enabled_default(dialog: ImagePickerDialog) -> None:
    """Checkbox is checked by default."""
    assert dialog.auto_ref_enabled is True


def test_auto_ref_reducer_default(dialog: ImagePickerDialog) -> None:
    """Reducer combobox defaults to 'mean'."""
    assert dialog.auto_ref_reducer == "mean"


def test_uncheck_disables_reducer(dialog: ImagePickerDialog) -> None:
    """Unchecking the checkbox disables the reducer combobox."""
    dialog._auto_ref_cb.setChecked(False)
    assert not dialog._reducer_cb.isEnabled()


def test_recheck_enables_reducer(dialog: ImagePickerDialog) -> None:
    """Re-checking the checkbox re-enables the reducer combobox."""
    dialog._auto_ref_cb.setChecked(False)
    dialog._auto_ref_cb.setChecked(True)
    assert dialog._reducer_cb.isEnabled()


def test_reducer_property_reflects_combobox(dialog: ImagePickerDialog) -> None:
    """auto_ref_reducer tracks the combobox selection."""
    dialog._reducer_cb.setCurrentText("median")
    assert dialog.auto_ref_reducer == "median"
