"""MainWindow batch disable/enable, ref auto-disable, compute guard.

Filename prefixed ``test_zz_`` so it runs *after* ``test_main_menus``: that
file toggles the global dark-mode palette via ``tokens.set_dark_mode``, which
fires every connection ever registered through ``tokens.on_theme_changed``.
Connections from MainWindow children of earlier-but-now-dead test instances
crash on dispatch, so we keep this module last.

Uses a single module-scoped MainWindow to limit observer leakage.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from openfcd.gui.controllers.session_controller import SessionController
from openfcd.gui.mainwindow import MainWindow


def _write_gray(path: Path, arr: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(arr).save(path)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    return MainWindow()


@pytest.fixture
def project_with_frames(main_window, tmp_path):
    """Wire a fresh project + on-disk frames into the shared MainWindow."""
    controller = SessionController()
    controller.new_project(
        name="batch_test",
        location=tmp_path,
        optical_preset="pattern_below_window",
        pattern_period_mm=1.2,
    )
    img_dir = tmp_path / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(5):
        p = img_dir / f"Img{i:04d}.png"
        _write_gray(p, np.zeros((16, 16), dtype=np.uint8))
        paths.append(p)

    proj = controller.project
    assert proj is not None
    proj.data.frames_dir = str(img_dir)
    proj.data.pattern = "Img*.png"

    main_window._session = controller
    main_window._frames = paths
    main_window._compute_queue = []
    main_window._single_frame_worker = None
    return main_window, proj


def test_batch_disable_writes_sorted_unique_indices(project_with_frames):
    win, proj = project_with_frames
    win._on_disable_frames([3, 1, 1])
    assert proj.data.disabled_frame_indices == [1, 3]


def test_batch_enable_removes_indices(project_with_frames):
    win, proj = project_with_frames
    proj.data.disabled_frame_indices = [0, 1, 2, 3]
    win._on_enable_frames([1, 3])
    assert proj.data.disabled_frame_indices == [0, 2]


def test_set_reference_auto_disables_ref_frame(project_with_frames):
    win, proj = project_with_frames
    win._on_set_reference(1)
    assert 1 in proj.data.disabled_frame_indices
    assert proj.reference.mode == "use_existing"


def test_compute_frame_skips_disabled(project_with_frames):
    win, proj = project_with_frames
    proj.reference.source = "Img0000.png"
    proj.data.disabled_frame_indices = [2]
    win._on_compute_frame(2)
    assert getattr(win, "_single_frame_worker", None) is None


def test_compute_frames_skips_disabled_and_queues_rest(project_with_frames, monkeypatch):
    win, proj = project_with_frames
    proj.reference.source = "Img0000.png"
    proj.data.disabled_frame_indices = [1, 3]
    called: list[int] = []
    monkeypatch.setattr(win, "_on_compute_frame", lambda idx: called.append(idx))
    win._on_compute_frames([0, 1, 2, 3])
    assert called == [0]
    assert win._compute_queue == [2]
