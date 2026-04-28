"""SceneContainer: routes scene type → dedicated view widget."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from openfcd.gui.scenes.scene_view_placeholder import SceneViewPlaceholder


class SceneContainer(QWidget):
    """Center-stack widget that holds all scene view types."""

    _IDX_PLACEHOLDER = 0
    _IDX_ETA_MAP = 1
    _IDX_PROFILE = 2
    _IDX_RMS = 3

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._placeholder = SceneViewPlaceholder()
        self._stack.addWidget(self._placeholder)  # 0

        from openfcd.gui.scenes.eta_map_scene import EtaMapSceneView
        self._eta_map_view = EtaMapSceneView()
        self._stack.addWidget(self._eta_map_view)  # 1

        from openfcd.gui.scenes.profile_scene import ProfileSceneView
        self._profile_view = ProfileSceneView()
        self._stack.addWidget(self._profile_view)  # 2

        from openfcd.gui.scenes.rms_scene import RmsSceneView
        self._rms_view = RmsSceneView()
        self._rms_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._stack.addWidget(self._rms_view)  # 3

    def show_scene(self, spec, project_path: Path) -> None:
        from openfcd.io.scene import SceneType
        if spec is None:
            self._placeholder.set_scene("—", "—", "No scene selected", 0)
            self._stack.setCurrentIndex(self._IDX_PLACEHOLDER)
            return
        if spec.type == SceneType.ETA_MAP:
            self._eta_map_view.load(spec, project_path)
            self._stack.setCurrentIndex(self._IDX_ETA_MAP)
        elif spec.type == SceneType.PROFILE:
            self._profile_view.load(spec, project_path)
            self._stack.setCurrentIndex(self._IDX_PROFILE)
        elif spec.type == SceneType.RMS:
            self._rms_view.load(spec, project_path)
            self._stack.setCurrentIndex(self._IDX_RMS)
        else:
            self._placeholder.set_scene(spec.id, spec.type, spec.name, len(spec.frame_indices))
            self._stack.setCurrentIndex(self._IDX_PLACEHOLDER)
