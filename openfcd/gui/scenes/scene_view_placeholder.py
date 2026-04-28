from __future__ import annotations
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from openfcd.gui import tokens


class SceneViewPlaceholder(QWidget):
    """Placeholder center panel for a selected Scene (pre-Phase-3)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_type = QLabel("—")
        self._lbl_name = QLabel("—")
        self._lbl_frames = QLabel("—")
        for lbl in (self._lbl_type, self._lbl_name, self._lbl_frames):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self._lbl_type.setStyleSheet(
            f"color:{tokens.ACCENT_CLAY};font-size:11px;font-weight:600;"
            f"letter-spacing:0.08em;text-transform:uppercase;"
        )
        self._lbl_name.setStyleSheet(
            f"color:{tokens.TEXT_PRIMARY};font-size:16px;font-weight:600;padding:4px 0;"
        )
        self._lbl_frames.setStyleSheet(
            f"color:{tokens.TEXT_MUTED};font-size:12px;"
        )

    def set_scene(self, scene_id: str, scene_type: str, name: str, frame_count: int) -> None:
        self._lbl_type.setText(scene_type.upper())
        self._lbl_name.setText(name)
        self._lbl_frames.setText(f"id: {scene_id}  ·  {frame_count} frames")
