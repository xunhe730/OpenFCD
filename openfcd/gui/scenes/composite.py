"""Composite scene widget — user-assembled figure grid."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from openfcd.gui import tokens


class CompositeScene(QWidget):
    """Scene for user-assembled multi-figure composite layouts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Composite")
        title.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {tokens.TEXT_PRIMARY}; "
            f"padding: 8px 16px; background: {tokens.BG_SECONDARY}; "
            f"border-bottom: 1px solid {tokens.BORDER_SUBTLE};"
        )
        layout.addWidget(title)

        placeholder = QLabel("Composite layout — add figures via Scenes menu")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-size: 12px; padding: 40px;"
        )
        layout.addWidget(placeholder)
        layout.addStretch()
