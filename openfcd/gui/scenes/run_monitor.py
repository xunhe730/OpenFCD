"""Run Monitor scene — shows live pipeline progress."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt

from openfcd.gui import tokens


class RunMonitor(QWidget):
    """Scene widget showing live run progress."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._title = QLabel("Run Monitor")
        self._title.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {tokens.TEXT_PRIMARY};"
        )
        layout.addWidget(self._title)

        self._stage_label = QLabel("Idle")
        self._stage_label.setStyleSheet(
            f"font-size: 12px; color: {tokens.TEXT_SECONDARY};"
            f"font-family: {tokens.FONT_MONO};"
        )
        layout.addWidget(self._stage_label)

        self._progress = QProgressBar()
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background: {tokens.BG_TERTIARY};
            }}
            QProgressBar::chunk {{
                background: {tokens.ACCENT_CLAY};
                border-radius: 3px;
            }}
        """)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet(
            f"font-size: 11px; color: {tokens.TEXT_MUTED};"
            f"font-family: {tokens.FONT_MONO};"
        )
        layout.addWidget(self._frame_label)

        layout.addStretch()

    def update_stage(self, stage: str, progress: float, frame: int | None = None) -> None:
        """Update display with new stage event."""
        self._stage_label.setText(f"Stage: {stage}")
        self._progress.setValue(int(progress * 100))
        if frame is not None:
            self._frame_label.setText(f"frame {frame}")

    def set_idle(self) -> None:
        """Reset to idle state."""
        self._stage_label.setText("Idle")
        self._progress.setValue(0)
        self._frame_label.setText("")
