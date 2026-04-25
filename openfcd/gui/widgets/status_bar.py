"""Status bar widget — displays pipeline status information + progress bar."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar

from openfcd.gui import tokens


class StatusBar(QWidget):
    """Bottom status bar with monospaced status items and optional progress bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(30)
        self._items: list[str] = []
        self._labels: list[QLabel] = []
        self._setup_ui()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()
        
    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"StatusBar {{ background: {tokens.BG_DOCK}; border-top: 1px solid {tokens.BORDER_SUBTLE}; }}"
        )
        self._log_label.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-size: 11px; font-family: {tokens.FONT_MONO};"
        )
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {tokens.BG_TERTIARY};
                border: none;
                border-radius: 4px;
                color: {tokens.TEXT_PRIMARY};
                font-size: 10.5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {tokens.ACCENT_CLAY};
                border-radius: 4px;
            }}
        """)
        # re-apply items to trigger label styling rebuild
        if self._items:
            self.set_items(self._items)

    def _setup_ui(self) -> None:
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(10, 0, 10, 0)
        self._layout.setSpacing(0)
        
        self._left_container = QHBoxLayout()
        self._left_container.setSpacing(0)
        self._layout.addLayout(self._left_container)
        
        self._layout.addStretch()

        # Progress bar (hidden by default) — prominent enough to notice during long runs
        self._progress = QProgressBar()
        self._progress.setFixedWidth(240)
        self._progress.setFixedHeight(16)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        self._progress.setVisible(False)
        self._layout.addWidget(self._progress)

        self._layout.addSpacing(8)
        
        self._log_label = QLabel()
        self._layout.addWidget(self._log_label)

    def set_log_text(self, text: str) -> None:
        self._log_label.setText(text)

    def set_items(self, items: list[str]) -> None:
        """Set status bar items (replaces all existing)."""
        self._items = items
        # Clear existing
        for lbl in self._labels:
            self._left_container.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()

        for i, text in enumerate(items):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {tokens.TEXT_SECONDARY}; font-size: 11px; "
                f"font-family: {tokens.FONT_MONO}; padding: 0 10px;"
            )
            self._left_container.addWidget(lbl)
            self._labels.append(lbl)
            if i < len(items) - 1:
                sep = QLabel("│")
                sep.setStyleSheet(f"color: {tokens.BORDER_STRONG}; font-size: 11px;")
                self._left_container.addWidget(sep)
                self._labels.append(sep)

    def show_progress(self, value: int = 0, maximum: int = 100) -> None:
        """Show determinate progress bar."""
        self._progress.setMaximum(maximum)
        self._progress.setValue(value)
        self._progress.setVisible(True)

    def show_busy(self) -> None:
        """Show indeterminate (marquee) progress bar — use when duration is unknown."""
        self._progress.setMaximum(0)
        self._progress.setValue(0)
        self._progress.setVisible(True)

    def set_progress(self, value: int) -> None:
        """Update progress bar value."""
        self._progress.setValue(value)

    def hide_progress(self) -> None:
        """Hide progress bar."""
        self._progress.setVisible(False)
        self._progress.setMaximum(100)
        self._progress.setValue(0)

