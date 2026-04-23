"""Scene tabs panel — QTabWidget for figure scenes."""

from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget, QWidget, QPushButton
from PyQt6.QtCore import Qt

from openfcd.gui import tokens


class SceneTabs(QTabWidget):
    """Tabbed container for visualization scenes."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.tabCloseRequested.connect(self._on_tab_close)
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(self._style())

    def _style(self) -> str:
        return f"""
            QTabWidget::pane {{
                border: none;
                background: {tokens.BG_PRIMARY};
                border-top: 1px solid {tokens.BORDER_SUBTLE};
            }}
            QTabBar {{
                background: {tokens.BG_SECONDARY};
            }}
            QTabBar::tab {{
                background: {tokens.BG_SECONDARY};
                color: {tokens.TEXT_SECONDARY};
                padding: 10px 20px;
                font-family: {tokens.FONT_UI};
                font-size: 12px;
                border: none;
                border-right: 1px solid {tokens.BORDER_SUBTLE};
                border-bottom: 1px solid {tokens.BORDER_SUBTLE};
                border-top: 2px solid transparent;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background: {tokens.BG_PRIMARY};
                color: {tokens.TEXT_PRIMARY};
                border-top: 2px solid {tokens.ACCENT_CLAY};
                border-bottom: 1px solid transparent;
            }}
            QTabBar::tab:hover:!selected {{
                color: {tokens.TEXT_PRIMARY};
                background: {tokens.BG_TERTIARY};
            }}
            QTabBar::close-button {{
                image: none; /* Can customize close button icon if needed */
            }}
        """

    def add_scene(self, widget: QWidget, title: str) -> int:
        """Add a scene widget with given title."""
        return self.addTab(widget, title)

    def _on_tab_close(self, index: int) -> None:
        """Handle tab close request."""
        self.removeTab(index)

    def show_eta(self, eta_mm) -> None:
        """Show or refresh the η Map tab with the given eta_mm array."""
        from openfcd.gui.scenes.eta_map import EtaMap

        # Find existing EtaMap tab
        eta_widget = None
        eta_idx = -1
        for i in range(self.count()):
            w = self.widget(i)
            if isinstance(w, EtaMap):
                eta_widget = w
                eta_idx = i
                break

        if eta_widget is None:
            eta_widget = EtaMap(self)
            eta_idx = self.addTab(eta_widget, "η Map")

        eta_widget.set_data(eta_mm)
        self.setCurrentIndex(eta_idx)

