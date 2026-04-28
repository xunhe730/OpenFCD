"""Toolbar widget — action buttons matching chrome.jsx spec."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QFrame, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, QEvent, QSize, QObject

from openfcd.gui import tokens
from openfcd.gui.icons import get_icon, ICON_NEW, ICON_OPEN, ICON_SAVE, ICON_RUN, ICON_STOP, ICON_LIGHT_MODE, ICON_DARK_MODE
from PyQt6.QtGui import QMouseEvent, QIcon

if TYPE_CHECKING:
    from PyQt6.QtCore import QEvent as QEventType


class _ToolbarButton(QPushButton):
    """Button with hover via eventFilter (no CSS :hover pseudo-class)."""

    _icon_text: str
    _icon: QIcon | None
    _hovered: bool
    _primary: bool
    _muted: bool


    def __init__(self, text: str = "", icon_text: str = "", icon: QIcon | None = None, parent=None) -> None:
        super().__init__(text, parent)
        self._icon_text = icon_text
        self._icon = icon
        self._hovered = False
        self._primary = False
        self._muted = False
        self.installEventFilter(self)
        tokens.on_theme_changed(self._apply_style)
        self._apply_style()
        if self._icon is not None:
            self.setIcon(self._icon)
            self.setIconSize(QSize(18, 18))

    def set_primary(self, value: bool) -> None:
        self._primary = value
        self._apply_style()

    def set_muted(self, value: bool) -> None:
        self._muted = value
        self._apply_style()

    def eventFilter(self, a0, a1: QEventType | None) -> bool:  # noqa: ANN001
        if a1 and a1.type() in (QEvent.Type.Enter, QEvent.Type.Leave):
            self._hovered = a1.type() == QEvent.Type.Enter
            self._apply_style()
        return super().eventFilter(a0, a1)

    def _apply_style(self) -> None:
        if self._primary:
            bg = tokens.ACCENT_CLAY_HOVER if self._hovered else tokens.ACCENT_CLAY
            fg = "#fff"
        else:
            bg = tokens.BORDER_SUBTLE if self._hovered else "transparent"
            fg = tokens.TEXT_MUTED if self._muted else tokens.TEXT_PRIMARY
        parts = [
            "background:" + bg,
            "color:" + fg,
            "border-radius:5px",
            "padding:5px 10px",
            "font-size:12px",
            "font-weight:500",
            "font-family:" + tokens.FONT_UI,
            "border:none",
        ]
        self.setStyleSheet(";" .join(parts) + ";")


class _SearchButton(QPushButton):
    """Search button with bg-tertiary and ⌘K badge."""

    _hovered: bool

    def __init__(self, parent=None) -> None:
        super().__init__("Search…", parent)
        self._hovered = False
        self.installEventFilter(self)
        self.setFixedHeight(28)
        self.setMinimumWidth(180)
        tokens.on_theme_changed(self._apply_style)
        self._apply_style()

    def eventFilter(self, a0, a1: QEventType | None) -> bool:  # noqa: ANN001
        if a1 and a1.type() in (QEvent.Type.Enter, QEvent.Type.Leave):
            self._hovered = a1.type() == QEvent.Type.Enter
            self._apply_style()
        return super().eventFilter(a0, a1)

    def _apply_style(self) -> None:
        bg = tokens.BORDER_SUBTLE if self._hovered else tokens.BG_TERTIARY
        parts = [
            "background:" + bg,
            "color:" + tokens.TEXT_SECONDARY,
            "border:1px solid " + tokens.BORDER_SUBTLE,
            "border-radius:6px",
            "padding:5px 10px",
            "font-size:12px",
            "font-family:" + tokens.FONT_UI,
        ]
        self.setStyleSheet(";".join(parts) + ";")


class _ThemeToggle(QPushButton):
    """Light/dark theme toggle button."""

    _dark: bool
    _hovered: bool

    def __init__(self, dark: bool = False, parent=None) -> None:
        super().__init__("", parent)
        self._dark = dark
        self._hovered = False
        self.installEventFilter(self)
        self.setFixedSize(28, 28)
        tokens.on_theme_changed(self._sync_theme)
        self._sync_theme()

    def _sync_theme(self) -> None:
        self._dark = tokens.is_dark
        self.setIcon(get_icon(ICON_LIGHT_MODE if self._dark else ICON_DARK_MODE))
        self.setIconSize(QSize(16, 16))
        self._apply_style()

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        tokens.set_dark_mode(not tokens.is_dark)
        super().mousePressEvent(ev)

    def eventFilter(self, a0, a1: QEventType | None) -> bool:  # noqa: ANN001
        if a1 and a1.type() in (QEvent.Type.Enter, QEvent.Type.Leave):
            self._hovered = a1.type() == QEvent.Type.Enter
            self._apply_style()
        return super().eventFilter(a0, a1)

    def _apply_style(self) -> None:
        bg = tokens.BORDER_SUBTLE if self._hovered else "transparent"
        parts = [
            "background:" + bg,
            "color:" + tokens.TEXT_SECONDARY,
            "border:none",
            "border-radius:5px",
            "font-size:13px",
        ]
        self.setStyleSheet(";".join(parts) + ";")


class _PreviewToggle(QPushButton):
    """ON/OFF preview toggle. When ON: η overlay shown. OFF: source only."""

    toggled_preview = pyqtSignal(bool)  # True=on, False=off

    def __init__(self, parent=None) -> None:
        super().__init__("Preview", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setEnabled(False)
        self.setFixedHeight(26)
        self.setMinimumWidth(70)
        tokens.on_theme_changed(self._apply_style)
        self.toggled.connect(self._on_toggled)
        self._apply_style()

    def _on_toggled(self, checked: bool) -> None:
        self._apply_style()
        self.toggled_preview.emit(checked)

    def _apply_style(self) -> None:
        if not self.isEnabled():
            bg, fg = tokens.BG_TERTIARY, tokens.TEXT_MUTED
        elif self.isChecked():
            bg, fg = tokens.ACCENT_CLAY, "#fff"
        else:
            bg, fg = tokens.BG_TERTIARY, tokens.TEXT_PRIMARY
        self.setStyleSheet(
            f"background:{bg};color:{fg};border-radius:5px;"
            f"padding:3px 10px;font-size:12px;font-weight:{'600' if self.isChecked() else '500'};"
            f"font-family:{tokens.FONT_UI};border:1px solid {tokens.BORDER_SUBTLE};"
        )


class Toolbar(QWidget):
    """Action toolbar: New, Open, Save | Run, Cancel | Preview + Overlap + Colorbar | Theme Toggle."""

    new_project_clicked = pyqtSignal()
    open_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    run_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()
    preview_changed = pyqtSignal(bool)    # True=on, False=off
    overlap_changed = pyqtSignal(bool)
    colorbar_changed = pyqtSignal(bool)

    _running: bool
    _btn_open: _ToolbarButton
    _btn_save: _ToolbarButton
    _btn_run: _ToolbarButton
    _btn_cancel: _ToolbarButton
    _running_label: QLabel
    _theme_toggle: _ThemeToggle

    def __init__(self, running: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self._running = running
        self._dividers = []
        self._setup_ui()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            "background:" + tokens.BG_PRIMARY + ";"
            "border-bottom:1px solid " + tokens.BORDER_SUBTLE + ";"
        )
        self._running_label.setStyleSheet(
            "color:" + tokens.ACCENT_CLAY + ";font-size:11px;"
            "font-family:" + tokens.FONT_MONO + ";padding:0 6px;"
        )
        for sep in self._dividers:
            sep.setStyleSheet("background:" + tokens.BORDER_SUBTLE + ";border:none;")
        for cb in (self._overlap_cb, self._colorbar_cb):
            cb.setStyleSheet(f"color:{tokens.TEXT_SECONDARY};font-size:12px;")
        self.set_running(self._running)

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(2)

        # Left buttons
        self._btn_new_project = _ToolbarButton("New…", icon=get_icon(ICON_NEW))
        self._btn_new_project.clicked.connect(self.new_project_clicked)
        layout.addWidget(self._btn_new_project)

        self._btn_open = _ToolbarButton("Open", icon=get_icon(ICON_OPEN))
        self._btn_open.clicked.connect(self.open_clicked)
        layout.addWidget(self._btn_open)

        self._btn_save = _ToolbarButton("Save", icon=get_icon(ICON_SAVE))
        self._btn_save.clicked.connect(self.save_clicked)
        layout.addWidget(self._btn_save)

        layout.addWidget(self._divider())

        # Run / Cancel
        self._btn_run = _ToolbarButton("Run", icon=get_icon(ICON_RUN))
        self._btn_run.set_primary(True)
        self._btn_run.clicked.connect(self.run_clicked)
        layout.addWidget(self._btn_run)

        self._btn_cancel = _ToolbarButton("Cancel", icon=get_icon(ICON_STOP))
        self._btn_cancel.set_muted(True)
        self._btn_cancel.clicked.connect(self.cancel_clicked)
        layout.addWidget(self._btn_cancel)

        # Running indicator (hidden by default)
        self._running_label = QLabel()
        self._running_label.setVisible(False)
        layout.addWidget(self._running_label)

        layout.addWidget(self._divider())

        # Preview toggle + Overlap + Colorbar inline
        self._btn_preview_toggle = _PreviewToggle()
        self._btn_preview_toggle.toggled_preview.connect(self._on_preview_toggled)
        self._btn_preview_toggle.toggled_preview.connect(self.preview_changed)
        layout.addWidget(self._btn_preview_toggle)

        self._overlap_cb = QCheckBox("Overlap")
        self._overlap_cb.setChecked(True)
        self._overlap_cb.setEnabled(False)
        self._overlap_cb.setStyleSheet(f"color:{tokens.TEXT_SECONDARY};font-size:12px;")
        self._overlap_cb.toggled.connect(self.overlap_changed)
        layout.addWidget(self._overlap_cb)

        self._colorbar_cb = QCheckBox("Colorbar")
        self._colorbar_cb.setChecked(False)
        self._colorbar_cb.setEnabled(False)
        self._colorbar_cb.setStyleSheet(f"color:{tokens.TEXT_SECONDARY};font-size:12px;")
        self._colorbar_cb.toggled.connect(self.colorbar_changed)
        layout.addWidget(self._colorbar_cb)

        # Spacer
        layout.addStretch()

        # Theme toggle
        self._theme_toggle = _ThemeToggle()
        layout.addWidget(self._theme_toggle)

        # Managed by _apply_theme
        self.set_running(self._running)

    def _on_preview_toggled(self, on: bool) -> None:
        self._overlap_cb.setEnabled(on)
        self._colorbar_cb.setEnabled(on)
        self._apply_theme()

    def _divider(self) -> QFrame:
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(18)
        sep.setStyleSheet("background:" + tokens.BORDER_SUBTLE + ";border:none;")
        sep.setContentsMargins(6, 0, 6, 0)
        self._dividers.append(sep)
        return sep

    def set_running(self, running: bool, elapsed: str = "") -> None:
        """Toggle Run/Cancel visibility and show running indicator."""
        self._running = running
        self._btn_run.setVisible(not running)
        self._btn_cancel.setVisible(running)   # Cancel shows while running
        if running:
            self._btn_cancel.set_muted(False)
            self._btn_cancel.setStyleSheet(
                "background:" + tokens.ERROR + ";color:#fff;border-radius:5px;"
                "padding:5px 10px;font-size:12px;font-weight:500;"
                "font-family:" + tokens.FONT_UI + ";border:none;"
            )
            self._running_label.setVisible(True)
            self._running_label.setText("● running" + (f" · {elapsed}" if elapsed else ""))
        else:
            self._btn_cancel.set_muted(True)
            self._btn_cancel.setStyleSheet("")
            self._btn_cancel.set_muted(True)
            self._running_label.setVisible(False)

    def set_preview_available(self, available: bool) -> None:
        """Enable/disable the Preview toggle button."""
        self._btn_preview_toggle.setEnabled(available)
        if not available:
            self._btn_preview_toggle.blockSignals(True)
            self._btn_preview_toggle.setChecked(False)
            self._btn_preview_toggle.blockSignals(False)
            self._btn_preview_toggle._apply_style()
            self._overlap_cb.setEnabled(False)
            self._colorbar_cb.setEnabled(False)

    def set_preview_on(self, on: bool) -> None:
        """Programmatically set preview on/off without emitting toggled_preview."""
        self._btn_preview_toggle.blockSignals(True)
        self._btn_preview_toggle.setChecked(on)
        self._btn_preview_toggle.blockSignals(False)
        self._overlap_cb.setEnabled(on)
        self._colorbar_cb.setEnabled(on)
        self._btn_preview_toggle._apply_style()

    def preview_on(self) -> bool:
        """Return True if preview is currently ON."""
        return self._btn_preview_toggle.isChecked()

    def overlap_on(self) -> bool:
        """Return True if Overlap checkbox is checked."""
        return self._overlap_cb.isChecked()

    def colorbar_on(self) -> bool:
        """Return True if Colorbar checkbox is checked."""
        return self._colorbar_cb.isChecked()
