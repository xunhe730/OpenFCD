"""TitleBar widget — clean/dirty/saving three states matching chrome.jsx."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter
from PyQt6.QtSvg import QSvgRenderer

from openfcd.gui import tokens
from openfcd.gui.icons import LOGO_MARK_PATH


class _Logo(QWidget):
    """Compact OpenFCD brand mark rendered from SVG."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._renderer = QSvgRenderer(str(LOGO_MARK_PATH))

    def paintEvent(self, a0) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(painter)
        painter.end()


class _MenuLabel(QLabel):
    """Clickable menu label with cursor change on hover."""

    clicked = pyqtSignal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        super().mousePressEvent(ev)
        self.clicked.emit()


class TitleBarWidget(QWidget):
    """Title bar with 3 states: clean, dirty (clay dot), saving (serif italic).

    Matches design/chrome.jsx TitleBar spec (32px height).
    """

    # Signals
    menu_requested = pyqtSignal(str)  # menu name (File, Edit, etc.)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)
        self._dirty = False
        self._saving = False
        self._title = "OpenFCD"
        self._setup_ui()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.brand.setStyleSheet(
            "color:" + tokens.TEXT_PRIMARY + ";font-size:12px;"
            "font-weight:600;letter-spacing:0.01em;"
        )
        for lbl in self._menus:
            lbl.setStyleSheet(
                "color:" + tokens.TEXT_SECONDARY + ";font-size:12px;"
                "background:transparent;border:none;padding:0 2px;"
            )
        self._title_label.setStyleSheet(
            "color:" + tokens.TEXT_SECONDARY + ";font-size:12px;"
            "background:transparent;border:none;"
        )
        self._dirty_dot.setStyleSheet(
            "background:" + tokens.ACCENT_CLAY + ";border-radius:3px;"
        )
        self._saving_label.setStyleSheet(
            "color:" + tokens.TEXT_MUTED + ";font-size:12px;"
            "font-style:italic;font-family:" + tokens.FONT_SERIF + ";"
            "margin-left:2px;"
        )
        for lbl in self._right_controls:
            lbl.setStyleSheet("color:" + tokens.TEXT_MUTED + ";font-size:12px;")
        self.setStyleSheet(
            "background:" + tokens.BG_DOCK + ";"
            "border-bottom:1px solid " + tokens.BORDER_SUBTLE + ";"
        )
        self._logo.update()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(0)

        # Left: Logo + Menus
        left = QHBoxLayout()
        left.setSpacing(18)

        # Logo group
        logo_grp = QHBoxLayout()
        logo_grp.setSpacing(6)
        self._logo = _Logo()
        logo_grp.addWidget(self._logo)

        self.brand = QLabel("OpenFCD")
        logo_grp.addWidget(self.brand)
        left.addLayout(logo_grp)

        # Menu items
        self._menus: list[_MenuLabel] = []
        for name in ["File", "Edit", "Run", "Scenes", "Tools", "View", "Help"]:
            lbl = _MenuLabel(name)
            lbl.clicked.connect(lambda n=name: self.menu_requested.emit(n))
            self._menus.append(lbl)
            left.addWidget(lbl)

        layout.addLayout(left)

        # Center: Title + state indicators
        layout.addSpacing(18)
        layout.addStretch()

        self._title_label = QLabel(self._title)
        layout.addWidget(self._title_label)

        # Dirty indicator (6px clay dot)
        self._dirty_dot = QLabel()
        self._dirty_dot.setFixedSize(6, 6)
        self._dirty_dot.setVisible(False)
        self._dirty_dot.setToolTip("Unsaved changes · ⌘S to save")
        layout.addWidget(self._dirty_dot)

        # Saving indicator (serif italic)
        self._saving_label = QLabel("Saving…")
        self._saving_label.setVisible(False)
        layout.addWidget(self._saving_label)

        layout.addStretch()

        # Native title bars already provide platform window controls.  Keep
        # this custom strip focused on app navigation so Windows does not show
        # a second set of inert minimize/maximize/close labels.
        self._right_controls = []

    def set_project_title(self, title: str) -> None:
        """Set the project title shown in the bar."""
        self._title = title
        self._title_label.setText(title)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @is_dirty.setter
    def is_dirty(self, value: bool) -> None:
        self._dirty = value
        self._dirty_dot.setVisible(value and not self._saving)

    @property
    def is_saving(self) -> bool:
        return self._saving

    @is_saving.setter
    def is_saving(self, value: bool) -> None:
        self._saving = value
        self._saving_label.setVisible(value)
        if value:
            self._dirty_dot.setVisible(False)
