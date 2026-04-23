"""Central icon registry with theme-aware QIcon factory for OpenFCD GUI."""

from pathlib import Path
from typing import Final

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from openfcd.gui import tokens

# ---------------------------------------------------------------------------
# Icon name constants
# ---------------------------------------------------------------------------

ICON_NEW: Final = "note_add"
ICON_OPEN: Final = "folder_open"
ICON_SAVE: Final = "save"
ICON_RUN: Final = "play_arrow"
ICON_STOP: Final = "stop"
ICON_LIGHT_MODE: Final = "light_mode"
ICON_DARK_MODE: Final = "dark_mode"
ICON_CROP: Final = "crop_square"
ICON_BRUSH: Final = "brush"
ICON_CLOSE: Final = "close"
ICON_STAR: Final = "star"
ICON_PAN_TOOL: Final = "pan_tool"
ICON_PENTAGON: Final = "pentagon"
ICON_RECTANGLE: Final = "rectangle"
ICON_WARNING: Final = "warning"
ICON_ARROW_BACK: Final = "arrow_back"
ICON_ARROW_FORWARD: Final = "arrow_forward"
ICON_SPEED: Final = "speed"
ICON_PIE_CHART: Final = "pie_chart"
ICON_PHOTO_LIBRARY: Final = "photo_library"
ICON_PHOTO: Final = "photo"
ICON_CHECK_CIRCLE: Final = "check_circle"
ICON_ERROR: Final = "error"
ICON_SCHEDULE: Final = "schedule"

ICON_NAMES: Final = [
    ICON_NEW,
    ICON_OPEN,
    ICON_SAVE,
    ICON_RUN,
    ICON_STOP,
    ICON_LIGHT_MODE,
    ICON_DARK_MODE,
    ICON_CROP,
    ICON_BRUSH,
    ICON_CLOSE,
    ICON_STAR,
    ICON_PAN_TOOL,
    ICON_PENTAGON,
    ICON_RECTANGLE,
    ICON_WARNING,
    ICON_ARROW_BACK,
    ICON_ARROW_FORWARD,
    ICON_SPEED,
    ICON_PIE_CHART,
    ICON_PHOTO_LIBRARY,
    ICON_PHOTO,
    ICON_CHECK_CIRCLE,
    ICON_ERROR,
    ICON_SCHEDULE,
]

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

ICONS_DIR: Final = Path(__file__).parent / "resources" / "icons"

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_icon_cache: dict[str, QIcon] = {}

# ---------------------------------------------------------------------------
# Theme change hook
# ---------------------------------------------------------------------------


def _on_theme_change() -> None:
    """Clear icon cache when theme changes."""
    _icon_cache.clear()


tokens.on_theme_changed(_on_theme_change)

# ---------------------------------------------------------------------------
# Icon factory
# ---------------------------------------------------------------------------


def _create_icon(name: str) -> QIcon:
    """Load an SVG, tint it with the current theme color, and return a QIcon.

    Args:
        name: Icon name (without extension), e.g. "note_add".

    Returns:
        A theme-tinted QIcon at 24x24 pixels.
    """
    svg_path = str(ICONS_DIR / f"{name}.svg")
    icon_color = tokens.TEXT_SECONDARY

    renderer = QSvgRenderer(svg_path)
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(icon_color))
    painter.end()

    return QIcon(pixmap)


def get_icon(name: str) -> QIcon:
    """Return a theme-aware QIcon for the given icon name.

    Icons are cached on first load. The cache is cleared automatically when
    the theme changes via the tokens signal.

    Args:
        name: Icon name (without extension), e.g. "note_add".

    Returns:
        A themed QIcon.
    """
    if name not in _icon_cache:
        _icon_cache[name] = _create_icon(name)
    return _icon_cache[name]
