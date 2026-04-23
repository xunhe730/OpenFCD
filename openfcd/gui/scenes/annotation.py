"""Annotation Scene — interactive polygon mask drawing for FCD preprocessing.

Translates the v3 design bundle `SceneAnnotation` (scenes.jsx L285-357) to PyQt6.
Features:
  - Image canvas with pan/zoom
  - Polygon drawing/editing (vertex drag, add/delete)
  - ROI rectangle display (dashed yellow)
  - Dilate preview (semi-transparent outer ring)
  - Floating tool rail (pan / zoom / draw / edit / ROI / anchor)
  - STALE banner when annotation has changed since last run
  - Frame slider for anchor frame navigation
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (
    QPixmap, QPainter, QColor, QPen, QBrush, QFont,
    QMouseEvent, QWheelEvent, QPolygonF, QPainterPath, QCursor, QIcon,
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QSlider, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem,
    QGraphicsEllipseItem, QGraphicsItem,
)

from openfcd.gui import tokens
from openfcd.gui.icons import get_icon, ICON_PAN_TOOL, ICON_PENTAGON, ICON_RECTANGLE, ICON_WARNING, ICONS_DIR
from openfcd.io.annotation import AnnotationSchema, PolygonData, ROIData

if TYPE_CHECKING:
    pass


class AnnotationTool(Enum):
    PAN = auto()
    DRAW_POLYGON = auto()
    DRAW_ROI = auto()


_TOOL_ICONS = {
    AnnotationTool.PAN: (ICON_PAN_TOOL, "Pan / Select"),
    AnnotationTool.DRAW_POLYGON: (ICON_PENTAGON, "Draw Polygon"),
    AnnotationTool.DRAW_ROI: (ICON_RECTANGLE, "Draw ROI (Compute Limits)"),
}


class _ToolRailButton(QPushButton):
    """Single tool button in the floating rail."""

    def __init__(self, icon_name: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self.setIcon(get_icon(icon_name))
        self.setIconSize(QSize(18, 18))
        self.setToolTip(tooltip)
        self.setFixedSize(30, 30)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._active = False
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {tokens.ACCENT_CLAY};
                    border-radius: 5px;
                    border: none;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border-radius: 5px;
                    border: none;
                }}
                QPushButton:hover {{
                    background: {tokens.BG_SECONDARY};
                }}
            """)

    def setChecked(self, checked: bool) -> None:
        super().setChecked(checked)
        self._active = checked
        self._apply_theme()


class _ToolRail(QFrame):
    """Floating vertical tool palette (left side of annotation canvas)."""

    tool_changed = pyqtSignal(object)  # AnnotationTool

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[AnnotationTool, _ToolRailButton] = {}
        self._current_tool = AnnotationTool.DRAW_POLYGON
        self._setup_ui()

        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        for tool, (icon, tooltip) in _TOOL_ICONS.items():
            btn = _ToolRailButton(icon, tooltip, self)
            btn.clicked.connect(lambda checked, t=tool: self._on_tool_clicked(t))
            layout.addWidget(btn)
            self._buttons[tool] = btn

        if self._current_tool in self._buttons:
            self._buttons[self._current_tool].setChecked(True)

    def _apply_theme(self) -> None:
        for btn in self._buttons.values():
            btn._apply_theme()
        self.setStyleSheet(f"""
            QFrame {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)

    def _on_tool_clicked(self, tool: AnnotationTool) -> None:
        self._current_tool = tool
        for t, btn in self._buttons.items():
            btn.setChecked(t == tool)
        self.tool_changed.emit(tool)

    @property
    def current_tool(self) -> AnnotationTool:
        return self._current_tool


class _VertexHandle(QGraphicsEllipseItem):
    """Draggable vertex handle on the polygon."""

    def __init__(self, x: float, y: float, idx: int, parent=None) -> None:
        size = 8.0
        super().__init__(-size / 2, -size / 2, size, size, parent)
        self.setPos(x, y)
        self.vertex_idx = idx
        self.setBrush(QBrush(QColor("#fff")))
        self.setPen(QPen(QColor("#2A9AB8"), 1.2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self.setZValue(10)


class _AnnotationCanvas(QGraphicsView):
    """QGraphicsView-based canvas for polygon annotation.

    Handles:
      - Background image display
      - Polygon drawing / editing
      - ROI rect display
      - Dilate preview ring
      - Pan and zoom
    """

    vertex_moved = pyqtSignal(int, float, float)  # idx, new_x, new_y
    polygon_completed = pyqtSignal(list)  # list of (x, y) tuples
    roi_drawn = pyqtSignal(float, float, float, float)  # x, y, w, h

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gscene = QGraphicsScene(self)
        self.setScene(self._gscene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        self._bg_item: QGraphicsPixmapItem | None = None
        self._polygon_item: QGraphicsPolygonItem | None = None
        self._dilate_item: QGraphicsPolygonItem | None = None
        self._roi_item: QGraphicsRectItem | None = None
        self._vertex_handles: list[_VertexHandle] = []
        self._drawing_points: list[QPointF] = []
        self._drawing_polygon: QGraphicsPolygonItem | None = None

        self._current_tool = AnnotationTool.DRAW_POLYGON
        self._is_panning = False
        self._pan_start = QPointF()

        self._frame_label = QLabel("", self)

        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        bg_col = tokens.BG_SECONDARY
        self.setStyleSheet(f"""
            QGraphicsView {{
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                background: {bg_col};
            }}
        """)
        if hasattr(self, '_scale_label'):
            self._scale_label.setStyleSheet(
                f"font-family: {tokens.FONT_MONO}; font-size: 9.5px; "
                f"color: {tokens.TEXT_PRIMARY}; background: transparent; padding: 2px;"
            )
        if hasattr(self, '_frame_label'):
            self._frame_label.setStyleSheet(f"""
                padding: 3px 8px;
                background: {tokens.BG_TERTIARY};
                border-radius: 3px;
                font-family: {tokens.FONT_MONO};
                font-size: 10.5px;
                color: {tokens.TEXT_PRIMARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
            """)

    def set_image(self, pixmap: QPixmap) -> None:
        """Set the background image."""
        if self._bg_item:
            self._gscene.removeItem(self._bg_item)
        self._bg_item = self._gscene.addPixmap(pixmap)
        self._bg_item.setZValue(-1)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._bg_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_frame_label(self, text: str) -> None:
        self._frame_label.setText(text)
        self._frame_label.adjustSize()
        self._frame_label.move(self.width() - self._frame_label.width() - 16, 16)

    def set_tool(self, tool: AnnotationTool) -> None:
        self._current_tool = tool
        if tool == AnnotationTool.PAN:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        for h in self._vertex_handles:
            h.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                tool == AnnotationTool.EDIT_VERTEX,
            )

    def display_polygon(self, vertices: list[list[float]], dilate_px: float = 0) -> None:
        """Show a polygon on the canvas."""
        self._clear_polygon()
        if not vertices or len(vertices) < 3:
            return

        points = [QPointF(v[1], v[0]) for v in vertices]  # col, row → x, y
        qpoly = QPolygonF(points)

        # Polygon outline (cyan)
        self._polygon_item = self._gscene.addPolygon(
            qpoly,
            QPen(QColor("#2A9AB8"), 2.0),
            QBrush(Qt.BrushStyle.NoBrush),
        )
        self._polygon_item.setZValue(5)

        # Dilate preview if applicable
        if dilate_px > 0:
            # Approximate by drawing a wider polygon
            path = QPainterPath()
            path.addPolygon(qpoly)
            stroker_pen = QPen()
            stroker_pen.setWidthF(dilate_px * 2)
            stroker_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

            from PyQt6.QtGui import QPainterPathStroker
            stroker = QPainterPathStroker(stroker_pen)
            outline = stroker.createStroke(path)
            dilate_poly = outline.toFillPolygon()

            self._dilate_item = self._gscene.addPolygon(
                dilate_poly,
                QPen(QColor(204, 120, 92, 128), 0.5, Qt.PenStyle.DashLine),
                QBrush(QColor(204, 120, 92, 56)),
            )
            self._dilate_item.setZValue(4)

        # Vertex handles
        for i, v in enumerate(vertices):
            handle = _VertexHandle(v[1], v[0], i)  # col, row
            self._gscene.addItem(handle)
            self._vertex_handles.append(handle)

    def display_roi(self, roi: ROIData) -> None:
        """Show a dashed ROI rectangle."""
        if self._roi_item:
            self._gscene.removeItem(self._roi_item)
            self._roi_item = None

        if roi.is_empty:
            return

        pen = QPen(QColor("#D4A24C"), 1.5, Qt.PenStyle.DashLine)
        self._roi_item = self._gscene.addRect(
            roi.x, roi.y, roi.width, roi.height, pen
        )
        self._roi_item.setZValue(3)

    def _clear_polygon(self) -> None:
        if self._polygon_item:
            self._gscene.removeItem(self._polygon_item)
            self._polygon_item = None
        if self._dilate_item:
            self._gscene.removeItem(self._dilate_item)
            self._dilate_item = None
        for h in self._vertex_handles:
            self._gscene.removeItem(h)
        self._vertex_handles.clear()

    # ── Mouse events ────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or self._current_tool == AnnotationTool.PAN:
            if event.button() == Qt.MouseButton.MiddleButton:
                # Force pan mode temporarily for middle mouse drag
                self._original_drag_mode = self.dragMode()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        if self._current_tool == AnnotationTool.DRAW_POLYGON:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check if we clicked on an item like a handle, if so let the graphics scene handle the drag
                item = self.itemAt(event.pos())
                if isinstance(item, _VertexHandle):
                    super().mousePressEvent(event)
                    return

                pos = self.mapToScene(event.pos())
                self._drawing_points.append(pos)
                self._update_drawing_polygon()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # Complete polygon
                if len(self._drawing_points) >= 3:
                    pts = [(p.y(), p.x()) for p in self._drawing_points]  # row, col
                    self.polygon_completed.emit(pts)
                self._drawing_points.clear()
                if self._drawing_polygon:
                    self._gscene.removeItem(self._drawing_polygon)
                    self._drawing_polygon = None
                return

        elif self._current_tool == AnnotationTool.DRAW_ROI:
            if event.button() == Qt.MouseButton.LeftButton:
                # Same check for handles
                item = self.itemAt(event.pos())
                if isinstance(item, _VertexHandle):
                    super().mousePressEvent(event)
                    return
                self._roi_start = self.mapToScene(event.pos())
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._current_tool == AnnotationTool.DRAW_ROI and hasattr(self, "_roi_start"):
            pos = self.mapToScene(event.pos())
            x = min(self._roi_start.x(), pos.x())
            y = min(self._roi_start.y(), pos.y())
            w = abs(pos.x() - self._roi_start.x())
            h = abs(pos.y() - self._roi_start.y())
            if self._roi_item:
                self._roi_item.setRect(x, y, w, h)
            else:
                pen = QPen(QColor("#D4A24C"), 1.5, Qt.PenStyle.DashLine)
                self._roi_item = self._gscene.addRect(x, y, w, h, pen)
                self._roi_item.setZValue(3)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and hasattr(self, "_original_drag_mode"):
            self.setDragMode(self._original_drag_mode)
            del self._original_drag_mode
            super().mouseReleaseEvent(event)
            return

        # If we were panning manually via PAN tool, let the super class handle the release
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag and event.button() == Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._current_tool == AnnotationTool.DRAW_ROI and hasattr(self, "_roi_start"):
            pos = self.mapToScene(event.pos())
            x = min(self._roi_start.x(), pos.x())
            y = min(self._roi_start.y(), pos.y())
            w = abs(pos.x() - self._roi_start.x())
            h = abs(pos.y() - self._roi_start.y())
            self.roi_drawn.emit(x, y, w, h)
            del self._roi_start
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def _update_drawing_polygon(self) -> None:
        """Update the in-progress polygon preview."""
        if len(self._drawing_points) < 2:
            return
        if self._drawing_polygon:
            self._gscene.removeItem(self._drawing_polygon)

        qpoly = QPolygonF(self._drawing_points)
        self._drawing_polygon = self._gscene.addPolygon(
            qpoly,
            QPen(QColor("#2A9AB8"), 1.5, Qt.PenStyle.DashLine),
            QBrush(QColor(42, 154, 184, 30)),
        )
        self._drawing_polygon.setZValue(8)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._bg_item:
            self.fitInView(self._bg_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._frame_label.move(self.width() - self._frame_label.width() - 16, 16)


class _StaleBanner(QWidget):
    """Warning banner shown when annotation changed since last run."""

    re_run_clicked = pyqtSignal()
    discard_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(32)
        self._setup_ui()
        self.setVisible(False)

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(10)

        self.setObjectName("StaleBanner")

        icon_pixmap = QPixmap(16, 16)
        icon_pixmap.fill(Qt.GlobalColor.transparent)
        svg_path = str(ICONS_DIR / "warning.svg")
        renderer = QSvgRenderer(svg_path)
        painter = QPainter(icon_pixmap)
        renderer.render(painter)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(icon_pixmap.rect(), QColor(tokens.ACCENT_CLAY))
        painter.end()
        self.icon = QLabel()
        self.icon.setPixmap(icon_pixmap)
        self.icon.setFixedSize(16, 16)
        layout.addWidget(self.icon)

        self._msg = QLabel("Annotation changed since last run · results stale")
        layout.addWidget(self._msg)

        layout.addStretch()

        self.rerun = QPushButton("Re-run")
        self.rerun.clicked.connect(self.re_run_clicked.emit)
        layout.addWidget(self.rerun)

        self.discard = QPushButton("Discard")
        self.discard.clicked.connect(self.discard_clicked.emit)
        layout.addWidget(self.discard)
        
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()
        
    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            #StaleBanner {{
                background: {tokens.ACCENT_CLAY_BG};
                border-bottom: 1px solid {tokens.ACCENT_CLAY};
            }}
        """)
        self.icon.setStyleSheet("border: none; background: transparent;")
        self._msg.setStyleSheet(
            f"font-size: 11.5px; color: {tokens.ACCENT_CLAY}; border: none; background: transparent;"
        )
        self.rerun.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {tokens.ACCENT_CLAY}; "
            f"text-decoration: underline; background: transparent; border: none;"
        )
        self.discard.setStyleSheet(
            f"font-size: 11px; color: {tokens.TEXT_SECONDARY}; "
            f"background: transparent; border: none;"
        )

class _AnnotationActionsBar(QWidget):
    auto_mask_current = pyqtSignal()
    auto_mask_all = pyqtSignal()
    clear_current = pyqtSignal()
    clear_all = pyqtSignal()
    scope_toggled = pyqtSignal(bool) # True = This Frame, False = Global

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("AnnotationActionsBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(10)

        self.btn_auto_curr = QPushButton("Auto Mask Current")
        self.btn_auto_curr.clicked.connect(self.auto_mask_current.emit)
        layout.addWidget(self.btn_auto_curr)

        self.btn_auto_all = QPushButton("Auto Mask All Selected")
        self.btn_auto_all.clicked.connect(self.auto_mask_all.emit)
        layout.addWidget(self.btn_auto_all)

        layout.addStretch()
        
        self.btn_scope = QPushButton("Save target: Global")
        self.btn_scope.setCheckable(True)
        self.btn_scope.toggled.connect(self._on_scope_toggled)
        layout.addWidget(self.btn_scope)

        layout.addStretch()

        self.btn_clr_curr = QPushButton("Clear Current")
        self.btn_clr_curr.clicked.connect(self.clear_current.emit)
        layout.addWidget(self.btn_clr_curr)

        self.btn_clr_all = QPushButton("Clear All Masks")
        self.btn_clr_all.clicked.connect(self.clear_all.emit)
        layout.addWidget(self.btn_clr_all)
        
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _on_scope_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_scope.setText("Save target: This Frame")
        else:
            self.btn_scope.setText("Save target: Global")
        self.scope_toggled.emit(checked)

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            #AnnotationActionsBar {{
                background: {tokens.BG_PRIMARY};
                border-bottom: 1px solid {tokens.BORDER_SUBTLE};
            }}
            QPushButton {{
                background: {tokens.BG_TERTIARY};
                color: {tokens.TEXT_PRIMARY};
                font-size: 11px;
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background: {tokens.BORDER_STRONG};
            }}
        """)
        self.btn_scope.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {tokens.TEXT_PRIMARY};
                border: 1px dashed {tokens.TEXT_MUTED};
                border-radius: 4px;
                padding: 4px 10px;
            }}
            QPushButton:checked {{
                background: {tokens.ACCENT_CLAY_BG};
                border: 1px solid {tokens.ACCENT_CLAY};
                color: {tokens.ACCENT_CLAY};
            }}
        """)
class _FrameSliderBar(QWidget):
    """Bottom frame navigation slider."""

    frame_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self._frame = 0
        self._total = 1
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setObjectName("FrameSliderBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(10)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, 1)

        self._frame_label = QLabel("0000 / 1")
        layout.addWidget(self._frame_label)

        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            #FrameSliderBar {{
                background: {tokens.BG_SECONDARY};
                border-top: 1px solid {tokens.BORDER_SUBTLE};
            }}
        """)
        self._frame_label.setStyleSheet(
            f"font-family: {tokens.FONT_MONO}; font-size: 11px; "
            f"color: {tokens.TEXT_SECONDARY}; min-width: 84px; border: none; background: transparent;"
        )
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {tokens.BORDER_SUBTLE};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
                background: {tokens.TEXT_PRIMARY};
                border: 2px solid {tokens.ACCENT_CLAY};
            }}
            QSlider::sub-page:horizontal {{
                background: {tokens.ACCENT_CLAY};
                border-radius: 2px;
            }}
        """)

    def set_range(self, total: int) -> None:
        self._total = max(1, total)
        self._slider.setMaximum(self._total - 1)
        self._update_label()

    def set_frame(self, idx: int) -> None:
        self._frame = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._update_label()

    def _on_slider(self, value: int) -> None:
        self._frame = value
        self._update_label()
        self.frame_changed.emit(value)

    def _update_label(self) -> None:
        self._frame_label.setText(
            f"{str(self._frame).zfill(4)} / {self._total}"
        )


class AnnotationScene(QWidget):
    """Annotation Scene — interactive polygon mask drawing.

    Provides a full annotation workflow:
    1. Load an image (anchor frame)
    2. Draw/edit polygon mask for occluding objects
    3. Draw ROI rectangle
    4. Save annotation to JSON
    """

    annotation_changed = pyqtSignal()
    re_run_requested = pyqtSignal()
    discard_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._annotation = AnnotationSchema()
        self._image_path: Path | None = None
        self._frames: list[Path] = []
        self._current_frame_idx: int = 0
        self._is_global_scope: bool = True
        self._total_frames = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # STALE banner
        self._stale_banner = _StaleBanner(self)
        self._stale_banner.re_run_clicked.connect(self.re_run_requested.emit)
        self._stale_banner.discard_clicked.connect(self.discard_requested.emit)
        layout.addWidget(self._stale_banner)

        # Actions bar
        self._actions_bar = _AnnotationActionsBar(self)
        self._actions_bar.auto_mask_current.connect(self._on_auto_mask_current)
        self._actions_bar.auto_mask_all.connect(self._on_auto_mask_all)
        self._actions_bar.clear_current.connect(self._on_clear_current)
        self._actions_bar.clear_all.connect(self._on_clear_all)
        self._actions_bar.scope_toggled.connect(self._on_scope_toggled)
        layout.addWidget(self._actions_bar)

        # Main area with canvas and tool rail
        main_area = QWidget()
        main_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout = QHBoxLayout(main_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Tool rail (floating left)
        rail_container = QWidget()
        rail_container.setFixedWidth(56)
        rail_layout = QVBoxLayout(rail_container)
        rail_layout.setContentsMargins(12, 0, 8, 0)
        rail_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._tool_rail = _ToolRail(self)
        self._tool_rail.tool_changed.connect(self._on_tool_changed)
        rail_layout.addWidget(self._tool_rail)

        main_layout.addWidget(rail_container)

        # Canvas
        self._canvas = _AnnotationCanvas(self)
        self._canvas.polygon_completed.connect(self._on_polygon_drawn)
        self._canvas.roi_drawn.connect(self._on_roi_drawn)
        main_layout.addWidget(self._canvas, 1)

        layout.addWidget(main_area, 1)

        # Frame slider
        self._frame_slider = _FrameSliderBar(self)
        self._frame_slider.frame_changed.connect(self._on_frame_changed)
        self._frame_slider.setVisible(False)
        layout.addWidget(self._frame_slider)

    # ── Public API ──────────────────────────────────────────────────
    def set_frame_list(self, frames: list[Path]) -> None:
        """Set the list of compute frames (def) selected by the user."""
        self._frames = frames
        self._total_frames = len(frames)
        self._frame_slider.set_range(self._total_frames)
        self._frame_slider.setVisible(self._total_frames > 1)
        if frames:
            # Reload current frame or start at 0
            idx = min(self._current_frame_idx, max(0, self._total_frames - 1))
            self.load_image(frames[idx], idx)

    def load_image(self, path: str | Path, idx: int = 0) -> None:
        """Load an image to annotate."""
        path = Path(path)
        self._image_path = path
        self._current_frame_idx = idx
        pm = QPixmap(str(path))
        self._canvas.set_image(pm)
        self._canvas.set_frame_label(f"[{idx+1}/{self._total_frames}] · {path.name}")
        self._display_annotation()

    def set_annotation(self, ann: AnnotationSchema) -> None:
        """Load an existing annotation."""
        self._annotation = ann
        self._display_annotation()

    def get_annotation(self) -> AnnotationSchema:
        """Return the current annotation data."""
        return self._annotation

    def set_total_frames(self, n: int) -> None:
        self._total_frames = n
        self._frame_slider.set_range(n)

    def set_stale(self, stale: bool, message: str = "") -> None:
        self._stale_banner.setVisible(stale)
        if message:
            self._stale_banner._msg.setText(message)

    # ── Internal ────────────────────────────────────────────────────
    def _display_annotation(self) -> None:
        """Refresh canvas from current annotation data."""
        if not self._image_path: return
        frame_name = self._image_path.name
        
        poly_list = self._annotation.polygons
        is_override = False
        if frame_name in self._annotation.frame_polygons:
            poly_list = self._annotation.frame_polygons[frame_name]
            is_override = True

        pdata = next((p for p in poly_list if p.visible), None)
        if pdata:
            dilate_px = int(self._annotation.dilate_cells * self._annotation.cell_mm * 7.27 + 0.5) # approximate px/mm for GUI display
            self._canvas.display_polygon(pdata.vertices, dilate_px)
        else:
            self._canvas._clear_polygon()
            
        self._canvas.display_roi(self._annotation.roi)

        # Sync scope button
        self._actions_bar.btn_scope.blockSignals(True)
        self._actions_bar.btn_scope.setChecked(is_override)
        self._is_global_scope = not is_override
        if is_override:
            self._actions_bar.btn_scope.setText("Save target: This Frame")
        else:
            self._actions_bar.btn_scope.setText("Save target: Global")
        self._actions_bar.btn_scope.blockSignals(False)

    def _on_tool_changed(self, tool: AnnotationTool) -> None:
        self._canvas.set_tool(tool)
        if tool == AnnotationTool.PAN:
            self._canvas.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self._canvas.setDragMode(QGraphicsView.DragMode.NoDrag)

    def _on_polygon_drawn(self, vertices: list[tuple[float, float]]) -> None:
        from openfcd.io.annotation import PolygonData

        poly = PolygonData(vertices=vertices, visible=True, label="object")
        if self._is_global_scope:
            self._annotation.polygons = [poly]
        else:
            if not self._image_path: return
            self._annotation.frame_polygons[self._image_path.name] = [poly]

        self._display_annotation()
        self.annotation_changed.emit()

    def _on_roi_drawn(self, x: float, y: float, w: float, h: float) -> None:
        """Called when user draws a ROI rectangle."""
        self._annotation.roi = ROIData(x=x, y=y, width=w, height=h)
        self._canvas.display_roi(self._annotation.roi)
        self.annotation_changed.emit()

    def _on_frame_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._frames):
            self.load_image(self._frames[idx], idx)

    def _on_scope_toggled(self, checked: bool) -> None:
        self._is_global_scope = not checked
        if not self._image_path: return
        frame_name = self._image_path.name
        
        if checked: # Switch to This Frame
            if frame_name not in self._annotation.frame_polygons:
                import copy
                self._annotation.frame_polygons[frame_name] = copy.deepcopy(self._annotation.polygons)
        else:
            if frame_name in self._annotation.frame_polygons:
                del self._annotation.frame_polygons[frame_name]

        self._display_annotation()
        self.annotation_changed.emit()

    def _on_auto_mask_current(self) -> None:
        if not self._image_path: return
        self._run_auto_mask([self._image_path], global_scope=self._is_global_scope)

    def _on_auto_mask_all(self) -> None:
        if not self._frames: return
        self._run_auto_mask(self._frames, global_scope=False)

    def _run_auto_mask(self, paths: list[Path], global_scope: bool) -> None:
        from openfcd.core.mask import auto_mask, find_oriented_polygon
        from openfcd.core.fcd import calculate_carriers
        from openfcd.core.flatfield import flatfield_normalize
        from skimage.io import imread
        import numpy as np
        from openfcd.io.annotation import PolygonData

        for p in paths:
            try:
                img = imread(str(p))
                if img.ndim == 3: img = np.mean(img, axis=2)
                ff = flatfield_normalize(img, sigma=300)
                c = calculate_carriers(ff - ff.mean())
                m = auto_mask(ff, c, threshold_ratio=0.2, dilate_px=8)
                poly = find_oriented_polygon(m)
                if poly:
                    pdata = PolygonData(vertices=[list(v) for v in poly.vertices], label="auto")
                    if global_scope:
                        self._annotation.polygons = [pdata]
                    else:
                        self._annotation.frame_polygons[p.name] = [pdata]
            except Exception as e:
                print(f"Auto mask failed for {p.name}: {e}")
        
        self._display_annotation()
        self.annotation_changed.emit()

    def _on_clear_current(self) -> None:
        if not self._image_path: return
        if self._is_global_scope:
            self._annotation.polygons.clear()
        else:
            if self._image_path.name in self._annotation.frame_polygons:
                self._annotation.frame_polygons[self._image_path.name].clear()
        self._display_annotation()
        self.annotation_changed.emit()

    def _on_clear_all(self) -> None:
        self._annotation.polygons.clear()
        self._annotation.frame_polygons.clear()
        self._display_annotation()
        self.annotation_changed.emit()
