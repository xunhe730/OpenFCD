"""Preview widget — QGraphicsView with crosshair, frame slider, ROI/Mask annotation.

Annotation modes:
  - ROI: 2-click diagonal corners → rectangle
  - MASK: 4-click clockwise (FL→FR→BR→BL) → polygon with forward direction
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
import io
import math

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSlider, QLabel, QHBoxLayout,
    QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsPathItem,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer
from PyQt6.QtGui import (
    QPixmap, QImage, QPen, QColor, QPainter, QWheelEvent,
    QMouseEvent, QKeyEvent, QBrush, QPolygonF, QPainterPath, QPainterPathStroker,
)

from openfcd.gui import tokens
from openfcd.gui.widgets.display_mode import DisplayMode, DisplayModeRegistry, DisplayModeSpec

try:
    from matplotlib import cm
except ImportError:  # pragma: no cover - matplotlib is a declared dependency
    cm = None


class AnnotationMode(Enum):
    NONE = auto()
    ROI = auto()      # 2-click rectangle
    MASK = auto()     # 4-click clockwise polygon


# ── Vertex handle ──────────────────────────────────────────────────
class _Handle(QGraphicsEllipseItem):
    """Draggable vertex dot."""
    def __init__(self, x: float, y: float, idx: int, color: str = "#FFD700"):
        r = 5.0
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(x, y)
        self.idx = idx
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#fff"), 1.2))
        self.setZValue(20)
        pen = QPen(QColor(color), 1)
        pen.setCosmetic(True)
        self.setPen(pen)


# ── Image View ─────────────────────────────────────────────────────
class _ImageView(QGraphicsView):
    """Zoomable/pannable image viewer with ROI/Mask annotation."""

    pixel_hovered = pyqtSignal(int, int, int)
    # ROI completed: (row0, col0, height, width)
    roi_completed = pyqtSignal(int, int, int, int)
    # Mask completed: list of 4 [row, col] pairs + forward_direction [dy, dx]
    mask_completed = pyqtSignal(list, list)
    # Point placed feedback (for status bar): point_index, row, col
    point_placed = pyqtSignal(int, int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._qimage: QImage | None = None
        self._zoom_factor = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._eta_overlay_item: QGraphicsPixmapItem | None = None
        self._eta_overlay_array: np.ndarray | None = None
        self._eta_array: np.ndarray | None = None
        self._eta_cmap: str = "RdBu_r"
        self._eta_opacity: float = 0.55
        self._overlap_on: bool = True

        # Crosshair
        self.crosshair_enabled = False
        self._crosshair_h = None
        self._crosshair_v = None
        self._ch_pen = QPen(QColor("#FFD700"))
        self._ch_pen.setWidth(1)
        self._ch_pen.setStyle(Qt.PenStyle.DashLine)
        self._ch_pen.setCosmetic(True)

        # Annotation state
        self._mode = AnnotationMode.NONE
        self._click_points: list[QPointF] = []   # scene coords
        self._handles: list[_Handle] = []
        self._preview_lines: list[QGraphicsLineItem] = []
        self._roi_rect: QGraphicsRectItem | None = None
        self._mask_poly: QGraphicsPolygonItem | None = None

        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"QGraphicsView {{ background: {tokens.BG_PRIMARY}; border: none; }}")

    # ── Image ──────────────────────────────────────────────────────
    def _reset_item_refs(self) -> None:
        self._pixmap_item = None
        self._eta_overlay_item = None
        self._eta_overlay_array = None
        self._eta_array = None
        self._roi_rect = None
        if hasattr(self, '_roi_mask_item'): self._roi_mask_item = None
        self._mask_poly = None
        if hasattr(self, '_mask_dilate_item'): self._mask_dilate_item = None
        self._handles.clear()
        self._preview_lines.clear()
        self._crosshair_h = None
        self._crosshair_v = None
        if hasattr(self, '_cursor_line'): self._cursor_line = None

    def set_image(self, path: Path | str) -> None:
        qimg = QImage(str(path))
        if qimg.isNull():
            return
        self.set_annotation_mode(AnnotationMode.NONE)
        self._qimage = qimg
        pixmap = QPixmap.fromImage(qimg)
        self._scene.clear()
        self._reset_item_refs()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        # 延迟到下一帧执行，确保 viewport 已完成布局后再 fitInView
        QTimer.singleShot(0, self.fit_in_view)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.set_annotation_mode(AnnotationMode.NONE)
        self._scene.clear()
        self._reset_item_refs()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        QTimer.singleShot(0, self.fit_in_view)

    def fit_in_view(self) -> None:
        """将图片适配到当前视口大小，保持纵横比。"""
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = 1.0

    def resizeEvent(self, event) -> None:
        """窗口调整大小时自动重新适配图片。"""
        super().resizeEvent(event)
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_image(self) -> None:
        self.set_annotation_mode(AnnotationMode.NONE)
        self._scene.clear()
        self._reset_item_refs()
        self._qimage = None

    # ── Annotation mode ────────────────────────────────────────────
    def set_annotation_mode(self, mode: AnnotationMode) -> None:
        self._mode = mode
        self._click_points.clear()
        self._clear_annotation_items()
        if mode != AnnotationMode.NONE:
            self.crosshair_enabled = True
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.crosshair_enabled = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._remove_crosshair()

    def _clear_annotation_items(self) -> None:
        for h in self._handles:
            try:
                if h.scene():
                    self._scene.removeItem(h)
            except RuntimeError:
                pass
        self._handles.clear()
        for ln in self._preview_lines:
            try:
                if ln.scene():
                    self._scene.removeItem(ln)
            except RuntimeError:
                pass
        self._preview_lines.clear()
        self._crosshair_h = None
        self._crosshair_v = None

    def clear_eta_overlay(self) -> None:
        """Remove only the η overlay — leaves ROI rect and mask intact."""
        try:
            if self._eta_overlay_item and self._eta_overlay_item.scene():
                self._scene.removeItem(self._eta_overlay_item)
        except RuntimeError:
            pass
        self._eta_overlay_item = None
        self._eta_overlay_array = None
        self._eta_array = None
        if self._pixmap_item:
            self._pixmap_item.setVisible(True)

    def clear_overlays(self) -> None:
        """Remove ROI rect and Mask polygon overlays."""
        self.clear_eta_overlay()


        try:
            if self._roi_rect and self._roi_rect.scene():
                self._scene.removeItem(self._roi_rect)
        except RuntimeError: pass
        self._roi_rect = None
        
        try:
            if hasattr(self, '_roi_mask_item') and self._roi_mask_item and self._roi_mask_item.scene():
                self._scene.removeItem(self._roi_mask_item)
        except RuntimeError: pass
        self._roi_mask_item = None
        
        try:
            if self._mask_poly and self._mask_poly.scene():
                self._scene.removeItem(self._mask_poly)
        except RuntimeError: pass
        self._mask_poly = None
        
        try:
            if hasattr(self, '_mask_dilate_item') and self._mask_dilate_item and self._mask_dilate_item.scene():
                self._scene.removeItem(self._mask_dilate_item)
        except RuntimeError: pass
        self._mask_dilate_item = None
        
        self._clear_annotation_items()

    def show_eta_overlay(
        self,
        eta_mm: np.ndarray,
        *,
        opacity: float = 0.55,
        cmap_name: str = "RdBu_r",
    ) -> tuple[float, float] | None:
        """Store eta array and render the heatmap overlay; returns (vmin, vmax) or None."""
        if self._pixmap_item is None or self._qimage is None:
            return None
        if cm is None:
            return None
        arr = np.asarray(eta_mm, dtype=np.float64)
        if arr.ndim != 2:
            return None
        self._eta_array = arr
        self._eta_cmap = cmap_name
        self._eta_opacity = opacity
        self._eta_overlay_array = arr
        return self._draw_eta()

    def _draw_eta(self) -> tuple[float, float] | None:
        """Re-render the eta overlay using current _overlap_on / _eta_array state."""
        if self._eta_array is None or self._pixmap_item is None:
            return None
        if cm is None:
            return None

        arr = self._eta_array
        try:
            if self._eta_overlay_item and self._eta_overlay_item.scene():
                self._scene.removeItem(self._eta_overlay_item)
        except RuntimeError:
            pass
        self._eta_overlay_item = None

        valid = np.isfinite(arr)
        if self._pixmap_item:
            self._pixmap_item.setVisible(self._overlap_on)

        if not valid.any():
            return None

        vmin = float(np.nanpercentile(arr, 5))
        vmax = float(np.nanpercentile(arr, 95))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = cm.get_cmap(self._eta_cmap)(norm)

        if self._overlap_on:
            rgba[..., 3] = np.where(valid, self._eta_opacity, 0.0)
        else:
            # Standalone mode: full opacity, theme background for NaN
            from openfcd.gui import tokens as _tok
            _h = _tok.BG_PRIMARY.lstrip('#')
            _r, _g, _b = int(_h[0:2], 16) / 255.0, int(_h[2:4], 16) / 255.0, int(_h[4:6], 16) / 255.0
            rgba[..., 3] = 1.0
            rgba[~valid, :3] = [_r, _g, _b]

        rgba_u8 = np.ascontiguousarray(np.round(rgba * 255).astype(np.uint8))
        h, w = rgba_u8.shape[:2]
        qimg = QImage(
            rgba_u8.data, w, h, rgba_u8.strides[0], QImage.Format.Format_RGBA8888
        ).copy()
        pixmap = QPixmap.fromImage(qimg)
        self._eta_overlay_item = self._scene.addPixmap(pixmap)
        self._eta_overlay_item.setZValue(6)
        return vmin, vmax

    def set_overlap(self, val: bool) -> tuple[float, float] | None:
        """Toggle overlap mode and re-render; returns (vmin, vmax) or None."""
        self._overlap_on = val
        return self._draw_eta()

    # ── Display existing annotation ────────────────────────────────
    def show_roi(self, row0: int, col0: int, h: int, w: int) -> None:
        try:
            if self._roi_rect and self._roi_rect.scene():
                self._scene.removeItem(self._roi_rect)
        except RuntimeError: pass
        try:
            if hasattr(self, '_roi_mask_item') and self._roi_mask_item and self._roi_mask_item.scene():
                self._scene.removeItem(self._roi_mask_item)
        except RuntimeError: pass
            
        pen = QPen(QColor("#FFD700"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._roi_rect = self._scene.addRect(col0, row0, w, h, pen)
        self._roi_rect.setZValue(10)
        
        # Add gray-out mask for the outside
        scene_rect = self._scene.sceneRect()
        if not scene_rect.isEmpty():
            path = QPainterPath()
            path.addRect(scene_rect)
            roi_path = QPainterPath()
            roi_path.addRect(col0, row0, w, h)
            path.addPath(roi_path)
            path.setFillRule(Qt.FillRule.OddEvenFill)
            self._roi_mask_item = self._scene.addPath(path, QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0, 150)))
            self._roi_mask_item.setZValue(9)

    def show_mask(self, vertices: list[list[int | float]], dilate_px: float = 0.0) -> None:
        try:
            if self._mask_poly and self._mask_poly.scene():
                self._scene.removeItem(self._mask_poly)
        except RuntimeError: pass
        try:
            if hasattr(self, '_mask_dilate_item') and self._mask_dilate_item and self._mask_dilate_item.scene():
                self._scene.removeItem(self._mask_dilate_item)
        except RuntimeError: pass
            
        if len(vertices) < 3:
            return
            
        pts = [QPointF(v[1], v[0]) for v in vertices]  # col, row → x, y
        poly = QPolygonF(pts)
        
        # Dilate preview
        if dilate_px > 0:
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            stroker = QPainterPathStroker()
            stroker.setWidth(dilate_px * 2)
            stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
            dilate_path = stroker.createStroke(path)
            self._mask_dilate_item = self._scene.addPath(dilate_path, QPen(Qt.PenStyle.NoPen), QBrush(QColor(42, 154, 184, 80)))
            self._mask_dilate_item.setZValue(9)
            
        pen = QPen(QColor("#2A9AB8"), 2.0)
        pen.setCosmetic(True)
        self._mask_poly = self._scene.addPolygon(
            poly, pen, QBrush(QColor(42, 154, 184, 40))
        )
        self._mask_poly.setZValue(10)
        # vertex labels
        labels = ["FL", "FR", "BR", "BL"]
        for i, v in enumerate(vertices[:4]):
            handle = _Handle(v[1], v[0], i, "#2A9AB8")
            self._scene.addItem(handle)
            self._handles.append(handle)

    # ── Zoom ───────────────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None: return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom_factor *= factor
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None: return
        mod = event.modifiers()
        key = event.key()
        if mod == Qt.KeyboardModifier.ControlModifier:
            if key in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
                self.scale(1.15, 1.15); self._zoom_factor *= 1.15; return
            elif key == Qt.Key.Key_Minus:
                self.scale(1/1.15, 1/1.15); self._zoom_factor /= 1.15; return
            elif key == Qt.Key.Key_0:
                self.fit_in_view(); return
            elif key == Qt.Key.Key_Z:
                self._undo_last_point(); return
        if key == Qt.Key.Key_Escape:
            self.set_annotation_mode(AnnotationMode.NONE); return
        super().keyPressEvent(event)

    # ── Pan ────────────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None: return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # Right-click = undo last point
        if event.button() == Qt.MouseButton.RightButton and self._mode != AnnotationMode.NONE:
            self._undo_last_point()
            return

        # Left-click annotation
        if event.button() == Qt.MouseButton.LeftButton and self._mode != AnnotationMode.NONE:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._place_point(scene_pos)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None: return
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        # Pixel hover
        if self._qimage and self._pixmap_item:
            x, y = int(scene_pos.x()), int(scene_pos.y())
            if 0 <= x < self._qimage.width() and 0 <= y < self._qimage.height():
                c = self._qimage.pixelColor(x, y)
                gray = int(0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue())
                self.pixel_hovered.emit(y, x, gray)

        # Crosshair
        if self.crosshair_enabled and self._pixmap_item:
            self._update_crosshair(scene_pos)

        # Preview line from last point to cursor
        if self._mode != AnnotationMode.NONE and self._click_points:
            self._update_preview_line(scene_pos)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None: return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self._mode != AnnotationMode.NONE else Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if self._mode == AnnotationMode.NONE:
            self.fit_in_view()

    # ── Point placement ────────────────────────────────────────────
    def _place_point(self, scene_pos: QPointF) -> None:
        row, col = int(scene_pos.y()), int(scene_pos.x())
        self._click_points.append(scene_pos)
        idx = len(self._click_points) - 1

        # Visual handle
        color = "#FFD700" if self._mode == AnnotationMode.ROI else "#2A9AB8"
        handle = _Handle(scene_pos.x(), scene_pos.y(), idx, color)
        self._scene.addItem(handle)
        self._handles.append(handle)

        # Draw connecting line to previous point
        if idx > 0:
            prev = self._click_points[idx - 1]
            pen = QPen(QColor(color), 1.5)
            pen.setCosmetic(True)
            ln = self._scene.addLine(prev.x(), prev.y(), scene_pos.x(), scene_pos.y(), pen)
            ln.setZValue(15)
            self._preview_lines.append(ln)

        self.point_placed.emit(idx, row, col)

        # Check completion
        if self._mode == AnnotationMode.ROI and len(self._click_points) == 2:
            self._complete_roi()
        elif self._mode == AnnotationMode.MASK and len(self._click_points) == 4:
            self._complete_mask()

    def _undo_last_point(self) -> None:
        if not self._click_points:
            return
        self._click_points.pop()
        # Remove last handle and preview line
        if self._handles:
            h = self._handles.pop()
            try:
                if h.scene(): self._scene.removeItem(h)
            except RuntimeError: pass
        if self._preview_lines:
            ln = self._preview_lines.pop()
            try:
                if ln.scene(): self._scene.removeItem(ln)
            except RuntimeError: pass

    def _update_preview_line(self, cursor_pos: QPointF) -> None:
        """Draw a dashed line from last placed point to cursor."""
        # We use a temporary line stored as the last item in _preview_lines tagged
        tag = "_cursor_preview"
        # Remove old cursor preview if any
        try:
            if hasattr(self, "_cursor_line") and self._cursor_line and self._cursor_line.scene():
                self._scene.removeItem(self._cursor_line)
        except RuntimeError: pass
        last = self._click_points[-1]
        color = "#FFD700" if self._mode == AnnotationMode.ROI else "#2A9AB8"
        pen = QPen(QColor(color), 1.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._cursor_line = self._scene.addLine(last.x(), last.y(), cursor_pos.x(), cursor_pos.y(), pen)
        self._cursor_line.setZValue(15)

    # ── Completion ─────────────────────────────────────────────────
    def _complete_roi(self) -> None:
        p0, p1 = self._click_points
        row0 = int(min(p0.y(), p1.y()))
        col0 = int(min(p0.x(), p1.x()))
        row1 = int(max(p0.y(), p1.y()))
        col1 = int(max(p0.x(), p1.x()))
        h = row1 - row0
        w = col1 - col0

        # Clear temp items FIRST
        self._clear_annotation_items()
        self._click_points.clear()
        self._mode = AnnotationMode.NONE
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.crosshair_enabled = False
        self._remove_crosshair()

        # Draw final rect
        self.show_roi(row0, col0, h, w)

        self.roi_completed.emit(row0, col0, h, w)

    def _complete_mask(self) -> None:
        # 4 points: FL(0) → FR(1) → BR(2) → BL(3)
        verts = [[int(p.y()), int(p.x())] for p in self._click_points]

        # Forward direction = normalize(centroid(0,1) - centroid(2,3))
        cy_front = (verts[0][0] + verts[1][0]) / 2.0
        cx_front = (verts[0][1] + verts[1][1]) / 2.0
        cy_back = (verts[2][0] + verts[3][0]) / 2.0
        cx_back = (verts[2][1] + verts[3][1]) / 2.0
        dy = cy_front - cy_back
        dx = cx_front - cx_back
        length = math.sqrt(dy * dy + dx * dx)
        if length > 0:
            fwd = [round(dy / length, 4), round(dx / length, 4)]
        else:
            fwd = [0.0, 0.0]

        # Clear temp items FIRST
        self._clear_annotation_items()
        self._click_points.clear()
        self._mode = AnnotationMode.NONE
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.crosshair_enabled = False
        self._remove_crosshair()

        # Draw final polygon
        self.show_mask(verts)

        self.mask_completed.emit(verts, fwd)

    # ── Crosshair ──────────────────────────────────────────────────
    def _update_crosshair(self, scene_pos: QPointF) -> None:
        rect = self._scene.sceneRect()
        if rect.isEmpty(): return
        x, y = scene_pos.x(), scene_pos.y()
        
        if self._crosshair_h is None:
            self._crosshair_h = self._scene.addLine(rect.left(), y, rect.right(), y, self._ch_pen)
            self._crosshair_h.setZValue(30)
        else:
            self._crosshair_h.setLine(rect.left(), y, rect.right(), y)
            
        if self._crosshair_v is None:
            self._crosshair_v = self._scene.addLine(x, rect.top(), x, rect.bottom(), self._ch_pen)
            self._crosshair_v.setZValue(30)
        else:
            self._crosshair_v.setLine(x, rect.top(), x, rect.bottom())

    def _remove_crosshair(self) -> None:
        try:
            if self._crosshair_h and self._crosshair_h.scene():
                self._scene.removeItem(self._crosshair_h)
        except RuntimeError: pass
        
        try:
            if self._crosshair_v and self._crosshair_v.scene():
                self._scene.removeItem(self._crosshair_v)
        except RuntimeError: pass
        
        self._crosshair_h = None
        self._crosshair_v = None

    def set_crosshair(self, enabled: bool) -> None:
        self.crosshair_enabled = enabled
        if not enabled:
            self._remove_crosshair()


# ── Preview Widget ─────────────────────────────────────────────────
class PreviewWidget(QWidget):
    """Central preview: image viewer + frame slider + annotation signals.

    Signals:
        frame_changed(int): slider frame change.
        pixel_hovered(int, int, int): row, col, grayscale.
        roi_completed(int, int, int, int): row0, col0, height, width.
        mask_completed(list, list): 4 vertices [[row,col],...], fwd [dy,dx].
        point_placed(int, int, int): point_index, row, col.
    """
    frame_changed = pyqtSignal(int)
    pixel_hovered = pyqtSignal(int, int, int)
    roi_completed = pyqtSignal(int, int, int, int)
    mask_completed = pyqtSignal(list, list)
    point_placed = pyqtSignal(int, int, int)
    preview_mode_changed = pyqtSignal(bool)  # True=per-frame eta, False=mean eta
    export_png_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame_count = 0
        self._eta_state: dict | None = None
        self._display_mode_registry = DisplayModeRegistry()
        self._current_display_mode = DisplayMode.ORIGINAL
        # Seed registry with ORIGINAL mode (no-op painter; image already shown by _view)
        self._display_mode_registry.register(DisplayModeSpec(
            mode=DisplayMode.ORIGINAL,
            label="Original",
            tooltip="Show source image only",
            paint=lambda w, eta: None,
        ))
        self._setup_ui()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Display controls (shown only when eta is active) ─────────
        self._display_bar = QWidget()
        db = QHBoxLayout(self._display_bar)
        db.setContentsMargins(10, 3, 10, 3)
        db.setSpacing(18)

        self._overlap_cb = QCheckBox("Overlap")
        self._overlap_cb.setChecked(True)
        self._overlap_cb.toggled.connect(self._on_overlap_toggled)
        db.addWidget(self._overlap_cb)

        self._colorbar_cb = QCheckBox("Colorbar")
        self._colorbar_cb.setChecked(False)
        self._colorbar_cb.toggled.connect(self._on_colorbar_toggled)
        db.addWidget(self._colorbar_cb)

        # Per-frame preview toggle (relevant only after Run; instant toggle
        # between showing each frame's η and the aggregate eta_mean)
        self._preview_cb = QCheckBox("Preview")
        self._preview_cb.setChecked(True)
        self._preview_cb.setToolTip(
            "ON: show this frame's η from the last Run\n"
            "OFF: show the aggregate eta_mean across all frames\n"
            "(toggle is instant — no recomputation)"
        )
        self._preview_cb.toggled.connect(self.preview_mode_changed)
        self._preview_cb.setVisible(False)  # hidden until a Run completes
        db.addWidget(self._preview_cb)

        db.addStretch()
        self._display_bar.setVisible(False)
        layout.addWidget(self._display_bar)

        # ── Image view ───────────────────────────────────────────────
        self._view = _ImageView(self)
        self._view.pixel_hovered.connect(self.pixel_hovered)
        self._view.roi_completed.connect(self.roi_completed)
        self._view.mask_completed.connect(self.mask_completed)
        self._view.point_placed.connect(self.point_placed)
        layout.addWidget(self._view, 1)

        # ── Colorbar strip ───────────────────────────────────────────
        self._colorbar_label = QLabel()
        self._colorbar_label.setFixedHeight(52)
        self._colorbar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._colorbar_label.setScaledContents(False)
        self._colorbar_label.setVisible(False)
        layout.addWidget(self._colorbar_label)

        # ── Frame slider bar ─────────────────────────────────────────
        self._slider_bar = QWidget()
        sl = QHBoxLayout(self._slider_bar)
        sl.setContentsMargins(8, 2, 8, 2)
        sl.setSpacing(6)

        self._frame_label = QLabel("Frame:")
        sl.addWidget(self._frame_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        sl.addWidget(self._slider, 1)

        self._index_label = QLabel("0 / 0")
        self._index_label.setFixedWidth(80)
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        sl.addWidget(self._index_label)

        self._slider_bar.setVisible(False)
        layout.addWidget(self._slider_bar)

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background: {tokens.BG_PRIMARY};")
        self._display_bar.setStyleSheet(
            f"background: {tokens.BG_SECONDARY}; border-bottom: 1px solid {tokens.BORDER_SUBTLE};"
        )
        for cb in (self._overlap_cb, self._colorbar_cb):
            cb.setStyleSheet(f"font-size: 11px; color: {tokens.TEXT_PRIMARY};")
        self._colorbar_label.setStyleSheet(f"background: {tokens.BG_SECONDARY};")
        self._slider_bar.setStyleSheet(
            f"background: {tokens.BG_SECONDARY}; border-top: 1px solid {tokens.BORDER_SUBTLE};"
        )
        self._frame_label.setStyleSheet(f"font-size: 11px; color: {tokens.TEXT_SECONDARY};")
        self._index_label.setStyleSheet(
            f"font-size: 11px; font-family: {tokens.FONT_MONO}; color: {tokens.TEXT_SECONDARY};"
        )
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background: {tokens.BG_TERTIARY}; height: 4px; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {tokens.ACCENT_CLAY}; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }}
            QSlider::sub-page:horizontal {{ background: {tokens.ACCENT_CLAY}; border-radius: 2px; }}
        """)

    # ── Public API ─────────────────────────────────────────────────
    def set_image(self, path: Path | str) -> None:
        self._view.set_image(path)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._view.set_pixmap(pixmap)

    def clear(self) -> None:
        self._view.clear_image()

    def fit(self) -> None:
        self._view.fit_in_view()

    def set_crosshair(self, enabled: bool) -> None:
        self._view.set_crosshair(enabled)

    def start_roi_mode(self) -> None:
        """Enter ROI annotation: 2-click rectangle."""
        self._view.set_annotation_mode(AnnotationMode.ROI)

    def start_mask_mode(self) -> None:
        """Enter Mask annotation: 4-click clockwise FL→FR→BR→BL."""
        self._view.set_annotation_mode(AnnotationMode.MASK)

    def cancel_annotation(self) -> None:
        self._view.set_annotation_mode(AnnotationMode.NONE)

    def show_roi(self, row0: int, col0: int, h: int, w: int) -> None:
        self._view.show_roi(row0, col0, h, w)

    def show_mask(self, vertices: list[list[int | float]], dilate_px: float = 0.0) -> None:
        self._view.show_mask(vertices, dilate_px)

    def show_eta_overlay(
        self,
        eta_mm: np.ndarray,
        *,
        opacity: float = 0.55,
        cmap_name: str = "RdBu_r",
        show_overlap: bool = True,
    ) -> None:
        # Sync the _ImageView overlap flag before rendering.
        self._view._overlap_on = show_overlap
        result = self._view.show_eta_overlay(eta_mm, opacity=opacity, cmap_name=cmap_name)
        # display_bar (Overlap/Colorbar) now lives in the toolbar — keep hidden.
        if result is not None:
            vmin, vmax = result
            self._eta_state = {"vmin": vmin, "vmax": vmax, "cmap_name": cmap_name}
            if self._colorbar_cb.isChecked():
                self._render_colorbar(vmin, vmax, cmap_name)
                self._colorbar_label.setVisible(True)

    def clear_eta_overlay(self) -> None:
        """Remove only the η overlay — leaves ROI and mask intact."""
        self._view.clear_eta_overlay()
        self._eta_state = None
        self._colorbar_label.setVisible(False)
        self._colorbar_label.clear()

    def clear_overlays(self) -> None:
        self._view.clear_overlays()
        self._eta_state = None
        self._display_bar.setVisible(False)
        self._colorbar_label.setVisible(False)
        self._colorbar_label.clear()

    def _on_overlap_toggled(self, val: bool) -> None:
        self._view.set_overlap(val)

    def _on_colorbar_toggled(self, val: bool) -> None:
        if val and self._eta_state:
            self._render_colorbar(
                self._eta_state["vmin"],
                self._eta_state["vmax"],
                self._eta_state["cmap_name"],
            )
            self._colorbar_label.setVisible(True)
        else:
            self._colorbar_label.setVisible(False)

    def _render_colorbar(self, vmin: float, vmax: float, cmap_name: str) -> None:
        try:
            from matplotlib.figure import Figure
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            from openfcd.gui import tokens as _tok
            bg_hex = _tok.BG_SECONDARY
            fg_hex = _tok.TEXT_SECONDARY

            dpr = float(self.devicePixelRatioF() or 1.0)
            label_w_px = max(self._colorbar_label.width(), 600)
            label_h_px = max(self._colorbar_label.height(), 52)

            dpi = 110.0
            fig_w_in = label_w_px / dpi
            fig_h_in = label_h_px / dpi

            fig = Figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            fig.patch.set_facecolor(bg_hex)
            ax = fig.add_axes([0.04, 0.42, 0.92, 0.30])
            ax.set_facecolor(bg_hex)
            sm = ScalarMappable(cmap=cmap_name, norm=Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
            cb.set_label("η (mm)", color=fg_hex, fontsize=9)
            ax.tick_params(labelsize=8, colors=fg_hex)
            for spine in ax.spines.values():
                spine.set_edgecolor(fg_hex)

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                facecolor=fig.get_facecolor(),
                dpi=dpi * dpr,
            )
            buf.seek(0)
            qimg = QImage()
            qimg.loadFromData(buf.read())
            qimg.setDevicePixelRatio(dpr)
            self._colorbar_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def set_overlap(self, val: bool) -> None:
        self._overlap_cb.setChecked(val)

    def set_colorbar(self, val: bool) -> None:
        self._colorbar_cb.setChecked(val)

    def set_preview_toggle_visible(self, visible: bool) -> None:
        """Show/hide the per-frame preview toggle (shown after a Run completes)."""
        self._preview_cb.setVisible(visible)

    def set_preview_mode(self, enabled: bool) -> None:
        self._preview_cb.blockSignals(True)
        self._preview_cb.setChecked(enabled)
        self._preview_cb.blockSignals(False)

    def preview_mode(self) -> bool:
        return self._preview_cb.isChecked()

    def setup_slider(self, frame_count: int, current: int = 0) -> None:
        self._frame_count = frame_count
        if frame_count > 1:
            self._slider.setMaximum(frame_count - 1)
            self._slider.setValue(current)
            self._index_label.setText(f"{current + 1} / {frame_count}")
            self._slider_bar.setVisible(True)
        else:
            self._slider_bar.setVisible(False)

    def hide_slider(self) -> None:
        self._slider_bar.setVisible(False)

    def set_slider_value(self, index: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._index_label.setText(f"{index + 1} / {self._frame_count}")
        self._slider.blockSignals(False)

    def set_scene_slider(self, frame_indices: list[int]) -> None:
        """Remap slider to a scene's frame subset (0..len-1 ticks)."""
        self._scene_frame_indices = list(frame_indices)
        if frame_indices:
            self._slider.blockSignals(True)
            self._slider.setRange(0, len(frame_indices) - 1)
            self._slider.setValue(0)
            self._slider.blockSignals(False)
            self._slider.setVisible(len(frame_indices) > 1)
        else:
            self._slider.setVisible(False)

    def restore_full_slider(self, total_frames: int) -> None:
        """Restore slider to full frame range after leaving a scene view."""
        self._scene_frame_indices = []
        if total_frames > 0:
            self._slider.blockSignals(True)
            self._slider.setRange(0, total_frames - 1)
            self._slider.blockSignals(False)
            self._slider.setVisible(total_frames > 1)

    def current_slider_value(self) -> int:
        """Return current slider position."""
        return self._slider.value()

    def _on_slider_changed(self, value: int) -> None:
        self._index_label.setText(f"{value + 1} / {self._frame_count}")
        self.frame_changed.emit(value)

    def contextMenuEvent(self, event) -> None:
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Export PNG…", lambda: self.export_png_requested.emit())
        menu.exec(event.globalPos())
