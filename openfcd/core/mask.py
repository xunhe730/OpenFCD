"""Detect occluding objects in an FCD image."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.morphology import dilation, binary_erosion, disk

from openfcd.core.fcd import Carrier, carrier_amplitude


@dataclass
class Box:
    """Axis-aligned box, image coords (row0, col0, height, width)."""
    row0: int
    col0: int
    height: int
    width: int


@dataclass
class Polygon:
    """Closed polygon, vertices as list of (row, col) tuples."""
    vertices: list   # [(row, col), ...] in image coordinates

    @classmethod
    def from_box(cls, b: Box) -> "Polygon":
        r0, c0 = b.row0, b.col0
        r1, c1 = r0 + b.height, c0 + b.width
        return cls([(r0, c0), (r0, c1), (r1, c1), (r1, c0)])

    @classmethod
    def from_oriented_rect(cls, center_rc: tuple, size_hw: tuple,
                            angle_deg: float) -> "Polygon":
        """center_rc=(row, col), size_hw=(h, w), angle in degrees CCW."""
        cy, cx = center_rc
        h, w = size_hw
        a = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        dx = w / 2.0
        dy = h / 2.0
        local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        world = local @ rot.T
        verts = [(cy + p[1], cx + p[0]) for p in world]
        return cls(verts)

    def bbox(self) -> Box:
        rs = [v[0] for v in self.vertices]
        cs = [v[1] for v in self.vertices]
        r0 = int(min(rs)); c0 = int(min(cs))
        return Box(r0, c0, int(max(rs)) - r0, int(max(cs)) - c0)

    def centroid(self) -> tuple:
        rs = [v[0] for v in self.vertices]
        cs = [v[1] for v in self.vertices]
        return (sum(rs) / len(rs), sum(cs) / len(cs))

    def shifted(self, drow: int, dcol: int) -> "Polygon":
        return Polygon([(v[0] + drow, v[1] + dcol) for v in self.vertices])


def auto_mask(img: np.ndarray,
              carriers: Optional[Sequence[Carrier]] = None,
              carrier_period_px: float = 14.0,
              threshold_ratio: float = 0.35,
              dilate_px: int = 8) -> np.ndarray:
    """Automatic occlusion mask.

    If ``carriers`` is provided, use the demodulated envelope (more
    physically grounded). Otherwise fall back to a simple bandpass-energy
    metric.
    """
    if carriers is not None:
        energy = carrier_amplitude(img, carriers)
    else:
        bg = gaussian_filter(img.astype(float), sigma=carrier_period_px * 2)
        hp = img.astype(float) - bg
        energy = uniform_filter(hp ** 2, size=int(carrier_period_px * 2)) ** 0.5

    pos = energy[energy > 0]
    ref = float(np.median(pos)) if pos.size else 1.0
    raw = energy < (threshold_ratio * ref)
    raw = binary_erosion(raw, disk(2))     # remove speckles
    if raw.any():
        raw = dilation(raw, disk(dilate_px))
    return raw


def detect_filament_occluders(
    img: np.ndarray,
    carriers: Sequence[Carrier],
    *,
    threshold_ratio: float = 0.78,
    min_area_px: int = 12,
    min_length_px: int = 10,
    max_width_px: int = 14,
    min_aspect: float = 2.5,
    dilate_px: int = 2,
) -> np.ndarray:
    """Detect thin elongated occluders such as wires.

    Physically these are regions where the checkerboard carrier envelope
    collapses along a narrow filament. We intentionally *do not* erode first,
    because that would erase the very structures we want to preserve.
    """
    from skimage.measure import label, regionprops

    energy = carrier_amplitude(img, carriers)
    pos = energy[energy > 0]
    ref = float(np.median(pos)) if pos.size else 1.0
    raw = energy < (threshold_ratio * ref)

    lab = label(raw.astype(np.uint8))
    out = np.zeros_like(raw, dtype=bool)

    for region in regionprops(lab):
        minr, minc, maxr, maxc = region.bbox
        h = maxr - minr
        w = maxc - minc
        long_axis = max(h, w)
        short_axis = max(1, min(h, w))
        aspect = long_axis / short_axis

        if region.area < min_area_px:
            continue
        if long_axis < min_length_px:
            continue
        if short_axis > max_width_px:
            continue
        if aspect < min_aspect:
            continue

        out |= lab == region.label

    if dilate_px > 0 and out.any():
        out = dilation(out, disk(dilate_px))
    return out


def manual_mask(shape: tuple, boxes: Sequence[Box],
                dilate_px: int = 0) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for b in boxes:
        r0 = max(0, b.row0)
        c0 = max(0, b.col0)
        r1 = min(shape[0], b.row0 + b.height)
        c1 = min(shape[1], b.col0 + b.width)
        m[r0:r1, c0:c1] = True
    if dilate_px > 0 and m.any():
        m = dilation(m, disk(dilate_px))
    return m


def rectangular_mask(shape: tuple, box: Box, pad_px: int = 0) -> np.ndarray:
    """Single rectangular boolean mask. Identical shape across frames."""
    return manual_mask(shape, [Box(box.row0 - pad_px, box.col0 - pad_px,
                                    box.height + 2 * pad_px,
                                    box.width + 2 * pad_px)])


def polygon_mask(shape: tuple, poly: Polygon,
                 dilate_px: int = 0) -> np.ndarray:
    """Boolean mask filled inside the polygon."""
    from skimage.draw import polygon as draw_polygon
    rr, cc = draw_polygon([v[0] for v in poly.vertices],
                          [v[1] for v in poly.vertices],
                          shape=shape)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    if dilate_px > 0 and m.any():
        m = dilation(m, disk(dilate_px))
    return m


def find_oriented_polygon(mask: np.ndarray,
                           edge_margin: int = 20) -> Optional[Polygon]:
    """OpenCV minAreaRect on the largest interior mask blob → oriented polygon."""
    import cv2
    from skimage.measure import label, regionprops
    lab = label(mask.astype(np.uint8))
    h, w = mask.shape
    biggest = None
    best_area = 0
    for r in regionprops(lab):
        r0, c0, r1, c1 = r.bbox
        if (r0 < edge_margin or c0 < edge_margin
                or r1 > h - edge_margin or c1 > w - edge_margin):
            continue
        if r.area > best_area:
            best_area = r.area
            biggest = r
    if biggest is None:
        return None
    blob = (lab == biggest.label).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    big_contour = max(contours, key=cv2.contourArea)
    (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(big_contour)
    pts = cv2.boxPoints(((cx, cy), (rect_w, rect_h), angle))
    verts = [(float(p[1]), float(p[0])) for p in pts]   # (row, col)
    return Polygon(verts)


def estimate_object_dimensions(body_mask: np.ndarray,
                                center_rc: tuple[float, float],
                                angle_deg: float,
                                percentile: float = 98.0,
                                min_pixels: int = 30
                                ) -> Optional[tuple[float, float]]:
    """Measure the object's long / short extent from mask pixels.

    Given a tight body mask and a long-axis angle (deg, from +col axis,
    (row, col) basis), project every mask pixel onto the long and short
    axis directions and return ``(L, S)`` as the robust span (p98-p2 by
    default). Returns ``None`` if the mask has fewer than ``min_pixels``
    active pixels.

    The percentile trim suppresses a few stray pixels (wire remnants,
    speckle) without needing shape priors, so this works for any object
    shape as long as the mask roughly covers the body.
    """
    if not body_mask.any():
        return None
    ys, xs = np.where(body_mask)
    if ys.size < min_pixels:
        return None
    cy, cx = center_rc
    a = np.deg2rad(angle_deg)
    dy = ys - cy
    dx = xs - cx
    proj_long = dy * np.sin(a) + dx * np.cos(a)
    proj_short = dy * np.cos(a) - dx * np.sin(a)
    lo = 100.0 - percentile
    hi = percentile
    span_long = float(np.percentile(proj_long, hi) - np.percentile(proj_long, lo))
    span_short = float(np.percentile(proj_short, hi) - np.percentile(proj_short, lo))
    L = max(span_long, span_short)
    S = min(span_long, span_short)
    if L < 1.0 or S < 1.0:
        return None
    return (L, S)


def union(*masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m
    return out


def find_largest_interior_blob(mask: np.ndarray, edge_margin: int = 20,
                               cluster_dilate_px: int = 6) -> Optional[Box]:
    """Largest mask blob whose bbox does not touch image edges.

    The hand enters from a corner; the robot lives in the middle. So this
    returns the robot bbox while ignoring the hand.

    `cluster_dilate_px` lightly closes nearby specks so the robot's
    fragmented mask becomes one connected component before regionprops.
    """
    from skimage.morphology import dilation
    work = dilation(mask, disk(cluster_dilate_px)) if mask.any() else mask
    from skimage.measure import label, regionprops
    lab = label(work.astype(np.uint8))
    h, w = mask.shape
    candidates = []
    for r in regionprops(lab):
        r0, c0, r1, c1 = r.bbox
        if (r0 < edge_margin or c0 < edge_margin
                or r1 > h - edge_margin or c1 > w - edge_margin):
            continue
        candidates.append(r)
    if not candidates:
        return None
    biggest = max(candidates, key=lambda r: r.area)
    r0, c0, r1, c1 = biggest.bbox
    pad = cluster_dilate_px
    return Box(int(r0 + pad), int(c0 + pad),
               int(max(1, r1 - r0 - 2 * pad)),
               int(max(1, c1 - c0 - 2 * pad)))
