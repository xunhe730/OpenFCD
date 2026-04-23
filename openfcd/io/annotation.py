"""Annotation data model — polygons, ROIs, and mask generation.

Stores per-condition annotation data: manual polygon vertices, ROI rect,
anchor frame, and dilate settings.  Supports conversion to binary mask
arrays for the FCD pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator


class PolygonData(BaseModel):
    """A single closed polygon defined by (row, col) vertices."""
    vertices: list[list[float]]
    label: str = "body"
    visible: bool = True

    @field_validator("vertices")
    @classmethod
    def _require_min_vertices(cls, v: list) -> list:
        if len(v) < 3:
            raise ValueError(f"A polygon needs at least 3 vertices, got {len(v)}")
        return v

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def as_rc_array(self) -> np.ndarray:
        """Return vertices as (N, 2) array of (row, col)."""
        return np.array(self.vertices, dtype=np.float64)


class ROIData(BaseModel):
    """Axis-aligned region of interest rectangle."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0


class AnnotationSchema(BaseModel):
    """Complete annotation for one experimental condition / batch."""
    condition: str = "default"
    frame_range: str = ""
    anchor_frame: str = ""
    polygons: list[PolygonData] = Field(default_factory=list)
    frame_polygons: dict[str, list[PolygonData]] = Field(default_factory=dict)
    roi: ROIData = Field(default_factory=ROIData)
    dilate_cells: float = 3.0
    cell_mm: float = 1.2
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    note: str = ""

    # ── Legacy compat: accept old flat polygon format ───────────────
    @field_validator("polygons", mode="before")
    @classmethod
    def _coerce_legacy(cls, v):
        """Accept a bare list-of-lists (old manual_polygon format)."""
        if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
            # Check if it's already a list of PolygonData dicts
            if isinstance(v[0], dict):
                return v
            # It's a plain list of [row, col] pairs → wrap in one PolygonData
            return [{"vertices": v, "label": "body"}]
        return v

    # ── Derived ────────────────────────────────────────────────────
    @property
    def dilate_px(self) -> float:
        """Convert dilate_cells × cell_mm to rough pixel estimate (needs px_per_mm)."""
        return self.dilate_cells * self.cell_mm

    def primary_polygon(self) -> PolygonData | None:
        """Return the first visible polygon, or None."""
        for p in self.polygons:
            if p.visible:
                return p
        return None


# ── IO ──────────────────────────────────────────────────────────────────

def load(path: str | Path) -> AnnotationSchema:
    """Load annotation from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle legacy format: if 'manual_polygon' key exists, convert
    if "manual_polygon" in data and "polygons" not in data:
        data["polygons"] = [{"vertices": data.pop("manual_polygon"), "label": "body"}]
    if isinstance(data.get("roi"), dict) and "x" not in data["roi"]:
        # Old format had arbitrary dict → convert
        roi = data["roi"]
        data["roi"] = {
            "x": roi.get("x_mm", roi.get("x", 0)),
            "y": roi.get("y_mm", roi.get("y", 0)),
            "width": roi.get("width", 0),
            "height": roi.get("height", 0),
        }
    return AnnotationSchema.model_validate(data)


def save(path: str | Path, ann: AnnotationSchema) -> None:
    """Save annotation to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ann.model_dump(), f, indent=2, ensure_ascii=False)


def validate(path: str | Path) -> list[str]:
    """Validate annotation file, returning list of error messages."""
    try:
        load(path)
        return []
    except Exception as e:
        return [str(e)]


# ── Mask generation ─────────────────────────────────────────────────────

def polygon_to_mask(
    shape: tuple[int, int],
    polygon: PolygonData,
    dilate_px: int = 0,
) -> np.ndarray:
    """Convert a PolygonData to a boolean mask array.

    Parameters
    ----------
    shape : (H, W) of the output mask
    polygon : PolygonData with vertices in (row, col) image coordinates
    dilate_px : optional dilation radius

    Returns
    -------
    np.ndarray of bool, shape (H, W)
    """
    from skimage.draw import polygon as draw_polygon

    verts = polygon.as_rc_array()
    rr, cc = draw_polygon(verts[:, 0], verts[:, 1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True

    if dilate_px > 0 and mask.any():
        from skimage.morphology import dilation, disk
        mask = dilation(mask, disk(dilate_px))

    return mask


def annotation_to_mask(
    shape: tuple[int, int],
    ann: AnnotationSchema,
    px_per_mm: float = 7.27,
    frame_name: str | None = None,
) -> np.ndarray:
    """Generate combined mask from all visible polygons in the annotation.

    Parameters
    ----------
    shape : (H, W) image shape
    ann : AnnotationSchema
    px_per_mm : pixels per mm for dilation conversion

    Returns
    -------
    np.ndarray of bool — True = masked / occluded region
    """
    dilate_px = int(ann.dilate_cells * ann.cell_mm * px_per_mm + 0.5)
    combined = np.zeros(shape, dtype=bool)

    poly_list = ann.polygons
    if frame_name and frame_name in ann.frame_polygons:
        poly_list = ann.frame_polygons[frame_name]

    for poly in poly_list:
        if poly.visible and poly.vertex_count >= 3:
            m = polygon_to_mask(shape, poly, dilate_px=dilate_px)
            combined |= m

    return combined


def roi_to_slice(
    roi: ROIData,
    shape: tuple[int, int],
    px_per_mm: float = 1.0,
) -> tuple[slice, slice]:
    """Convert ROI to numpy slices (row_slice, col_slice).

    If ROI is empty, returns slices covering the full image.
    """
    if roi.is_empty:
        return slice(None), slice(None)

    r0 = max(0, int(roi.y * px_per_mm))
    c0 = max(0, int(roi.x * px_per_mm))
    r1 = min(shape[0], int((roi.y + roi.height) * px_per_mm))
    c1 = min(shape[1], int((roi.x + roi.width) * px_per_mm))
    return slice(r0, r1), slice(c0, c1)
