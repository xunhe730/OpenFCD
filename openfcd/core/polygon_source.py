"""PolygonSource Protocol — decouples polygon supply from FCD core."""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np

from openfcd.core.mask import Polygon


def _frame_id_from_filename(stem: str) -> int:
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else -1


@runtime_checkable
class PolygonSource(Protocol):
    def get(self, frame_id: int) -> Polygon | None: ...
    def kind(self) -> str: ...


class FixedPolygonSource:
    def __init__(self, polygon: Polygon) -> None:
        self._polygon = polygon

    def get(self, frame_id: int) -> Polygon:
        return self._polygon

    def kind(self) -> str:
        return "fixed"


class TrackedPolygonSource:
    def __init__(self, by_frame: dict[int, Polygon]) -> None:
        self._by_frame = by_frame

    @classmethod
    def from_polygons_json(cls, path: str, *,
                           inflated: bool = True) -> "TrackedPolygonSource":
        with open(path) as f:
            records = json.load(f)
        poly_key = "polygon_inflated_full_frame" if inflated else "polygon_full_frame"
        by_frame: dict[int, Polygon] = {}
        for rec in records:
            verts_data = rec.get(poly_key)
            if verts_data is None:
                continue
            fid = _frame_id_from_filename(rec["frame"])
            by_frame[fid] = Polygon([tuple(v) for v in verts_data])
        return cls(by_frame)

    def get(self, frame_id: int) -> Polygon | None:
        return self._by_frame.get(frame_id)

    def kind(self) -> str:
        return "tracked"


class AutoPolygonSource:
    def __init__(self, ref_image: np.ndarray, carriers: list, *,
                 threshold_ratio: float = 0.3, dilate_px: int = 4) -> None:
        self._ref = ref_image
        self._carriers = carriers
        self._threshold_ratio = threshold_ratio
        self._dilate_px = dilate_px
        self._call_count = 0

    def get(self, frame_id: int) -> Polygon | None:
        from openfcd.core.mask import auto_mask, find_oriented_polygon
        self._call_count += 1
        if self._call_count == 11:
            print(
                "[AutoPolygonSource] warning: auto-detecting polygon on every frame "
                "is slow. Pre-compute with scripts/track_polygon.py and pass "
                "--polygons-json to use TrackedPolygonSource instead."
            )
        mask = auto_mask(self._ref, self._carriers,
                         threshold_ratio=self._threshold_ratio,
                         dilate_px=self._dilate_px)
        return find_oriented_polygon(mask, edge_margin=20)

    def kind(self) -> str:
        return "bootstrap"


def find_oriented_polygon(mask: np.ndarray,
                           edge_margin: int = 20) -> Polygon | None:
    """OpenCV minAreaRect on the largest interior mask blob → oriented polygon."""
    from openfcd.core.mask import find_oriented_polygon as _impl
    return _impl(mask, edge_margin=edge_margin)


def estimate_object_dimensions(body_mask: np.ndarray,
                                center_rc: tuple[float, float],
                                angle_deg: float,
                                percentile: float = 98.0,
                                min_pixels: int = 30
                                ) -> tuple[float, float] | None:
    """Measure the object's long / short extent from mask pixels."""
    from openfcd.core.mask import estimate_object_dimensions as _impl
    return _impl(body_mask, center_rc, angle_deg, percentile, min_pixels)
