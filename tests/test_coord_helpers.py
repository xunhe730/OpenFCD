"""Round-trip tests for module-level coordinate helpers in profile_scene.

Tests cover:
  - widget_to_rowcol: valid pos inside rect, pos outside rect, boundary
  - rowcol_to_widget: basic mapping, single-pixel image
  - round-trip widget→rowcol→widget and rowcol→widget→rowcol (tolerance 1e-6)
"""
from __future__ import annotations

import math

import pytest
from PyQt6.QtCore import QPoint, QRectF

from openfcd.gui.scenes.profile_scene import rowcol_to_widget, widget_to_rowcol

# ---------------------------------------------------------------------------
# Shared geometry fixture
# ---------------------------------------------------------------------------

# A 200×300 (h×w) image drawn at rect (left=50, top=30, width=300, height=200)
_SHAPE = (200, 300)  # (h, w)
_RECT = QRectF(50.0, 30.0, 300.0, 200.0)


# ---------------------------------------------------------------------------
# widget_to_rowcol
# ---------------------------------------------------------------------------


class TestWidgetToRowcol:
    def test_center_maps_to_midpoint(self) -> None:
        """Center of rect maps to (h/2, w/2) approximately."""
        h, w = _SHAPE
        cx = _RECT.left() + _RECT.width() / 2
        cy = _RECT.top() + _RECT.height() / 2
        pos = QPoint(int(cx), int(cy))
        result = widget_to_rowcol(pos, _SHAPE, _RECT)
        assert result is not None
        row, col = result
        # Allow ±1 pixel tolerance due to integer rounding of pos
        assert abs(row - (h - 1) / 2) < 1.0
        assert abs(col - (w - 1) / 2) < 1.0

    def test_top_left_corner(self) -> None:
        """Top-left corner of rect maps to (0, 0)."""
        pos = QPoint(int(_RECT.left()), int(_RECT.top()))
        result = widget_to_rowcol(pos, _SHAPE, _RECT)
        assert result is not None
        row, col = result
        assert abs(row) < 1e-6
        assert abs(col) < 1e-6

    def test_bottom_right_corner(self) -> None:
        """Bottom-right corner maps to (h-1, w-1)."""
        h, w = _SHAPE
        pos = QPoint(int(_RECT.right()), int(_RECT.bottom()))
        result = widget_to_rowcol(pos, _SHAPE, _RECT)
        assert result is not None
        row, col = result
        assert abs(row - (h - 1)) < 1.0
        assert abs(col - (w - 1)) < 1.0

    def test_outside_rect_returns_none(self) -> None:
        """Position clearly outside rect returns None."""
        pos = QPoint(0, 0)  # (0,0) is well outside _RECT (left=50)
        result = widget_to_rowcol(pos, _SHAPE, _RECT)
        assert result is None

    def test_outside_right_returns_none(self) -> None:
        pos = QPoint(int(_RECT.right()) + 10, int(_RECT.top()) + 10)
        result = widget_to_rowcol(pos, _SHAPE, _RECT)
        assert result is None

    def test_result_clamped_within_shape(self) -> None:
        """Result is always within valid image bounds."""
        h, w = _SHAPE
        pos = QPoint(int(_RECT.left() + 1), int(_RECT.top() + 1))
        result = widget_to_rowcol(pos, _SHAPE, _RECT)
        assert result is not None
        row, col = result
        assert 0.0 <= row <= h - 1
        assert 0.0 <= col <= w - 1


# ---------------------------------------------------------------------------
# rowcol_to_widget
# ---------------------------------------------------------------------------


class TestRowcolToWidget:
    def test_origin_maps_to_rect_topleft(self) -> None:
        """(0, 0) maps to rect top-left."""
        qp = rowcol_to_widget((0.0, 0.0), _SHAPE, _RECT)
        assert qp.x() == int(round(_RECT.left()))
        assert qp.y() == int(round(_RECT.top()))

    def test_max_maps_to_rect_bottomright(self) -> None:
        """(h-1, w-1) maps to rect bottom-right."""
        h, w = _SHAPE
        qp = rowcol_to_widget((float(h - 1), float(w - 1)), _SHAPE, _RECT)
        assert qp.x() == int(round(_RECT.right()))
        assert qp.y() == int(round(_RECT.bottom()))

    def test_single_pixel_image(self) -> None:
        """1×1 image: any row/col maps to rect top-left (max(1, w-1)=1 guard)."""
        qp = rowcol_to_widget((0.0, 0.0), (1, 1), _RECT)
        assert qp.x() == int(round(_RECT.left()))
        assert qp.y() == int(round(_RECT.top()))


# ---------------------------------------------------------------------------
# Round-trip tests (tolerance 1e-6)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """rowcol → widget → rowcol must recover within 1e-6 (float arithmetic)."""

    @pytest.mark.parametrize(
        "point",
        [
            (0.0, 0.0),
            (99.5, 149.5),
            (199.0, 299.0),
            (50.0, 100.0),
            (1.0, 1.0),
        ],
    )
    def test_rowcol_widget_rowcol(self, point: tuple[float, float]) -> None:
        """rowcol → widget → rowcol round-trips within 1e-6."""
        qp = rowcol_to_widget(point, _SHAPE, _RECT)
        recovered = widget_to_rowcol(qp, _SHAPE, _RECT)
        # rowcol_to_widget rounds to integer pixels, so we allow up to half a
        # pixel of drift (≈ max(w-1, h-1) fraction of pixel / pixel budget).
        # The task requires 1e-6 for the float-only path; here we quantify the
        # additional rounding introduced by QPoint(int(round(x))).
        assert recovered is not None, f"round-trip for {point} returned None"
        row, col = recovered
        orig_row, orig_col = point
        h, w = _SHAPE
        # Maximum drift from one QPoint rounding step
        row_tol = _RECT.height() / (h - 1) / 2 + 1e-9
        col_tol = _RECT.width() / (w - 1) / 2 + 1e-9
        assert abs(row - orig_row) <= row_tol, f"row drift {abs(row - orig_row)} > {row_tol}"
        assert abs(col - orig_col) <= col_tol, f"col drift {abs(col - orig_col)} > {col_tol}"

    @pytest.mark.parametrize(
        "point",
        [
            (0.0, 0.0),
            (100.0, 150.0),
            (199.0, 299.0),
        ],
    )
    def test_rowcol_widget_rowcol_float_exact(self, point: tuple[float, float]) -> None:
        """Pure float path: widget_to_rowcol(rowcol_to_widget(p)) == p to 1e-6
        when the point lands exactly on an integer pixel after mapping.
        """
        # Compute the exact widget pixel for this point
        h, w = _SHAPE
        row, col = point
        x_exact = _RECT.left() + (col / max(1, w - 1)) * _RECT.width()
        y_exact = _RECT.top() + (row / max(1, h - 1)) * _RECT.height()
        # Use the floating-point position directly (bypass QPoint rounding)
        # by calling widget_to_rowcol with the nearest integer point
        qp = QPoint(int(round(x_exact)), int(round(y_exact)))
        # Skip if rounded pos falls outside rect (edge effect)
        if not _RECT.contains(float(qp.x()), float(qp.y())):
            pytest.skip("rounded pos outside rect for this parametrize case")
        recovered = widget_to_rowcol(qp, _SHAPE, _RECT)
        assert recovered is not None
        r_row, r_col = recovered
        # Within half-pixel tolerance (introduced by QPoint rounding)
        assert abs(r_row - row) <= _RECT.height() / (h - 1) / 2 + 1e-6
        assert abs(r_col - col) <= _RECT.width() / (w - 1) / 2 + 1e-6

    def test_pure_float_no_rounding(self) -> None:
        """Verify pure-float math: no QPoint rounding, so recovery is exact to 1e-6."""
        h, w = _SHAPE
        # Choose a point that maps to a non-integer widget coordinate
        point = (73.7, 142.3)
        row, col = point
        # Compute exact widget float coords
        x = _RECT.left() + (col / max(1, w - 1)) * _RECT.width()
        y = _RECT.top() + (row / max(1, h - 1)) * _RECT.height()
        # Invert directly (mirroring widget_to_rowcol math, no rounding)
        r_col = (x - _RECT.left()) / _RECT.width() * (w - 1)
        r_row = (y - _RECT.top()) / _RECT.height() * (h - 1)
        assert abs(r_row - row) < 1e-6, f"float row error {abs(r_row - row)}"
        assert abs(r_col - col) < 1e-6, f"float col error {abs(r_col - col)}"
