from __future__ import annotations

import json
import math
from pathlib import Path

import typer


Point = tuple[float, float]


def parse_point_pair(value: str) -> tuple[Point, Point]:
    """Parse 'x1,y1,x2,y2' into two image points."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("points must be formatted as x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError("points must contain numeric coordinates") from exc
    return (x1, y1), (x2, y2)


def grid_pixel_calibration(
    p0: Point,
    p1: Point,
    *,
    cells: float,
    cell_mm: float,
) -> dict[str, float]:
    """Calculate pixel/mm from two manually marked endpoints."""
    cells = float(cells)
    cell_mm = float(cell_mm)
    if cells <= 0:
        raise ValueError("cells must be positive")
    if cell_mm <= 0:
        raise ValueError("cell_mm must be positive")
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    pixel_distance = math.hypot(dx, dy)
    if pixel_distance <= 0:
        raise ValueError("marked points must not be identical")
    length_mm = cells * cell_mm
    return {
        "pixel_distance": pixel_distance,
        "cells": cells,
        "cell_mm": cell_mm,
        "length_mm": length_mm,
        "pixel_per_mm": pixel_distance / length_mm,
    }


def _pick_points_interactive(image_path: Path) -> tuple[Point, Point]:
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    img = imread(str(image_path))
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title("Click the two endpoints of a known checkerboard span")
    ax.set_axis_off()
    typer.echo("Click two endpoints in the image window, then close is not needed.")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("expected exactly two clicked points")
    return (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))


def _format_report(result: dict[str, float], compare_px_per_mm: float | None) -> list[str]:
    lines = [
        f"pixel_distance_px: {result['pixel_distance']:.3f}",
        f"physical_length_mm: {result['length_mm']:.6g}",
        f"pixel_per_mm: {result['pixel_per_mm']:.6f}",
    ]
    if compare_px_per_mm and compare_px_per_mm > 0:
        measured = result["pixel_per_mm"]
        ratio = compare_px_per_mm / measured
        lines.extend(
            [
                f"compare_pixel_per_mm: {compare_px_per_mm:.6f}",
                f"px_per_mm_ratio_compare_over_measured: {ratio:.6f}",
                f"eta_scale_if_replace_compare_with_measured: {ratio * ratio:.6f}",
            ]
        )
    return lines


def calibrate_grid_cmd(
    image: Path,
    *,
    cells: float,
    cell_mm: float,
    points: str | None = None,
    compare_px_per_mm: float | None = None,
    json_output: bool = False,
) -> dict[str, float]:
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    if points:
        p0, p1 = parse_point_pair(points)
    else:
        p0, p1 = _pick_points_interactive(image)
    result = grid_pixel_calibration(p0, p1, cells=cells, cell_mm=cell_mm)
    result["x0"] = float(p0[0])
    result["y0"] = float(p0[1])
    result["x1"] = float(p1[0])
    result["y1"] = float(p1[1])
    if compare_px_per_mm and compare_px_per_mm > 0:
        ratio = float(compare_px_per_mm) / result["pixel_per_mm"]
        result["compare_pixel_per_mm"] = float(compare_px_per_mm)
        result["px_per_mm_ratio_compare_over_measured"] = ratio
        result["eta_scale_if_replace_compare_with_measured"] = ratio * ratio
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        for line in _format_report(result, compare_px_per_mm):
            typer.echo(line)
    return result
