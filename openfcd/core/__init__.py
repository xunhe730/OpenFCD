# Import submodule first so `from openfcd.core import fcd` resolves to the module.
from openfcd.core import fcd  # noqa: F401 — exposes openfcd.core.fcd as a module

# Re-export individual symbols for convenience imports.
from openfcd.core.fcd import (
    calculate_carriers,
    fcd as fcd_fn,
    fftinvgrad,
    carrier_wavelength_mm,
)
from openfcd.core.flatfield import flatfield_normalize
from openfcd.core.mask import polygon_mask, auto_mask
from openfcd.core.temporal import time_mean, time_median, time_rms
from openfcd.core.inpaint import inpaint_fft
from openfcd.core.heading import detect_heading
from openfcd.core.polygon_source import (
    PolygonSource, FixedPolygonSource, TrackedPolygonSource,
    AutoPolygonSource, find_oriented_polygon, estimate_object_dimensions,
)

__all__ = [
    "fcd",
    "calculate_carriers", "fcd_fn", "fftinvgrad", "carrier_wavelength_mm",
    "flatfield_normalize",
    "polygon_mask", "auto_mask",
    "time_mean", "time_median", "time_rms",
    "inpaint_fft",
    "detect_heading",
    "PolygonSource", "FixedPolygonSource", "TrackedPolygonSource",
    "AutoPolygonSource", "find_oriented_polygon", "estimate_object_dimensions",
]
