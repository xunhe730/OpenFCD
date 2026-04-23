from __future__ import annotations
from openfcd.io.project import GeometryConfig, OpticalLayer


_VALID_PRESETS = frozenset({"pattern_below_window", "immersed_pattern", "custom"})


def get_default_layers(
    preset: str,
    *,
    glass_thickness_mm: float | None = None,
    fluid_depth_mm: float | None = None,
) -> list[OpticalLayer]:
    """Return default optical layers for a given preset name.

    Args:
        preset: One of 'pattern_below_window', 'immersed_pattern', or 'custom'.

    Returns:
        List of OpticalLayer instances. Empty list for 'custom' or invalid presets.
    """
    if preset == "pattern_below_window":
        glass_mm = 3.0 if glass_thickness_mm is None else glass_thickness_mm
        fluid_mm = 12.0 if fluid_depth_mm is None else fluid_depth_mm
        return [
            OpticalLayer(thickness_mm=glass_mm, medium="glass", n=1.5),
            OpticalLayer(thickness_mm=fluid_mm, medium="water", n=1.333),
        ]
    if preset == "immersed_pattern":
        fluid_mm = 10.0 if fluid_depth_mm is None else fluid_depth_mm
        return [
            OpticalLayer(thickness_mm=fluid_mm, medium="water", n=1.333),
        ]
    # 'custom' or any invalid preset → empty list (graceful fallback)
    return []


class OpticalGeometry:
    """Compute derived optical parameters from optical stack config."""

    def __init__(self, geometry: GeometryConfig):
        layers = geometry.optical_stack.layers
        if not layers:
            raise ValueError("optical_stack.layers must not be empty")
        n_top = layers[-1].n
        n_air = 1.0
        self._alpha = 1.0 - n_air / n_top
        self._h_p_eff_mm = sum(L.thickness_mm * L.n / n_top for L in layers)
        self._K_per_mm = 1.0 / (self._alpha * self._h_p_eff_mm)
        self._pattern_period_mm = geometry.pattern_period_mm

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def h_p_eff_mm(self) -> float:
        return self._h_p_eff_mm

    @property
    def K_per_mm(self) -> float:
        return self._K_per_mm

    @property
    def pattern_period_mm(self) -> float:
        return self._pattern_period_mm
