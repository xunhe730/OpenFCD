"""Display-mode registry for PreviewWidget.

Instead of hardcoding QCheckBox widgets for Overlap / Colorbar /
Preview, PreviewWidget consults this registry to build its display
bar. Each mode registers a DisplayModeSpec. Downstream features
(Preview toggle, Scene η rendering) extend the registry without
touching PreviewWidget._setup_ui.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Callable
import numpy as np

if TYPE_CHECKING:
    from openfcd.gui.widgets.preview_widget import PreviewWidget


class DisplayMode(StrEnum):
    ORIGINAL = "original"
    OVERLAY  = "overlay"
    ETA_ONLY = "eta_only"


@dataclass
class DisplayModeSpec:
    mode: DisplayMode
    label: str
    tooltip: str
    paint: Callable[[PreviewWidget, np.ndarray | None], None] = field(repr=False)


class DisplayModeRegistry:
    def __init__(self) -> None:
        self._specs: dict[DisplayMode, DisplayModeSpec] = {}

    def register(self, spec: DisplayModeSpec) -> None:
        self._specs[spec.mode] = spec

    def get(self, mode: DisplayMode) -> DisplayModeSpec | None:
        return self._specs.get(mode)

    def all_modes(self) -> list[DisplayModeSpec]:
        return list(self._specs.values())
