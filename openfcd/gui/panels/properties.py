"""Properties panel — QStackedWidget showing context-sensitive properties.

Switches between sub-panels based on SimTree node selection:
  - EmptyPanel (default)
  - ImportPanel (Images parent)
  - FrameInfoPanel (Images/frame)
  - AnnotationPanel (Annotations)
  - ComputePanel (Runs parent)
  - RunSummaryPanel (Runs/item)
  - VizSettingsPanel (Scenes/item)
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QFrame,
    QPushButton, QSlider, QDoubleSpinBox, QSpinBox,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QIcon

from openfcd.gui import tokens
from openfcd.gui.icons import (
    get_icon, ICON_CROP, ICON_BRUSH, ICON_CLOSE, ICON_STAR, ICON_RUN,
)


# ── Reusable form primitives ───────────────────────────────────────

class PropGroup(QWidget):
    def __init__(self, title: str, accent: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._accent = accent
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self.header = QLabel(title)
        self.header.setContentsMargins(14, 10, 14, 6)
        self._layout.addWidget(self.header)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(14, 2, 14, 12)
        self.content_layout.setSpacing(4)
        self._layout.addWidget(self.content)

        self.sep = QFrame()
        self.sep.setFixedHeight(1)
        self._layout.addWidget(self.sep)

        self._row_labels: list[QLabel] = []
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def add_row(self, label: str, widget: QWidget) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 3, 0, 3)
        row_layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet(f"font-size: 11.5px; color: {tokens.TEXT_SECONDARY};")
        self._row_labels.append(lbl)

        row_layout.addWidget(lbl)
        row_layout.addWidget(widget)
        self.content_layout.addWidget(row)

    def _apply_theme(self) -> None:
        c = tokens.ACCENT_CLAY if self._accent else tokens.TEXT_SECONDARY
        self.header.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {c}; "
            f"letter-spacing: 0.06em; text-transform: uppercase;"
        )
        self.sep.setStyleSheet(f"background: {tokens.BORDER_SUBTLE}; border: none;")
        for lbl in self._row_labels:
            lbl.setStyleSheet(f"font-size: 11.5px; color: {tokens.TEXT_SECONDARY};")


class PropInput(QLineEdit):
    def __init__(self, text: str = "", mono: bool = False, parent=None) -> None:
        super().__init__(text, parent)
        self._mono = mono
        self.setFixedHeight(24)
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        font = tokens.FONT_MONO if self._mono else "inherit"
        self.setStyleSheet(f"""
            QLineEdit {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 3px 7px;
                color: {tokens.TEXT_PRIMARY};
                font-family: {font};
                font-size: 11.5px;
            }}
            QLineEdit:focus {{
                border: 1px solid {tokens.ACCENT_CLAY};
            }}
        """)


class PropSelect(QComboBox):
    def __init__(self, items: list[str], parent=None) -> None:
        super().__init__(parent)
        self.addItems(items)
        self.setFixedHeight(24)
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QComboBox {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 3px 7px;
                color: {tokens.TEXT_PRIMARY};
                font-size: 11.5px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """)


class PropCheck(QCheckBox):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QCheckBox {{
                color: {tokens.TEXT_PRIMARY};
                font-size: 12px;
                padding: 2px 0;
            }}
            QCheckBox::indicator {{
                width: 13px;
                height: 13px;
                border-radius: 3px;
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_STRONG};
            }}
            QCheckBox::indicator:checked {{
                background: {tokens.ACCENT_CLAY};
                border: 1px solid {tokens.ACCENT_CLAY};
            }}
        """)


def _action_button(text: str, primary: bool = False, icon: QIcon | None = None) -> QPushButton:
    """Create a styled action button."""
    btn = QPushButton(text)
    if icon is not None:
        btn.setIcon(icon)
        btn.setIconSize(QSize(16, 16))
    btn.setFixedHeight(28)

    def apply():
        if primary:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {tokens.ACCENT_CLAY};
                    color: #fff;
                    border: none;
                    border-radius: 5px;
                    padding: 4px 14px;
                    font-size: 12px;
                    font-weight: 600;
                    font-family: {tokens.FONT_UI};
                }}
                QPushButton:hover {{
                    background: {tokens.ACCENT_CLAY_HOVER};
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {tokens.BG_TERTIARY};
                    color: {tokens.TEXT_PRIMARY};
                    border: 1px solid {tokens.BORDER_SUBTLE};
                    border-radius: 5px;
                    padding: 4px 14px;
                    font-size: 12px;
                    font-family: {tokens.FONT_UI};
                }}
                QPushButton:hover {{
                    background: {tokens.BORDER_SUBTLE};
                }}
            """)
    apply()
    tokens.on_theme_changed(apply)
    return btn


# ── Sub-Panels ─────────────────────────────────────────────────────

class _EmptyPanel(QWidget):
    """Default panel when nothing is selected."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        lbl = QLabel("Select an item in the tree to view its properties.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(lbl)
        layout.addStretch()


class ImagePropertiesPanel(QWidget):
    """Properties for a single image frame: info + annotation tools."""

    set_ref_clicked = pyqtSignal()
    compute_frame_clicked = pyqtSignal()
    draw_roi_clicked = pyqtSignal()
    draw_mask_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    dilate_changed = pyqtSignal(float)
    highpass_sigma_changed = pyqtSignal(float)
    taper_alpha_changed = pyqtSignal(float)
    edge_nan_changed = pyqtSignal(float)
    small_hole_fill_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Info group ──
        g_info = PropGroup("Frame Info")
        self._name_input = PropInput("", mono=True)
        self._name_input.setReadOnly(True)
        g_info.add_row("Filename", self._name_input)
        self._size_input = PropInput("", mono=True)
        self._size_input.setReadOnly(True)
        g_info.add_row("Size", self._size_input)
        self._index_input = PropInput("", mono=True)
        self._index_input.setReadOnly(True)
        g_info.add_row("Index", self._index_input)
        layout.addWidget(g_info)

        # ── Annotation Tools ──
        g_act = PropGroup("Annotation Tools", accent=True)
        
        info = QLabel("(Left click to place point, Right click to undo)")
        info.setWordWrap(True)
        info.setStyleSheet(f"font-size: 11px; color: {tokens.TEXT_SECONDARY}; padding: 0 0 4px 0;")
        g_act.content_layout.addWidget(info)
        
        self._btn_roi = _action_button("Draw ROI (2 clicks)", primary=True, icon=get_icon(ICON_CROP))
        self._btn_roi.clicked.connect(self.draw_roi_clicked)
        g_act.content_layout.addWidget(self._btn_roi)
        self._btn_mask = _action_button("Draw Mask (4 clicks)", icon=get_icon(ICON_BRUSH))
        self._btn_mask.clicked.connect(self.draw_mask_clicked)
        g_act.content_layout.addWidget(self._btn_mask)
        self._btn_clear = _action_button("Clear Annotations", icon=get_icon(ICON_CLOSE))
        self._btn_clear.clicked.connect(self.clear_clicked)
        g_act.content_layout.addWidget(self._btn_clear)
        layout.addWidget(g_act)

        # ── ROI Stats ──
        g_roi = PropGroup("ROI")
        self._roi_x = PropInput("—", mono=True)
        self._roi_x.setReadOnly(True)
        g_roi.add_row("X (col0)", self._roi_x)
        self._roi_y = PropInput("—", mono=True)
        self._roi_y.setReadOnly(True)
        g_roi.add_row("Y (row0)", self._roi_y)
        self._roi_w = PropInput("—", mono=True)
        self._roi_w.setReadOnly(True)
        g_roi.add_row("Width", self._roi_w)
        self._roi_h = PropInput("—", mono=True)
        self._roi_h.setReadOnly(True)
        g_roi.add_row("Height", self._roi_h)
        layout.addWidget(g_roi)

        # ── Mask Stats & Dilate ──
        g_mask = PropGroup("Mask (4-point clockwise)")
        self._mask_status = PropInput("Not set", mono=True)
        self._mask_status.setReadOnly(True)
        g_mask.add_row("Status", self._mask_status)

        dilate_row = QWidget()
        dr_layout = QHBoxLayout(dilate_row)
        dr_layout.setContentsMargins(0, 3, 0, 3)
        dr_layout.setSpacing(8)
        lbl = QLabel("Dilate pixels")
        lbl.setFixedWidth(80)
        lbl.setStyleSheet(f"font-size: 11.5px; color: {tokens.TEXT_SECONDARY};")
        dr_layout.addWidget(lbl)

        self._dilate_spin = QDoubleSpinBox()
        self._dilate_spin.setRange(0, 20)
        self._dilate_spin.setValue(3.0)
        self._dilate_spin.setSingleStep(0.5)
        self._dilate_spin.setFixedHeight(24)
        self._dilate_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 2px 6px;
                color: {tokens.TEXT_PRIMARY};
                font-size: 11.5px;
            }}
        """)
        self._dilate_spin.valueChanged.connect(self.dilate_changed)
        dr_layout.addWidget(self._dilate_spin)
        g_mask.content_layout.addWidget(dilate_row)
        layout.addWidget(g_mask)

        # ── Debug / tuning ──
        g_tune = PropGroup("Compute Tuning (single-frame debug)")
        hint = QLabel("Drift cutoff below shortest physical wavelength;\n"
                      "0 disables. Try 200–500 for cross-session refs.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {tokens.TEXT_SECONDARY}; padding: 0 0 4px 0;")
        g_tune.content_layout.addWidget(hint)

        hp_row = QWidget()
        hp_layout = QHBoxLayout(hp_row)
        hp_layout.setContentsMargins(0, 3, 0, 3)
        hp_layout.setSpacing(8)
        hp_lbl = QLabel("Highpass σ (px)")
        hp_lbl.setFixedWidth(110)
        hp_lbl.setStyleSheet(f"font-size: 11.5px; color: {tokens.TEXT_SECONDARY};")
        hp_layout.addWidget(hp_lbl)

        self._hp_slider = QSlider(Qt.Orientation.Horizontal)
        self._hp_slider.setRange(0, 1000)
        self._hp_slider.setSingleStep(10)
        self._hp_slider.setPageStep(50)
        self._hp_slider.setValue(0)
        hp_layout.addWidget(self._hp_slider, 1)

        self._hp_spin = QSpinBox()
        self._hp_spin.setRange(0, 2000)
        self._hp_spin.setSingleStep(10)
        self._hp_spin.setFixedWidth(64)
        self._hp_spin.setFixedHeight(24)
        self._hp_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 2px 4px;
                color: {tokens.TEXT_PRIMARY};
                font-size: 11.5px;
            }}
        """)
        hp_layout.addWidget(self._hp_spin)

        self._hp_syncing = False
        self._hp_slider.valueChanged.connect(self._on_hp_slider)
        self._hp_spin.valueChanged.connect(self._on_hp_spin)
        g_tune.content_layout.addWidget(hp_row)

        # Taper alpha (0 → Moisan periodic BC, >0 → cosine taper)
        self._taper_alpha_spin = QDoubleSpinBox()
        self._taper_alpha_spin.setRange(0.0, 0.5)
        self._taper_alpha_spin.setSingleStep(0.01)
        self._taper_alpha_spin.setDecimals(3)
        self._taper_alpha_spin.setFixedHeight(24)
        self._taper_alpha_spin.setToolTip(
            "0 = Moisan periodic boundary (preserves edges)\n"
            ">0 = cosine taper (zeros edges; 0.08 typical)"
        )
        self._taper_alpha_spin.setStyleSheet(
            f"QDoubleSpinBox {{ background: {tokens.BG_TERTIARY}; "
            f"border: 1px solid {tokens.BORDER_SUBTLE}; border-radius: 4px; "
            f"padding: 2px 4px; color: {tokens.TEXT_PRIMARY}; font-size: 11.5px; }}"
        )
        self._taper_alpha_spin.valueChanged.connect(
            lambda v: self.taper_alpha_changed.emit(float(v))
        )
        g_tune.add_row("Taper α", self._taper_alpha_spin)

        # Edge NaN margin (mm)
        self._edge_nan_spin = QDoubleSpinBox()
        self._edge_nan_spin.setRange(0.0, 50.0)
        self._edge_nan_spin.setSingleStep(0.5)
        self._edge_nan_spin.setDecimals(1)
        self._edge_nan_spin.setValue(3.0)
        self._edge_nan_spin.setFixedHeight(24)
        self._edge_nan_spin.setToolTip(
            "Physical margin (mm) NaN'd around the ROI edge after FCD"
        )
        self._edge_nan_spin.setStyleSheet(
            f"QDoubleSpinBox {{ background: {tokens.BG_TERTIARY}; "
            f"border: 1px solid {tokens.BORDER_SUBTLE}; border-radius: 4px; "
            f"padding: 2px 4px; color: {tokens.TEXT_PRIMARY}; font-size: 11.5px; }}"
        )
        self._edge_nan_spin.valueChanged.connect(
            lambda v: self.edge_nan_changed.emit(float(v))
        )
        g_tune.add_row("Edge NaN mm", self._edge_nan_spin)

        # Small enclosed NaN hole fill radius (mm)
        self._small_hole_spin = QDoubleSpinBox()
        self._small_hole_spin.setRange(0.0, 2.0)
        self._small_hole_spin.setSingleStep(0.1)
        self._small_hole_spin.setDecimals(1)
        self._small_hole_spin.setValue(1.0)
        self._small_hole_spin.setFixedHeight(24)
        self._small_hole_spin.setToolTip(
            "Fill small enclosed NaN holes in eta; 0 disables. Capped at 2 mm to preserve robot masks."
        )
        self._small_hole_spin.setStyleSheet(
            f"QDoubleSpinBox {{ background: {tokens.BG_TERTIARY}; "
            f"border: 1px solid {tokens.BORDER_SUBTLE}; border-radius: 4px; "
            f"padding: 2px 4px; color: {tokens.TEXT_PRIMARY}; font-size: 11.5px; }}"
        )
        self._small_hole_spin.valueChanged.connect(
            lambda v: self.small_hole_fill_changed.emit(float(v))
        )
        g_tune.add_row("Small hole fill mm", self._small_hole_spin)

        layout.addWidget(g_tune)

        # ── Compute actions ──
        g2 = PropGroup("Actions")
        self._btn_ref = _action_button("Set as Reference", icon=get_icon(ICON_STAR))
        self._btn_ref.clicked.connect(self.set_ref_clicked)
        g2.content_layout.addWidget(self._btn_ref)
        self._btn_compute = _action_button("Compute This Frame", icon=get_icon(ICON_RUN))
        self._btn_compute.clicked.connect(self.compute_frame_clicked)
        g2.content_layout.addWidget(self._btn_compute)
        layout.addWidget(g2)

        layout.addStretch()

    def _on_hp_slider(self, v: int) -> None:
        if self._hp_syncing:
            return
        self._hp_syncing = True
        self._hp_spin.setValue(v)
        self._hp_syncing = False
        self.highpass_sigma_changed.emit(float(v))

    def _on_hp_spin(self, v: int) -> None:
        if self._hp_syncing:
            return
        self._hp_syncing = True
        clamped = min(max(v, 0), self._hp_slider.maximum())
        self._hp_slider.setValue(clamped)
        self._hp_syncing = False
        self.highpass_sigma_changed.emit(float(v))

    def set_highpass_sigma(self, v: float) -> None:
        self._hp_syncing = True
        self._hp_spin.setValue(int(round(v)))
        self._hp_slider.setValue(min(max(int(round(v)), 0), self._hp_slider.maximum()))
        self._hp_syncing = False

    def highpass_sigma(self) -> float:
        return float(self._hp_spin.value())

    def set_taper_alpha(self, v: float) -> None:
        self._taper_alpha_spin.blockSignals(True)
        self._taper_alpha_spin.setValue(float(v))
        self._taper_alpha_spin.blockSignals(False)

    def taper_alpha(self) -> float:
        return float(self._taper_alpha_spin.value())

    def set_edge_nan_mm(self, v: float) -> None:
        self._edge_nan_spin.blockSignals(True)
        self._edge_nan_spin.setValue(float(v))
        self._edge_nan_spin.blockSignals(False)

    def edge_nan_mm(self) -> float:
        return float(self._edge_nan_spin.value())

    def set_small_hole_fill_radius(self, v: float) -> None:
        self._small_hole_spin.blockSignals(True)
        self._small_hole_spin.setValue(float(v))
        self._small_hole_spin.blockSignals(False)

    def small_hole_fill_radius(self) -> float:
        return float(self._small_hole_spin.value())

    def set_frame_info(self, name: str, size: str, index: int, total: int) -> None:
        self._name_input.setText(name)
        self._size_input.setText(size)
        self._index_input.setText(f"{index + 1} / {total}")

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi_x.setText(str(x))
        self._roi_y.setText(str(y))
        self._roi_w.setText(str(w))
        self._roi_h.setText(str(h))

    def set_mask_status(self, status: str) -> None:
        self._mask_status.setText(status)



class ComputePanel(QWidget):
    """Compute parameters: user-required + system defaults."""

    run_all_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Required params
        g1 = PropGroup("Required Parameters", accent=True)
        self._period_mm = PropInput("", mono=True)
        self._period_mm.setPlaceholderText("e.g. 1.2")
        g1.add_row("Checker cell side mm", self._period_mm)
        self._glass_mm = PropInput("3.0", mono=True)
        g1.add_row("Window glass mm", self._glass_mm)
        self._fluid_mm = PropInput("12.0", mono=True)
        g1.add_row("Fluid depth mm", self._fluid_mm)

        self._preset = PropSelect([
            "pattern_below_window",
            "immersed_pattern",
            "custom",
        ])
        g1.add_row("Optical preset", self._preset)
        layout.addWidget(g1)

        # System defaults
        g2 = PropGroup("System Defaults (adjustable)")
        self._flatfield = PropInput("300.0", mono=True)
        g2.add_row("Flatfield σ (px)", self._flatfield)
        self._detrend = PropSelect(["plane", "none"])
        g2.add_row("Detrend", self._detrend)
        self._taper_alpha = PropInput("0.08", mono=True)
        g2.add_row("Taper α", self._taper_alpha)
        self._edge_nan = PropInput("3.0", mono=True)
        g2.add_row("Edge NaN mm", self._edge_nan)
        self._workers = PropInput("auto", mono=True)
        g2.add_row("Workers", self._workers)
        layout.addWidget(g2)

        # Run button
        g3 = PropGroup("Execute")
        self._btn_run = _action_button("Run All Frames", primary=True, icon=get_icon(ICON_RUN))
        self._btn_run.clicked.connect(self.run_all_clicked)
        g3.content_layout.addWidget(self._btn_run)
        layout.addWidget(g3)

        layout.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)


class WaveStatsPanel(QWidget):
    """Wave Stats configuration sub-panel (v2).

    Provides a single peak_prominence_k spinbox and a Recompute button.
    Per-segment editing lives in the ProfileSceneView table.
    """

    prominence_k_changed = pyqtSignal(float)
    recompute_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        g = PropGroup("Wave Stats", accent=True)
        self._prom_spin = QDoubleSpinBox()
        self._prom_spin.setRange(0.05, 2.0)
        self._prom_spin.setSingleStep(0.01)
        self._prom_spin.setDecimals(2)
        self._prom_spin.setValue(0.30)
        self._prom_spin.setFixedHeight(24)
        self._prom_spin.setStyleSheet(self._spin_ss())
        g.add_row("Prominence k", self._prom_spin)
        layout.addWidget(g)

        g_act = PropGroup("Postprocess")
        self._btn_recompute = _action_button(
            "Recompute Wave Stats", primary=True, icon=get_icon(ICON_RUN)
        )
        self._btn_recompute.setEnabled(False)
        self._btn_recompute.clicked.connect(self.recompute_clicked)
        g_act.content_layout.addWidget(self._btn_recompute)
        layout.addWidget(g_act)

        layout.addStretch()

        self._prom_timer = QTimer()
        self._prom_timer.setInterval(400)
        self._prom_timer.setSingleShot(True)
        self._prom_timer.timeout.connect(
            lambda: self.prominence_k_changed.emit(float(self._prom_spin.value()))
        )

        self._has_wave_stats = False
        self._has_run = False
        self._is_running = False

        self._prom_spin.valueChanged.connect(lambda _: self._prom_timer.start())

    @staticmethod
    def _spin_ss() -> str:
        return (
            f"QDoubleSpinBox {{ background: {tokens.BG_TERTIARY}; "
            f"border: 1px solid {tokens.BORDER_SUBTLE}; border-radius: 4px; "
            f"padding: 2px 4px; color: {tokens.TEXT_PRIMARY}; font-size: 11.5px; }}"
        )

    def _update_recompute_state(self) -> None:
        enabled = self._has_wave_stats and self._has_run and not self._is_running
        self._btn_recompute.setEnabled(enabled)

    # ── Public API ────────────────────────────────────────────────

    def set_annotation(self, annotation) -> None:
        """Populate prominence spinbox from annotation.wave_stats (v2)."""
        self._prom_timer.stop()
        ws = annotation.wave_stats if annotation is not None else None
        # A wave_stats config without segments is treated as "no wave stats"
        # for the purpose of enabling the recompute button.
        self._has_wave_stats = ws is not None and any(
            segs for segs in ws.segments_by_frame.values()
        )
        if ws is not None:
            self._prom_spin.blockSignals(True)
            self._prom_spin.setValue(float(ws.peak_prominence_k))
            self._prom_spin.blockSignals(False)
        self._update_recompute_state()

    def set_has_run(self, has_run: bool) -> None:
        self._has_run = has_run
        self._update_recompute_state()

    def set_running(self, is_running: bool) -> None:
        self._is_running = is_running
        self._update_recompute_state()


class RunSummaryPanel(QWidget):
    """Summary for a completed/failed run, with embedded Wave Stats panel."""

    wave_stats_prominence_changed = pyqtSignal(float)
    wave_stats_recompute_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        g = PropGroup("Run Summary")
        self._status = PropInput("", mono=True)
        self._status.setReadOnly(True)
        g.add_row("Status", self._status)
        self._frames = PropInput("", mono=True)
        self._frames.setReadOnly(True)
        g.add_row("Frames", self._frames)
        self._duration = PropInput("", mono=True)
        self._duration.setReadOnly(True)
        g.add_row("Duration", self._duration)
        self._fingerprint = PropInput("", mono=True)
        self._fingerprint.setReadOnly(True)
        g.add_row("Config hash", self._fingerprint)
        layout.addWidget(g)

        # ── embedded Wave Stats sub-panel ──
        self._wave_stats = WaveStatsPanel()
        self._wave_stats.prominence_k_changed.connect(self.wave_stats_prominence_changed)
        self._wave_stats.recompute_clicked.connect(self.wave_stats_recompute_clicked)
        layout.addWidget(self._wave_stats)

        layout.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_run_info(self, status: str, frames: str, duration: str, fp: str = "") -> None:
        self._status.setText(status)
        self._frames.setText(frames)
        self._duration.setText(duration)
        self._fingerprint.setText(fp)

    @property
    def wave_stats_panel(self) -> WaveStatsPanel:
        return self._wave_stats


class VizSettingsPanel(QWidget):
    """Visualization settings for a scene."""

    export_png_clicked = pyqtSignal()
    export_pdf_clicked = pyqtSignal()
    export_csv_clicked = pyqtSignal()
    apply_clicked = pyqtSignal(dict)   # emits full viz_params dict
    reset_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene_type = "eta_map"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        g = PropGroup("Visualization", accent=True)

        # Colormap
        self._cmap = PropSelect(["RdBu_r", "viridis", "plasma", "inferno", "coolwarm", "magma", "seismic"])
        g.add_row("Colormap", self._cmap)

        # vmin with auto toggle
        vmin_row = QWidget()
        vrl = QHBoxLayout(vmin_row)
        vrl.setContentsMargins(0, 0, 0, 0)
        vrl.setSpacing(4)
        self._vmin = PropInput("auto", mono=True)
        self._vmin.setFixedWidth(70)
        self._vmin_auto = QCheckBox("auto")
        self._vmin_auto.setChecked(True)
        self._vmin_auto.toggled.connect(lambda c: self._vmin.setEnabled(not c))
        self._vmin.setEnabled(False)
        vrl.addWidget(self._vmin)
        vrl.addWidget(self._vmin_auto)
        vrl.addStretch()
        g.add_row("vmin", vmin_row)

        # vmax with auto toggle
        vmax_row = QWidget()
        vxl = QHBoxLayout(vmax_row)
        vxl.setContentsMargins(0, 0, 0, 0)
        vxl.setSpacing(4)
        self._vmax = PropInput("auto", mono=True)
        self._vmax.setFixedWidth(70)
        self._vmax_auto = QCheckBox("auto")
        self._vmax_auto.setChecked(True)
        self._vmax_auto.toggled.connect(lambda c: self._vmax.setEnabled(not c))
        self._vmax.setEnabled(False)
        vxl.addWidget(self._vmax)
        vxl.addWidget(self._vmax_auto)
        vxl.addStretch()
        g.add_row("vmax", vmax_row)

        # Colorbar position
        self._colorbar_pos = PropSelect(["right", "bottom", "none"])
        g.add_row("Colorbar", self._colorbar_pos)

        # Title
        self._title_edit = PropInput("", mono=False)
        self._title_edit.setPlaceholderText("(auto)")
        g.add_row("Title", self._title_edit)

        # DPI
        self._dpi = PropInput("150", mono=True)
        g.add_row("DPI", self._dpi)

        # Inline help tooltips on viz fields
        self._cmap.setToolTip("Color palette used for the η/RMS heatmap")
        self._vmin.setToolTip("Minimum value for colorbar (2nd percentile when auto)")
        self._vmax.setToolTip("Maximum value for colorbar (98th percentile when auto)")
        self._colorbar_pos.setToolTip("Colorbar position: right, bottom, or hidden")
        self._title_edit.setToolTip("Optional plot title (leave blank for auto-generated)")
        self._dpi.setToolTip("Output resolution in dots per inch (default 150)")

        layout.addWidget(g)

        # Type-specific params
        g_type = PropGroup("Type Parameters")
        self._type_stack = QStackedWidget()

        # η_map panel (index 0)
        p_eta = QWidget()
        pl = QVBoxLayout(p_eta)
        pl.setContentsMargins(0, 0, 0, 0)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setValue(0.8)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setFixedHeight(24)
        ar = QWidget()
        arl = QHBoxLayout(ar)
        arl.setContentsMargins(0, 3, 0, 3)
        arl.setSpacing(8)
        albl = QLabel("Alpha")
        albl.setFixedWidth(110)
        albl.setStyleSheet(f"font-size:11.5px;color:{tokens.TEXT_SECONDARY};")
        arl.addWidget(albl)
        arl.addWidget(self._alpha_spin)
        pl.addWidget(ar)
        self._type_stack.addWidget(p_eta)  # 0: eta_map

        # profile panel (index 1)
        p_prof = QWidget()
        ppl = QVBoxLayout(p_prof)
        ppl.setContentsMargins(0, 0, 0, 0)
        self._line_color = PropInput("red")
        self._profile_color = PropInput("blue")
        self._strip_mm = PropInput("2.0", mono=True)
        self._y_range_mm = PropInput("30.0", mono=True)
        self._min_roi_width_mm = PropInput("160.0", mono=True)
        self._min_roi_height_mm = PropInput("50.0", mono=True)
        self._x_padding_mm = PropInput("5.0", mono=True)
        self._auto_crop = QCheckBox("auto")
        self._auto_crop.setChecked(True)
        self._x_range_mm = PropInput("", mono=True)
        self._x_range_mm.setPlaceholderText("auto or min:max")
        prof_group = PropGroup("Profile Composite")
        prof_group.add_row("Line color", self._line_color)
        prof_group.add_row("Profile color", self._profile_color)
        prof_group.add_row("Strip mm", self._strip_mm)
        prof_group.add_row("y range mm", self._y_range_mm)
        prof_group.add_row("min ROI width mm", self._min_roi_width_mm)
        prof_group.add_row("min ROI height mm", self._min_roi_height_mm)
        prof_group.add_row("x padding mm", self._x_padding_mm)
        prof_group.add_row("Auto crop", self._auto_crop)
        prof_group.add_row("x range mm", self._x_range_mm)
        ppl.addWidget(prof_group)
        self._type_stack.addWidget(p_prof)  # 1: profile

        # rms panel (index 2)
        p_rms = QWidget()
        rpl = QVBoxLayout(p_rms)
        rpl.setContentsMargins(0, 0, 0, 0)
        self._clip_pct = QSpinBox()
        self._clip_pct.setRange(0, 100)
        self._clip_pct.setValue(98)
        self._clip_pct.setFixedHeight(24)
        cr = QWidget()
        crl = QHBoxLayout(cr)
        crl.setContentsMargins(0, 3, 0, 3)
        crl.setSpacing(8)
        clbl = QLabel("Clip pct")
        clbl.setFixedWidth(110)
        clbl.setStyleSheet(f"font-size:11.5px;color:{tokens.TEXT_SECONDARY};")
        crl.addWidget(clbl)
        crl.addWidget(self._clip_pct)
        rpl.addWidget(cr)
        self._type_stack.addWidget(p_rms)  # 2: rms

        g_type.content_layout.addWidget(self._type_stack)
        layout.addWidget(g_type)

        # Apply / Reset buttons
        g_act = PropGroup("Apply")
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setFixedHeight(28)
        self._btn_apply.setStyleSheet(
            f"background:{tokens.BG_TERTIARY};color:{tokens.TEXT_PRIMARY};"
            f"border:1px solid {tokens.BORDER_SUBTLE};border-radius:5px;"
            f"padding:4px 14px;font-size:12px;"
        )
        self._btn_apply.clicked.connect(self._on_apply)
        g_act.content_layout.addWidget(self._btn_apply)
        self._btn_reset = QPushButton("Reset Defaults")
        self._btn_reset.setFixedHeight(28)
        self._btn_reset.clicked.connect(self.reset_clicked)
        g_act.content_layout.addWidget(self._btn_reset)
        layout.addWidget(g_act)

        # Export buttons
        g2 = PropGroup("Export")
        self._btn_png = _action_button("Export PNG", primary=True)
        self._btn_png.clicked.connect(self.export_png_clicked)
        g2.content_layout.addWidget(self._btn_png)
        self._btn_pdf = _action_button("Export PDF")
        self._btn_pdf.clicked.connect(self.export_pdf_clicked)
        g2.content_layout.addWidget(self._btn_pdf)
        self._btn_csv = _action_button("Export CSV")
        self._btn_csv.clicked.connect(self.export_csv_clicked)
        g2.content_layout.addWidget(self._btn_csv)
        layout.addWidget(g2)

        layout.addStretch()

        # Track dirty state
        self._dirty = False
        for widget in [
            self._cmap,
            self._vmin,
            self._vmax,
            self._colorbar_pos,
            self._title_edit,
            self._dpi,
            self._line_color,
            self._profile_color,
            self._strip_mm,
            self._y_range_mm,
            self._min_roi_width_mm,
            self._min_roi_height_mm,
            self._x_padding_mm,
            self._auto_crop,
            self._x_range_mm,
        ]:
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._mark_dirty)
            elif hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._mark_dirty)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._mark_dirty)

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self._btn_apply.setText("Apply *")
            self._btn_apply.setStyleSheet(
                f"background:{tokens.ACCENT_CLAY_BG};color:{tokens.ACCENT_CLAY};"
                f"border:1px solid {tokens.ACCENT_CLAY};border-radius:5px;"
                f"padding:4px 14px;font-size:12px;font-weight:600;"
            )

    def _clear_dirty(self) -> None:
        self._dirty = False
        self._btn_apply.setText("Apply")
        self._btn_apply.setStyleSheet(
            f"background:{tokens.BG_TERTIARY};color:{tokens.TEXT_PRIMARY};"
            f"border:1px solid {tokens.BORDER_SUBTLE};border-radius:5px;"
            f"padding:4px 14px;font-size:12px;"
        )

    def _on_apply(self) -> None:
        self._clear_dirty()
        self.apply_clicked.emit(self.get_viz_params())

    @staticmethod
    def _parse_float(text: str, default: float | None = None) -> float | None:
        try:
            return float(text)
        except (TypeError, ValueError):
            return default

    def _current_defaults(self) -> dict:
        from openfcd.gui.scenes.viz_defaults import scene_defaults
        return scene_defaults(self._scene_type)

    def _parse_profile_x_range_value(self) -> tuple[float, float] | None:
        from openfcd.core.profile_composite import parse_profile_x_range

        raw = self._x_range_mm.text().strip()
        parsed = parse_profile_x_range(raw)
        if raw and parsed is None:
            self._x_range_mm.setText("")
            return None
        return parsed

    @staticmethod
    def _positive(value: float | None, default: float) -> float:
        if value is None or value <= 0:
            return default
        return value

    @staticmethod
    def _nonnegative(value: float | None, default: float) -> float:
        if value is None or value < 0:
            return default
        return value

    def get_viz_params(self) -> dict:
        defaults = self._current_defaults()
        default_dpi = int(defaults.get("dpi", 150))
        params = {
            "cmap": self._cmap.currentText(),
            "vmin": None if self._vmin_auto.isChecked() else self._vmin.text(),
            "vmax": None if self._vmax_auto.isChecked() else self._vmax.text(),
            "colorbar": self._colorbar_pos.currentText(),
            "title": self._title_edit.text() or None,
            "dpi": int(self._dpi.text()) if self._dpi.text().isdigit() and int(self._dpi.text()) > 0 else default_dpi,
            "alpha": self._alpha_spin.value(),
            "line_color": self._line_color.text() or "red",
            "profile_color": self._profile_color.text() or "blue",
            "strip_mm": self._nonnegative(
                self._parse_float(self._strip_mm.text()),
                float(defaults.get("strip_mm", 0.0)),
            ),
            "y_range_mm": self._positive(
                self._parse_float(self._y_range_mm.text()),
                float(defaults.get("y_range_mm", 30.0)),
            ),
            "min_roi_width_mm": self._positive(
                self._parse_float(self._min_roi_width_mm.text()),
                float(defaults.get("min_roi_width_mm", 160.0)),
            ),
            "min_roi_height_mm": self._positive(
                self._parse_float(self._min_roi_height_mm.text()),
                float(defaults.get("min_roi_height_mm", 50.0)),
            ),
            "x_padding_mm": self._nonnegative(
                self._parse_float(self._x_padding_mm.text()),
                float(defaults.get("x_padding_mm", 0.0)),
            ),
            "auto_crop": self._auto_crop.isChecked(),
            "show_measurements": False,
        }
        x_range = self._parse_profile_x_range_value()
        params["x_range_mm"] = list(x_range) if x_range is not None else None
        return params

    def set_scene_type(self, scene_type: str) -> None:
        """Switch type-specific panel based on scene type."""
        self._scene_type = scene_type
        idx_map = {"eta_map": 0, "profile": 1, "rms": 2}
        self._type_stack.setCurrentIndex(idx_map.get(scene_type, 0))

    def load_viz_params(self, scene_type_or_params, params: dict | None = None) -> None:
        """Populate fields from a viz_params dict."""
        if params is None:
            scene_type = self._scene_type
            user_params = dict(scene_type_or_params or {})
        else:
            scene_type = str(scene_type_or_params)
            self.set_scene_type(scene_type)
            user_params = dict(params or {})
        from openfcd.gui.scenes.viz_defaults import scene_defaults
        params = scene_defaults("profile")
        params.update(scene_defaults(scene_type))
        params.update(user_params)
        if "cmap" in params:
            idx = self._cmap.findText(params["cmap"])
            if idx >= 0:
                self._cmap.setCurrentIndex(idx)
        if params.get("vmin") is not None:
            self._vmin_auto.setChecked(False)
            self._vmin.setText(str(params["vmin"]))
        else:
            self._vmin_auto.setChecked(True)
        if params.get("vmax") is not None:
            self._vmax_auto.setChecked(False)
            self._vmax.setText(str(params["vmax"]))
        else:
            self._vmax_auto.setChecked(True)
        if "dpi" in params:
            self._dpi.setText(str(params["dpi"]))
        if "colorbar" in params:
            idx = self._colorbar_pos.findText(str(params["colorbar"]))
            if idx >= 0:
                self._colorbar_pos.setCurrentIndex(idx)
        self._title_edit.setText(str(params.get("title") or ""))
        if "line_color" in params:
            self._line_color.setText(str(params["line_color"]))
        if "profile_color" in params:
            self._profile_color.setText(str(params["profile_color"]))
        if "strip_mm" in params:
            self._strip_mm.setText(str(params["strip_mm"]))
        if "y_range_mm" in params:
            self._y_range_mm.setText(str(params["y_range_mm"]))
        if "min_roi_width_mm" in params:
            self._min_roi_width_mm.setText(str(params["min_roi_width_mm"]))
        if "min_roi_height_mm" in params:
            self._min_roi_height_mm.setText(str(params["min_roi_height_mm"]))
        if "x_padding_mm" in params:
            self._x_padding_mm.setText(str(params["x_padding_mm"]))
        if "auto_crop" in params:
            self._auto_crop.setChecked(bool(params["auto_crop"]))
        from openfcd.core.profile_composite import format_profile_x_range
        self._x_range_mm.setText(format_profile_x_range(params.get("x_range_mm")))
        self._clear_dirty()


class ImportPanel(QWidget):
    """Panel shown when Images parent node is selected."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        g = PropGroup("Image Source")
        self._folder = PropInput("", mono=True)
        self._folder.setReadOnly(True)
        g.add_row("Folder", self._folder)
        self._pattern = PropInput("Img*.jpg", mono=True)
        g.add_row("Pattern", self._pattern)
        self._count = PropInput("0", mono=True)
        self._count.setReadOnly(True)
        g.add_row("Frame count", self._count)
        layout.addWidget(g)
        layout.addStretch()

    def set_info(self, folder: str, pattern: str, count: int) -> None:
        self._folder.setText(folder)
        self._pattern.setText(pattern)
        self._count.setText(str(count))


# ── Main Panel ─────────────────────────────────────────────────────

class PropertiesPanel(QStackedWidget):
    """Context-switching properties panel.

    Maps node type keys to sub-panels.
    """

    # Expose sub-panel signals for MainWindow wiring
    set_ref_clicked = pyqtSignal()
    compute_frame_clicked = pyqtSignal()
    run_all_clicked = pyqtSignal()
    draw_roi_clicked = pyqtSignal()
    draw_mask_clicked = pyqtSignal()
    clear_annotation_clicked = pyqtSignal()
    dilate_changed = pyqtSignal(float)
    highpass_sigma_changed = pyqtSignal(float)
    taper_alpha_changed = pyqtSignal(float)
    edge_nan_changed = pyqtSignal(float)
    small_hole_fill_changed = pyqtSignal(float)

    # Wave Stats signals (relayed from RunSummaryPanel → WaveStatsPanel)
    wave_stats_prominence_changed = pyqtSignal(float)
    wave_stats_recompute_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(250)

        # Create all sub-panels
        self._empty = _EmptyPanel()
        self._import = ImportPanel()
        self._image_properties = ImagePropertiesPanel()
        self._compute = ComputePanel()
        self._run_summary = RunSummaryPanel()
        self._viz_settings = VizSettingsPanel()

        # Add in fixed order
        self.addWidget(self._empty)             # 0
        self.addWidget(self._import)            # 1
        self.addWidget(self._image_properties)  # 2
        self.addWidget(self._compute)           # 3
        self.addWidget(self._run_summary)       # 4
        self.addWidget(self._viz_settings)      # 5

        self.setCurrentIndex(0)

        # Wire signals
        self._image_properties.set_ref_clicked.connect(self.set_ref_clicked)
        self._image_properties.compute_frame_clicked.connect(self.compute_frame_clicked)
        self._image_properties.draw_roi_clicked.connect(self.draw_roi_clicked)
        self._image_properties.draw_mask_clicked.connect(self.draw_mask_clicked)
        self._image_properties.clear_clicked.connect(self.clear_annotation_clicked)
        self._image_properties.dilate_changed.connect(self.dilate_changed)
        self._image_properties.highpass_sigma_changed.connect(self.highpass_sigma_changed)
        self._image_properties.taper_alpha_changed.connect(self.taper_alpha_changed)
        self._image_properties.edge_nan_changed.connect(self.edge_nan_changed)
        self._image_properties.small_hole_fill_changed.connect(self.small_hole_fill_changed)

        self._compute.run_all_clicked.connect(self.run_all_clicked)

        # Wire wave stats signals from RunSummaryPanel
        self._run_summary.wave_stats_prominence_changed.connect(self.wave_stats_prominence_changed)
        self._run_summary.wave_stats_recompute_clicked.connect(self.wave_stats_recompute_clicked)

        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"background: {tokens.BG_SECONDARY};")

    # Node type key → panel index mapping
    _KEY_MAP = {
        "root": 0,
        "images": 1,
        "image_frame": 2,
        "runs": 3,
        "run_item": 4,
        "scenes": 0,
        "scene_item": 5,
    }

    def show_figure_properties(self, key: str) -> None:
        """Switch to the appropriate sub-panel based on node type key."""
        idx = self._KEY_MAP.get(key.lower(), 0)
        self.setCurrentIndex(idx)

    # ── Accessors for sub-panels ───────────────────────────────────
    @property
    def import_panel(self) -> ImportPanel:
        return self._import

    @property
    def image_properties_panel(self) -> ImagePropertiesPanel:
        return self._image_properties

    @property
    def compute_panel(self) -> ComputePanel:
        return self._compute

    @property
    def run_summary_panel(self) -> RunSummaryPanel:
        return self._run_summary

    @property
    def viz_settings_panel(self) -> VizSettingsPanel:
        return self._viz_settings

    # ── Wave Stats delegation ──────────────────────────────────────

    def set_annotation(self, annotation) -> None:
        """Propagate annotation to the wave stats panel."""
        self._run_summary.wave_stats_panel.set_annotation(annotation)

    def set_run_state(self, has_run: bool, is_running: bool = False) -> None:
        """Update the recompute-button enable guards."""
        self._run_summary.wave_stats_panel.set_has_run(has_run)
        self._run_summary.wave_stats_panel.set_running(is_running)

