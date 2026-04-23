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
from PyQt6.QtCore import Qt, pyqtSignal, QSize
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
        g1.add_row("Pattern period mm", self._period_mm)
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


class RunSummaryPanel(QWidget):
    """Summary for a completed/failed run."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
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
        layout.addStretch()

    def set_run_info(self, status: str, frames: str, duration: str, fp: str = "") -> None:
        self._status.setText(status)
        self._frames.setText(frames)
        self._duration.setText(duration)
        self._fingerprint.setText(fp)


class VizSettingsPanel(QWidget):
    """Visualization settings for a scene."""

    export_png_clicked = pyqtSignal()
    export_pdf_clicked = pyqtSignal()
    export_csv_clicked = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        g = PropGroup("Visualization", accent=True)
        self._cmap = PropSelect(["viridis", "plasma", "inferno", "coolwarm", "RdBu_r", "seismic"])
        g.add_row("Colormap", self._cmap)

        self._vmin = PropInput("auto", mono=True)
        g.add_row("vmin", self._vmin)
        self._vmax = PropInput("auto", mono=True)
        g.add_row("vmax", self._vmax)

        self._dpi = PropInput("150", mono=True)
        g.add_row("DPI", self._dpi)
        layout.addWidget(g)

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
        
        self._compute.run_all_clicked.connect(self.run_all_clicked)

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
