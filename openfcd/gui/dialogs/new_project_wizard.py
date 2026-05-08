"""New Project Wizard — 3-step dialog (Name & Path → Image source → Optical setup)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QStackedWidget,
    QLineEdit, QPushButton, QLabel, QFileDialog, QComboBox,
    QSpinBox, QGroupBox, QWidget, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon

from openfcd.gui import tokens
from openfcd.gui.icons import get_icon, ICON_ARROW_BACK, ICON_ARROW_FORWARD
from openfcd.gui.preferences import UserPrefs, get_prefs

# Hardcoded fallbacks for first-run when no prefs exist.
_FALLBACK_PERIOD_MM = 1.2
_FALLBACK_GLASS_MM = 3.0
_FALLBACK_FLUID_MM = 12.0
_FALLBACK_PATTERN = "Img*.jpg"
_FALLBACK_PRESET = "pattern_below_window"


class NewProjectWizard(QDialog):
    """3-step wizard: Name & path → Image source & reference → Optical setup."""

    def __init__(self, parent=None, prefs: UserPrefs | None = None) -> None:
        super().__init__(parent)
        self._prefs = prefs if prefs is not None else get_prefs()
        self._step = 0  # 0-indexed
        self._setup_ui()
        self._update_nav()

    def _setup_ui(self) -> None:
        self.setWindowTitle("New OpenFCD Project")
        self.setMinimumSize(640, 440)
        self.setStyleSheet(f"""
            QDialog {{
                background: {tokens.BG_PRIMARY};
                color: {tokens.TEXT_PRIMARY};
                font-family: {tokens.FONT_UI};
                font-size: 12px;
            }}
            QGroupBox {{
                font-weight: 600;
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: {tokens.RADIUS_MD}px;
                margin-top: 8px;
                padding-top: 18px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {tokens.TEXT_SECONDARY};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Step indicator
        self._step_bar = QLabel()
        layout.addWidget(self._step_bar)

        # Stacked content
        self._stack = QStackedWidget()
        self._build_step1()
        self._build_step2()
        self._build_step3()
        layout.addWidget(self._stack, 1)

        # Navigation buttons
        btn_layout = QHBoxLayout()
        self._step_label = QLabel()
        self._step_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 11px;")
        btn_layout.addWidget(self._step_label)
        btn_layout.addStretch()

        self._back_btn = QPushButton("Back")
        self._back_btn.setIcon(get_icon(ICON_ARROW_BACK))
        self._back_btn.setIconSize(QSize(16, 16))
        self._back_btn.clicked.connect(self._prev_step)
        btn_layout.addWidget(self._back_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setIcon(get_icon(ICON_ARROW_FORWARD))
        self._next_btn.setIconSize(QSize(16, 16))
        self._next_btn.setStyleSheet(
            f"background: {tokens.ACCENT_CLAY}; color: #fff; "
            f"font-weight: 600; padding: 6px 16px; border-radius: 4px;"
        )
        self._next_btn.clicked.connect(self._next_step)
        btn_layout.addWidget(self._next_btn)

        layout.addLayout(btn_layout)

    # ── Step 1: Name & Path ─────────────────────────────────────────
    def _build_step1(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("Project Identity")
        form = QFormLayout(grp)
        form.setSpacing(10)

        self.name_edit = QLineEdit("my_experiment")
        self.name_edit.setStyleSheet(self._input_style())
        self.name_edit.textChanged.connect(self._update_preview)
        form.addRow("Project name:", self.name_edit)

        path_layout = QHBoxLayout()
        initial_location = self._prefs.last_project_dir or str(Path.home() / "Documents")
        self.location_edit = QLineEdit(initial_location)
        self.location_edit.setStyleSheet(self._input_style())
        self.location_edit.textChanged.connect(self._update_preview)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_location)
        path_layout.addWidget(self.location_edit)
        path_layout.addWidget(browse_btn)
        form.addRow("Location:", path_layout)

        self._path_preview = QLabel("")
        self._path_preview.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-family: {tokens.FONT_MONO}; font-size: 11px;"
        )
        form.addRow("Will create:", self._path_preview)

        layout.addWidget(grp)
        layout.addStretch()

        self._stack.addWidget(page)
        self._update_preview()

    # ── Step 2: Image source & reference ────────────────────────────
    def _build_step2(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("Image Source")
        form = QFormLayout(grp)
        form.setSpacing(10)

        img_layout = QHBoxLayout()
        self.image_folder_edit = QLineEdit(self._prefs.last_image_folder)
        self.image_folder_edit.setPlaceholderText("/path/to/image/folder")
        self.image_folder_edit.setStyleSheet(self._input_style())
        img_browse = QPushButton("Browse…")
        img_browse.clicked.connect(self._browse_image_folder)
        img_layout.addWidget(self.image_folder_edit)
        img_layout.addWidget(img_browse)
        form.addRow("Image folder:", img_layout)

        self.pattern_edit = QLineEdit(self._prefs.last_file_pattern or _FALLBACK_PATTERN)
        self.pattern_edit.setStyleSheet(self._input_style())
        self.pattern_edit.textChanged.connect(self._on_pattern_changed)
        form.addRow("File pattern:", self.pattern_edit)

        self._image_count_label = QLabel("—")
        self._image_count_label.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-family: {tokens.FONT_MONO}; font-size: 11px;"
        )
        form.addRow("Matched files:", self._image_count_label)

        layout.addWidget(grp)

        self._stack.addWidget(page)

    # ── Step 3: Optical setup ───────────────────────────────────────
    def _build_step3(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("Optical Stack Preset")
        form = QFormLayout(grp)
        form.setSpacing(10)

        self._optical_preset = QComboBox()
        self._optical_preset.addItems(["pattern_below_window", "immersed_pattern", "custom"])
        self._optical_preset.setStyleSheet(self._input_style())
        preset_idx = self._optical_preset.findText(
            self._prefs.last_optical_preset or _FALLBACK_PRESET
        )
        if preset_idx >= 0:
            self._optical_preset.setCurrentIndex(preset_idx)
        form.addRow("Preset:", self._optical_preset)

        self._pattern_period = QDoubleSpinBox()
        self._pattern_period.setRange(0.01, 50.0)
        self._pattern_period.setValue(self._prefs.last_pattern_period_mm or _FALLBACK_PERIOD_MM)
        self._pattern_period.setSuffix(" mm")
        self._pattern_period.setDecimals(2)
        form.addRow("Checker cell side:", self._pattern_period)

        self._glass_thickness = QDoubleSpinBox()
        self._glass_thickness.setRange(0.0, 100.0)
        self._glass_thickness.setValue(self._prefs.last_glass_thickness_mm or _FALLBACK_GLASS_MM)
        self._glass_thickness.setSuffix(" mm")
        self._glass_thickness.setDecimals(2)
        form.addRow("Window glass:", self._glass_thickness)

        self._fluid_depth = QDoubleSpinBox()
        self._fluid_depth.setRange(0.01, 500.0)
        self._fluid_depth.setValue(self._prefs.last_fluid_depth_mm or _FALLBACK_FLUID_MM)
        self._fluid_depth.setSuffix(" mm")
        self._fluid_depth.setDecimals(2)
        form.addRow("Fluid depth:", self._fluid_depth)

        layout.addWidget(grp)

        hint = QLabel(
            "These values define the pattern-to-surface optical stack used for\n"
            "η amplitude calibration. You can fine-tune them later in Properties."
        )
        hint.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 11px; padding: 8px 4px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()

        self._stack.addWidget(page)

    # ── navigation ──────────────────────────────────────────────────
    def _update_nav(self) -> None:
        steps = ["Name & path", "Images", "Optical setup"]
        parts = []
        for i, label in enumerate(steps):
            if i == self._step:
                parts.append(f"<b>{i+1}. {label}</b>")
            elif i < self._step:
                parts.append(f"<span style='color: {tokens.SUCCESS}'>✓ {label}</span>")
            else:
                parts.append(f"<span style='color: {tokens.TEXT_MUTED}'>{i+1}. {label}</span>")
        self._step_bar.setText(" &nbsp;→&nbsp; ".join(parts))
        self._step_label.setText(f"Step {self._step + 1} of 3")

        self._back_btn.setVisible(self._step > 0)
        self._stack.setCurrentIndex(self._step)

        if self._step == 2:
            self._next_btn.setText("Create project")
            self._next_btn.setIcon(QIcon())  # Clear icon for create button
        else:
            self._next_btn.setText("Next")
            self._next_btn.setIcon(get_icon(ICON_ARROW_FORWARD))

    def _next_step(self) -> None:
        if self._step < 2:
            self._step += 1
            self._update_nav()
        else:
            self.persist_to_prefs()
            self.accept()

    def persist_to_prefs(self) -> None:
        """Save the wizard's current values back to UserPrefs."""
        self._prefs.last_project_dir = self.project_location
        if self.image_folder:
            self._prefs.last_image_folder = self.image_folder
        if self.file_pattern:
            self._prefs.last_file_pattern = self.file_pattern
        self._prefs.last_optical_preset = self.optical_preset
        self._prefs.last_pattern_period_mm = float(self.pattern_period_mm)
        self._prefs.last_glass_thickness_mm = float(self.glass_thickness_mm)
        self._prefs.last_fluid_depth_mm = float(self.fluid_depth_mm)

    def _prev_step(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._update_nav()

    # ── helpers ─────────────────────────────────────────────────────
    def _update_preview(self) -> None:
        name = self.name_edit.text().strip() or "my_experiment"
        loc = self.location_edit.text().strip() or "."
        self._path_preview.setText(str(Path(loc) / f"{name}.ofcd"))

    def _browse_location(self) -> None:
        start = (
            self.location_edit.text().strip()
            or self._prefs.last_project_dir
            or str(Path.home() / "Documents")
        )
        path = QFileDialog.getExistingDirectory(self, "Select Project Location", start)
        if path:
            self.location_edit.setText(path)

    def _browse_image_folder(self) -> None:
        start = (
            self.image_folder_edit.text().strip()
            or self._prefs.last_image_folder
            or str(Path.home())
        )
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder", start)
        if path:
            self.image_folder_edit.setText(path)
            self._auto_detect_pattern(path)
            self._count_images(path)

    def _auto_detect_pattern(self, folder: str) -> None:
        """Fill the pattern field with the most common image type found in *folder*."""
        from openfcd.io.image import detect_pattern
        detected = detect_pattern(folder)
        self.pattern_edit.setText(detected)

    def _count_images(self, folder: str) -> None:
        """Scan folder and show matched count; warn when nothing matches."""
        from openfcd.io.image import scan_frames
        try:
            pattern = self.pattern_edit.text().strip() or "Img*.jpg"
            frames = scan_frames(folder, pattern)
            if frames:
                self._image_count_label.setText(f"{len(frames)} files")
                self._image_count_label.setStyleSheet(
                    f"color: {tokens.TEXT_MUTED}; font-family: {tokens.FONT_MONO}; font-size: 11px;"
                )
            else:
                self._image_count_label.setText(f"0 files — no match for \"{pattern}\"")
                self._image_count_label.setStyleSheet(
                    "color: #e05252; font-family: monospace; font-size: 11px;"
                )
        except Exception:
            self._image_count_label.setText("scan failed")
            self._image_count_label.setStyleSheet(
                "color: #e05252; font-family: monospace; font-size: 11px;"
            )

    def _on_pattern_changed(self, _text: str) -> None:
        folder = self.image_folder_edit.text().strip()
        if folder:
            self._count_images(folder)

    @staticmethod
    def _input_style() -> str:
        return f"""
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {tokens.BG_TERTIARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                border-radius: {tokens.RADIUS_SM}px;
                padding: 4px 8px;
            }}
        """

    # ── public accessors (read by MainWindow after accept) ──────────
    @property
    def project_name(self) -> str:
        return self.name_edit.text().strip() or "my_experiment"

    @property
    def project_location(self) -> str:
        return self.location_edit.text().strip() or str(Path.home() / "Documents")

    @property
    def image_folder(self) -> str:
        return self.image_folder_edit.text().strip()

    @property
    def file_pattern(self) -> str:
        return self.pattern_edit.text().strip() or "Img*.jpg"

    @property
    def optical_preset(self) -> str:
        return self._optical_preset.currentText()

    @property
    def pattern_period_mm(self) -> float:
        return self._pattern_period.value()

    @property
    def glass_thickness_mm(self) -> float:
        return self._glass_thickness.value()

    @property
    def fluid_depth_mm(self) -> float:
        return self._fluid_depth.value()
