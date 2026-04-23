from __future__ import annotations

from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout,
    QStackedWidget, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from openfcd.gui import tokens
from openfcd.gui.panels import SimTree, SceneTabs, PropertiesPanel
from openfcd.gui.panels.sim_tree import NodeType
from openfcd.gui.widgets.title_bar import TitleBarWidget
from openfcd.gui.widgets.status_bar import StatusBar
from openfcd.gui.widgets.toolbar import Toolbar
from openfcd.gui.widgets.preview_widget import PreviewWidget
from openfcd.gui.controllers import RunController, SessionController


def _apply_optical_preset(project, preset: str) -> bool:
    """Update preset and keep preset-backed layer defaults consistent."""
    from openfcd.geometry.optical import get_default_layers

    valid_presets = {"pattern_below_window", "immersed_pattern", "custom"}
    if preset not in valid_presets:
        return False
    return _apply_optical_preset_with_dimensions(project, preset)


def _apply_optical_preset_with_dimensions(
    project,
    preset: str,
    *,
    glass_thickness_mm: float | None = None,
    fluid_depth_mm: float | None = None,
) -> bool:
    """Update preset-backed optical layers from explicit geometry dimensions."""
    from openfcd.geometry.optical import get_default_layers

    valid_presets = {"pattern_below_window", "immersed_pattern", "custom"}
    if preset not in valid_presets:
        return False

    stack = project.geometry.optical_stack
    preset_changed = stack.preset != preset
    stack.preset = preset
    dimensions_overridden = glass_thickness_mm is not None or fluid_depth_mm is not None

    # Preset-backed stacks should always carry their default layer layout.
    if preset != "custom" and (preset_changed or not stack.layers or dimensions_overridden):
        stack.layers = get_default_layers(
            preset,
            glass_thickness_mm=glass_thickness_mm,
            fluid_depth_mm=fluid_depth_mm,
        )
    return True


def _repair_optical_config(project, session) -> bool:
    """Auto-repair old projects with empty optical layers.

    Only repairs preset!=custom with layers=[] — silent + dirty flag.
    Returns True if repair was performed.
    """
    stack = project.geometry.optical_stack
    if stack.preset != "custom" and not stack.layers:
        if _apply_optical_preset(project, stack.preset):
            session.mark_dirty()
            return True
    return False


def _parse_workers_value(workers_str: str) -> int | None:
    """Parse workers string from ComputePanel.

    Returns:
        int: number of workers (-1 for auto), or None if invalid.
    """
    workers_str = workers_str.strip()
    if workers_str.lower() in ("auto", ""):
        return -1
    try:
        return int(workers_str)
    except ValueError:
        return None


class _SingleFrameWorker(QThread):
    """Background thread for single-frame compute preview."""
    frame_done = pyqtSignal(object, object)  # eta_mm overlay, eta_mm cropped
    frame_failed = pyqtSignal(str)           # error message
    frame_progress = pyqtSignal(int, str)    # pct (0-100), stage label

    def __init__(
        self,
        project,
        project_dir: Path,
        frame_path: Path,
        annotation,
    ) -> None:
        super().__init__()
        self._project = project
        self._project_dir = project_dir
        self._frame_path = frame_path
        self._annotation = annotation

    def run(self) -> None:
        try:
            from openfcd.cli.cmd_run import (
                _build_geom_params,
                _compute_single_frame,
                _resolve_reference,
            )
            from openfcd.core.mask import Box, Polygon
            from openfcd.pipeline.compute import load_gray

            geom = _build_geom_params(self._project)
            ref_img = _resolve_reference(self._project, self._project_dir)
            def_img = load_gray(self._frame_path)

            if ref_img.shape != def_img.shape:
                self.frame_failed.emit(
                    f"Shape mismatch: ref={ref_img.shape}, frame={def_img.shape}"
                )
                return

            roi_box = None
            frame_poly = None
            if self._annotation is not None:
                roi = self._annotation.roi
                if not roi.is_empty:
                    roi_box = Box(
                        row0=int(roi.y),
                        col0=int(roi.x),
                        height=int(roi.height),
                        width=int(roi.width),
                    )

                polys = self._annotation.frame_polygons.get(self._frame_path.name, [])
                if polys and polys[0].vertices:
                    frame_poly = Polygon(
                        [(float(v[0]), float(v[1])) for v in polys[0].vertices]
                    )

            eta_mm = _compute_single_frame(
                ref_img,
                def_img,
                geom,
                self._project,
                roi_box=roi_box,
                robot_poly=frame_poly,
                fast_preview=True,
                progress_cb=lambda pct, lbl: self.frame_progress.emit(pct, lbl),
            )
            eta_overlay = eta_mm
            if roi_box is not None:
                eta_overlay = np.full(ref_img.shape, np.nan, dtype=np.float64)
                r0 = max(0, roi_box.row0)
                c0 = max(0, roi_box.col0)
                r1 = min(ref_img.shape[0], roi_box.row0 + roi_box.height)
                c1 = min(ref_img.shape[1], roi_box.col0 + roi_box.width)
                eta_overlay[r0:r1, c0:c1] = eta_mm

            self.frame_done.emit(eta_overlay, eta_mm)

        except Exception as exc:
            self.frame_failed.emit(str(exc))

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenFCD")
        self.resize(1400, 900)
        
        tokens.set_dark_mode(tokens.is_dark)
        
        self._session = SessionController(self)
        self._run_ctrl = RunController(self)
        self._setup_ui()
        self._connect_signals()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar (32px) ──
        self._title_bar = TitleBarWidget(self)
        root.addWidget(self._title_bar)

        self._toolbar = Toolbar(parent=self)
        self._toolbar.new_project_clicked.connect(self._on_new)
        self._toolbar.open_clicked.connect(self._on_open)
        self._toolbar.save_clicked.connect(self._session.save)
        self._toolbar.run_clicked.connect(self._on_run)
        self._toolbar.cancel_clicked.connect(self._run_ctrl.cancel)
        root.addWidget(self._toolbar)

        # ── Center splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: SimTree (280px)
        self._sim_tree = SimTree(self)
        self._sim_tree.populate()
        splitter.addWidget(self._sim_tree)

        # Center: Stacked preview area
        center_col = QWidget()
        center_layout = QVBoxLayout(center_col)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Central stack: index 0 = placeholder, expanded later with
        # PreviewWidget / thumbnail grid / SceneTabs (post-processing only)
        self._center_stack = QStackedWidget()

        # Placeholder welcome label
        from PyQt6.QtWidgets import QLabel
        self._welcome = QLabel("Open or create a project to get started.")
        self._welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._welcome.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-size: 14px; padding: 40px;"
        )
        self._center_stack.addWidget(self._welcome)  # index 0

        # PreviewWidget — image viewer + frame slider
        self._preview = PreviewWidget(self)
        self._center_stack.addWidget(self._preview)  # index 1

        # SceneTabs — hidden by default, shown only for post-processing viz
        self._scene_tabs = SceneTabs(self)
        self._scene_tabs.setVisible(False)
        self._center_stack.addWidget(self._scene_tabs)  # index 2

        center_layout.addWidget(self._center_stack, 1)
        splitter.addWidget(center_col)

        # Right: PropertiesPanel
        self._properties = PropertiesPanel(self)
        splitter.addWidget(self._properties)

        splitter.setSizes([280, 770, 350])
        root.addWidget(splitter, 1)

        # Track loaded frames for slider/preview
        self._frames: list[Path] = []
        self._current_frame_idx: int = -1

        # ── Status bar (22px) ──
        self._status_bar = StatusBar(self)
        self._status_bar.set_items(["Ready"])
        root.addWidget(self._status_bar)

    def _connect_signals(self) -> None:
        self._sim_tree.node_selected.connect(self._on_node_selected)
        self._sim_tree.set_reference_requested.connect(self._on_set_reference)
        self._sim_tree.compute_frame_requested.connect(self._on_compute_frame)
        self._title_bar.menu_requested.connect(self._on_menu_requested)
        self._session.session_opened.connect(self._on_session_opened)
        self._session.session_modified.connect(self._on_session_dirty)
        self._session.session_saved.connect(self._on_session_saved)
        self._run_ctrl.stage_event.connect(self._on_stage_event)
        self._run_ctrl.run_finished.connect(self._on_run_finished)
        self._run_ctrl.run_failed.connect(self._on_run_failed)
        # Preview pixel hover → status bar
        self._preview.pixel_hovered.connect(self._on_pixel_hover)
        # Preview frame slider → show that frame
        self._preview.frame_changed.connect(self._on_slider_frame_changed)
        # Properties panel action buttons
        self._properties.set_ref_clicked.connect(
            lambda: self._on_set_reference(self._current_frame_idx)
        )
        self._properties.compute_frame_clicked.connect(
            lambda: self._on_compute_frame(self._current_frame_idx)
        )
        self._properties.run_all_clicked.connect(self._on_run)
        # Annotation buttons → preview annotation mode
        self._properties.draw_roi_clicked.connect(self._on_start_roi)
        self._properties.draw_mask_clicked.connect(self._on_start_mask)
        self._properties.clear_annotation_clicked.connect(self._on_clear_annotation)
        # Preview annotation completion → update session + panel
        self._preview.roi_completed.connect(self._on_roi_completed)
        self._preview.mask_completed.connect(self._on_mask_completed)
        self._preview.point_placed.connect(self._on_point_placed)
        self._properties.dilate_changed.connect(self._on_dilate_changed)

    # ── Node routing ────────────────────────────────────────────────
    def _on_node_selected(self, key: str) -> None:
        """Route tree selection to center preview + right panel."""
        # Right panel routing
        self._properties.show_figure_properties(key)

        # Center preview routing
        k = key.lower()
        if k == "image_frame":
            # Show the selected frame image
            item = self._sim_tree.currentItem()
            if item:
                from openfcd.gui.panels.sim_tree import ROLE_DATA
                idx = item.data(0, ROLE_DATA)
                if idx is not None and 0 <= idx < len(self._frames):
                    self._current_frame_idx = idx
                    self._preview.set_image(self._frames[idx])
                    self._preview.set_slider_value(idx)
                    self._center_stack.setCurrentWidget(self._preview)
                    # Update frame info panel
                    fpath = self._frames[idx]
                    from PyQt6.QtGui import QImage
                    qimg = QImage(str(fpath))
                    size_str = f"{qimg.width()}×{qimg.height()}" if not qimg.isNull() else "?"
                    panel = self._properties.image_properties_panel
                    panel.set_frame_info(
                        fpath.name, size_str, idx, len(self._frames)
                    )
                    
                    # Update annotation state
                    self._preview.clear_overlays()
                    if self._session.has_project:
                        ann = self._session.annotation
                        
                        # ROI
                        roi = ann.roi
                        if not roi.is_empty:
                            self._preview.show_roi(int(roi.y), int(roi.x), int(roi.height), int(roi.width))
                            panel.set_roi(int(roi.x), int(roi.y), int(roi.width), int(roi.height))
                        else:
                            panel.set_roi(0, 0, 0, 0)
                            
                        # Dilate value
                        panel._dilate_spin.blockSignals(True)
                        panel._dilate_spin.setValue(ann.dilate_cells)
                        panel._dilate_spin.blockSignals(False)
                        dilate_px = ann.dilate_cells * ann.cell_mm * 7.27 # rough estimation
                            
                        # Mask (per frame)
                        if fpath.name in ann.frame_polygons:
                            polys = ann.frame_polygons[fpath.name]
                            if polys and polys[0].vertices:
                                self._preview.show_mask(polys[0].vertices, dilate_px=dilate_px)
                                panel.set_mask_status(f"{len(polys[0].vertices)} pts")
                            else:
                                panel.set_mask_status("Not set")
                        else:
                            panel.set_mask_status("Not set")
                            
        elif k == "images":
            # Show welcome or thumbnail grid (TODO: Phase 5)
            self._center_stack.setCurrentIndex(0)
        elif k == "scene_item":
            # Post-processing viz
            self._scene_tabs.setVisible(True)
            self._center_stack.setCurrentIndex(2)  # SceneTabs
        else:
            # Default: keep current view
            pass

    def _on_pixel_hover(self, row: int, col: int, value: int) -> None:
        """Update status bar with pixel coordinates."""
        base = ["Ready"]
        if self._frames:
            base.append(f"{len(self._frames)} frames")
        base.append(f"({row}, {col})  val={value}")
        self._status_bar.set_items(base)

    def _on_slider_frame_changed(self, index: int) -> None:
        """Frame slider changed — show that frame."""
        if 0 <= index < len(self._frames):
            self._current_frame_idx = index
            self._preview.set_image(self._frames[index])
            # Also select the corresponding tree node
            # (without re-triggering a full route)

    def _on_set_reference(self, idx: int) -> None:
        """Set frame as reference."""
        if not self._session.has_project or idx < 0:
            return
        proj = self._session.project
        if proj and 0 <= idx < len(self._frames):
            proj.reference.mode = "use_existing"
            proj.reference.source = self._frames[idx].name
            self._sim_tree.update_ref_mark(idx)
            self._session.mark_dirty()

    def _on_compute_frame(self, idx: int) -> None:
        """Compute a single frame in a background thread and show the result."""
        if not self._session.has_project:
            return
        proj = self._session.project
        if proj and not proj.reference.source:
            QMessageBox.warning(
                self, "Reference Required",
                "请先指定参考帧。\nNo reference frame set."
            )
            return
        if idx < 0 or idx >= len(self._frames):
            return

        # Sync ComputePanel params before running
        self._sync_compute_panel_to_project()
        if not self._ensure_optical_config_ready(interactive=True):
            return

        # Warn when reference and deformed carriers differ by >10% (zoom mismatch).
        frame_path = self._frames[idx]
        scale_warning = self._check_carrier_scale(proj, frame_path)
        if scale_warning:
            reply = QMessageBox.warning(
                self, "Scale Mismatch Detected",
                f"{scale_warning}\n\nScale normalization will be applied automatically.\nContinue?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Ok,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return

        self._toolbar.set_running(True, "--:--")
        self._status_bar.set_items(["Running", f"Computing frame {idx + 1}...", "0%"])
        self._status_bar.show_progress(0, 100)

        worker = _SingleFrameWorker(
            project=proj,
            project_dir=self._session.project_path,
            frame_path=frame_path,
            annotation=self._session.annotation,
        )
        self._single_frame_worker = worker
        worker.frame_done.connect(self._on_single_frame_done)
        worker.frame_failed.connect(self._on_single_frame_failed)
        worker.frame_progress.connect(self._on_single_frame_progress)
        worker.finished.connect(lambda: setattr(self, "_single_frame_worker", None))
        worker.start()

    def _check_carrier_scale(self, proj, frame_path) -> str | None:
        """Return a warning string if ref/def carrier period mismatch >10%, else None."""
        try:
            from openfcd.pipeline.compute import load_gray
            from openfcd.core.flatfield import flatfield_normalize
            from openfcd.core.fcd import calculate_carriers
            from openfcd.core.registration import _mean_carrier_period_px
            from openfcd.cli.cmd_run import _resolve_reference

            ref_img = _resolve_reference(proj, self._session.project_path)
            def_img = load_gray(frame_path)
            h, w = ref_img.shape
            sigma = float(np.clip(max(h, w) * 0.06, 100.0, 2000.0))
            ref_ff = flatfield_normalize(ref_img, sigma=sigma)
            def_ff = flatfield_normalize(def_img, sigma=sigma, bg_src=ref_img)
            c_ref = calculate_carriers(ref_ff - ref_ff.mean())
            c_def = calculate_carriers(def_ff - def_ff.mean())
            p_ref = _mean_carrier_period_px(c_ref)
            p_def = _mean_carrier_period_px(c_def)
            deviation = abs(p_def / p_ref - 1.0) if p_ref > 0 else 0.0
            if deviation >= 0.10:
                return (
                    f"Carrier period mismatch: reference={p_ref:.0f} px, "
                    f"frame={p_def:.0f} px ({deviation*100:.0f}% difference).\n"
                    "This often indicates a camera zoom change between sessions."
                )
        except Exception:
            pass
        return None

    def _on_single_frame_done(self, eta_overlay, eta_mm) -> None:
        """Single frame computation finished — show eta map."""
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        import numpy as np
        vmin = float(np.nanpercentile(eta_mm, 2)) if not np.all(np.isnan(eta_mm)) else -0.35
        vmax = float(np.nanpercentile(eta_mm, 98)) if not np.all(np.isnan(eta_mm)) else 0.35
        self._status_bar.set_items([
            "Done",
            f"η ∈ [{vmin:.3f}, {vmax:.3f}] mm",
        ])
        self._center_stack.setCurrentWidget(self._preview)
        self._preview.show_eta_overlay(eta_overlay)

    def _on_single_frame_failed(self, error: str) -> None:
        """Single frame computation failed."""
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        self._status_bar.set_items(["Failed", error[:80]])
        QMessageBox.warning(self, "Compute Failed", f"单帧计算失败:\n{error}")

    def _on_single_frame_progress(self, pct: int, label: str) -> None:
        """Forward single-frame stage progress to status bar."""
        self._status_bar.set_progress(pct)
        self._status_bar.set_items(["Running", label, f"{pct}%"])

    def _sync_compute_panel_to_project(self) -> None:
        """Sync ComputePanel UI values → ProjectModel before any run."""
        if not self._session.has_project:
            return
        panel = self._properties.compute_panel
        proj = self._session.project

        # Pattern period
        try:
            val = float(panel._period_mm.text())
            if val > 0:
                proj.geometry.pattern_period_mm = val
        except ValueError:
            pass

        # Flatfield sigma
        try:
            proj.process.flatfield_sigma = float(panel._flatfield.text())
        except ValueError:
            pass

        # Taper alpha
        try:
            proj.process.taper.alpha = float(panel._taper_alpha.text())
        except ValueError:
            pass

        # Edge NaN mm
        try:
            proj.process.edge_nan_mm = float(panel._edge_nan.text())
        except ValueError:
            pass

        # Detrend — validate against schema
        detrend = panel._detrend.currentText()
        VALID_DETRENDS = {"plane", "none"}
        if detrend in VALID_DETRENDS:
            proj.process.detrend = detrend

        # Optical preset — validate against schema
        preset = panel._preset.currentText()
        glass_thickness_mm = None
        fluid_depth_mm = None
        try:
            parsed = float(panel._glass_mm.text())
            if parsed >= 0:
                glass_thickness_mm = parsed
        except ValueError:
            pass
        try:
            parsed = float(panel._fluid_mm.text())
            if parsed > 0:
                fluid_depth_mm = parsed
        except ValueError:
            pass
        _apply_optical_preset_with_dimensions(
            proj,
            preset,
            glass_thickness_mm=glass_thickness_mm,
            fluid_depth_mm=fluid_depth_mm,
        )

        self._session.mark_dirty()

    def _populate_compute_panel_from_project(self) -> None:
        """Refresh ComputePanel UI from the current project config."""
        if not self._session.has_project:
            return

        proj = self._session.project
        panel = self._properties.compute_panel
        stack = proj.geometry.optical_stack
        layers = stack.layers

        panel._period_mm.setText(f"{proj.geometry.pattern_period_mm:g}")
        glass_mm = 0.0
        fluid_mm = 0.0
        if stack.preset == "pattern_below_window" and len(layers) >= 2:
            glass_mm = layers[0].thickness_mm
            fluid_mm = layers[-1].thickness_mm
        elif stack.preset == "immersed_pattern" and layers:
            fluid_mm = layers[-1].thickness_mm
        elif stack.preset == "custom":
            glass_mm = next((layer.thickness_mm for layer in layers if layer.medium == "glass"), 0.0)
            fluid_mm = layers[-1].thickness_mm if layers else 0.0
        panel._glass_mm.setText(f"{glass_mm:g}")
        panel._fluid_mm.setText(f"{fluid_mm:g}")
        panel._flatfield.setText(f"{proj.process.flatfield_sigma:g}")
        panel._taper_alpha.setText(f"{proj.process.taper.alpha:g}")
        panel._edge_nan.setText(f"{proj.process.edge_nan_mm:g}")
        panel._workers.setText(
            "auto" if proj.run.workers is None else str(proj.run.workers)
        )

        preset_idx = panel._preset.findText(proj.geometry.optical_stack.preset)
        if preset_idx >= 0:
            panel._preset.setCurrentIndex(preset_idx)

        detrend_idx = panel._detrend.findText(proj.process.detrend)
        if detrend_idx >= 0:
            panel._detrend.setCurrentIndex(detrend_idx)

    def _ensure_optical_config_ready(self, *, interactive: bool) -> bool:
        """Repair legacy preset-backed stacks and reject empty custom stacks."""
        if not self._session.has_project:
            return False

        proj = self._session.project
        repaired = _repair_optical_config(proj, self._session)
        if repaired:
            self._populate_compute_panel_from_project()

        stack = proj.geometry.optical_stack
        if stack.preset == "custom" and not stack.layers:
            self._status_bar.set_items(["Error", "Custom optical stack has no layers"])
            if interactive:
                QMessageBox.warning(
                    self,
                    "Optical Stack Incomplete",
                    "当前项目使用 custom optical stack，但没有配置任何 layers。\n"
                    "Please configure at least one optical layer before computing.",
                )
            return False
        return True

    # ── Theme ───────────────────────────────────────────────────────
    def _apply_theme(self) -> None:
        t = tokens
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {t.BG_PRIMARY};
                color: {t.TEXT_PRIMARY};
                font-family: {t.FONT_UI};
                font-size: 13px;
            }}
            QSplitter::handle {{
                background: {t.BORDER_SUBTLE};
                width: 1px;
            }}
        """)
        # Refresh welcome label color
        if hasattr(self, "_welcome"):
            self._welcome.setStyleSheet(
                f"color: {t.TEXT_MUTED}; font-size: 14px; padding: 40px;"
            )

    # ── Application Events ──────────────────────────────────────────
    def closeEvent(self, event) -> None:
        """Handle window close event, prompting if there are unsaved changes."""
        if self._session.is_dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )
            if reply == QMessageBox.StandardButton.Save:
                self._session.save()
                event.accept()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()

    def _on_menu_requested(self, menu_name: str) -> None:
        """Handle custom menu requests from the title bar."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QCursor
        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {tokens.BG_PRIMARY};
                color: {tokens.TEXT_PRIMARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
            }}
            QMenu::item:selected {{
                background-color: {tokens.ACCENT_CLAY};
                color: white;
            }}
        """)
        
        if menu_name == "File":
            menu.addAction("New Project...", self._on_new)
            menu.addAction("Open Project...", self._on_open)
            menu.addSeparator()
            menu.addAction("Save", self._session.save)
            menu.addSeparator()
            menu.addAction("Exit", self.close)
        elif menu_name == "Run":
            menu.addAction("Run All Frames", self._on_run)
            menu.addSeparator()
            menu.addAction("Cancel", self._run_ctrl.cancel)
        elif menu_name == "View":
            menu.addAction("Dark Mode", lambda: tokens.set_dark_mode(not tokens.is_dark))
        else:
            act = menu.addAction(f"Not applicable for {menu_name} in v0.0.1")
            act.setEnabled(False)
            
        menu.exec(QCursor.pos())

    # ── Session events ──────────────────────────────────────────────
    def _on_session_opened(self, path: str) -> None:
        proj_path = Path(path)
        name = proj_path.name
        self._title_bar.set_project_title(f"{name} — OpenFCD")

        # Scan frames
        proj = self._session.project
        repaired = False
        if proj:
            repaired = _repair_optical_config(proj, self._session)
        frames: list[Path] = []
        ref_idx = -1
        if proj and proj.data.frames_dir:
            from openfcd.io.image import scan_frames
            frames = scan_frames(proj.data.frames_dir, proj.data.pattern)
            # Find ref index
            if proj.reference.source:
                for i, f in enumerate(frames):
                    if f.name == proj.reference.source:
                        ref_idx = i
                        break

        self._frames = frames

        # Data-driven tree population
        self._sim_tree.populate_from_session(
            project_name=name.replace(".ofcd", ""),
            frames=frames,
            ref_index=ref_idx,
        )

        # Setup preview slider
        self._preview.setup_slider(len(frames))
        self._populate_compute_panel_from_project()

        # Update import panel info
        if proj:
            self._properties.import_panel.set_info(
                proj.data.frames_dir or "",
                proj.data.pattern,
                len(frames),
            )

        # Update status bar
        if frames:
            frame_msg = f"{len(frames)} frames"
        else:
            pattern = proj.data.pattern if proj else "Img*.jpg"
            frame_msg = f'0 frames — pattern "{pattern}" matched nothing'
        self._status_bar.set_items([
            "Ready",
            frame_msg,
            "Optical defaults repaired" if repaired else "",
        ])

        # Load annotation if exists
        ann_path = proj_path / "annotations" / "default.json"
        if ann_path.exists():
            from openfcd.io.annotation import load as load_annotation
            ann = load_annotation(ann_path)
            # TODO: populate AnnotationPanel ROI/Mask from ann

    def _on_session_dirty(self) -> None:
        self._title_bar.is_dirty = self._session.is_dirty

    def _on_session_saved(self) -> None:
        self._title_bar.is_saving = False

        # Also save annotation
        if self._session.project_path:
            from openfcd.io.annotation import save as save_annotation
            ann_dir = self._session.project_path / "annotations"
            ann_dir.mkdir(exist_ok=True)
            # TODO: get annotation from AnnotationPanel once built

    # ── Run events ──────────────────────────────────────────────────
    def _on_stage_event(self, event) -> None:
        stage = getattr(event, "stage", "unknown")
        progress = getattr(event, "progress", 0.0)
        frame = getattr(event, "frame", None)

        self._status_bar.set_progress(int(progress * 100))
        if frame is not None:
            self._status_bar.set_items(["Running", stage, f"frame {frame}", f"{progress:.0%}"])
        else:
            self._status_bar.set_items(["Running", stage, f"{progress:.0%}"])

    def _on_run_finished(self, _run_id: str) -> None:
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        self._status_bar.set_items(["Ready", "Computation complete"])

        if self._session.project_path:
            from openfcd.io.result import HDF5ResultStore
            results_path = self._session.project_path / "runs" / _run_id / "results.h5"
            if results_path.exists():
                result_store = None
                try:
                    result_store = HDF5ResultStore.open(results_path, "r")
                    eta_mm = None
                    for batch_name in result_store.list_batches():
                        try:
                            eta_mm = result_store.read_summary(batch_name, "eta_mean")
                        except Exception:
                            frame_ids = result_store.list_frames(batch_name)
                            for frame_id in frame_ids:
                                candidate = result_store.read_frame(batch_name, frame_id)
                                if candidate.ndim == 2 and candidate.size > 1:
                                    eta_mm = candidate
                                    break
                        if eta_mm is not None:
                            self._scene_tabs.setVisible(True)
                            self._center_stack.setCurrentWidget(self._scene_tabs)
                            self._scene_tabs.show_eta(eta_mm)
                            break
                except Exception as e:
                    import logging
                    logging.exception(f"Failed to load run results: {e}")
                finally:
                    if result_store is not None:
                        result_store.close()

    def _on_run_failed(self, error: str) -> None:
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        self._status_bar.set_items(["Failed", error[:60]])

    # ── Actions ─────────────────────────────────────────────────────
    def _on_open(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Open .ofcd Project", ""
        )
        if path:
            try:
                self._session.open_project(path)
            except (FileNotFoundError, ValueError) as e:
                QMessageBox.warning(self, "Open Failed", str(e))

    def _on_new(self) -> None:
        from openfcd.gui.dialogs.new_project_wizard import NewProjectWizard
        wizard = NewProjectWizard(self)
        if wizard.exec() == NewProjectWizard.DialogCode.Accepted:
            try:
                self._session.new_project(
                    name=wizard.project_name,
                    location=wizard.project_location,
                    image_folder=wizard.image_folder,
                    file_pattern=wizard.file_pattern,
                    optical_preset=wizard.optical_preset,
                    pattern_period_mm=wizard.pattern_period_mm,
                    glass_thickness_mm=wizard.glass_thickness_mm,
                    fluid_depth_mm=wizard.fluid_depth_mm,
                )
            except Exception as e:
                QMessageBox.warning(self, "Create Failed", str(e))

    def _on_run(self) -> None:
        if not self._session.has_project:
            QMessageBox.information(
                self, "No Project",
                "请先创建或打开一个项目。\nPlease create or open a project first."
            )
            return
        proj = self._session.project
        if proj and not proj.reference.source:
            QMessageBox.warning(
                self, "Reference Required",
                "请先在 Images 列表中右键指定参考帧 (Reference)。\n"
                "No reference frame set. Right-click an image in the tree → Set as Reference."
            )
            return

        # Sync ComputePanel UI params to project before starting run
        self._sync_compute_panel_to_project()
        if not self._ensure_optical_config_ready(interactive=True):
            return
        self._session.save()  # Persist to disk before run

        # Parse workers from ComputePanel
        workers_str = self._properties.compute_panel._workers.text().strip()
        workers = _parse_workers_value(workers_str)
        if workers is None:
            self._status_bar.set_items(["Error", f"Invalid workers value: {workers_str}"])
            return

        self._toolbar.set_running(True, "00:00")
        self._status_bar.set_items(["Running", "Initializing..."])
        self._status_bar.show_progress(0, 100)
        self._run_ctrl.start_run(str(self._session.project_path), workers=workers)

    # ── Annotation handlers ────────────────────────────────────────
    def _on_start_roi(self) -> None:
        """Enter ROI drawing mode on the preview."""
        try:
            self._center_stack.setCurrentWidget(self._preview)
            self._preview.start_roi_mode()
            self._status_bar.set_items(["ROI mode", "Click and drag to draw region of interest"])
        except Exception as e:
            import traceback, logging
            logging.error(f"ROI mode error: {traceback.format_exc()}")
            print(f"ROI mode error: {traceback.format_exc()}")

    def _on_start_mask(self) -> None:
        """Enter Mask drawing mode on the preview."""
        try:
            self._center_stack.setCurrentWidget(self._preview)
            self._preview.start_mask_mode()
            self._status_bar.set_items(["Mask mode", "Click FL → FR → BR → BL (4 points clockwise)"])
        except Exception as e:
            import traceback, logging
            logging.error(f"Mask mode error: {traceback.format_exc()}")
            print(f"Mask mode error: {traceback.format_exc()}")

    def _on_clear_annotation(self) -> None:
        """Clear all annotation overlays."""
        self._preview.clear_overlays()
        panel = self._properties.image_properties_panel
        panel.set_roi(0, 0, 0, 0)
        panel.set_mask_status("Not set")
        
        if self._session.has_project:
            ann = self._session.annotation
            ann.roi.x = 0
            ann.roi.y = 0
            ann.roi.width = 0
            ann.roi.height = 0
            
            if self._current_frame_idx >= 0:
                fpath = self._frames[self._current_frame_idx]
                ann.frame_polygons.pop(fpath.name, None)
                
            self._session.mark_dirty()
            
        self._status_bar.set_items(["Ready", "Annotations cleared"])

    def _on_roi_completed(self, row0: int, col0: int, h: int, w: int) -> None:
        """ROI rectangle completed."""
        self._properties.image_properties_panel.set_roi(col0, row0, w, h)
        if self._session.has_project:
            ann = self._session.annotation
            ann.roi.x = float(col0)
            ann.roi.y = float(row0)
            ann.roi.width = float(w)
            ann.roi.height = float(h)
            self._session.mark_dirty()
            
        self._status_bar.set_items([
            "ROI set",
            f"({col0}, {row0}) {w}×{h}",
        ])

    def _on_mask_completed(self, verts: list, fwd: list) -> None:
        """Mask polygon completed."""
        self._properties.image_properties_panel.set_mask_status(
            f"4 pts, fwd=[{fwd[0]:.2f}, {fwd[1]:.2f}]"
        )
        if self._session.has_project and self._current_frame_idx >= 0:
            fpath = self._frames[self._current_frame_idx]
            ann = self._session.annotation
            
            from openfcd.io.annotation import PolygonData
            poly = PolygonData(vertices=[[float(r), float(c)] for r, c in verts], label="body")
            ann.frame_polygons[fpath.name] = [poly]
            self._session.mark_dirty()
            
            # Redraw to show the dilate preview if applicable
            dilate_px = self._properties.image_properties_panel._dilate_spin.value() * ann.cell_mm * 7.27
            self._preview.show_mask(verts, dilate_px=dilate_px)
            
        self._status_bar.set_items([
            "Mask set",
            f"4 vertices, forward=({fwd[0]:.2f}, {fwd[1]:.2f})",
        ])

    def _on_dilate_changed(self, value: float) -> None:
        if self._session.has_project:
            ann = self._session.annotation
            ann.dilate_cells = value
            self._session.mark_dirty()
            
            # Update preview if mask exists
            if self._current_frame_idx >= 0:
                fpath = self._frames[self._current_frame_idx]
                if fpath.name in ann.frame_polygons:
                    polys = ann.frame_polygons[fpath.name]
                    if polys and polys[0].vertices:
                        dilate_px = value * ann.cell_mm * 7.27
                        self._preview.show_mask(polys[0].vertices, dilate_px=dilate_px)

    def _on_point_placed(self, idx: int, row: int, col: int) -> None:
        """A single annotation point was placed — update status bar."""
        mode_labels = {0: "1/2", 1: "2/2"}  # ROI
        if idx < 4:
            mask_labels = {0: "FL 1/4", 1: "FR 2/4", 2: "BR 3/4", 3: "BL 4/4"}
            label = mask_labels.get(idx, f"{idx+1}")
        else:
            label = f"{idx+1}"
        self._status_bar.set_items([
            f"Point {label}",
            f"({row}, {col})",
        ])
