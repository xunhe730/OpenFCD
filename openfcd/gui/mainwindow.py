from __future__ import annotations

from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout,
    QStackedWidget, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from openfcd.gui import tokens
from openfcd.gui.icons import get_app_icon
from openfcd.gui.panels import SimTree, SceneTabs, PropertiesPanel
from openfcd.gui.panels.sim_tree import NodeType
from openfcd.gui.widgets.title_bar import TitleBarWidget
from openfcd.gui.widgets.status_bar import StatusBar
from openfcd.gui.widgets.toolbar import Toolbar
from openfcd.gui.widgets.preview_widget import PreviewWidget
from openfcd.gui.controllers import RunController, SessionController
from openfcd.gui.controllers.view_router import ViewRouter
from openfcd.gui.preferences import get_prefs


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
    reference_resolved = pyqtSignal(object)  # np.ndarray, for cache write-back

    def __init__(
        self,
        project,
        project_dir: Path,
        frame_path: Path,
        annotation,
        cached_reference=None,
    ) -> None:
        super().__init__()
        self._project = project
        self._project_dir = project_dir
        self._frame_path = frame_path
        self._annotation = annotation
        self._cached_reference = cached_reference
        from openfcd.pipeline.base import CancelToken
        self._cancel_token = CancelToken()

    def cancel(self) -> None:
        """Request cancellation of the current single-frame compute."""
        self._cancel_token.cancel()

    def run(self) -> None:
        try:
            from openfcd.cli.cmd_run import (
                _build_geom_params,
                _compute_single_frame,
                _embed_eta_in_frame,
                _resolve_reference,
            )
            from openfcd.core.mask import Box, Polygon
            from openfcd.pipeline.base import CancelledError
            from openfcd.pipeline.compute import load_gray

            geom = _build_geom_params(self._project)
            self.frame_progress.emit(2, "Loading reference…")
            self._cancel_token.check()
            if self._cached_reference is not None:
                ref_img = self._cached_reference
            else:
                ref_img = _resolve_reference(self._project, self._project_dir)
                self.reference_resolved.emit(ref_img)
            self.frame_progress.emit(8, "Loading frame…")
            self._cancel_token.check()
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

            computation = _compute_single_frame(
                ref_img,
                def_img,
                geom,
                self._project,
                roi_box=roi_box,
                robot_poly=frame_poly,
                fast_preview=True,
                progress_cb=lambda pct, lbl: self.frame_progress.emit(pct, lbl),
                cancel=self._cancel_token,
            )
            eta_mm = getattr(computation, "eta_mm", computation)
            # η may be smaller than ref_img when scale normalization cropped
            # (ref zoomed down to match def's carrier period).  Re-centre it
            # inside the ROI (or the full frame if no ROI) to preserve alignment
            # with the original pixel grid.
            eta_overlay = _embed_eta_in_frame(eta_mm, ref_img.shape, roi_box)

            self.frame_done.emit(eta_overlay, eta_mm)

        except CancelledError:
            self.frame_failed.emit("Compute cancelled")
        except Exception as exc:
            self.frame_failed.emit(str(exc))

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenFCD")
        self.setWindowIcon(get_app_icon())
        self.resize(1400, 900)
        
        tokens.set_dark_mode(tokens.is_dark)
        
        self._prefs = get_prefs()
        self._session = SessionController(self)
        self._run_ctrl = RunController(self)
        # Run-result state (populated on Run completion; used for per-frame η preview)
        self._run_eta_frames = None      # np.ndarray (N,H,W) or None
        self._run_eta_mean = None        # np.ndarray (H,W) or None
        # Per-frame η cache: populated by single-frame compute AND full Run.
        # Keyed by frame index; cleared when a new project opens.
        self._frame_eta_cache: dict[int, np.ndarray] = {}
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
        self._toolbar.cancel_clicked.connect(self._on_cancel)
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
        self._view_router = ViewRouter(self._center_stack)

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

        # SceneContainer — routes scene type to dedicated view (index 3)
        from openfcd.gui.scenes.scene_container import SceneContainer
        self._scene_container = SceneContainer(self)
        self._center_stack.addWidget(self._scene_container)  # index 3

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

        # Track current scene selection and per-scene slider positions
        self._current_scene_id: str | None = None
        self._scene_slider_pos: dict[str, int] = {}   # scene_id → last slider index

        # Register view handlers
        self._view_router.register("image_frame", lambda s, c: self._handle_image_frame(s, c))
        self._view_router.register("images",      lambda s, c: self._handle_images(s, c))
        self._view_router.register("scene_item",  lambda s, c: self._handle_scene_item(s, c))
        self._view_router.set_default(lambda s, c: None)

        # ── Status bar (22px) ──
        self._status_bar = StatusBar(self)
        self._status_bar.set_items(["Ready"])
        root.addWidget(self._status_bar)

    def _connect_signals(self) -> None:
        self._sim_tree.node_selected.connect(self._on_node_selected)
        self._sim_tree.set_reference_requested.connect(self._on_set_reference)
        self._sim_tree.compute_frame_requested.connect(self._on_compute_frame)
        self._sim_tree.new_scene_requested.connect(self._on_new_scene_requested)
        self._sim_tree.delete_scene_requested.connect(self._on_delete_scene_requested)
        self._sim_tree.duplicate_scene_requested.connect(self._on_duplicate_scene_requested)
        self._sim_tree.export_scene_requested.connect(self._on_export_scene_png)
        self._sim_tree.reveal_scene_requested.connect(self._on_reveal_scene)
        self._scene_container.scene_changed.connect(self._on_scene_changed)
        self._sim_tree.frame_disabled_requested.connect(self._on_disable_frame)
        self._sim_tree.frame_enabled_requested.connect(self._on_enable_frame)
        self._title_bar.menu_requested.connect(self._on_menu_requested)
        self._session.session_opened.connect(self._on_session_opened)
        self._session.session_modified.connect(self._on_session_dirty)
        self._session.session_saved.connect(self._on_session_saved)
        self._run_ctrl.stage_event.connect(self._on_stage_event)
        self._run_ctrl.run_finished.connect(self._on_run_finished)
        self._run_ctrl.run_failed.connect(self._on_run_failed)
        self._toolbar.preview_changed.connect(self._on_preview_changed)
        self._toolbar.overlap_changed.connect(self._on_overlap_changed)
        self._toolbar.colorbar_changed.connect(self._on_colorbar_changed)
        # Preview export PNG via right-click context menu
        self._preview.export_png_requested.connect(self._export_current_view_png)
        # Preview pixel hover → status bar
        self._preview.pixel_hovered.connect(self._on_pixel_hover)
        # Preview frame slider → show that frame
        self._preview.frame_changed.connect(self._on_slider_frame_changed)
        self._preview.preview_mode_changed.connect(self._on_preview_mode_changed)
        # Properties panel action buttons
        self._properties.set_ref_clicked.connect(
            lambda: self._on_set_reference(self._current_frame_idx)
        )
        self._properties.compute_frame_clicked.connect(
            lambda: self._on_compute_frame(self._current_frame_idx)
        )
        self._properties.run_all_clicked.connect(self._on_run)
        viz = self._properties.viz_settings_panel
        viz.apply_clicked.connect(self._on_viz_apply)
        viz.reset_clicked.connect(self._on_viz_reset)
        viz.export_png_clicked.connect(lambda: self._export_current_scene_rendered("png"))
        viz.export_pdf_clicked.connect(lambda: self._export_current_scene_rendered("pdf"))
        viz.export_csv_clicked.connect(lambda: self._status_bar.set_items(["Ready", "CSV export not available for this scene"]))
        # Annotation buttons → preview annotation mode
        self._properties.draw_roi_clicked.connect(self._on_start_roi)
        self._properties.draw_mask_clicked.connect(self._on_start_mask)
        self._properties.clear_annotation_clicked.connect(self._on_clear_annotation)
        # Preview annotation completion → update session + panel
        self._preview.roi_completed.connect(self._on_roi_completed)
        self._preview.mask_completed.connect(self._on_mask_completed)
        self._preview.point_placed.connect(self._on_point_placed)
        self._properties.dilate_changed.connect(self._on_dilate_changed)
        self._properties.highpass_sigma_changed.connect(self._on_highpass_sigma_changed)
        self._properties.taper_alpha_changed.connect(self._on_taper_alpha_changed)
        self._properties.edge_nan_changed.connect(self._on_edge_nan_changed)
        self._properties.small_hole_fill_changed.connect(self._on_small_hole_fill_changed)

        # ⌘N shortcut: New η Map when a Scene node is selected, else New Project
        from PyQt6.QtGui import QShortcut, QKeySequence
        self._shortcut_new_eta = QShortcut(QKeySequence("Ctrl+N"), self)
        self._shortcut_new_eta.activated.connect(self._on_shortcut_new_eta)

    def _on_shortcut_new_eta(self) -> None:
        """⌘N — trigger New η Map when a Scene node is selected."""
        item = self._sim_tree.currentItem()
        if item is None:
            self._on_new()  # no selection → fall back to New Project
            return
        from openfcd.gui.panels.sim_tree import ROLE_NODE_TYPE, NodeType
        node_type = item.data(0, ROLE_NODE_TYPE)
        if node_type in (NodeType.SCENE_ITEM, NodeType.SCENES):
            self._on_new_scene_requested("eta_map")
        else:
            self._on_new()

    # ── Scene signal handlers ────────────────────────────────────────
    def _on_new_scene_requested(self, scene_type: str) -> None:
        """Open FramePickerDialog, create a SceneSpec, add to session, navigate."""
        if not self._session.has_project:
            return
        if not self._frames:
            QMessageBox.information(self, "No Frames",
                "请先打开一个包含图像帧的项目。\nNo image frames loaded.")
            return

        # Check if a Run exists (needed to display η)
        run_exists = bool(self._run_eta_frames is not None or self._run_eta_mean is not None)
        if not run_exists and self._session.project_path:
            runs_dir = self._session.project_path / "runs"
            run_exists = runs_dir.is_dir() and any(runs_dir.iterdir())

        from openfcd.gui.dialogs.frame_picker import FramePickerDialog
        if not run_exists:
            QMessageBox.information(self, "Run First",
                f"创建 {scene_type} Scene 需要先完成一次 Run。\n"
                "Please run a computation first.")
            return

        dialog = FramePickerDialog(
            self,
            frames=self._frames,
            run_exists=run_exists,
            preselected=list(range(len(self._frames))),
        )
        if dialog.exec() != FramePickerDialog.DialogCode.Accepted:
            return

        indices = dialog.selected_indices()
        if not indices:
            return

        import uuid, datetime
        from openfcd.io.scene import SceneSpec, SceneType
        type_map = {"eta_map": SceneType.ETA_MAP, "profile": SceneType.PROFILE, "rms": SceneType.RMS}
        name_map = {"eta_map": "η Map", "profile": "Profile", "rms": "RMS"}

        # Bind to latest run_id if available
        run_id = None
        if self._session.project_path:
            runs_dir = self._session.project_path / "runs"
            if runs_dir.is_dir():
                dirs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
                run_id = dirs[-1] if dirs else None

        spec = SceneSpec(
            id=str(uuid.uuid4())[:8],
            name=f"{name_map.get(scene_type, scene_type)} {len(self._session.scenes) + 1}",
            type=type_map.get(scene_type, SceneType.ETA_MAP),
            frame_indices=indices,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            run_id=run_id,
        )
        self._session.add_scene(spec)
        self._sim_tree.populate_scenes(self._session.scenes)
        self._status_bar.set_items(["Ready", f"Scene created: {spec.name}"])

        # Navigate immediately to the new scene
        self._scene_container.show_scene(spec, self._session.project_path, 0)
        self._center_stack.setCurrentWidget(self._scene_container)
        self._current_scene_id = spec.id
        self._sync_viz_panel(spec)
        self._toolbar.set_preview_available(False)

    def _on_delete_scene_requested(self, scene_id: str) -> None:
        reply = QMessageBox.question(
            self, "Delete Scene",
            "Delete this scene? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._session.remove_scene(scene_id)
            self._sim_tree.populate_scenes(self._session.scenes)
            self._status_bar.set_items(["Ready", "Scene deleted"])

    def _on_duplicate_scene_requested(self, scene_id: str) -> None:
        from openfcd.io.scene import SceneSpec
        import uuid
        import datetime
        for spec in self._session.scenes:
            if spec.id == scene_id:
                new_spec = SceneSpec(
                    id=str(uuid.uuid4())[:8],
                    name=f"{spec.name} (copy)",
                    type=spec.type,
                    frame_indices=list(spec.frame_indices),
                    viz_params=dict(spec.viz_params),
                    profile_lines=spec.profile_lines,
                    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    run_id=spec.run_id,
                )
                self._session.add_scene(new_spec)
                self._sim_tree.populate_scenes(self._session.scenes)
                self._status_bar.set_items(["Ready", f"Duplicated: {new_spec.name}"])
                return

    def _on_reveal_scene(self, scene_id: str) -> None:
        """Open the OS file browser at the scene's JSON file location."""
        if not self._session.project_path:
            return
        scenes_dir = self._session.project_path / "scenes"
        scene_file = scenes_dir / f"{scene_id}.json"
        target = scene_file if scene_file.exists() else scenes_dir
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _current_scene_spec(self):
        if not self._current_scene_id:
            return None
        return next((s for s in self._session.scenes if s.id == self._current_scene_id), None)

    def _persist_scene(self, spec) -> None:
        self._session.update_scene(spec)
        self._session.save()
        self._sim_tree.populate_scenes(self._session.scenes)
        if "_properties" in self.__dict__ and hasattr(self._properties, "show_figure_properties"):
            self._properties.show_figure_properties("scene_item")
        if spec is not None and hasattr(self._sim_tree, "find_scene_item"):
            item = self._sim_tree.find_scene_item(spec.id)
            if item is not None:
                if hasattr(self._sim_tree, "blockSignals"):
                    self._sim_tree.blockSignals(True)
                if hasattr(self._sim_tree, "setCurrentItem"):
                    self._sim_tree.setCurrentItem(item)
                if hasattr(self._sim_tree, "blockSignals"):
                    self._sim_tree.blockSignals(False)
                self._current_scene_id = spec.id
                if "_properties" in self.__dict__:
                    self._sync_viz_panel(spec)

    def _on_scene_changed(self, spec) -> None:
        self._current_scene_id = spec.id
        self._persist_scene(spec)
        self._status_bar.set_items(["Ready", f"Scene updated: {spec.name}"])

    def _sync_viz_panel(self, spec) -> None:
        panel = self._properties.viz_settings_panel
        scene_type = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        panel.load_viz_params(scene_type, spec.viz_params or {})

    def _on_viz_apply(self, params: dict) -> None:
        spec = self._current_scene_spec()
        if spec is None:
            return
        updated = spec.model_copy(update={"viz_params": dict(params)})
        self._persist_scene(updated)
        self._scene_container.apply_viz(params)
        if "_center_stack" in self.__dict__:
            self._center_stack.setCurrentWidget(self._scene_container)
        self._status_bar.set_items(["Ready", f"Visualization updated: {updated.name}"])

    def _on_viz_reset(self) -> None:
        spec = self._current_scene_spec()
        if spec is None:
            return
        from openfcd.gui.scenes.viz_defaults import scene_defaults
        scene_type = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
        params = scene_defaults(scene_type)
        self._properties.viz_settings_panel.load_viz_params(scene_type, params)
        self._on_viz_apply(params)

    @staticmethod
    def _figure_id_for_scene(spec) -> str | None:
        from openfcd.io.scene import SceneType
        mapping = {
            SceneType.ETA_MAP: "eta_heatmap",
            SceneType.PROFILE: "wavelength_profile",
            SceneType.RMS: "rms_map",
        }
        return mapping.get(spec.type)

    # ── Node routing ────────────────────────────────────────────────
    def _on_node_selected(self, key: str) -> None:
        """Route tree selection to center preview + right panel."""
        self._properties.show_figure_properties(key)
        self._view_router.route(key, {"key": key})

    def _handle_image_frame(self, stack, ctx: dict) -> None:
        """Show the selected frame image in the preview."""
        item = self._sim_tree.currentItem()
        if item:
            from openfcd.gui.panels.sim_tree import ROLE_DATA
            idx = item.data(0, ROLE_DATA)
            if idx is not None and 0 <= idx < len(self._frames):
                self._current_frame_idx = idx
                self._preview.set_image(self._frames[idx])
                self._preview.set_slider_value(idx)
                stack.setCurrentWidget(self._preview)
                # Restore full-frame slider range when returning from a scene view
                self._preview.restore_full_slider(len(self._frames))
                # Update frame info panel
                fpath = self._frames[idx]
                from PyQt6.QtGui import QImage
                qimg = QImage(str(fpath))
                size_str = f"{qimg.width()}×{qimg.height()}" if not qimg.isNull() else "?"
                panel = self._properties.image_properties_panel
                panel.set_frame_info(
                    fpath.name, size_str, idx, len(self._frames)
                )

                # Re-enable toolbar preview if a run exists
                if self._run_eta_frames is not None or self._run_eta_mean is not None:
                    self._toolbar.set_preview_available(True)

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
                    dilate_px = ann.dilate_cells * ann.cell_mm * 7.27  # rough estimation

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

                # Re-apply η overlay according to current toolbar mode.
                # MUST come after annotation setup so η renders on top.
                self._reapply_run_eta()

    def _handle_images(self, stack, ctx: dict) -> None:
        """Show welcome or thumbnail grid (TODO: Phase 5)."""
        stack.setCurrentIndex(0)
        self._preview.restore_full_slider(len(self._frames))
        if self._run_eta_frames is not None or self._run_eta_mean is not None:
            self._toolbar.set_preview_available(True)

    def _handle_scene_item(self, stack, ctx: dict) -> None:
        """Show scene view for selected Scene (routes by SceneType)."""
        from openfcd.gui.panels.sim_tree import ROLE_DATA
        self._toolbar.set_preview_available(False)

        item = self._sim_tree.currentItem()
        if item is None:
            stack.setCurrentIndex(0)
            return

        scene_id: str | None = item.data(0, ROLE_DATA)
        if not scene_id or not self._session.has_project:
            stack.setCurrentIndex(0)
            return

        # Save previous scene slider pos
        if self._current_scene_id and self._current_scene_id != scene_id:
            current_slider = self._scene_container.current_slider_value()
            if current_slider is not None:
                self._scene_slider_pos[self._current_scene_id] = current_slider

        self._current_scene_id = scene_id

        spec = next((s for s in self._session.scenes if s.id == scene_id), None)
        self._scene_container.show_scene(
            spec,
            self._session.project_path,
            self._scene_slider_pos.get(scene_id),
        )
        if spec is not None:
            self._sync_viz_panel(spec)
        stack.setCurrentWidget(self._scene_container)

    def _on_preview_changed(self, on: bool) -> None:
        """Toolbar Preview ON/OFF toggled — re-apply eta overlay."""
        self._reapply_run_eta()
        self._status_bar.set_items(["Ready", "Preview: ON" if on else "Preview: OFF"])

    def _on_overlap_changed(self, on: bool) -> None:
        """Toolbar Overlap checkbox changed — re-apply eta overlay."""
        self._reapply_run_eta()

    def _on_colorbar_changed(self, on: bool) -> None:
        """Toolbar Colorbar checkbox changed — re-apply eta overlay."""
        self._reapply_run_eta()

    def _on_pixel_hover(self, row: int, col: int, value: int) -> None:
        """Update status bar with pixel coordinates."""
        base = ["Ready"]
        if self._frames:
            base.append(f"{len(self._frames)} frames")
        base.append(f"({row}, {col})  val={value}")
        self._status_bar.set_items(base)

    def _on_slider_frame_changed(self, index: int) -> None:
        """Frame slider changed — show that frame's source image and (after Run)
        re-apply the η overlay for that frame (or eta_mean if preview is off)."""
        if 0 <= index < len(self._frames):
            self._current_frame_idx = index
            self._preview.set_image(self._frames[index])
            # set_image wipes the scene → re-apply η overlay from run results
            self._reapply_run_eta()

    def _on_disable_frame(self, idx: int) -> None:
        self._sim_tree.set_frame_disabled(idx, True)
        if self._session.has_project and self._session.project:
            proj = self._session.project
            disabled = list(proj.data.disabled_frame_indices)
            if idx not in disabled:
                disabled.append(idx)
            proj.data.disabled_frame_indices = disabled
            self._session.mark_dirty()

    def _on_enable_frame(self, idx: int) -> None:
        self._sim_tree.set_frame_disabled(idx, False)
        if self._session.has_project and self._session.project:
            proj = self._session.project
            proj.data.disabled_frame_indices = [
                i for i in proj.data.disabled_frame_indices if i != idx
            ]
            self._session.mark_dirty()

    def _on_set_reference(self, idx: int) -> None:
        """Set frame as reference."""
        if not self._session.has_project or idx < 0:
            return
        proj = self._session.project
        if proj and 0 <= idx < len(self._frames):
            proj.reference.mode = "use_existing"
            proj.reference.source = self._frames[idx].name
            self._sim_tree.update_ref_mark(idx)
            self._session.invalidate_reference_cache()
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

        # Reflect "running" state immediately so the user gets instant feedback,
        # before any blocking sync/validation runs.
        self._toolbar.set_running(True)
        self._status_bar.set_items(["Running", f"Preparing frame {idx + 1}…"])
        self._status_bar.show_progress(0, 100)

        # Sync ComputePanel params before running. These are cheap UI reads.
        self._sync_compute_panel_to_project()
        if not self._ensure_optical_config_ready(interactive=True):
            self._toolbar.set_running(False)
            self._status_bar.hide_progress()
            return

        frame_path = self._frames[idx]
        self._status_bar.set_items(["Running", f"Computing frame {idx + 1}…"])

        worker = _SingleFrameWorker(
            project=proj,
            project_dir=self._session.project_path,
            frame_path=frame_path,
            annotation=self._session.annotation,
            cached_reference=self._session.get_cached_reference(),
        )
        self._single_frame_worker = worker
        worker.frame_done.connect(self._on_single_frame_done)
        worker.frame_failed.connect(self._on_single_frame_failed)
        worker.frame_progress.connect(self._on_single_frame_progress)
        worker.reference_resolved.connect(self._session.cache_reference)
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
        """Single frame computation finished — store result and display."""
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        import numpy as np
        vmin = float(np.nanpercentile(eta_mm, 2)) if not np.all(np.isnan(eta_mm)) else -0.35
        vmax = float(np.nanpercentile(eta_mm, 98)) if not np.all(np.isnan(eta_mm)) else 0.35
        self._status_bar.set_items(["Done", f"η ∈ [{vmin:.3f}, {vmax:.3f}] mm"])

        # Store in unified cache so frame navigation can re-show this η.
        self._frame_eta_cache[self._current_frame_idx] = eta_overlay

        # Activate toolbar if not already done.
        self._toolbar.set_preview_available(True)
        if not self._toolbar.preview_on():
            self._toolbar.set_preview_on(True)

        self._center_stack.setCurrentWidget(self._preview)
        self._reapply_run_eta()

    def _on_single_frame_failed(self, error: str) -> None:
        """Single frame computation failed."""
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        if "cancel" in error.lower():
            self._status_bar.set_items(["Cancelled", "Single frame"])
            return
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

        # Checker cell side length (legacy project key: pattern_period_mm)
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

        # Sync Image panel's Compute Tuning fields with project config
        img_panel = self._properties.image_properties_panel
        img_panel.set_highpass_sigma(float(getattr(proj.process, "highpass_sigma_px", 0.0)))
        img_panel.set_taper_alpha(float(proj.process.taper.alpha))
        img_panel.set_edge_nan_mm(float(proj.process.edge_nan_mm))
        img_panel.set_small_hole_fill_radius(
            float(getattr(proj.process, "small_hole_fill_radius_mm", 1.0))
        )

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

        builders = {
            "File": self._build_file_menu,
            "Edit": self._build_edit_menu,
            "Run": self._build_run_menu,
            "Scenes": self._build_scenes_menu,
            "Tools": self._build_tools_menu,
            "View": self._build_view_menu,
            "Help": self._build_help_menu,
        }
        builder = builders.get(menu_name)
        if builder is None:
            act = menu.addAction(f"Unknown menu: {menu_name}")
            act.setEnabled(False)
        else:
            builder(menu)

        menu.exec(QCursor.pos())

    # ── menu builders ────────────────────────────────────────────────
    def _export_current_view_png(self) -> None:
        """Export the currently visible central widget as PNG."""
        from PyQt6.QtWidgets import QFileDialog
        import datetime
        project_name = (
            self._session.project_path.name.replace(".ofcd", "")
            if self._session.project_path else "openfcd"
        )
        scene_name = "view"
        if self._current_scene_id:
            for spec in self._session.scenes:
                if spec.id == self._current_scene_id:
                    scene_name = spec.name.replace(" ", "_")
                    break
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{project_name}_{scene_name}_{ts}.png"
        start_dir = self._prefs.last_export_dir or self._prefs.last_project_dir or str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", str(Path(start_dir) / default_name), "PNG files (*.png)"
        )
        if not path:
            return
        self._prefs.last_export_dir = str(Path(path).parent)
        current = self._center_stack.currentWidget()
        if current is None:
            return
        from PyQt6.QtGui import QPixmap
        pixmap = current.grab()
        if pixmap.save(path, "PNG"):
            self._status_bar.set_items(["Exported", Path(path).name])
        else:
            QMessageBox.warning(self, "Export Failed", f"Could not save PNG to:\n{path}")

    def _export_current_scene_rendered(self, fmt: str = "png") -> None:
        """Export the selected scene through the registered renderer."""
        spec = self._current_scene_spec()
        if spec is None or not self._session.project_path:
            self._export_current_view_png()
            return
        figure_id = self._figure_id_for_scene(spec)
        if figure_id is None:
            QMessageBox.warning(self, "Export Failed", "No renderer for this scene type.")
            return

        default_name = f"{spec.name.replace(' ', '_')}.{fmt}"
        start_dir = self._prefs.last_export_dir or self._prefs.last_project_dir or str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {fmt.upper()}",
            str(Path(start_dir) / default_name),
            f"{fmt.upper()} files (*.{fmt})",
        )
        if not path:
            return
        self._prefs.last_export_dir = str(Path(path).parent)
        try:
            import importlib
            from types import SimpleNamespace
            from openfcd.core.figures import RENDERERS
            from openfcd.io.result import HDF5ResultStore

            importlib.import_module("openfcd.gui.renderers")  # registers built-in renderers
            run_id = spec.run_id
            if run_id is None:
                runs_dir = self._session.project_path / "runs"
                dirs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir()) if runs_dir.exists() else []
                run_id = dirs[-1] if dirs else None
            if run_id is None:
                raise RuntimeError("No run available for this scene")
            results_path = self._session.project_path / "runs" / run_id / "results.h5"
            renderer = RENDERERS[figure_id]
            with HDF5ResultStore.open(results_path, "r") as results:
                batches = results.list_batches()
                if not batches:
                    raise RuntimeError("No batches in results.h5")
                viz_params = dict(spec.viz_params or {})
                if hasattr(self._scene_container, "export_context"):
                    viz_params.update(self._scene_container.export_context())
                viz_params["layout_mode"] = "export"
                viz = SimpleNamespace(**viz_params)
                fig = renderer.render(results, batches[0], viz)
                fig.savefig(path, dpi=getattr(viz, "dpi", 150), bbox_inches="tight")
            self._status_bar.set_items(["Exported", Path(path).name])
            self._persist_scene(spec)
            self._center_stack.setCurrentWidget(self._scene_container)
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))

    def _on_export_scene_png(self, scene_id: str) -> None:
        self._current_scene_id = scene_id
        spec = self._current_scene_spec()
        if spec is not None and self._session.project_path:
            self._scene_container.show_scene(
                spec,
                self._session.project_path,
                self._scene_slider_pos.get(scene_id),
            )
            self._center_stack.setCurrentWidget(self._scene_container)
            self._sync_viz_panel(spec)
        self._export_current_scene_rendered("png")

    def _build_file_menu(self, menu) -> None:
        menu.addAction("New Project...", self._on_new)
        menu.addAction("Open Project...", self._on_open)
        recent_menu = menu.addMenu("Recent Projects")
        self._populate_recent_projects_menu(recent_menu)
        menu.addSeparator()
        save_act = menu.addAction("Save", self._session.save)
        save_act.setEnabled(self._session.has_project)
        export_act = menu.addAction("Export Current View as PNG…", self._export_current_view_png)
        export_act.setEnabled(self._center_stack.currentIndex() > 0)
        menu.addSeparator()
        menu.addAction("Exit", self.close)

    def _populate_recent_projects_menu(self, recent_menu) -> None:
        recents = self._prefs.recent_projects
        if not recents:
            none_act = recent_menu.addAction("(none)")
            none_act.setEnabled(False)
        else:
            for path in recents:
                exists = Path(path).exists()
                label = path if exists else f"{path}  (missing)"
                act = recent_menu.addAction(label)
                act.triggered.connect(lambda _checked=False, p=path: self._open_project_path(p))
        recent_menu.addSeparator()
        clear_act = recent_menu.addAction("Clear Recent")
        clear_act.setEnabled(bool(recents))
        clear_act.triggered.connect(self._on_clear_recent)

    def _on_clear_recent(self) -> None:
        self._prefs.clear_recent_projects()
        self._status_bar.set_items(["Ready", "Recent projects cleared"])

    def _build_edit_menu(self, menu) -> None:
        clear_act = menu.addAction("Clear Recent Projects", self._on_clear_recent)
        clear_act.setEnabled(bool(self._prefs.recent_projects))
        menu.addSeparator()
        menu.addAction("Reset UI Preferences", self._on_reset_prefs)

    def _on_reset_prefs(self) -> None:
        reply = QMessageBox.question(
            self, "Reset UI Preferences",
            "Reset all UI preferences (last directories, last parameters, recent list)?\n"
            "This does not modify any project files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._prefs.reset()
            self._status_bar.set_items(["Ready", "UI preferences reset"])

    def _build_run_menu(self, menu) -> None:
        run_act = menu.addAction("Run All Frames", self._on_run)
        run_act.setEnabled(self._session.has_project)
        menu.addSeparator()
        menu.addAction("Cancel", self._run_ctrl.cancel)

    def _build_scenes_menu(self, menu) -> None:
        menu.addAction("Show Welcome", lambda: self._center_stack.setCurrentIndex(0))
        menu.addAction(
            "Show Image Preview",
            lambda: self._center_stack.setCurrentWidget(self._preview),
        )

        def _show_scene_tabs() -> None:
            self._scene_tabs.setVisible(True)
            self._center_stack.setCurrentWidget(self._scene_tabs)

        menu.addAction("Show Scene Tabs", _show_scene_tabs)

    def _build_tools_menu(self, menu) -> None:
        validate_act = menu.addAction(
            "Validate Optical Stack",
            lambda: self._ensure_optical_config_ready(interactive=True),
        )
        validate_act.setEnabled(self._session.has_project)

        reset_act = menu.addAction(
            "Reset Compute Defaults",
            self._populate_compute_panel_from_project,
        )
        reset_act.setEnabled(self._session.has_project)

        menu.addSeparator()
        fast_act = menu.addAction("Fast Compute (use all CPU cores)")
        fast_act.setCheckable(True)
        fast_act.setChecked(self._prefs.fast_compute)
        fast_act.setToolTip(
            "Let BLAS/OpenMP fan out across cores during a Run.\n"
            "Faster on large frame sets but breaks bit-equal serial↔parallel."
        )
        fast_act.toggled.connect(self._on_fast_compute_toggled)

    def _on_fast_compute_toggled(self, enabled: bool) -> None:
        self._prefs.fast_compute = enabled
        self._status_bar.set_items([
            "Ready",
            "Fast compute: ON" if enabled else "Fast compute: OFF",
        ])

    def _build_view_menu(self, menu) -> None:
        menu.addAction("Toggle Dark Mode", lambda: tokens.set_dark_mode(not tokens.is_dark))

    def _build_help_menu(self, menu) -> None:
        menu.addAction("About OpenFCD", self._on_about)
        menu.addAction("Open Documentation", self._on_open_docs)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About OpenFCD",
            "<b>OpenFCD</b> v0.0.1<br>"
            "Free-surface height reconstruction from synthetic-Schlieren video.<br><br>"
            "Pipeline: Preprocess → Compute → Postprocess.<br>"
            "GUI · CLI parity via shared stage chain.",
        )

    def _on_open_docs(self) -> None:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        # Walk up from this module to find the repo root containing docs/.
        candidate = Path(__file__).resolve()
        for parent in candidate.parents:
            docs = parent / "docs"
            if docs.is_dir():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs)))
                return
        self._status_bar.set_items(["Help", "docs/ folder not found"])

    # ── Session events ──────────────────────────────────────────────
    def _on_session_opened(self, path: str) -> None:
        # New project → clear η caches from any previous session.
        self._frame_eta_cache = {}
        self._run_eta_frames = None
        self._run_eta_mean = None
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

            # Show image picker to let user filter frames before import.
            if frames:
                from openfcd.gui.dialogs.image_picker import ImagePickerDialog
                picker = ImagePickerDialog(self, frames=frames)
                if picker.exec() == ImagePickerDialog.DialogCode.Accepted:
                    frames = picker.selected_frames

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

        # Restore disabled frame state
        if proj and proj.data.disabled_frame_indices:
            for idx in proj.data.disabled_frame_indices:
                self._sim_tree.set_frame_disabled(idx, True)

        # Load scenes into SimTree
        self._sim_tree.populate_scenes(self._session.scenes)

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
    # Map each stage's local 0..1 progress onto a slice of the global bar so
    # Preprocess (fast) doesn't shoot to 100% before Compute starts.
    _STAGE_RANGES = {
        "preprocess": (0.0, 0.05),
        "compute":    (0.05, 0.95),
        "postprocess":(0.95, 1.0),
    }

    def _on_stage_event(self, event) -> None:
        stage = getattr(event, "stage", "unknown") or "unknown"
        progress = getattr(event, "progress", None)
        frame = getattr(event, "frame", None)

        lo, hi = self._STAGE_RANGES.get(stage.lower(), (0.0, 1.0))
        if progress is None:
            # Info/heartbeat event — keep the bar at the stage floor instead
            # of snapping to 0% (which used to make the bar look frozen).
            global_pct = int(lo * 100)
        else:
            local = max(0.0, min(1.0, float(progress)))
            global_pct = int((lo + (hi - lo) * local) * 100)

        self._status_bar.set_progress(global_pct)
        label_pct = f"{global_pct}%"
        if frame is not None:
            self._status_bar.set_items(["Running", stage, f"frame {frame}", label_pct])
        else:
            self._status_bar.set_items(["Running", stage, label_pct])

    def _compute_missing_counts(self, run_id: str) -> dict[str, int]:
        """For each scene, count how many frame_indices are absent in results.h5."""
        if not self._session.project_path:
            return {}
        h5_path = self._session.project_path / "runs" / run_id / "results.h5"
        if not h5_path.exists():
            return {}
        try:
            from openfcd.io.result import HDF5ResultStore
            store = HDF5ResultStore.open(h5_path, "r")
            available = set(store.list_frames("default"))
            store.close()
        except Exception:
            return {}
        result = {}
        for spec in self._session.scenes:
            missing = sum(1 for idx in spec.frame_indices if idx not in available)
            if missing:
                result[spec.id] = missing
        return result

    def _on_run_finished(self, _run_id: str) -> None:
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()

        # Refresh all scenes to use the new run
        self._session.refresh_scenes(_run_id)
        # Invalidate RMS view caches (force recompute with new data)
        if hasattr(self, "_scene_container"):
            self._scene_container._rms_view._cached_spec_id = None
            self._scene_container._rms_view._cached_rms = None
        # Re-populate SimTree scene nodes with updated specs + badges
        missing_counts = self._compute_missing_counts(_run_id)
        self._sim_tree.populate_scenes(self._session.scenes, missing_counts)
        n_scenes = len(self._session.scenes)
        if n_scenes > 0:
            self._status_bar.set_items(["Ready", "Computation complete", f"{n_scenes} scenes refreshed"])
        else:
            self._status_bar.set_items(["Ready", "Computation complete"])

        if not self._session.project_path:
            return

        from openfcd.io.result import HDF5ResultStore
        results_path = self._session.project_path / "runs" / _run_id / "results.h5"
        if not results_path.exists():
            return

        # Load per-frame η by filename. HDF5 frame IDs are run-local ordinals;
        # the GUI's image list can be filtered/reordered, so never bind by idx.
        self._run_eta_frames = None
        self._run_eta_mean = None
        result_store = None
        try:
            result_store = HDF5ResultStore.open(results_path, "r")
            batches = result_store.list_batches()
            if not batches:
                return
            batch = batches[0]

            try:
                self._run_eta_mean = np.asarray(result_store.read_summary(batch, "eta_mean"))
            except Exception:
                self._run_eta_mean = None

            frame_ids = result_store.list_frames(batch)
            # η arrays are full-frame with NaN outside ROI. For typical camera
            # images (3712×5568 float64 ≈ 165 MB/frame), allow up to 4 GB so
            # experiments with up to ~24 frames cache per-frame previews.
            _guard_bytes = 4_000_000_000
            eta_by_name: dict[str, np.ndarray] = {}
            _per_frame_bytes = 0
            for fid in frame_ids:
                try:
                    arr = result_store.read_frame(batch, fid)
                    if arr.ndim == 2 and arr.size > 1:
                        if _per_frame_bytes == 0:
                            _per_frame_bytes = arr.nbytes
                        if _per_frame_bytes * len(frame_ids) > _guard_bytes:
                            eta_by_name = {}
                            break
                        attrs = result_store.read_frame_attrs(batch, fid)
                        frame_name = Path(str(attrs.get("frame_path", ""))).name
                        if frame_name:
                            eta_by_name[frame_name] = np.asarray(arr, dtype=np.float64)
                except Exception:
                    pass

            matched = [
                eta_by_name.get(f.name)
                for f in self._frames
            ]
            valid = [a for a in matched if a is not None]
            if valid:
                shape = valid[0].shape
                stack = np.full((len(self._frames),) + shape, np.nan, dtype=np.float64)
                for i, arr in enumerate(matched):
                    if arr is not None and arr.shape == shape:
                        stack[i] = arr
                self._run_eta_frames = stack
        except Exception as e:
            import logging
            logging.exception(f"Failed to load run results: {e}")
        finally:
            if result_store is not None:
                result_store.close()

        # Populate per-frame cache from Run results so frame navigation works.
        if self._run_eta_frames is not None:
            for i, eta in enumerate(self._run_eta_frames):
                if 0 <= i < len(self._frames) and not np.all(np.isnan(eta)):
                    self._frame_eta_cache[i] = eta

        # Display in the preview widget.
        if self._run_eta_frames is not None:
            self._center_stack.setCurrentWidget(self._preview)
            self._preview.set_preview_toggle_visible(self._run_eta_frames is not None)
            self._preview.set_preview_mode(True)
            self._toolbar.set_preview_available(True)
            self._toolbar.set_preview_on(True)
            self._status_bar.set_items(["Run done", "Preview: ON"])
            self._reapply_run_eta()

        # Show "N scenes refreshed" for 3 seconds then restore to "Ready"
        from PyQt6.QtCore import QTimer
        n_scenes = len(self._session.scenes)
        if n_scenes > 0:
            self._status_bar.set_items(["Run done", f"{n_scenes} scenes refreshed"])
            QTimer.singleShot(3000, lambda: self._status_bar.set_items(["Ready"]))
        else:
            self._status_bar.set_items(["Ready", "Computation complete"])

    def _on_preview_mode_changed(self, enabled: bool) -> None:
        """Preview toggle flipped — re-apply per-frame η if available."""
        self._reapply_run_eta()

    def _reapply_run_eta(self) -> None:
        """Apply η overlay for the current frame using unified cache.

        Priority: unified per-frame cache > loaded run stack.
        Respects the toolbar Preview ON/OFF and Overlap toggle.
        """
        preview_on = self._toolbar.preview_on() if hasattr(self._toolbar, "preview_on") else False
        if not preview_on:
            self._preview.clear_eta_overlay()
            return

        eta = None

        # 1. Single-frame compute cache — user explicitly computed this frame;
        #    takes priority so fresh results always appear over stale Run data.
        if self._current_frame_idx in self._frame_eta_cache:
            eta = self._frame_eta_cache[self._current_frame_idx]

        # 2. Run per-frame result for this frame
        if eta is None and (self._run_eta_frames is not None
                and 0 <= self._current_frame_idx < self._run_eta_frames.shape[0]):
            candidate = self._run_eta_frames[self._current_frame_idx]
            if not np.all(np.isnan(candidate)):
                eta = candidate

        # 3. Run mean (fallback when per-frame unavailable)
        # Do not fall back to eta_mean for an individual frame. If a frame has
        # no matching run result, showing the aggregate makes the frame look
        # incorrectly computed.

        if eta is None:
            self._preview.clear_eta_overlay()
            return

        overlap = self._toolbar.overlap_on() if hasattr(self._toolbar, "overlap_on") else True
        self._preview.show_eta_overlay(np.asarray(eta), show_overlap=overlap)

    def _on_run_failed(self, error: str) -> None:
        self._toolbar.set_running(False)
        self._status_bar.hide_progress()
        if "cancel" in error.lower():
            self._status_bar.set_items(["Cancelled"])
        else:
            self._status_bar.set_items(["Failed", error[:60]])

    def _on_cancel(self) -> None:
        """Cancel whichever compute path currently owns the toolbar."""
        single_worker = getattr(self, "_single_frame_worker", None)
        if single_worker is not None and single_worker.isRunning():
            single_worker.cancel()
            self._status_bar.set_items(["Cancelling", "single frame"])
            return
        if self._run_ctrl.is_running:
            self._run_ctrl.cancel()
            self._status_bar.set_items(["Cancelling", "run"])

    # ── Actions ─────────────────────────────────────────────────────
    def _on_open(self) -> None:
        start_dir = self._prefs.last_project_dir or ""
        path = QFileDialog.getExistingDirectory(
            self, "Open .ofcd Project", start_dir
        )
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path: str) -> None:
        try:
            self._session.open_project(path)
        except (FileNotFoundError, ValueError) as err:
            QMessageBox.warning(self, "Open Failed", str(err))
            return
        self._record_recent_project(path)

    def _record_recent_project(self, path: str) -> None:
        proj_path = Path(path)
        self._prefs.add_recent_project(str(proj_path))
        if proj_path.parent.exists():
            self._prefs.last_project_dir = str(proj_path.parent)

    def _on_new(self) -> None:
        from openfcd.gui.dialogs.new_project_wizard import NewProjectWizard
        wizard = NewProjectWizard(self, prefs=self._prefs)
        if wizard.exec() == NewProjectWizard.DialogCode.Accepted:
            try:
                project_dir = self._session.new_project(
                    name=wizard.project_name,
                    location=wizard.project_location,
                    image_folder=wizard.image_folder,
                    file_pattern=wizard.file_pattern,
                    optical_preset=wizard.optical_preset,
                    pattern_period_mm=wizard.pattern_period_mm,
                    glass_thickness_mm=wizard.glass_thickness_mm,
                    fluid_depth_mm=wizard.fluid_depth_mm,
                )
                self._record_recent_project(str(project_dir))
            except Exception as err:
                QMessageBox.warning(self, "Create Failed", str(err))

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

        # A new Run invalidates all previously displayed η overlays, including
        # single-frame preview cache entries. Otherwise frame navigation can
        # show stale masks/results even after the HDF5 run has been replaced.
        self._frame_eta_cache.clear()
        self._run_eta_frames = None
        self._run_eta_mean = None
        self._preview.clear_eta_overlay()

        self._toolbar.set_running(True)
        self._status_bar.set_items(["Running", "Initializing..."])
        self._status_bar.show_progress(0, 100)

        # Apply fast-compute toggle: lets BLAS/OpenMP fan out across cores
        # (gives N-way parallelism even on a single Run). Off → bit-equal mode.
        import os as _os
        if self._prefs.fast_compute:
            _os.environ["OPENFCD_BLAS_THREADS"] = "auto"
            _os.environ["OPENFCD_STREAM_RESULTS"] = "1"
        else:
            _os.environ.pop("OPENFCD_BLAS_THREADS", None)
            _os.environ.pop("OPENFCD_STREAM_RESULTS", None)

        self._run_ctrl.start_run(
            str(self._session.project_path),
            workers=workers,
            disabled_indices=self._sim_tree.disabled_indices,
            frame_paths=tuple(self._frames),
        )

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

    def _on_highpass_sigma_changed(self, value: float) -> None:
        """Live-update project.process.highpass_sigma_px from the slider.

        Value is applied on the next Compute This Frame — the user tweaks the
        slider, clicks compute, and sees the effect immediately.
        """
        if self._session.has_project:
            self._session.project.process.highpass_sigma_px = float(value)
            self._session.mark_dirty()

    def _on_taper_alpha_changed(self, value: float) -> None:
        """Live-update project.process.taper.alpha (0 = Moisan periodic BC)."""
        if self._session.has_project:
            self._session.project.process.taper.alpha = float(value)
            self._session.mark_dirty()

    def _on_edge_nan_changed(self, value: float) -> None:
        """Live-update project.process.edge_nan_mm."""
        if self._session.has_project:
            self._session.project.process.edge_nan_mm = float(value)
            self._session.mark_dirty()

    def _on_small_hole_fill_changed(self, value: float) -> None:
        """Live-update project.process.small_hole_fill_radius_mm."""
        if self._session.has_project:
            self._session.project.process.small_hole_fill_radius_mm = float(value)
            self._session.mark_dirty()

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
