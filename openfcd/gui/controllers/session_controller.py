"""Session controller — new/open/save/close project lifecycle.

Delegates all persistence to FileSessionStore, keeping the GUI controller
focused on Qt signals, dirty-tracking, and dialog coordination.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from openfcd.geometry.optical import get_default_layers
from openfcd.io.project import ProjectModel
from openfcd.io.store import FileSessionStore
from openfcd.io.annotation import (
    AnnotationSchema,
    ProfileLineData,
    WaveStatsConfig,
)


class SessionController(QObject):
    """Manages project session lifecycle via FileSessionStore."""

    session_opened = pyqtSignal(str)   # project_path
    session_closed = pyqtSignal()
    session_saved = pyqtSignal()
    session_modified = pyqtSignal()    # dirty state changed
    wave_stats_updated = pyqtSignal()          # annotation.wave_stats replaced
    wave_stats_recomputed = pyqtSignal(str)    # recompute done; emits run_id

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._store: FileSessionStore | None = None
        self._dirty = False
        # Cached reference image: (key tuple, np.ndarray). Invalidated when the
        # project's reference config changes or the project is closed.
        self._ref_cache: tuple[tuple, object] | None = None

    def _ref_cache_key(self) -> tuple | None:
        proj = self.project
        if proj is None:
            return None
        ref = proj.reference
        return (
            ref.mode,
            ref.source or "",
            tuple(sorted((ref.build_params or {}).items())),
        )

    def get_cached_reference(self):
        """Return a cached np.ndarray reference or None if not yet resolved."""
        if self._ref_cache is None:
            return None
        key, arr = self._ref_cache
        if key != self._ref_cache_key():
            self._ref_cache = None
            return None
        return arr

    def cache_reference(self, arr) -> None:
        key = self._ref_cache_key()
        if key is not None:
            self._ref_cache = (key, arr)

    def invalidate_reference_cache(self) -> None:
        self._ref_cache = None

    # ── properties ──────────────────────────────────────────────────
    @property
    def project_path(self) -> Path | None:
        return self._store._dir if self._store else None

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def has_project(self) -> bool:
        return self._store is not None

    @property
    def store(self) -> FileSessionStore | None:
        return self._store

    @property
    def project(self) -> ProjectModel | None:
        return self._store.project if self._store else None

    @property
    def annotation(self) -> AnnotationSchema | None:
        return self._store.annotation if self._store else None

    @property
    def last_run_id(self) -> str | None:
        return self._store._last_run_id if self._store else None

    @last_run_id.setter
    def last_run_id(self, value: str | None) -> None:
        if self._store is not None:
            self._store._last_run_id = value
            self.mark_dirty()

    # ── lifecycle ───────────────────────────────────────────────────
    def new_project(
        self,
        name: str,
        location: str | Path,
        *,
        image_folder: str = "",
        file_pattern: str = "Img*.jpg",
        pattern_period_mm: float = 1.2,
        glass_thickness_mm: float = 3.0,
        fluid_depth_mm: float = 12.0,
        optical_preset: str = "custom",
    ) -> Path:
        """Create a new .ofcd project folder via FileSessionStore.new()."""
        loc = Path(location)
        project_dir = loc / f"{name}.ofcd"

        store = FileSessionStore.new(project_dir, name)

        # Apply initial config from wizard
        proj = store.project
        proj.data.frames_dir = image_folder
        proj.data.pattern = file_pattern
        proj.geometry.pattern_period_mm = pattern_period_mm
        proj.geometry.optical_stack.preset = optical_preset  # type: ignore[assignment]
        proj.geometry.optical_stack.layers = get_default_layers(
            optical_preset,
            glass_thickness_mm=glass_thickness_mm,
            fluid_depth_mm=fluid_depth_mm,
        )
        store.save()

        self._store = store
        self._dirty = False
        self.session_opened.emit(str(project_dir))
        return project_dir

    def open_project(self, project_path: str | Path) -> None:
        """Open an existing .ofcd project via FileSessionStore.open()."""
        path = Path(project_path)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {path}")

        # Accept both a .ofcd directory and any child file (e.g. project.yaml)
        if path.is_file():
            path = path.parent
        if not path.name.endswith(".ofcd"):
            raise ValueError(f"Not a .ofcd folder: {path}")

        # Close current session if any
        if self._store is not None:
            self._store.close()

        self._store = FileSessionStore.open(path)
        self._dirty = False
        self.invalidate_reference_cache()
        self.session_opened.emit(str(path))

    def save(self) -> None:
        """Save current session state to disk."""
        if self._store is None:
            return
        self._store.save()
        self._dirty = False
        self.session_saved.emit()

    def save_as(self, new_path: str | Path) -> None:
        """Save session to a new location."""
        new = Path(new_path)
        if self._store is not None:
            # Copy entire .ofcd tree
            old = self._store._dir
            if old != new:
                shutil.copytree(old, new, dirs_exist_ok=True)
            self._store = FileSessionStore.open(new)
        self._dirty = False
        self.session_saved.emit()

    def close(self) -> None:
        """Close current project."""
        if self._store is not None:
            self._store.close()
        self._store = None
        self._dirty = False
        self.invalidate_reference_cache()
        self.session_closed.emit()

    # ── dirty tracking ──────────────────────────────────────────────
    def mark_dirty(self) -> None:
        """Mark session as having unsaved changes."""
        if not self._dirty:
            self._dirty = True
            self.session_modified.emit()

    def project_name(self) -> str:
        """Return project folder name or empty string."""
        if self._store is None:
            return ""
        return self._store._dir.name

    @property
    def scenes(self) -> list:
        """Return current list of SceneSpec (empty if no project or no scenes loaded)."""
        if self._store is None:
            return []
        return self._store.scenes

    def set_scenes(self, scenes: list) -> None:
        """Bulk-replace scenes (kept for test compatibility)."""
        if self._store is not None:
            self._store._scenes = list(scenes)
            self._store._scenes_dirty = {s.id for s in scenes}

    def add_scene(self, spec) -> None:
        if self._store is None:
            return
        self._store.add_scene(spec)
        self.mark_dirty()

    def update_scene(self, spec) -> None:
        if self._store is None:
            return
        self._store.update_scene(spec)
        self.mark_dirty()

    def remove_scene(self, scene_id: str) -> None:
        if self._store is None:
            return
        self._store.remove_scene(scene_id)
        self.mark_dirty()

    def flush_annotation_to_disk(self) -> "Path | None":
        """Write live in-memory annotation to annotations/default.json.

        Called on the main thread before _RunWorker starts so the worker's
        fresh FileSessionStore.open() picks up the current in-memory state.
        Returns the path written, or None if no project is open.
        """
        if self._store is None:
            return None
        from openfcd.io.annotation import save as save_annotation

        ann_path = self._store._dir / "annotations" / "default.json"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        save_annotation(ann_path, self._store.annotation)
        return ann_path

    def refresh_scenes(self, run_id: str) -> None:
        """Called after a Run completes. Update run_id on unbound scenes."""
        if self._store is None:
            return
        updated = []
        dirty = False
        for spec in self._store.scenes:
            if spec.run_id is None:
                updated.append(spec.model_copy(update={"run_id": run_id}))
                dirty = True
            else:
                updated.append(spec)
        if dirty:
            self._store._scenes = updated
            self._store._scenes_dirty = {s.id for s in updated}

    # ── Wave Stats ───────────────────────────────────────────────────

    def update_wave_stats_config(self, config: WaveStatsConfig) -> None:
        """Replace wave_stats on the live annotation and mark dirty.

        Emits ``wave_stats_updated``.  The caller is responsible for
        constructing a valid (frozen) ``WaveStatsConfig`` instance.
        """
        if self._store is None:
            return
        ann = self._store.annotation
        ann.wave_stats = config
        self.mark_dirty()
        self.wave_stats_updated.emit()

    def set_profile_line(self, line: ProfileLineData) -> None:
        """Update profile_line on the live annotation and mark dirty."""
        if self._store is None:
            return
        ann = self._store.annotation
        ann.profile_line = line
        self.mark_dirty()

    def recompute_wave_stats(self, run_id: str | None = None) -> bool:
        """Recompute wave_stats for *run_id* (default: latest run).

        Uses the same logic as ``openfcd postprocess`` but via direct import
        (no subprocess) for lower overhead and proper exception propagation.

        Returns ``True`` on success; ``False`` when pre-conditions are not met
        or an error occurs.  Emits ``wave_stats_recomputed(run_id)`` on success.
        """
        if self._store is None:
            return False
        annotation = self._store.annotation
        if annotation.wave_stats is None:
            return False

        ofcd_path = self._store._dir
        runs = self._store.list_runs()
        if not runs:
            return False

        if run_id is None:
            target_run: str = runs[-1]["run_id"]
        else:
            available = [r["run_id"] for r in runs]
            if run_id not in available:
                return False
            target_run = run_id

        h5_path = ofcd_path / "runs" / target_run / "results.h5"
        if not h5_path.exists():
            return False

        # Flush current in-memory annotation to disk so the recompute picks it up
        self.flush_annotation_to_disk()

        import os
        import h5py
        import numpy as np
        from openfcd.io.result import HDF5ResultStore
        from openfcd.cli.cmd_postprocess import _copy_skipping_wave_stats
        from openfcd.pipeline.wave_stats_pipeline import compute_and_write_wave_stats

        tmp_path = h5_path.with_name(h5_path.name + ".tmp")
        try:
            with h5py.File(h5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
                _copy_skipping_wave_stats(src, dst)

            store = HDF5ResultStore.open_existing_for_append(tmp_path)
            try:
                batches = store.list_batches()

                px_per_mm_by_batch: dict[str, float] = {}
                for b in batches:
                    meta = store.read_batch_meta(b)
                    px_per_mm_by_batch[b] = float(meta.get("pixel_per_mm_median", 1.0))

                body_polygons_by_batch_frame: dict[str, dict[str, np.ndarray | None]] = {}
                for b in batches:
                    frame_poly_map: dict[str, np.ndarray | None] = {}
                    for fid in store.list_frames(b):
                        fid_str = str(fid)
                        if (
                            fid_str in annotation.frame_polygons
                            and annotation.frame_polygons[fid_str]
                        ):
                            frame_poly_map[fid_str] = (
                                annotation.frame_polygons[fid_str][0].as_rc_array()
                            )
                        elif annotation.polygons:
                            frame_poly_map[fid_str] = annotation.polygons[0].as_rc_array()
                        else:
                            frame_poly_map[fid_str] = None
                    body_polygons_by_batch_frame[b] = frame_poly_map

                def _eta_loader(batch: str, fid_str: str) -> np.ndarray:
                    return store._file[f"batches/{batch}/frames/{fid_str}"][:]  # type: ignore[index]

                compute_and_write_wave_stats(
                    annotation=annotation,
                    store=store,
                    batches=batches,
                    px_per_mm_by_batch=px_per_mm_by_batch,
                    body_polygons_by_batch_frame=body_polygons_by_batch_frame,
                    eta_loader=_eta_loader,
                )
            finally:
                store.close()

            os.replace(tmp_path, h5_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False

        self.wave_stats_recomputed.emit(target_run)
        return True
