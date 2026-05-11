from __future__ import annotations
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from openfcd.io.project import ProjectModel
from openfcd.io.annotation import AnnotationSchema, load as load_annotation, save as save_annotation
from openfcd.io.scene import load_scenes, save_scene as _save_scene, delete_scene as _delete_scene


_LOCK = threading.Lock()


class FileSessionStore:
    """Filesystem .ofcd/ directory session manager."""

    def __init__(
        self, 
        dir_path: Path, 
        project: ProjectModel, 
        annotation: AnnotationSchema | None = None,
        read_only: bool = False
    ):
        self._dir = Path(dir_path)
        self._project = project
        self._annotation = annotation or AnnotationSchema()
        self._read_only = read_only
        self._dirty = False
        self._last_run_id: str | None = None

    @classmethod
    def new(cls, dir_path: str | Path, name: str) -> "FileSessionStore":
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "annotations", "runs", "autosave", "exports"):
            (dir_path / sub).mkdir(exist_ok=True)

        project = ProjectModel(
            name=name,
            created=datetime.now(timezone.utc).isoformat(),
            geometry={
                "pattern_period_mm": 1.2,
                "optical_stack": {"preset": "custom", "layers": []},
            },
            data={"frames_dir": ""},
        )
        project.to_yaml(dir_path / "project.yaml")
        session = {"scenes": {}, "ui_state": {}, "scenes_order": []}
        (dir_path / "session.json").write_text(json.dumps(session, indent=2))
        store = cls(dir_path, project)
        store._scenes: list = []
        store._scenes_dirty: set[str] = set()
        return store  # _last_run_id stays None for new projects

    @classmethod
    def open(cls, dir_path: str | Path, read_only: bool = False) -> "FileSessionStore":
        dir_path = Path(dir_path)
        project = ProjectModel.from_yaml(dir_path / "project.yaml")
        
        ann_path = dir_path / "annotations" / "default.json"
        if ann_path.exists():
            annotation = load_annotation(ann_path)
        else:
            annotation = AnnotationSchema()
            
        store = cls(dir_path, project, annotation, read_only)
        if not read_only:
            lock_path = dir_path / "session.lock"
            lock_path.write_text(str(threading.get_ident()))

        # Load scenes
        scenes_dir = dir_path / "scenes"
        loaded_scenes = load_scenes(scenes_dir)
        # Load scenes_order from session.json
        session_data: dict = {}
        session_path = dir_path / "session.json"
        if session_path.exists():
            try:
                session_data = json.loads(session_path.read_text())
            except Exception:
                pass
        store._last_run_id = session_data.get("ui_state", {}).get("last_run_id")
        scenes_order = session_data.get("scenes_order", [])
        # Reorder loaded_scenes according to scenes_order
        id_to_spec = {s.id: s for s in loaded_scenes}
        ordered = [id_to_spec[sid] for sid in scenes_order if sid in id_to_spec]
        unordered = [s for s in loaded_scenes if s.id not in set(scenes_order)]
        store._scenes = ordered + unordered
        store._scenes_dirty: set[str] = set()
        return store

    def save(self) -> None:
        if self._read_only:
            return
        yaml_path = self._dir / "project.yaml"
        tmp = yaml_path.with_suffix(".yaml.tmp")
        self._project.to_yaml(tmp)
        os.replace(tmp, yaml_path)

        save_annotation(self._dir / "annotations" / "default.json", self._annotation)

        # Persist dirty scenes
        scenes_dir = self._dir / "scenes"
        for spec in getattr(self, "_scenes", []):
            if spec.id in getattr(self, "_scenes_dirty", set()):
                _save_scene(spec, scenes_dir)
        if hasattr(self, "_scenes_dirty"):
            self._scenes_dirty.clear()

        # Write scenes_order into session.json
        session_path = self._dir / "session.json"
        try:
            session_data = json.loads(session_path.read_text()) if session_path.exists() else {}
        except Exception:
            session_data = {}
        session_data["scenes_order"] = [s.id for s in getattr(self, "_scenes", [])]
        session_data.setdefault("ui_state", {})["last_run_id"] = self._last_run_id
        tmp_s = session_path.with_suffix(".json.tmp")
        tmp_s.write_text(json.dumps(session_data, indent=2))
        os.replace(tmp_s, session_path)

        self._dirty = False

    def save_as(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._project.to_yaml(path / "project.yaml")
        (path / "annotations").mkdir(exist_ok=True)
        save_annotation(path / "annotations" / "default.json", self._annotation)

    def close(self) -> None:
        lock_path = self._dir / "session.lock"
        if lock_path.exists():
            lock_path.unlink(missing_ok=True)

    def record_run(self, run_id: str, manifest: dict) -> None:
        run_dir = self._dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "config_fingerprint": self.config_fingerprint(),
            **manifest,
        }
        (run_dir / "manifest.yaml").write_text(json.dumps(payload, indent=2))
        self._last_run_id = run_id

    def list_runs(self) -> list[dict]:
        runs_dir = self._dir / "runs"
        result: list[dict] = []
        if not runs_dir.exists():
            return result
        for run_dir in sorted(runs_dir.iterdir()):
            manifest = run_dir / "manifest.yaml"
            if manifest.exists():
                result.append(json.loads(manifest.read_text()))
        return result

    def config_fingerprint(self) -> str:
        payload = (
            json.dumps(self._project.geometry.model_dump(), sort_keys=True)
            + json.dumps(self._project.process.model_dump(), sort_keys=True)
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def is_stale(self, run_id: str) -> bool:
        runs = {r["run_id"]: r for r in self.list_runs()}
        if run_id not in runs:
            return True
        return runs[run_id].get("config_fingerprint") != self.config_fingerprint()

    @property
    def project(self) -> ProjectModel:
        return self._project

    @property
    def annotation(self) -> AnnotationSchema:
        return self._annotation

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def last_run_id(self) -> str | None:
        return self._last_run_id

    @last_run_id.setter
    def last_run_id(self, value: str | None) -> None:
        self._last_run_id = value

    # ── Scene mutation API ───────────────────────────────────────────
    @property
    def scenes(self) -> list:
        return list(getattr(self, "_scenes", []))

    def add_scene(self, spec) -> None:
        scenes = list(getattr(self, "_scenes", []))
        scenes.append(spec)
        self._scenes = scenes
        if not hasattr(self, "_scenes_dirty"):
            self._scenes_dirty = set()
        self._scenes_dirty.add(spec.id)

    def update_scene(self, spec) -> None:
        scenes = getattr(self, "_scenes", [])
        self._scenes = [spec if s.id == spec.id else s for s in scenes]
        if not hasattr(self, "_scenes_dirty"):
            self._scenes_dirty = set()
        self._scenes_dirty.add(spec.id)

    def remove_scene(self, scene_id: str) -> None:
        self._scenes = [s for s in getattr(self, "_scenes", []) if s.id != scene_id]
        if hasattr(self, "_scenes_dirty"):
            self._scenes_dirty.discard(scene_id)
        _delete_scene(scene_id, self._dir / "scenes")
