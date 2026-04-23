from __future__ import annotations
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from openfcd.io.project import ProjectModel
from openfcd.io.annotation import AnnotationSchema, load as load_annotation, save as save_annotation


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
        session = {"scenes": {}, "ui_state": {}}
        (dir_path / "session.json").write_text(json.dumps(session, indent=2))
        return cls(dir_path, project)

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
        return store

    def save(self) -> None:
        if self._read_only:
            return
        yaml_path = self._dir / "project.yaml"
        tmp = yaml_path.with_suffix(".yaml.tmp")
        self._project.to_yaml(tmp)
        os.replace(tmp, yaml_path)
        
        save_annotation(self._dir / "annotations" / "default.json", self._annotation)
        
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
