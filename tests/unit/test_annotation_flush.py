"""AC-B5: flush_annotation_to_disk() writes in-memory annotation before Run.
AC-B6: CLI run_cmd loads annotations/default.json into ctx["annotation"].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave


def _make_project_dir(tmp_path: Path) -> Path:
    from openfcd.io.store import FileSessionStore

    project_dir = tmp_path / "test.ofcd"
    store = FileSessionStore.new(project_dir, "test")
    store.close()
    return project_dir


# ---------------------------------------------------------------------------
# AC-B5: GUI annotation flush
# ---------------------------------------------------------------------------

class TestFlushAnnotationToDisk:
    def test_flush_writes_annotation_to_disk(self, tmp_path: Path) -> None:
        """flush_annotation_to_disk() persists in-memory annotation before worker starts."""
        from PyQt6.QtCore import QCoreApplication

        from openfcd.gui.controllers.session_controller import SessionController
        from openfcd.io.annotation import AnnotationSchema, PolygonData, load as load_annotation

        _app = QCoreApplication.instance() or QCoreApplication([])

        project_dir = _make_project_dir(tmp_path)
        ctrl = SessionController()
        ctrl.open_project(project_dir)

        # Inject in-memory annotation (not yet flushed to disk)
        new_ann = AnnotationSchema(
            frame_polygons={
                "frame_001.png": [
                    PolygonData(
                        vertices=[[10.0, 20.0], [10.0, 30.0], [20.0, 30.0]],
                        label="body",
                    )
                ]
            }
        )
        ctrl._store._annotation = new_ann

        # Flush to disk
        path = ctrl.flush_annotation_to_disk()
        assert path is not None
        assert path.exists()

        # Verify on-disk matches in-memory
        on_disk = load_annotation(path)
        assert "frame_001.png" in on_disk.frame_polygons
        verts = on_disk.frame_polygons["frame_001.png"][0].vertices
        assert len(verts) == 3
        ctrl._store.close()

    def test_flush_returns_none_without_project(self) -> None:
        from PyQt6.QtCore import QCoreApplication
        from openfcd.gui.controllers.session_controller import SessionController

        _app = QCoreApplication.instance() or QCoreApplication([])
        ctrl = SessionController()
        assert ctrl.flush_annotation_to_disk() is None

    def test_run_controller_calls_flush_before_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RunController.start_run() calls flush_annotation_to_disk on main thread."""
        from PyQt6.QtCore import QCoreApplication

        _app = QCoreApplication.instance() or QCoreApplication([])

        flushed: list[str] = []

        class _FakeSession:
            def flush_annotation_to_disk(self) -> None:
                flushed.append("flushed")

        class _FakeWorker:
            started = False
            stage_event = type("S", (), {"connect": lambda self, f: None})()
            run_finished = type("S", (), {"connect": lambda self, f: None})()
            run_failed = type("S", (), {"connect": lambda self, f: None})()
            finished = type("S", (), {"connect": lambda self, f: None})()

            def start(self) -> None:
                _FakeWorker.started = True

        from openfcd.gui.controllers import run_controller as rc

        monkeypatch.setattr(rc, "_RunWorker", lambda *a, **kw: _FakeWorker())

        from openfcd.gui.controllers.run_controller import RunController

        ctrl = RunController()
        ctrl.start_run(
            tmp_path,
            workers=1,
            session_controller=_FakeSession(),
        )
        assert flushed == ["flushed"]
        assert _FakeWorker.started


# ---------------------------------------------------------------------------
# AC-B6: CLI annotation load
# ---------------------------------------------------------------------------

class TestCliAnnotationLoad:
    def test_run_cmd_injects_annotation_into_ctx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_cmd loads annotations/default.json and passes it in ctx["annotation"]."""
        from openfcd.io.store import FileSessionStore
        from openfcd.io.annotation import AnnotationSchema, PolygonData, save as save_annotation

        project_dir = tmp_path / "cli_ann_test.ofcd"
        store = FileSessionStore.new(project_dir, "cli_ann_test")

        img_dir = tmp_path / "frames"
        img_dir.mkdir()
        imsave(str(img_dir / "ref.png"),
               (np.ones((32, 32)) * 128).astype(np.uint8))
        imsave(str(img_dir / "frame_000.png"),
               (np.ones((32, 32)) * 120).astype(np.uint8))

        store.project.data.frames_dir = str(img_dir)
        store.project.data.pattern = "frame_*.png"
        store.project.reference.mode = "use_existing"
        store.project.reference.source = "ref.png"
        store.save()
        store.close()

        # Write annotation with a known polygon
        ann_path = project_dir / "annotations" / "default.json"
        ann = AnnotationSchema(
            frame_polygons={
                "frame_000.png": [
                    PolygonData(
                        vertices=[[5.0, 5.0], [5.0, 10.0], [10.0, 10.0]],
                        label="body",
                    )
                ]
            }
        )
        save_annotation(ann_path, ann)

        captured_ctx: list[dict] = []

        # Patch PipelineRunner to capture ctx without running compute
        class _FakeRunner:
            def __init__(self, *a, **kw):
                pass

            def iter(self, ctx, cancel):
                captured_ctx.append(dict(ctx))
                return iter([])

        from openfcd.cli import cmd_run as cr
        monkeypatch.setattr(cr, "PipelineRunner", _FakeRunner)

        from typer.testing import CliRunner
        from openfcd.cli import app

        runner = CliRunner()
        runner.invoke(app, ["run", str(project_dir)])

        assert captured_ctx, "ctx was never captured"
        ctx = captured_ctx[0]
        assert "annotation" in ctx, "annotation key missing from ctx"
        loaded_ann = ctx["annotation"]
        assert loaded_ann is not None
        assert "frame_000.png" in loaded_ann.frame_polygons
