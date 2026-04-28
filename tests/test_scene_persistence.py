"""End-to-end test: new project → add 2 scenes → save → reopen → scenes still there."""
from __future__ import annotations

from pathlib import Path

import pytest

from openfcd.io.store import FileSessionStore
from openfcd.io.scene import SceneSpec, SceneType


def _make_store(tmp: Path) -> FileSessionStore:
    store = FileSessionStore.new(tmp / "test.ofcd", "test")
    return store


def test_add_and_save_and_reload_scenes(tmp_path):
    store = _make_store(tmp_path)
    s1 = SceneSpec(id="s1", name="Wake η", type=SceneType.ETA_MAP, frame_indices=[0, 1, 2])
    s2 = SceneSpec(id="s2", name="RMS",    type=SceneType.RMS,     frame_indices=[0, 1])
    store.add_scene(s1)
    store.add_scene(s2)
    store.save()
    store.close()

    store2 = FileSessionStore.open(tmp_path / "test.ofcd")
    scenes = store2.scenes
    assert len(scenes) == 2
    assert scenes[0].id == "s1"
    assert scenes[1].id == "s2"
    assert scenes[0].frame_indices == [0, 1, 2]
    store2.close()


def test_scenes_order_preserved_on_reload(tmp_path):
    store = _make_store(tmp_path)
    for i in range(4):
        store.add_scene(SceneSpec(id=f"s{i}", name=f"Scene {i}", type=SceneType.ETA_MAP))
    store.save()
    store.close()

    store2 = FileSessionStore.open(tmp_path / "test.ofcd")
    assert [s.id for s in store2.scenes] == ["s0", "s1", "s2", "s3"]
    store2.close()


def test_remove_scene(tmp_path):
    store = _make_store(tmp_path)
    s1 = SceneSpec(id="keep", name="keep", type=SceneType.ETA_MAP)
    s2 = SceneSpec(id="del",  name="del",  type=SceneType.RMS)
    store.add_scene(s1)
    store.add_scene(s2)
    store.save()
    store.remove_scene("del")
    store.save()
    store.close()

    store2 = FileSessionStore.open(tmp_path / "test.ofcd")
    ids = [s.id for s in store2.scenes]
    assert "keep" in ids
    assert "del" not in ids
    store2.close()


def test_update_scene(tmp_path):
    store = _make_store(tmp_path)
    s = SceneSpec(id="u1", name="original", type=SceneType.ETA_MAP)
    store.add_scene(s)
    store.save()

    updated = s.model_copy(update={"name": "renamed"})
    store.update_scene(updated)
    store.save()
    store.close()

    store2 = FileSessionStore.open(tmp_path / "test.ofcd")
    assert store2.scenes[0].name == "renamed"
    store2.close()


def test_new_store_has_empty_scenes(tmp_path):
    store = _make_store(tmp_path)
    assert store.scenes == []
    store.close()
