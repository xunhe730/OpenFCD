from pathlib import Path
from openfcd.io.scene import SceneSpec, SceneType, ProfileLine, load_scenes, save_scene, delete_scene
import pytest


def test_scene_spec_round_trip():
    spec = SceneSpec(
        id="s1", name="η wake", type=SceneType.ETA_MAP,
        frame_indices=[0, 1, 2], created_at="2026-04-27T00:00:00Z"
    )
    reloaded = SceneSpec.model_validate_json(spec.model_dump_json())
    assert reloaded.id == "s1"
    assert reloaded.type == SceneType.ETA_MAP
    assert reloaded.frame_indices == [0, 1, 2]


def test_scene_spec_extra_fields_ignored():
    data = '{"id":"x","name":"n","type":"eta_map","unknown_future_field":42}'
    spec = SceneSpec.model_validate_json(data)
    assert spec.id == "x"


def test_model_json_schema_importable():
    schema = SceneSpec.model_json_schema()
    assert "properties" in schema


def test_save_and_load_round_trip(tmp_path):
    spec = SceneSpec(id="abc", name="test", type=SceneType.RMS, frame_indices=[1, 2])
    save_scene(spec, tmp_path)
    loaded = load_scenes(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].id == "abc"
    assert loaded[0].frame_indices == [1, 2]


def test_load_skips_corrupt_file(tmp_path):
    (tmp_path / "bad.json").write_text("not json")
    specs = load_scenes(tmp_path)
    assert specs == []


def test_delete_removes_file(tmp_path):
    spec = SceneSpec(id="del1", name="d", type=SceneType.ETA_MAP)
    save_scene(spec, tmp_path)
    delete_scene("del1", tmp_path)
    assert not (tmp_path / "del1.json").exists()
    delete_scene("del1", tmp_path)   # second call: silent


def test_load_empty_dir(tmp_path):
    assert load_scenes(tmp_path / "nonexistent") == []
