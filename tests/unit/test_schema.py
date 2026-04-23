from __future__ import annotations
import json
import pytest
from pydantic import ValidationError

from openfcd.io.project import (
    ProjectModel,
    GeometryConfig,
    OpticalStack,
    OpticalLayer,
    DataConfig,
)
from openfcd.io.annotation import AnnotationSchema, load, save, validate
from openfcd.geometry.optical import OpticalGeometry


def _make_project() -> ProjectModel:
    return ProjectModel(
        name="test_proj",
        created="2026-04-22T00:00:00",
        geometry=GeometryConfig(
            pattern_period_mm=0.5,
            optical_stack=OpticalStack(
                preset="pattern_below_window",
                layers=[
                    OpticalLayer(thickness_mm=5.0, medium="glass", n=1.5),
                    OpticalLayer(thickness_mm=10.0, medium="water", n=1.33),
                ],
            ),
        ),
        data=DataConfig(frames_dir="/tmp/frames"),
    )


def _make_annotation() -> AnnotationSchema:
    return AnnotationSchema(
        condition="20Hz_s2",
        frame_range="0001-0500",
        anchor_frame="Img0001.jpg",
        manual_polygon=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
        roi={"x_mm": 140.0, "y_mm": 0.0},
        created="2026-04-22T00:00:00",
        note="test",
    )


def test_project_yaml_roundtrip(tmp_path):
    p = _make_project()
    yaml_path = tmp_path / "project.yaml"
    p.to_yaml(yaml_path)
    loaded = ProjectModel.from_yaml(yaml_path)
    expected = set(ProjectModel.model_fields.keys())
    assert len(expected) == 14
    for field in expected:
        assert getattr(loaded, field) == getattr(p, field)


def test_project_missing_required_fields():
    with pytest.raises(ValidationError):
        ProjectModel.model_validate({"created": "2026-04-22"})


def test_roi_config_units_are_mm():
    p = _make_project()
    assert isinstance(p.roi.x_mm, float)
    assert isinstance(p.roi.y_mm, float)
    assert p.roi.x_mm > 1.0


def test_annotation_roundtrip(tmp_path):
    ann = _make_annotation()
    path = tmp_path / "ann.json"
    save(path, ann)
    loaded = load(path)
    assert loaded.condition == ann.condition
    assert loaded.frame_range == ann.frame_range
    assert loaded.anchor_frame == ann.anchor_frame
    assert loaded.manual_polygon == ann.manual_polygon
    assert loaded.roi == ann.roi
    assert loaded.created == ann.created
    assert loaded.note == ann.note


def test_annotation_anchor_frame_validation():
    with pytest.raises(ValidationError):
        AnnotationSchema(
            condition="x",
            frame_range="0-1",
            anchor_frame="Img0001.png",
            manual_polygon=[[0.0, 0.0]],
            roi={},
            created="2026-04-22",
        )


def test_annotation_validate_ok(tmp_path):
    ann = _make_annotation()
    path = tmp_path / "ann.json"
    save(path, ann)
    assert validate(path) == []


def test_annotation_validate_bad(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    errs = validate(bad)
    assert len(errs) > 0


def test_optical_geometry_known_values():
    geo = GeometryConfig(
        pattern_period_mm=0.5,
        optical_stack=OpticalStack(
            preset="custom",
            layers=[
                OpticalLayer(thickness_mm=4.0, medium="glass", n=1.5),
                OpticalLayer(thickness_mm=10.0, medium="water", n=1.33),
            ],
        ),
    )
    og = OpticalGeometry(geo)
    n_top = 1.33
    expected_alpha = 1.0 - 1.0 / n_top
    expected_h = 4.0 * 1.5 / n_top + 10.0 * 1.33 / n_top
    expected_K = 1.0 / (expected_alpha * expected_h)
    assert og.alpha == pytest.approx(expected_alpha)
    assert og.h_p_eff_mm == pytest.approx(expected_h)
    assert og.K_per_mm == pytest.approx(expected_K)
    assert og.pattern_period_mm == 0.5


def test_optical_geometry_empty_layers():
    geo = GeometryConfig(
        pattern_period_mm=0.5,
        optical_stack=OpticalStack(preset="custom", layers=[]),
    )
    with pytest.raises(ValueError):
        OpticalGeometry(geo)
