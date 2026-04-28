"""Scene persistence model and I/O stubs.

SceneSpec is the single source of truth for what a Scene is.
load_scenes / save_scene / delete_scene provide atomic JSON persistence.
"""
from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class SceneType(StrEnum):
    ETA_MAP = "eta_map"
    PROFILE = "profile"
    RMS     = "rms"


class ProfileLine(BaseModel):
    p0: tuple[float, float]  # (row, col) start
    p1: tuple[float, float]  # (row, col) end


class SceneSpec(BaseModel):
    model_config = {"extra": "ignore"}   # forward-compat: ignore unknown fields

    id: str
    name: str
    type: SceneType
    frame_indices: list[int] = Field(default_factory=list)
    viz_params: dict[str, Any] = Field(default_factory=dict)
    profile_lines: dict[int, ProfileLine] | None = None
    created_at: str = ""
    run_id: str | None = None


# ── I/O ───────────────────────────────────────────────────────────

def load_scenes(scenes_dir: Path) -> list[SceneSpec]:
    """Load all SceneSpec from scenes_dir/*.json. Tolerates corrupt files."""
    if not scenes_dir.is_dir():
        return []
    specs = []
    for p in sorted(scenes_dir.glob("*.json")):
        try:
            spec = SceneSpec.model_validate_json(p.read_text(encoding="utf-8"))
            specs.append(spec)
        except Exception as exc:
            _log.warning("Skipping corrupt scene file %s: %s", p, exc)
    return specs


def save_scene(spec: SceneSpec, scenes_dir: Path) -> Path:
    """Persist spec to scenes_dir/{spec.id}.json (atomic write)."""
    scenes_dir.mkdir(parents=True, exist_ok=True)
    target = scenes_dir / f"{spec.id}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def delete_scene(scene_id: str, scenes_dir: Path) -> None:
    """Remove scenes_dir/{scene_id}.json. Silent if not found."""
    p = scenes_dir / f"{scene_id}.json"
    p.unlink(missing_ok=True)
