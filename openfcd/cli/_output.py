"""StageEvent serialization and display helpers."""
from __future__ import annotations

import json

from openfcd.pipeline.base import StageEvent


def stage_event_to_json(event: StageEvent) -> str:
    """Serialize a StageEvent to a single-line JSON string with all 12 fields."""
    data: dict[str, object] = {
        "kind": event.kind,
        "stage": event.stage,
        "batch": event.batch,
        "frame_idx": event.frame_idx,
        "substage": event.substage,
        "progress": event.progress,
        "total": event.total,
        "completed": event.completed,
        "metrics": event.metrics,
        "run_id": event.run_id,
        "seq": event.seq,
        "timestamp": event.timestamp,
    }
    return json.dumps(data)


def format_progress(event: StageEvent) -> str:
    """Human-readable progress string for non-JSON mode."""
    if event.kind == "progress":
        pct = f"{event.progress:.0%}" if event.progress is not None else "??"
        frame = f"frame {event.frame_idx}" if event.frame_idx is not None else ""
        batch = f"[{event.batch}]" if event.batch else ""
        return f"[{event.stage}] {pct} {batch} {frame}"
    elif event.kind == "finish":
        return f"[{event.stage}] DONE"
    elif event.kind == "error":
        return f"[{event.stage}] ERROR: {event.metrics.get('message', 'unknown')}"
    elif event.kind == "cancel":
        return f"[{event.stage}] CANCELLED"
    else:
        return f"[{event.stage}] {event.kind}"
