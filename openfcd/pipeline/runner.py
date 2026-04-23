from __future__ import annotations
from typing import Iterator, Callable
from openfcd.pipeline.base import Stage, StageEvent, CancelToken


class PipelineRunner:
    """Orchestrates a sequence of Stages, serially or via loky parallel."""

    def __init__(self, stages: list[Stage], workers: int = 1):
        self._stages = stages
        self._workers = workers

    def run(
        self,
        ctx: dict,
        cancel: CancelToken | None = None,
        progress: Callable[[StageEvent], None] | None = None,
    ) -> None:
        for event in self.iter(ctx, cancel):
            if progress is not None:
                progress(event)

    def iter(self, ctx: dict, cancel: CancelToken | None = None) -> Iterator[StageEvent]:
        for stage in self._stages:
            if cancel and cancel.is_cancelled:
                return
            yield from stage.run(ctx, cancel)

    def dry_run(self, ctx: dict) -> list[dict]:
        result: list[dict] = []
        for stage in self._stages:
            for action in stage.dry_run(ctx):
                result.append({"stage": stage.name, "action": action})
        return result
