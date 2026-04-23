from dataclasses import dataclass, field
from typing import Protocol, Iterator
import threading
import time


@dataclass(frozen=True)
class StageEvent:
    kind: str              # "start" | "progress" | "log" | "finish" | "error" | "cancel"
    stage: str             # "preprocess" | "compute" | "postprocess"
    batch: str | None
    frame_idx: int | None
    substage: str | None
    progress: float | None
    total: int | None
    completed: int | None
    metrics: dict = field(default_factory=dict)
    run_id: str = ""
    seq: int = 0
    timestamp: float = field(default_factory=time.time)


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def check(self) -> None:
        if self._cancelled.is_set():
            raise CancelledError("Operation cancelled")

    def wait(self, timeout: float) -> bool:
        return self._cancelled.wait(timeout)


class CancelledError(Exception):
    pass


class Stage(Protocol):
    name: str

    def run(self, ctx: dict, cancel: CancelToken | None = None) -> Iterator[StageEvent]: ...
    def dry_run(self, ctx: dict) -> list[str]: ...
