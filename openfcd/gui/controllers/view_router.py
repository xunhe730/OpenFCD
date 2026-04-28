"""Central view-routing contract for MainWindow.

Each node key from SimTree is mapped to a handler function that
receives the QStackedWidget and a context dict. Handlers are
registered at MainWindow.__init__ time; _on_node_selected becomes
a one-line delegate call.
"""
from __future__ import annotations
from typing import Callable
from PyQt6.QtWidgets import QStackedWidget


class ViewRouter:
    def __init__(self, stack: QStackedWidget) -> None:
        self._stack = stack
        self._handlers: dict[str, Callable[[QStackedWidget, dict], None]] = {}
        self._default: Callable[[QStackedWidget, dict], None] | None = None

    def register(self, key: str, handler: Callable[[QStackedWidget, dict], None]) -> None:
        self._handlers[key.lower()] = handler

    def set_default(self, handler: Callable[[QStackedWidget, dict], None]) -> None:
        self._default = handler

    def route(self, key: str, ctx: dict | None = None) -> None:
        handler = self._handlers.get(key.lower(), self._default)
        if handler is not None:
            handler(self._stack, ctx or {})
