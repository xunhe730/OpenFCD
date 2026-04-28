"""Unit tests for ViewRouter."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call


def _make_router():
    """Create a ViewRouter with a mock QStackedWidget (no Qt display needed)."""
    from openfcd.gui.controllers.view_router import ViewRouter
    stack = MagicMock()
    return ViewRouter(stack), stack


def test_register_and_route_correct_handler():
    router, stack = _make_router()
    handler_a = MagicMock()
    handler_b = MagicMock()
    router.register("alpha", handler_a)
    router.register("beta", handler_b)

    router.route("alpha", {"key": "alpha"})

    handler_a.assert_called_once_with(stack, {"key": "alpha"})
    handler_b.assert_not_called()


def test_route_unknown_key_calls_default():
    router, stack = _make_router()
    known = MagicMock()
    default = MagicMock()
    router.register("known", known)
    router.set_default(default)

    router.route("unknown_key", {"key": "unknown_key"})

    known.assert_not_called()
    default.assert_called_once_with(stack, {"key": "unknown_key"})


def test_route_no_ctx_does_not_crash():
    router, stack = _make_router()
    handler = MagicMock()
    router.register("mykey", handler)

    router.route("mykey")  # no ctx argument

    handler.assert_called_once_with(stack, {})


def test_route_unknown_key_no_default_is_noop():
    router, stack = _make_router()
    # No default set, unknown key → should not raise
    router.route("nonexistent")


def test_key_lookup_is_case_insensitive():
    router, stack = _make_router()
    handler = MagicMock()
    router.register("Image_Frame", handler)

    router.route("image_frame", {"key": "image_frame"})

    handler.assert_called_once()
