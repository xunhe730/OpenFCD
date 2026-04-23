"""Tests for workers parsing in MainWindow._on_run()."""
from __future__ import annotations

import pytest

from openfcd.gui.mainwindow import _parse_workers_value


def test_parse_workers_auto() -> None:
    """'auto' should resolve to -1 (automatic worker count)."""
    assert _parse_workers_value("auto") == -1


def test_parse_workers_auto_uppercase() -> None:
    """'AUTO' should also resolve to -1 (case-insensitive)."""
    assert _parse_workers_value("AUTO") == -1


def test_parse_workers_empty() -> None:
    """Empty string should resolve to -1 (default auto)."""
    assert _parse_workers_value("") == -1


def test_parse_workers_whitespace() -> None:
    """Whitespace-padded 'auto' should resolve to -1."""
    assert _parse_workers_value("  auto  ") == -1


def test_parse_workers_integer() -> None:
    """Integer string should resolve to that integer."""
    assert _parse_workers_value("4") == 4


def test_parse_workers_integer_padded() -> None:
    """Whitespace-padded integer should resolve correctly."""
    assert _parse_workers_value("  8  ") == 8


def test_parse_workers_zero() -> None:
    """'0' should resolve to 0 (explicit zero workers)."""
    assert _parse_workers_value("0") == 0


def test_parse_workers_negative() -> None:
    """Negative integer should parse (caller decides validity)."""
    assert _parse_workers_value("-2") == -2


def test_parse_workers_invalid_alpha() -> None:
    """Non-numeric string should return None."""
    assert _parse_workers_value("abc") is None


def test_parse_workers_invalid_mixed() -> None:
    """Mixed alpha-numeric string should return None."""
    assert _parse_workers_value("4workers") is None


def test_parse_workers_invalid_float() -> None:
    """Float string should return None (only integers accepted)."""
    assert _parse_workers_value("4.5") is None
