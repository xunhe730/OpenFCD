"""User preferences — QSettings-backed singleton for persistent UX state.

Stores last-used directories, file patterns, and physical parameters so the
NewProject wizard, file dialogs, and parameter forms can pre-populate sensible
starting values rather than hardcoded defaults.

The class accepts a ``backend: dict | None`` injection so unit tests can run
without a QApplication. When ``backend`` is None we lazily create a QSettings
instance scoped to ('OpenFCD', 'OpenFCD').
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

_RECENT_PROJECTS_CAP = 8

_DEFAULTS: dict[str, Any] = {
    "last_project_dir": "",
    "last_image_folder": "",
    "last_file_pattern": "Img*.jpg",
    "last_optical_preset": "pattern_below_window",
    "last_pattern_period_mm": 0.0,
    "last_glass_thickness_mm": 0.0,
    "last_fluid_depth_mm": 0.0,
    "recent_projects": [],
    "fast_compute": False,
    "last_export_dir": "",
    "window_geometry": b"",
    "window_state": b"",
    "splitter_state": b"",
}


class _QSettingsBackend(MutableMapping[str, Any]):
    """Adapter making QSettings look like a dict for our purposes."""

    def __init__(self) -> None:
        from PyQt6.QtCore import QSettings  # imported lazily

        self._qs = QSettings("OpenFCD", "OpenFCD")

    def __getitem__(self, key: str) -> Any:
        if not self._qs.contains(key):
            raise KeyError(key)
        return self._qs.value(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._qs.setValue(key, value)
        self._qs.sync()

    def __delitem__(self, key: str) -> None:
        self._qs.remove(key)
        self._qs.sync()

    def __iter__(self):
        return iter(self._qs.allKeys())

    def __len__(self) -> int:
        return len(self._qs.allKeys())


class UserPrefs:
    """Type-safe accessor over a key/value backend (QSettings or a dict)."""

    def __init__(self, backend: MutableMapping[str, Any] | None = None) -> None:
        self._backend: MutableMapping[str, Any] = (
            backend if backend is not None else _QSettingsBackend()
        )

    # ── generic helpers ────────────────────────────────────────────────
    def _get(self, key: str, default: Any) -> Any:
        try:
            value = self._backend[key]
        except KeyError:
            return default
        if value is None:
            return default
        return value

    def _set(self, key: str, value: Any) -> None:
        self._backend[key] = value

    @staticmethod
    def _coerce_bytes(value: Any, default: bytes = b"") -> bytes:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        # QByteArray (optional dep — check by class name to avoid import at module level)
        if type(value).__name__ == "QByteArray":
            return bytes(value)
        if isinstance(value, str):
            # QSettings on macOS plist may round-trip QByteArray as a str;
            # latin-1 is byte-preserving (maps 0x00-0xFF identically).
            return value.encode("latin-1", errors="ignore")
        return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_str(value: Any, default: str) -> str:
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return []

    # ── string-valued preferences ──────────────────────────────────────
    @property
    def last_project_dir(self) -> str:
        return self._coerce_str(self._get("last_project_dir", ""), "")

    @last_project_dir.setter
    def last_project_dir(self, value: str) -> None:
        self._set("last_project_dir", str(value or ""))

    @property
    def last_image_folder(self) -> str:
        return self._coerce_str(self._get("last_image_folder", ""), "")

    @last_image_folder.setter
    def last_image_folder(self, value: str) -> None:
        self._set("last_image_folder", str(value or ""))

    @property
    def last_file_pattern(self) -> str:
        return self._coerce_str(
            self._get("last_file_pattern", _DEFAULTS["last_file_pattern"]),
            _DEFAULTS["last_file_pattern"],
        )

    @last_file_pattern.setter
    def last_file_pattern(self, value: str) -> None:
        self._set("last_file_pattern", str(value or _DEFAULTS["last_file_pattern"]))

    @property
    def last_export_dir(self) -> str:
        return self._coerce_str(self._get("last_export_dir", ""), "")

    @last_export_dir.setter
    def last_export_dir(self, value: str) -> None:
        self._set("last_export_dir", str(value or ""))

    @property
    def last_optical_preset(self) -> str:
        return self._coerce_str(
            self._get("last_optical_preset", _DEFAULTS["last_optical_preset"]),
            _DEFAULTS["last_optical_preset"],
        )

    @last_optical_preset.setter
    def last_optical_preset(self, value: str) -> None:
        self._set("last_optical_preset", str(value or _DEFAULTS["last_optical_preset"]))

    # ── float-valued physical parameters ───────────────────────────────
    @property
    def last_pattern_period_mm(self) -> float:
        return self._coerce_float(self._get("last_pattern_period_mm", 0.0), 0.0)

    @last_pattern_period_mm.setter
    def last_pattern_period_mm(self, value: float) -> None:
        self._set("last_pattern_period_mm", float(value))

    @property
    def last_glass_thickness_mm(self) -> float:
        return self._coerce_float(self._get("last_glass_thickness_mm", 0.0), 0.0)

    @last_glass_thickness_mm.setter
    def last_glass_thickness_mm(self, value: float) -> None:
        self._set("last_glass_thickness_mm", float(value))

    @property
    def last_fluid_depth_mm(self) -> float:
        return self._coerce_float(self._get("last_fluid_depth_mm", 0.0), 0.0)

    @last_fluid_depth_mm.setter
    def last_fluid_depth_mm(self, value: float) -> None:
        self._set("last_fluid_depth_mm", float(value))

    # ── parallel-compute toggle ────────────────────────────────────────
    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return default

    @property
    def fast_compute(self) -> bool:
        return self._coerce_bool(self._get("fast_compute", False), False)

    @fast_compute.setter
    def fast_compute(self, value: bool) -> None:
        self._set("fast_compute", bool(value))

    # ── recent projects (capped MRU list) ──────────────────────────────
    @property
    def recent_projects(self) -> list[str]:
        return self._coerce_str_list(self._get("recent_projects", []))

    def add_recent_project(self, path: str | Path) -> list[str]:
        """Push *path* to the front of the MRU list, dedup, cap at 8."""
        text = str(path).strip()
        if not text:
            return self.recent_projects
        existing = [p for p in self.recent_projects if p != text]
        updated = [text] + existing
        updated = updated[:_RECENT_PROJECTS_CAP]
        self._set("recent_projects", updated)
        return updated

    def clear_recent_projects(self) -> None:
        self._set("recent_projects", [])

    # ── window geometry / state ────────────────────────────────────────
    @property
    def window_geometry(self) -> bytes:
        return self._coerce_bytes(self._get("window_geometry", b""))

    @window_geometry.setter
    def window_geometry(self, value: bytes | bytearray) -> None:
        self._set("window_geometry", bytes(value or b""))

    @property
    def window_state(self) -> bytes:
        return self._coerce_bytes(self._get("window_state", b""))

    @window_state.setter
    def window_state(self, value: bytes | bytearray) -> None:
        self._set("window_state", bytes(value or b""))

    @property
    def splitter_state(self) -> bytes:
        return self._coerce_bytes(self._get("splitter_state", b""))

    @splitter_state.setter
    def splitter_state(self, value: bytes | bytearray) -> None:
        self._set("splitter_state", bytes(value or b""))

    # ── bulk reset ─────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset all known preferences to their defaults."""
        for key, default in _DEFAULTS.items():
            self._set(key, default)


_singleton: UserPrefs | None = None


def get_prefs() -> UserPrefs:
    """Return the process-wide UserPrefs singleton (lazy QSettings backed)."""
    global _singleton
    if _singleton is None:
        _singleton = UserPrefs()
    return _singleton


def reset_prefs_singleton() -> None:
    """Clear the cached singleton (intended for tests)."""
    global _singleton
    _singleton = None
