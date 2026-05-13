"""AST-based whitelist enforcement for profile_scene.py.

Tests:
(a) Only _request_render / _do_render / __show_chart / _ProfileSceneView__show_chart
    may call __show_chart variants.  All other function bodies are forbidden from
    referencing those symbols.
(b) _recompute_live_stats (and other whitelist overlay methods) must NOT call
    fig.clear / clf / add_axes / add_subplot, and must NOT construct new Artists
    (Line2D, Polygon, PathPatch, scatter, plot, imshow, pcolormesh).
    Only set_data / set_xy / set_offsets / set_array are allowed as mutations.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

SCENE_FILE = (
    Path(__file__).parent.parent / "openfcd" / "gui" / "scenes" / "profile_scene.py"
)

# Functions allowed to call __show_chart (in any form)
SHOW_CHART_CALLERS = frozenset(
    {
        "_request_render",
        "_do_render",
        "__show_chart",
        "_ProfileSceneView__show_chart",
        "_request_render_safe",  # future-proof
    }
)

# All name forms of __show_chart that must not appear elsewhere
SHOW_CHART_NAMES = frozenset(
    {
        "_show_chart",
        "__show_chart",
        "_ProfileSceneView__show_chart",
    }
)

# Overlay-only methods that must not create new figure structure or Artists
OVERLAY_WHITELIST_METHODS = frozenset({"_recompute_live_stats"})

# Banned function/method calls inside whitelist methods
AXES_STRUCTURE_BLACKLIST = frozenset({"clear", "clf", "add_axes", "add_subplot"})
ARTIST_CONSTRUCTOR_BLACKLIST = frozenset(
    {"Line2D", "Polygon", "PathPatch", "scatter", "plot", "imshow", "pcolormesh"}
)
ALLOWED_MUTATION_METHODS = frozenset(
    {"set_data", "set_xy", "set_offsets", "set_array", "axvspan", "axvline"}
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _load_tree() -> ast.Module:
    source = SCENE_FILE.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(SCENE_FILE))


def _iter_class_methods(tree: ast.Module):
    """Yield (class_node, funcdef_node) for every method in every class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    yield node, item


def _call_name(call_node: ast.Call) -> str | None:
    """Return a string identifier for a Call node, or None if unrecognisable."""
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_in_func(func_node: ast.FunctionDef):
    """Yield all ast.Call nodes inside func_node body (recursively)."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            yield node


# ─── Test (a): show_chart call-site enforcement ──────────────────────────────


def test_show_chart_only_called_from_allowed_callers():
    """No function body outside SHOW_CHART_CALLERS may reference __show_chart variants."""
    tree = _load_tree()
    violations = []

    for cls_node, func_node in _iter_class_methods(tree):
        func_name = func_node.name
        if func_name in SHOW_CHART_CALLERS:
            continue  # allowed callers are exempt

        for call in _calls_in_func(func_node):
            func_ref = call.func

            # Detect Attribute form: self.__show_chart(...)  → mangled or plain
            if isinstance(func_ref, ast.Attribute):
                attr = func_ref.attr
                if attr in SHOW_CHART_NAMES:
                    violations.append(
                        f"  {cls_node.name}.{func_name}() at line "
                        f"{call.lineno}: calls self.{attr}() — "
                        "only _request_render/_do_render/__show_chart may do this"
                    )
            # Detect Name form: __show_chart(...)  (bare call within class scope)
            elif isinstance(func_ref, ast.Name):
                if func_ref.id in SHOW_CHART_NAMES:
                    violations.append(
                        f"  {cls_node.name}.{func_name}() at line "
                        f"{call.lineno}: bare call to {func_ref.id}() — "
                        "only _request_render/_do_render/__show_chart may do this"
                    )

    assert not violations, (
        "SHOW_CHART WHITELIST VIOLATIONS — these function bodies must NOT call "
        "__show_chart variants:\n" + "\n".join(violations)
    )


# ─── Test (b): overlay whitelist — no Artist construction inside _recompute_live_stats


def test_overlay_methods_do_not_construct_new_artists():
    """_recompute_live_stats must not call fig.clear/add_axes/add_subplot
    or construct new Artists (Line2D, Polygon, PathPatch, scatter, plot,
    imshow, pcolormesh). Only set_data/set_xy/set_offsets/set_array/
    axvspan/axvline are permitted mutations.
    """
    tree = _load_tree()
    violations = []

    for cls_node, func_node in _iter_class_methods(tree):
        if func_node.name not in OVERLAY_WHITELIST_METHODS:
            continue

        for call in _calls_in_func(func_node):
            name = _call_name(call)
            if name is None:
                continue

            # Check axes-structure blacklist
            if name in AXES_STRUCTURE_BLACKLIST:
                violations.append(
                    f"  {cls_node.name}.{func_node.name}() at line "
                    f"{call.lineno}: calls {name!r} — "
                    "must not restructure figure axes"
                )
                continue

            # Check Artist constructor blacklist (bare Name calls like scatter(...))
            if name in ARTIST_CONSTRUCTOR_BLACKLIST:
                violations.append(
                    f"  {cls_node.name}.{func_node.name}() at line "
                    f"{call.lineno}: constructs new artist via {name!r}() — "
                    "overlay methods must only mutate existing artists"
                )

    assert not violations, (
        "OVERLAY WHITELIST VIOLATIONS — these methods must not restructure "
        "the figure or construct new Artists:\n" + "\n".join(violations)
    )
