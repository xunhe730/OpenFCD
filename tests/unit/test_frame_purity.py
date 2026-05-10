"""AC-A2: ``openfcd/pipeline/frame.py`` is import-pure.

It must not (at runtime) import ``Session``, ``FileSessionStore``,
``HDF5ResultStore``, ``h5py``, ``pathlib.Path`` for non-typing use, or any
``openfcd.gui.*`` module. This is enforced by AST-scanning the module's
top-level imports.
"""
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_MODULES = {
    "h5py",
    "openfcd.io.store",
    "openfcd.io.result",
}
FORBIDDEN_PREFIXES = ("openfcd.gui",)
FORBIDDEN_NAMES = {
    "Session",
    "FileSessionStore",
    "HDF5ResultStore",
}


def _frame_module_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "openfcd"
        / "pipeline"
        / "frame.py"
    )


def _walk_imports(tree: ast.AST):
    """Yield (module, names, in_type_checking) for every Import/ImportFrom node."""
    type_checking_stack: list[bool] = []

    class _V(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            test = node.test
            is_tc = (
                (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (
                    isinstance(test, ast.Attribute)
                    and test.attr == "TYPE_CHECKING"
                )
            )
            type_checking_stack.append(bool(is_tc))
            for child in node.body:
                self.visit(child)
            type_checking_stack.pop()
            for child in node.orelse:
                self.visit(child)

        def visit_Import(self, node: ast.Import) -> None:
            in_tc = bool(type_checking_stack and type_checking_stack[-1])
            for alias in node.names:
                yields.append((alias.name, [alias.asname or alias.name], in_tc))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            in_tc = bool(type_checking_stack and type_checking_stack[-1])
            module = node.module or ""
            names = [a.name for a in node.names]
            yields.append((module, names, in_tc))

    yields: list[tuple[str, list[str], bool]] = []
    _V().visit(tree)
    return yields


def test_frame_module_has_no_forbidden_runtime_imports() -> None:
    src = _frame_module_path().read_text()
    tree = ast.parse(src)
    violations: list[str] = []
    for module, names, in_tc in _walk_imports(tree):
        if in_tc:
            continue
        if module in FORBIDDEN_MODULES:
            violations.append(f"forbidden module import at runtime: {module}")
        if any(module.startswith(p) for p in FORBIDDEN_PREFIXES):
            violations.append(f"forbidden GUI module at runtime: {module}")
        for name in names:
            if name in FORBIDDEN_NAMES:
                violations.append(
                    f"forbidden symbol imported at runtime: {name} from {module!r}"
                )
    assert not violations, "frame.py purity violations: " + "; ".join(violations)
