"""`openfcd new` — create a new .ofcd project directory."""
from __future__ import annotations

from pathlib import Path

import typer

from openfcd.io.store import FileSessionStore


def new_cmd(
    name: str = typer.Argument(..., help="Project name"),
    path: Path = typer.Argument(..., help="Directory path for new .ofcd project"),
) -> None:
    """Create a new .ofcd project directory."""
    FileSessionStore.new(path, name)
    typer.echo(f"Created project '{name}' at {path}")
