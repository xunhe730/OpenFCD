"""T0 scaffold smoke tests — verify package is importable and version is correct."""

import tomllib
from pathlib import Path


def _pyproject_version() -> str:
    pp = Path(__file__).resolve().parent.parent / "pyproject.toml"
    return tomllib.loads(pp.read_text())["project"]["version"]


def test_version():
    import openfcd
    assert openfcd.__version__ == _pyproject_version()


def test_pipeline_importable():
    import openfcd.pipeline  # noqa: F401


def test_geometry_importable():
    import openfcd.geometry  # noqa: F401


def test_cli_importable():
    import openfcd.cli  # noqa: F401
    from openfcd.cli import app
    assert app is not None
