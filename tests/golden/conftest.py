"""Golden regression tests compare the FCD pipeline against a reference
implementation's output.  They require a local BOS dataset that ships golden
η fields in ``output/`` and matching raw frames in ``DATA/``.

Set ``OPENFCD_BOS_ROOT`` to the dataset root to enable these tests.  Without
it every golden test is skipped, so a fresh clone of OpenFCD still passes a
full ``pytest`` run.
"""
import os
import pathlib

import pytest

_BOS_ROOT = os.environ.get("OPENFCD_BOS_ROOT")
GOLDEN_ROOT = pathlib.Path(_BOS_ROOT) / "output" if _BOS_ROOT else None
DATA_ROOT = pathlib.Path(_BOS_ROOT) / "DATA" if _BOS_ROOT else None


def pytest_configure(config):
    config.addinivalue_line("markers", "golden: mark test as golden regression")


def _require_bos():
    if _BOS_ROOT is None:
        pytest.skip("OPENFCD_BOS_ROOT not set — golden tests skipped")


@pytest.fixture(scope="session")
def golden_dir():
    _require_bos()
    d = GOLDEN_ROOT / "20hz_s2"
    if not d.exists():
        pytest.skip(f"Golden output not found: {d}")
    return d


@pytest.fixture(scope="session")
def data_dir():
    _require_bos()
    d = DATA_ROOT / "20Hz" / "s2"
    if not d.exists():
        pytest.skip(f"Data not found: {d}")
    return d
