import pytest
import pathlib

GOLDEN_ROOT = pathlib.Path("/Volumes/ZXD_PKU/MAC_mini/Research/XJ-robot/BOS/output")
DATA_ROOT = pathlib.Path("/Volumes/ZXD_PKU/MAC_mini/Research/XJ-robot/BOS/DATA")


def pytest_configure(config):
    config.addinivalue_line("markers", "golden: mark test as golden regression")


@pytest.fixture(scope="session")
def golden_dir():
    d = GOLDEN_ROOT / "20hz_s2"
    if not d.exists():
        pytest.skip(f"Golden output not found: {d}")
    return d


@pytest.fixture(scope="session")
def data_dir():
    d = DATA_ROOT / "20Hz" / "s2"
    if not d.exists():
        pytest.skip(f"Data not found: {d}")
    return d
