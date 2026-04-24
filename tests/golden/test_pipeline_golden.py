"""
Golden regression tests for the full openfcd CLI pipeline (T4 gate).

All tests are skipped until openfcd.cli.app exposes a fully-implemented
`run` command (i.e. T4 is complete).

The gate checks that:
  1. openfcd.cli.app is importable, AND
  2. `openfcd run --help` exits 0, AND
  3. The run command is NOT still the scaffold placeholder
     (i.e. the source does not contain "not yet implemented").
"""
import pathlib
import textwrap

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T4 gate
# ---------------------------------------------------------------------------

def _can_import_cli() -> bool:
    try:
        from openfcd.cli import app  # noqa: F401
        import inspect
        import openfcd.cli as _cli_mod

        # Reject placeholder: the scaffold prints "not yet implemented"
        src = inspect.getsource(_cli_mod)
        if "not yet implemented" in src:
            return False

        import typer.testing
        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        return result.exit_code == 0
    except Exception:
        return False


_CLI_READY = _can_import_cli()
_SKIP_REASON = "openfcd.cli run not implemented (T4 pending)"

# Paths — set OPENFCD_BOS_ROOT to enable golden tests; skipped otherwise.
import os as _os
_BOS_ROOT_STR = _os.environ.get("OPENFCD_BOS_ROOT")
if _BOS_ROOT_STR is None:
    import pytest as _pt
    _pt.skip("OPENFCD_BOS_ROOT not set", allow_module_level=True)
_BOS_ROOT = pathlib.Path(_BOS_ROOT_STR)
_GOLDEN_BATCH = _BOS_ROOT / "output" / "20hz_s2" / "batch_2000606-2000746"
_DATA_DIR = _BOS_ROOT / "DATA" / "20Hz" / "s2"
_REF_IMAGE = _BOS_ROOT / "DATA" / "_ref_20hz_s2.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_project_yaml(tmp_path: pathlib.Path, data_dir: pathlib.Path) -> pathlib.Path:
    """
    Write a minimal .ofcd project directory with project.yaml pointing at
    the BOS DATA/20Hz/s2 image sequence and the reference image.

    Adjust field names to match the final Pydantic v2 schema when T4 lands.
    """
    ofcd_dir = tmp_path / "test_project.ofcd"
    ofcd_dir.mkdir()

    project_yaml = textwrap.dedent(f"""\
        format_version: 0
        name: golden_regression_test
        created: "2026-04-23T00:00:00+00:00"

        geometry:
          pattern_period_mm: 2.0
          optical_stack:
            preset: pattern_below_window
            layers: []

        data:
          frames_dir: "{data_dir}"
          pattern: "Img*.jpg"

        reference:
          mode: use_existing
          source: "{_REF_IMAGE}"

        batches:
          - range: "2000606-2000746"
    """)

    (ofcd_dir / "project.yaml").write_text(project_yaml)
    return ofcd_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.skipif(not _CLI_READY, reason=_SKIP_REASON)
def test_pipeline_run_produces_results_h5(tmp_path, data_dir, golden_dir):
    """openfcd run writes a results.h5 that contains an 'eta' dataset."""
    from openfcd.cli import app
    import typer.testing
    import h5py

    ofcd_dir = _make_minimal_project_yaml(tmp_path, data_dir)

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["run", str(ofcd_dir)])
    assert result.exit_code == 0, (
        f"openfcd run exited {result.exit_code}:\n{result.output}"
    )

    # Locate results.h5 in the run snapshot
    results_files = list(ofcd_dir.rglob("results.h5"))
    assert results_files, "No results.h5 found after openfcd run"

    with h5py.File(results_files[0], "r") as f:
        assert "eta" in f, f"'eta' dataset missing from results.h5 (keys: {list(f.keys())})"


@pytest.mark.golden
@pytest.mark.skipif(not _CLI_READY, reason=_SKIP_REASON)
def test_pipeline_eta_matches_golden(tmp_path, data_dir, golden_dir):
    """Pipeline results.h5 eta matches the golden eta_mean within atol=1e-6."""
    from openfcd.cli import app
    import typer.testing
    import h5py

    ofcd_dir = _make_minimal_project_yaml(tmp_path, data_dir)

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["run", str(ofcd_dir)])
    assert result.exit_code == 0, (
        f"openfcd run exited {result.exit_code}:\n{result.output}"
    )

    results_files = list(ofcd_dir.rglob("results.h5"))
    assert results_files, "No results.h5 found after openfcd run"

    with h5py.File(results_files[0], "r") as f:
        eta_new = f["eta"][:]

    eta_golden = np.load(_GOLDEN_BATCH / "eta_mean.npy")

    np.testing.assert_allclose(
        eta_new, eta_golden, atol=1e-6, equal_nan=True,
        err_msg="Pipeline eta does not match golden eta_mean",
    )


@pytest.mark.golden
@pytest.mark.skipif(not _CLI_READY, reason=_SKIP_REASON)
def test_pipeline_replay_byte_equal(tmp_path, data_dir, golden_dir):
    """
    openfcd replay produces the same eta array as the original run
    (byte-equal to results.h5 from test_pipeline_eta_matches_golden).
    """
    from openfcd.cli import app
    import typer.testing
    import h5py

    ofcd_dir = _make_minimal_project_yaml(tmp_path, data_dir)

    runner = typer.testing.CliRunner()

    # First run
    r1 = runner.invoke(app, ["run", str(ofcd_dir)])
    assert r1.exit_code == 0, f"run failed:\n{r1.output}"

    results_files = list(ofcd_dir.rglob("results.h5"))
    assert results_files
    with h5py.File(results_files[0], "r") as f:
        eta_run = f["eta"][:]

    # Replay — adjust command name/flags to match T4 CLI when implemented
    r2 = runner.invoke(app, ["replay", str(ofcd_dir)])
    assert r2.exit_code == 0, f"replay failed:\n{r2.output}"

    replay_files = list(ofcd_dir.rglob("replay_results.h5"))
    if not replay_files:
        replay_files = results_files  # some impls overwrite in-place

    with h5py.File(replay_files[0], "r") as f:
        eta_replay = f["eta"][:]

    np.testing.assert_array_equal(
        eta_replay, eta_run,
        err_msg="Replay eta is not byte-equal to original run eta",
    )
