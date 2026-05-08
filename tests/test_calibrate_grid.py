from pathlib import Path

from typer.testing import CliRunner


def test_grid_pixel_calibration_from_points() -> None:
    from openfcd.cli.cmd_calibrate import grid_pixel_calibration

    result = grid_pixel_calibration((10.0, 20.0), (130.0, 20.0), cells=10, cell_mm=1.2)

    assert result["pixel_distance"] == 120.0
    assert result["length_mm"] == 12.0
    assert result["pixel_per_mm"] == 10.0


def test_calibrate_grid_cli_points_mode(tmp_path: Path) -> None:
    from openfcd.cli import app

    image = tmp_path / "ref.png"
    image.write_bytes(b"placeholder")

    result = CliRunner().invoke(
        app,
        [
            "calibrate-grid",
            str(image),
            "--cells",
            "10",
            "--cell-mm",
            "1.2",
            "--points",
            "10,20,130,20",
            "--compare-px-per-mm",
            "31.04",
        ],
    )

    assert result.exit_code == 0
    assert "pixel_per_mm: 10.000000" in result.stdout
    assert "eta_scale_if_replace_compare_with_measured: 9.634816" in result.stdout

