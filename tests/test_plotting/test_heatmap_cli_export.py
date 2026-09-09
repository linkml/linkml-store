"""
Regression test for the heatmap CLI's --export-data path.

The export branch referenced `plt` without importing it in that function, so any
invocation with --export-data raised `NameError: name 'plt' is not defined`. The
existing plotting tests call `create_heatmap` and `export_heatmap_data` directly
and never go through the CLI command, which is why the crash was not caught. See
https://github.com/linkml/linkml-store/issues/72
"""

import csv
from pathlib import Path

import matplotlib
import pytest
from click.testing import CliRunner

matplotlib.use("Agg")  # no display in CI

from linkml_store.plotting.cli import plot_cli  # noqa: E402


@pytest.fixture
def input_csv(tmp_path: Path) -> Path:
    """A small x/y/value table, enough to produce a heatmap with more than one cell."""
    path = tmp_path / "input.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category_x", "category_y", "value"])
        for row in [
            ("A", "X", 1.0),
            ("B", "X", 2.0),
            ("A", "Y", 3.0),
            ("B", "Y", 4.0),
        ]:
            writer.writerow(row)
    return path


def test_heatmap_cli_export_data_does_not_crash(input_csv: Path, tmp_path: Path) -> None:
    """`--export-data` completes and writes the export file."""
    output_image = tmp_path / "heatmap.png"
    export_file = tmp_path / "exported.csv"

    result = CliRunner().invoke(
        plot_cli,
        [
            "heatmap",
            str(input_csv),
            "-x",
            "category_x",
            "-y",
            "category_y",
            "-v",
            "value",
            "-o",
            str(output_image),
            "-e",
            str(export_file),
        ],
    )

    # Surface the real traceback rather than a bare exit code if this regresses.
    assert result.exit_code == 0, result.output + "\n" + repr(result.exception)
    assert output_image.exists(), "heatmap image was not written"
    assert export_file.exists(), "export file was not written"
