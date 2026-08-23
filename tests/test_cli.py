"""Command line behaviour: where output lands."""

import matplotlib

matplotlib.use("Agg")

import pytest

from fractaldim.cli import DEFAULT_OUTPUT_DIR, main


@pytest.fixture(autouse=True)
def in_tmp_dir(tmp_path, monkeypatch):
    """Run each case in a scratch directory, so nothing lands in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_draw_defaults_to_the_output_directory(in_tmp_dir):
    assert main(["draw", "koch", "-n", "2", "--dpi", "40", "--size", "3"]) == 0
    assert (in_tmp_dir / DEFAULT_OUTPUT_DIR / "koch-2.png").exists()


def test_animate_defaults_to_the_output_directory(in_tmp_dir):
    assert main(["animate", "koch", "-n", "1", "--frames", "3", "--hold", "0",
                 "--dpi", "40", "--size", "3", "--quiet"]) == 0
    assert (in_tmp_dir / DEFAULT_OUTPUT_DIR / "koch-1.gif").exists()


def test_explicit_output_wins_and_its_directory_is_created(in_tmp_dir):
    target = in_tmp_dir / "nested" / "elsewhere" / "curve.png"
    assert main(["draw", "koch", "-n", "2", "-o", str(target),
                 "--dpi", "40", "--size", "3"]) == 0
    assert target.exists()
    assert not (in_tmp_dir / DEFAULT_OUTPUT_DIR).exists()


def test_output_dir_is_overridable(in_tmp_dir):
    assert main(["draw", "koch", "-n", "2", "--output-dir", "renders",
                 "--dpi", "40", "--size", "3"]) == 0
    assert (in_tmp_dir / "renders" / "koch-2.png").exists()


def test_unknown_name_is_reported_without_a_traceback(in_tmp_dir):
    assert main(["draw", "nonesuch"]) == 1


def test_animate_as_a_flag_on_draw_points_at_the_subcommand(in_tmp_dir, capsys):
    assert main(["draw", "koch", "--animate"]) == 1
    assert "is a subcommand" in capsys.readouterr().err
