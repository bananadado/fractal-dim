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


def test_rolloff_writes_a_figure_named_for_the_deepest_level(in_tmp_dir):
    assert main(["rolloff", "koch", "--levels", "4,5", "--below-step", "8",
                 "--dpi", "40", "--size", "4"]) == 0
    assert (in_tmp_dir / DEFAULT_OUTPUT_DIR / "koch-5-rolloff.png").exists()


def test_rolloff_refuses_a_system_with_no_exact_dimension(in_tmp_dir, capsys):
    """The roll-off is a comparison against ground truth, so a branching
    system with no Moran dimension has nothing to be measured against."""
    assert main(["rolloff", "plant"]) == 1
    assert "no exact dimension" in capsys.readouterr().err


def test_rolloff_rejects_a_malformed_level_list(in_tmp_dir, capsys):
    assert main(["rolloff", "koch", "--levels", "4,five"]) == 1
    assert "comma-separated" in capsys.readouterr().err


def test_settle_reports_a_verdict_from_two_shallower_levels(in_tmp_dir, capsys):
    """The settling test is the one diagnostic worth leading with, so the
    command has to be able to produce it."""
    assert main(["boxcount", "koch", "-n", "5", "--settle",
                 "--dpi", "40", "--size", "3"]) == 0
    out = capsys.readouterr().out
    assert "settling" in out and "levels 3/4/5" in out


def test_settle_declines_when_there_are_no_shallower_levels(in_tmp_dir, capsys):
    assert main(["boxcount", "koch", "-n", "2", "--settle",
                 "--dpi", "40", "--size", "3"]) == 0
    assert "-n is too small" in capsys.readouterr().out
