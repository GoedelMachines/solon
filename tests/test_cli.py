"""CLI layout: analysis commands live under `solon analyze`."""

import numpy as np
import pytest

from solon import cli


def _parse(argv, monkeypatch):
    seen = {}
    monkeypatch.setattr("sys.argv", ["solon", *argv])
    for name in ("cmd_compare_programs", "cmd_similarity", "cmd_decision_boundary"):
        monkeypatch.setattr(cli, name, lambda args, name=name: seen.update(func=name, args=args))
    cli.main()
    return seen


@pytest.mark.parametrize("sub,func", [
    ("compare", "cmd_compare_programs"),
    ("similarity", "cmd_similarity"),
    ("decision-boundary", "cmd_decision_boundary"),
])
def test_analyze_subcommands_dispatch(sub, func, monkeypatch):
    seen = _parse(["analyze", sub, "--dataset", "d.yaml", "--cases", "c.csv", "runs/a"], monkeypatch)
    assert seen["func"] == func and seen["args"].dirs == ["runs/a"]


def test_old_top_level_names_are_gone(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(["compare-programs", "--dataset", "d", "--cases", "c", "x"], monkeypatch)


def test_tsne_works_with_few_programs():
    from solon.analysis.program_similarity import tsne_2d

    decisions = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]] * 1)
    assert tsne_2d(decisions).shape == (4, 2)
