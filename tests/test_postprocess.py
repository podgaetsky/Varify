"""Tests for src.analysis.postprocess: load_xy, fit_spline, spline_mse.

Plain pytest-style ``test_*`` functions using synthetic sine-curve data in
tmp_path fixtures.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from varify.src.analysis.postprocess import fit_spline, load_xy, spline_mse


def _write_xy(path: Path, xs, ys, delimiter: str = " ", header: bool = True) -> None:
    lines = []
    if header:
        lines.append("# x y  (comment line)")
        lines.append("")
    for x, y in zip(xs, ys):
        lines.append(f"{x}{delimiter}{y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dense_sine(n: int = 200, lo: float = 0.0, hi: float = 10.0):
    xs = np.linspace(lo, hi, n)
    ys = np.sin(xs)
    return xs, ys


def test_spline_mse_identical_curves_near_zero(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case0"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys)

    mse = spline_mse(case_dir, exp_path, "output.dat")
    assert mse == mse  # not NaN
    assert mse < 1e-10


def test_spline_mse_constant_offset(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case1"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys + 0.5)

    mse = spline_mse(case_dir, exp_path, "output.dat")
    assert mse == pytest.approx(0.25, abs=1e-6)


def test_spline_mse_missing_sim_file(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case2"
    case_dir.mkdir()
    # No output.dat written in case_dir.

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys)

    mse = spline_mse(case_dir, exp_path, "output.dat")
    assert math.isnan(mse)


def test_spline_mse_no_domain_overlap(tmp_path: Path) -> None:
    sim_xs = np.linspace(0.0, 10.0, 100)
    sim_ys = np.sin(sim_xs)
    case_dir = tmp_path / "case3"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", sim_xs, sim_ys)

    exp_xs = np.linspace(100.0, 110.0, 50)
    exp_ys = np.sin(exp_xs)
    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, exp_xs, exp_ys)

    mse = spline_mse(case_dir, exp_path, "output.dat")
    assert math.isnan(mse)


def test_load_xy_comma_delimited_with_comments(tmp_path: Path) -> None:
    path = tmp_path / "comma.csv"
    path.write_text(
        "# comment line\n"
        "\n"
        "0.0,0.0\n"
        "1.0,1.0\n"
        "2.0,4.0\n",
        encoding="utf-8",
    )
    x, y = load_xy(path)
    assert list(x) == [0.0, 1.0, 2.0]
    assert list(y) == [0.0, 1.0, 4.0]


def test_load_xy_whitespace_delimited(tmp_path: Path) -> None:
    path = tmp_path / "ws.dat"
    path.write_text(
        "# comment line\n"
        "0.0   0.0\n"
        "\n"
        "1.0   1.0\n"
        "2.0   4.0\n",
        encoding="utf-8",
    )
    x, y = load_xy(path)
    assert list(x) == [0.0, 1.0, 2.0]
    assert list(y) == [0.0, 1.0, 4.0]


def test_load_xy_sorts_and_drops_duplicate_x(tmp_path: Path) -> None:
    path = tmp_path / "unsorted.dat"
    path.write_text(
        "2.0 4.0\n"
        "0.0 0.0\n"
        "1.0 1.0\n"
        "1.0 99.0\n",  # duplicate x=1.0, should be dropped
        encoding="utf-8",
    )
    x, y = load_xy(path)
    assert list(x) == [0.0, 1.0, 2.0]
    assert list(y) == [0.0, 1.0, 4.0]


def test_fit_spline_reduces_degree_for_few_points() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    spline = fit_spline(x, y, k=3, s=0.0)
    # With only 2 points, degree must fall back to 1 (linear) and evaluate.
    assert spline(0.5) == pytest.approx(0.5, abs=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
