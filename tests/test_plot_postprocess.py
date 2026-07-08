"""Tests for PlotSuite.plot_overlay / plot_postprocess.

Uses a small duck-typed ``SimpleNamespace`` config (matching the attributes
``PlotSuite`` actually reads: ``cfg.analysis.plots_dir`` and
``cfg.optimizer.{spline_k,spline_s,experimental_data,sim_output_file}``) and
synthetic damped-sine data in tmp_path fixtures, mirroring
``tests/test_postprocess.py``.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.plotting import PlotSuite


def _write_xy(path: Path, x, y) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# x, y\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.6f},{yi:.6f}\n")


def _make_cfg(plots_dir: Path, experimental_path, sim_output_file: str):
    return types.SimpleNamespace(
        analysis=types.SimpleNamespace(plots_dir=plots_dir),
        optimizer=types.SimpleNamespace(
            spline_k=3,
            spline_s=0.0,
            experimental_data=experimental_path,
            sim_output_file=sim_output_file,
        ),
    )


def test_plot_overlay_produces_png(tmp_path: Path) -> None:
    x_exp = np.linspace(0.0, 10.0, 80)
    y_exp = np.sin(x_exp) * np.exp(-0.1 * x_exp)
    experimental_path = tmp_path / "experimental.csv"
    _write_xy(experimental_path, x_exp, y_exp)

    case_dir = tmp_path / "case_best"
    x_sim = np.linspace(0.2, 9.6, 25)
    y_sim = np.sin(x_sim) * np.exp(-0.1 * x_sim)
    _write_xy(case_dir / "output.dat", x_sim, y_sim)

    plots_dir = tmp_path / "plots"
    cfg = _make_cfg(plots_dir, experimental_path, "output.dat")
    plotter = PlotSuite(cfg)

    plotter.plot_overlay(case_dir, experimental_path, "output.dat", name="overlay")

    assert (plots_dir / "overlay.png").exists()


def test_plot_overlay_missing_sim_file_does_not_raise(tmp_path: Path) -> None:
    x_exp = np.linspace(0.0, 10.0, 20)
    y_exp = np.sin(x_exp)
    experimental_path = tmp_path / "experimental.csv"
    _write_xy(experimental_path, x_exp, y_exp)

    case_dir = tmp_path / "case_missing"
    case_dir.mkdir()
    # No output.dat written — sim file is missing.

    plots_dir = tmp_path / "plots"
    cfg = _make_cfg(plots_dir, experimental_path, "output.dat")
    plotter = PlotSuite(cfg)

    plotter.plot_overlay(case_dir, experimental_path, "output.dat", name="overlay")

    assert not (plots_dir / "overlay.png").exists()


def test_plot_postprocess_produces_png(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    cfg = _make_cfg(plots_dir, tmp_path / "unused_experimental.csv", "output.dat")
    plotter = PlotSuite(cfg)

    n = 15
    objective = np.abs(1.0 * np.exp(-0.2 * np.arange(n)) + 0.01)
    history_df = pd.DataFrame({
        "eval": np.arange(1, n + 1),
        "objective": objective,
        "case_dir": [""] * n,
    })

    plotter.plot_postprocess(history_df)

    assert (plots_dir / "postprocess_convergence.png").exists()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
