"""Blueprint 6 — Post-processing overlay (spline vs. experiment).

Demonstrates :class:`~src.analysis.plotting.PlotSuite`'s post-processing
figures without needing a live optimizer run or SLURM cluster: a damped-sine
"experimental" curve is synthesized alongside a coarser, slightly offset
"simulated" curve, then ``plot_overlay`` fits a spline through the simulated
points and scores it against experiment (mean-squared error, annotated on
the figure), and ``plot_postprocess`` renders the MSE-per-evaluation /
running-best convergence panel from a small fabricated history.

Exports ``overlay.png``, ``postprocess_convergence.png`` and
``overlay_best.png`` to a fresh timestamped directory under ``results/``.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.analysis.plotting import PlotSuite  # noqa: E402


def _damped_sine(x: np.ndarray, amp: float = 1.0, decay: float = 0.15,
                  freq: float = 1.3, phase: float = 0.0) -> np.ndarray:
    return amp * np.exp(-decay * x) * np.sin(freq * x + phase)


def _write_xy(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# x, y\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.6f},{yi:.6f}\n")


def _make_cfg(plots_dir: Path, experimental_path: Path,
              sim_output_file: str) -> types.SimpleNamespace:
    """Minimal duck-typed stand-in for ``FrameworkConfig``.

    Only the attributes ``PlotSuite`` actually reads are populated:
    ``cfg.analysis.plots_dir`` and ``cfg.optimizer.{spline_k,spline_s,
    experimental_data,sim_output_file}``.
    """
    return types.SimpleNamespace(
        analysis=types.SimpleNamespace(plots_dir=plots_dir),
        optimizer=types.SimpleNamespace(
            spline_k=3,
            spline_s=0.0,
            experimental_data=experimental_path,
            sim_output_file=sim_output_file,
        ),
    )


def main() -> None:
    rng = np.random.default_rng(1234)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(__file__).resolve().parents[1] / "results" / f"example_06_{stamp}"
    plots_dir = run_dir / "plots"
    case_dir = run_dir / "case_best"

    # ── Synthesize "experimental" data: dense, noisy damped sine. ──────────────
    x_exp = np.linspace(0.0, 10.0, 80)
    y_exp = _damped_sine(x_exp) + rng.normal(scale=0.03, size=x_exp.size)
    experimental_path = run_dir / "experimental.csv"
    _write_xy(experimental_path, x_exp, y_exp)

    # ── Synthesize "simulated" data: coarser, slightly offset parameters. ──────
    sim_output_file = "output.dat"
    x_sim = np.linspace(0.2, 9.6, 25)
    y_sim = _damped_sine(x_sim, decay=0.17, freq=1.25, phase=0.05)
    _write_xy(case_dir / sim_output_file, x_sim, y_sim)

    cfg = _make_cfg(plots_dir, experimental_path, sim_output_file)
    plotter = PlotSuite(cfg)

    # ── Overlay: raw + spline-fit simulated curve vs. experimental scatter. ────
    plotter.plot_overlay(case_dir, experimental_path, sim_output_file, name="overlay")

    # ── Fabricated optimizer history with decreasing MSE. ───────────────────────
    n_evals = 20
    objective = 1.0 * np.exp(-0.18 * np.arange(n_evals)) + rng.normal(
        scale=0.01, size=n_evals
    ).clip(min=-0.005)
    objective = np.abs(objective)
    history_df = pd.DataFrame({
        "eval": np.arange(1, n_evals + 1),
        "objective": objective,
        "case_dir": [str(case_dir)] * n_evals,
    })

    plotter.plot_postprocess(history_df, best_case_dir=case_dir)

    print(f"experimental data → {experimental_path}")
    print(f"simulated data    → {case_dir / sim_output_file}")
    print(f"figures           → {plots_dir}")
    for png in sorted(plots_dir.glob("*.png")):
        print(f"  - {png.name}")


if __name__ == "__main__":
    main()
