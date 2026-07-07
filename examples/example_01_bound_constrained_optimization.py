"""Blueprint 1 — Bound-constrained optimization.

Minimizes the Rosenbrock function inside a box using the built-in
Nelder-Mead strategy (``--backend scipy`` upgrades to scipy when
installed).  Exports evaluation history, a running-best convergence matrix
and (when matplotlib is present) a convergence figure to ``results/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import RunSpec, WorkflowRunner  # noqa: E402


def rosenbrock(x: float, y: float) -> float:
    """Classic banana-valley test objective (global minimum at (1, 1))."""
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny budget")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime", choices=["local", "slurm"],
                        default="local")
    parser.add_argument("--backend", choices=["builtin", "scipy"],
                        default="builtin")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = RunSpec(
        name="opt_rosenbrock",
        strategy="optimize",
        model=rosenbrock,
        bounds={"x": (-2.0, 2.0), "y": (-1.0, 3.0)},
        options={
            "max_evaluations": 60 if args.smoke else 400,
            "backend": args.backend,
            "xatol": 1e-6,
            "fatol": 1e-10,
        },
        seed=args.seed,
        runtime=args.runtime,
        slurm={"job_name": "opt_rosenbrock", "time": "00:30:00"},
    )
    telemetry = WorkflowRunner(spec).run()
    if telemetry.get("mode") == "slurm_submission":
        return

    run_dir = Path(telemetry["run_dir"])
    _export_convergence_matrix(run_dir)
    result = telemetry["result"]
    print(f"\nbest = {result['best_params']}  "
          f"f* = {result['best_value']:.3e}  "
          f"({result['n_evaluations']} evals, converged={result['converged']})")
    print(f"artifacts → {run_dir}")


def _export_convergence_matrix(run_dir: Path) -> None:
    """Running-best matrix (eval #, current value, best-so-far) + figure."""
    import csv

    with open(run_dir / "history.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    best = float("inf")
    matrix = []
    for row in rows:
        value = float(row["value"])
        best = min(best, value)
        matrix.append([int(row["eval"]), value, best])
    with open(run_dir / "convergence_matrix.csv", "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["eval", "value", "running_best"])
        writer.writerows(matrix)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    evals = [m[0] for m in matrix]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(evals, [m[1] for m in matrix], "o", ms=3, alpha=0.4,
                label="evaluation")
    ax.semilogy(evals, [m[2] for m in matrix], "-", lw=2, color="crimson",
                label="running best")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("f(x, y)")
    ax.set_title("Bound-constrained Nelder-Mead on Rosenbrock")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
