"""Blueprint 7 — Hybrid Differential Evolution + Nelder-Mead optimization.

Minimizes the Rosenbrock function inside a box using the built-in
``hybrid`` strategy: a pure-Python rand/1/bin Differential Evolution
explores the box globally, then its champion seeds a Nelder-Mead simplex
for a local polish with the remaining evaluation budget. Exports the
globally-numbered, phase-tagged evaluation history and a running-best
convergence matrix (plus, when matplotlib is present, a figure coloring
the DE and NM phases) to ``results/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varify.runner import RunSpec, WorkflowRunner  # noqa: E402


def rosenbrock(x: float, y: float) -> float:
    """Classic banana-valley test objective (global minimum at (1, 1))."""
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny budget")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime", choices=["local", "slurm"],
                        default="local")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = RunSpec(
        name="hybrid_rosenbrock",
        strategy="hybrid",
        model=rosenbrock,
        bounds={"x": (-2.0, 2.0), "y": (-1.0, 3.0)},
        options={
            "max_evaluations": 80 if args.smoke else 400,
            "de_popsize": 12,
            "de_generations": 15,
            "de_f": 0.7,
            "de_cr": 0.9,
            "de_stall": 5,
            "xatol": 1e-6,
            "fatol": 1e-10,
        },
        seed=args.seed,
        runtime=args.runtime,
        slurm={"job_name": "hybrid_rosenbrock", "time": "00:30:00"},
    )
    telemetry = WorkflowRunner(spec).run()
    if telemetry.get("mode") == "slurm_submission":
        return

    run_dir = Path(telemetry["run_dir"])
    _export_convergence_matrix(run_dir)
    result = telemetry["result"]
    print(f"\nbest = {result['best_params']}  "
          f"f* = {result['best_value']:.3e}  "
          f"({result['n_evaluations']} evals: "
          f"{result['de_evaluations']} de / {result['nm_evaluations']} nm, "
          f"converged={result['converged']})")
    print(f"artifacts → {run_dir}")


def _export_convergence_matrix(run_dir: Path) -> None:
    """Running-best matrix (eval #, current value, best-so-far, phase) + figure."""
    import csv

    with open(run_dir / "history.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    best = float("inf")
    matrix = []
    for row in rows:
        value = float(row["value"])
        best = min(best, value)
        matrix.append([int(row["eval"]), value, best, row["phase"]])
    with open(run_dir / "convergence_matrix.csv", "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["eval", "value", "running_best", "phase"])
        writer.writerows(matrix)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    de_pts = [m for m in matrix if m[3] == "de"]
    nm_pts = [m for m in matrix if m[3] == "nm"]
    fig, ax = plt.subplots(figsize=(8, 5))
    if de_pts:
        ax.semilogy([m[0] for m in de_pts], [m[1] for m in de_pts], "o",
                    ms=3, alpha=0.4, color="steelblue", label="DE")
    if nm_pts:
        ax.semilogy([m[0] for m in nm_pts], [m[1] for m in nm_pts], "o",
                    ms=3, alpha=0.4, color="darkorange", label="Nelder-Mead")
    ax.semilogy([m[0] for m in matrix], [m[2] for m in matrix], "-", lw=2,
                color="crimson", label="running best")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("f(x, y)")
    ax.set_title("Hybrid DE + Nelder-Mead on Rosenbrock")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
