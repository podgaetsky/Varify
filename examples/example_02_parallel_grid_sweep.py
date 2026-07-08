"""Blueprint 2 — Highly concurrent grid run / parameter sweep.

Sweeps a 2-D damped-oscillator response surface over the Cartesian product
of two axes using a ``multiprocessing`` pool.  Chunk-level checkpointing
means a wall-time kill mid-sweep resumes with zero recomputation.  Exports
the full point table, a heatmap visualization matrix and (with matplotlib)
a rendered heatmap.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varify.runner import RunSpec, WorkflowRunner  # noqa: E402


def damped_response(frequency: float, damping: float) -> float:
    """Steady-state amplitude of a driven damped oscillator (ω₀ = 1)."""
    return 1.0 / math.sqrt(
        (1.0 - frequency ** 2) ** 2 + (2.0 * damping * frequency) ** 2
    )


def _linspace(lo: float, hi: float, num: int) -> list:
    step = (hi - lo) / (num - 1)
    return [lo + i * step for i in range(num)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="small grid")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime", choices=["local", "slurm"],
                        default="local")
    parser.add_argument("--workers", type=int,
                        default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n = 12 if args.smoke else 60
    spec = RunSpec(
        name="grid_oscillator",
        strategy="grid",
        model=damped_response,
        options={
            "axes": {
                "frequency": _linspace(0.1, 2.5, n),
                "damping": _linspace(0.05, 1.0, max(n // 2, 4)),
            },
            "workers": args.workers,
        },
        seed=args.seed,
        runtime=args.runtime,
        slurm={"job_name": "grid_oscillator", "cpus_per_task": args.workers},
    )
    telemetry = WorkflowRunner(spec).run()
    if telemetry.get("mode") == "slurm_submission":
        return

    run_dir = Path(telemetry["run_dir"])
    _plot_heatmap(run_dir)
    result = telemetry["result"]
    print(f"\n{result['n_completed']}/{result['n_points']} points "
          f"({result['workers']} workers) → {run_dir}")


def _plot_heatmap(run_dir: Path) -> None:
    matrix_csv = run_dir / "grid_matrix.csv"
    if not matrix_csv.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import csv

    with open(matrix_csv, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    col_labels = [float(v) for v in rows[0][1:]]
    row_labels = [float(r[0]) for r in rows[1:]]
    z = [[float(v) for v in r[1:]] for r in rows[1:]]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        z, aspect="auto", origin="lower", cmap="viridis",
        extent=(col_labels[0], col_labels[-1], row_labels[0], row_labels[-1]),
    )
    fig.colorbar(im, ax=ax, label="amplitude")
    ax.set_xlabel("damping ζ")
    ax.set_ylabel("frequency ratio ω/ω₀")
    ax.set_title("Driven damped-oscillator response surface")
    fig.tight_layout()
    fig.savefig(run_dir / "heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
