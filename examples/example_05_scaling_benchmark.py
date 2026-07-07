"""Blueprint 5 — Hardware scale benchmark run.

Profiles a CPU-bound numerical kernel (composite-trapezoid quadrature)
across a ladder of worker counts, measuring wall time, speedup vs. a
single worker and parallel efficiency.  Exports the scaling matrix and a
speedup/efficiency figure — the data needed to choose sane
``--cpus-per-task`` values before burning cluster allocation.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import RunSpec, WorkflowRunner  # noqa: E402


def quadrature_kernel(samples: int = 200_000) -> float:
    """CPU-bound kernel: trapezoid integral of sin(x)·exp(-x/π) on [0, π]."""
    h = math.pi / samples
    total = 0.5 * (math.sin(0.0) + math.sin(math.pi) * math.exp(-1.0))
    for i in range(1, samples):
        x = i * h
        total += math.sin(x) * math.exp(-x / math.pi)
    return total * h


def _worker_ladder(max_workers: int) -> list:
    ladder, n = [], 1
    while n <= max_workers:
        ladder.append(n)
        n *= 2
    if ladder[-1] != max_workers:
        ladder.append(max_workers)
    return ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny workload")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime", choices=["local", "slurm"],
                        default="local")
    parser.add_argument("--max-workers", type=int,
                        default=os.cpu_count() or 1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ladder = _worker_ladder(args.max_workers)
    spec = RunSpec(
        name="scaling_benchmark",
        strategy="benchmark",
        model=quadrature_kernel,
        options={
            "worker_counts": ladder,
            "task_count": 8 if args.smoke else 4 * max(ladder),
            "repeats": 2 if args.smoke else 3,
            "task_arg": {"samples": 20_000 if args.smoke else 200_000},
        },
        seed=args.seed,
        runtime=args.runtime,
        slurm={"job_name": "scaling_benchmark",
               "cpus_per_task": max(ladder)},
    )
    telemetry = WorkflowRunner(spec).run()
    if telemetry.get("mode") == "slurm_submission":
        return

    run_dir = Path(telemetry["run_dir"])
    result = telemetry["result"]
    _plot_scaling(run_dir)
    print(f"\n{'workers':>8s} {'mean_s':>9s} {'speedup':>8s} {'eff':>6s}")
    for n, row in result["scaling"].items():
        print(f"{n:>8s} {row['mean_s']:>9.3f} "
              f"{row['speedup']:>8.2f} {row['efficiency']:>6.2f}")
    print(f"artifacts → {run_dir}")


def _plot_scaling(run_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import csv

    with open(run_dir / "scaling_matrix.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    workers = [int(r[0]) for r in rows[1:]]
    speedup = [float(r[3]) for r in rows[1:]]
    eff = [float(r[4]) for r in rows[1:]]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0.plot(workers, speedup, "o-", lw=2, label="measured")
    ax0.plot(workers, workers, "--", color="grey", label="ideal")
    ax0.set_xlabel("workers")
    ax0.set_ylabel("speedup")
    ax0.legend()
    ax0.grid(alpha=0.4)
    ax1.plot(workers, eff, "s-", lw=2, color="darkorange")
    ax1.axhline(1.0, ls="--", color="grey")
    ax1.set_xlabel("workers")
    ax1.set_ylabel("parallel efficiency")
    ax1.set_ylim(0, 1.15)
    ax1.grid(alpha=0.4)
    fig.suptitle("Hardware scaling benchmark")
    fig.tight_layout()
    fig.savefig(run_dir / "scaling.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
