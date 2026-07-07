"""Blueprint 4 — Advanced diagnostic MCMC with auto-stopping.

Samples a banana-shaped (Rosenbrock-like) posterior while monitoring
split-chain Gelman-Rubin R̂ and effective sample size per parameter every
``check_every`` steps; the chain terminates itself the moment both
convergence boundaries are met.  Exports the diagnostics-history matrix
alongside the chain, and plots the R̂ / ESS trajectories.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner import RunSpec, WorkflowRunner  # noqa: E402


def banana_log_prob(x: float, y: float) -> float:
    """Curved 'banana' posterior — a stress test for convergence metrics."""
    return -0.5 * (x * x / 4.0 + (y - 0.5 * x * x) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="loose targets")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime", choices=["local", "slurm"],
                        default="local")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = RunSpec(
        name="mcmc_diag_banana",
        strategy="mcmc_diagnostic",
        model=banana_log_prob,
        bounds={"x": (-8.0, 8.0), "y": (-4.0, 20.0)},
        options={
            "walkers": 12,
            "steps": 600 if args.smoke else 6000,
            "check_every": 50 if args.smoke else 200,
            "min_steps": 200 if args.smoke else 800,
            "rhat_target": 1.10 if args.smoke else 1.02,
            "ess_target": 100.0 if args.smoke else 400.0,
            "burn_fraction": 0.3,
        },
        seed=args.seed,
        runtime=args.runtime,
        slurm={"job_name": "mcmc_diag_banana", "time": "04:00:00"},
    )
    telemetry = WorkflowRunner(spec).run()
    if telemetry.get("mode") == "slurm_submission":
        return

    run_dir = Path(telemetry["run_dir"])
    result = telemetry["result"]
    _plot_diagnostics(run_dir, list(spec.bounds))
    print(f"\nauto-stopped = {result['converged']}  "
          f"after {result['steps_run']} steps  "
          f"(targets: R̂ < {result['rhat_target']}, "
          f"ESS ≥ {result['ess_target']:.0f})")
    for name, stats in result["posterior"].items():
        print(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"artifacts → {run_dir}")


def _plot_diagnostics(run_dir: Path, names: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import csv

    with open(run_dir / "diagnostics_history.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return
    header, data = rows[0][1:], [[float(v) for v in r] for r in rows[1:]]
    steps = [r[0] for r in data]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i, name in enumerate(names):
        ax0.plot(steps, [r[1 + i] for r in data], "-o", ms=4,
                 label=f"R̂({name})")
        ax1.plot(steps, [r[1 + len(names) + i] for r in data], "-o", ms=4,
                 label=f"ESS({name})")
    ax0.axhline(1.0, color="k", lw=0.8)
    ax0.set_ylabel("split-chain R̂")
    ax0.legend()
    ax0.grid(alpha=0.35)
    ax1.set_ylabel("effective sample size")
    ax1.set_xlabel("chain step")
    ax1.legend()
    ax1.grid(alpha=0.35)
    ax0.set_title("Integrated convergence monitoring (auto-stop)")
    fig.tight_layout()
    fig.savefig(run_dir / "diagnostics.png", dpi=150)
    plt.close(fig)
    _ = header


if __name__ == "__main__":
    main()
