"""Blueprint 3 — Baseline MCMC sampling.

Samples a correlated 2-D Gaussian posterior with the standard
affine-invariant ensemble sampler (Goodman & Weare stretch move — the same
algorithm as emcee; pass ``--backend emcee`` to delegate to emcee when it
is installed).  Exports the raw chain, posterior summary, a marginal
histogram matrix and (with matplotlib) trace + scatter figures.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varify.runner import RunSpec, WorkflowRunner  # noqa: E402

_RHO = 0.8  # correlation between the two parameters


def log_prob(mu: float, sigma: float) -> float:
    """Log-density of a correlated bivariate Gaussian (means 1.0 / 2.0)."""
    a, b = mu - 1.0, sigma - 2.0
    return -0.5 * (a * a - 2.0 * _RHO * a * b + b * b) / (1.0 - _RHO * _RHO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="short chain")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runtime", choices=["local", "slurm"],
                        default="local")
    parser.add_argument("--backend", choices=["builtin", "emcee", "auto"],
                        default="builtin")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    steps = 200 if args.smoke else 2000
    spec = RunSpec(
        name="mcmc_gaussian",
        strategy="mcmc",
        model=log_prob,
        bounds={"mu": (-4.0, 6.0), "sigma": (-3.0, 7.0)},
        options={
            "walkers": 8,
            "steps": steps,
            "burn_in": steps // 5,
            "stretch_a": 2.0,
            "backend": args.backend,
        },
        seed=args.seed,
        runtime=args.runtime,
        slurm={"job_name": "mcmc_gaussian", "time": "02:00:00"},
    )
    telemetry = WorkflowRunner(spec).run()
    if telemetry.get("mode") == "slurm_submission":
        return

    run_dir = Path(telemetry["run_dir"])
    result = telemetry["result"]
    _plot_chain(run_dir, list(result["posterior"]), result["burn_in"])
    print(f"\nacceptance = {result['acceptance_rate']:.3f}  "
          f"backend = {result['backend']}")
    for name, stats in result["posterior"].items():
        print(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}  "
              f"(n = {stats['n']})")
    print(f"artifacts → {run_dir}")


def _plot_chain(run_dir: Path, names: list, burn_in: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import csv
    from collections import defaultdict

    walkers: dict = defaultdict(lambda: defaultdict(list))
    with open(run_dir / "chain.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for name in names:
                walkers[name][int(row["walker"])].append(float(row[name]))

    fig, axes = plt.subplots(len(names) + 1, 1,
                             figsize=(10, 3 * (len(names) + 1)))
    for ax, name in zip(axes, names):
        for chain in walkers[name].values():
            ax.plot(chain, lw=0.5, alpha=0.6)
        ax.axvline(burn_in, ls="--", color="k", lw=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    xs = [v for c in walkers[names[0]].values() for v in c[burn_in:]]
    ys = [v for c in walkers[names[1]].values() for v in c[burn_in:]]
    axes[-1].scatter(xs, ys, s=2, alpha=0.15, color="steelblue")
    axes[-1].set_xlabel(names[0])
    axes[-1].set_ylabel(names[1])
    axes[-1].grid(alpha=0.3)
    axes[0].set_title("Affine-invariant ensemble MCMC — traces & posterior")
    fig.tight_layout()
    fig.savefig(run_dir / "chain.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
