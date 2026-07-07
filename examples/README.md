# Blueprint workflows

Five standalone examples of the agnostic runner. Each writes telemetry
(`telemetry.json`, embedding the full provenance record), visualization
matrices (CSV) and — when matplotlib is installed — figures, into a fresh
timestamped directory under the central `results/` path.

| Example | Strategy | Demonstrates |
|---|---|---|
| `example_01_bound_constrained_optimization.py` | `optimize` | box-bounded Nelder-Mead (scipy upgrade optional), convergence telemetry |
| `example_02_parallel_grid_sweep.py` | `grid` | multiprocessed Cartesian sweep, chunk-level checkpoint/resume, heatmap matrix |
| `example_03_mcmc_baseline.py` | `mcmc` | affine-invariant ensemble sampling (emcee-compatible), posterior export |
| `example_04_mcmc_autostop_diagnostics.py` | `mcmc_diagnostic` | integrated split-chain R̂ / ESS monitoring with auto-stop |
| `example_05_scaling_benchmark.py` | `benchmark` | thread/core scaling, speedup & parallel-efficiency profiling |

Common flags: `--smoke` (tiny sizes for CI), `--seed N` (reproducibility),
`--runtime slurm` (self-submits via `sbatch_template.sh` instead of running
locally — the *only* change needed to move a workflow onto the cluster).
