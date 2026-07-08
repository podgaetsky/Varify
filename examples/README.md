# Blueprint workflows

Seven standalone examples. `example_01`–`example_05` and `example_07` drive
the agnostic runner: each writes telemetry (`telemetry.json`, embedding the
full provenance record), visualization matrices (CSV) and — when matplotlib
is installed — figures, into a fresh timestamped directory under the
central `results/` path. `example_06` instead drives
`src.analysis.plotting.PlotSuite` directly to demonstrate its
post-processing (spline-overlay) figures.

| Example | Strategy | Demonstrates |
|---|---|---|
| `example_01_bound_constrained_optimization.py` | `optimize` | box-bounded Nelder-Mead (scipy upgrade optional), convergence telemetry |
| `example_02_parallel_grid_sweep.py` | `grid` | multiprocessed Cartesian sweep, chunk-level checkpoint/resume, heatmap matrix |
| `example_03_mcmc_baseline.py` | `mcmc` | affine-invariant ensemble sampling (emcee-compatible), posterior export |
| `example_04_mcmc_autostop_diagnostics.py` | `mcmc_diagnostic` | integrated split-chain R̂ / ESS monitoring with auto-stop |
| `example_05_scaling_benchmark.py` | `benchmark` | thread/core scaling, speedup & parallel-efficiency profiling |
| `example_06_postprocess_overlay.py` | n/a (standalone) | `PlotSuite.plot_overlay` / `plot_postprocess`: spline-fit simulated curve vs. experimental data, MSE-annotated overlay + convergence panel |
| `example_07_hybrid_de_nm.py` | `hybrid` | global Differential Evolution seeding a Nelder-Mead local polish, phase-tagged convergence telemetry |

Common flags: `--smoke` (tiny sizes for CI), `--seed N` (reproducibility),
`--runtime slurm` (self-submits via `sbatch_template.sh` instead of running
locally — the *only* change needed to move a workflow onto the cluster).
