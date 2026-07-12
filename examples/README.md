# Blueprint workflows

Eight standalone examples. `example_01`–`example_05` and `example_07` drive
the agnostic runner: each writes telemetry (`telemetry.json`, embedding the
full provenance record), visualization matrices (CSV) and — when matplotlib
is installed — figures, into a fresh timestamped directory under the
central `results/` path. `example_06` instead drives
`src.analysis.plotting.PlotSuite` directly to demonstrate its
post-processing (spline-overlay) figures, and `example_08` drives the full
config-file workflow (`load_config` → `make_optimizer`) end-to-end with a
local fake-sbatch stand-in, no cluster required.

| Example | Strategy | Demonstrates |
|---|---|---|
| `example_01_bound_constrained_optimization.py` | `optimize` | box-bounded Nelder-Mead (scipy upgrade optional), convergence telemetry |
| `example_02_parallel_grid_sweep.py` | `grid` | multiprocessed Cartesian sweep, chunk-level checkpoint/resume, heatmap matrix |
| `example_03_mcmc_baseline.py` | `mcmc` | affine-invariant ensemble sampling (emcee-compatible), posterior export |
| `example_04_mcmc_autostop_diagnostics.py` | `mcmc_diagnostic` | integrated split-chain R̂ / ESS monitoring with auto-stop |
| `example_05_scaling_benchmark.py` | `benchmark` | thread/core scaling, speedup & parallel-efficiency profiling |
| `example_06_postprocess_overlay.py` | n/a (standalone) | `PlotSuite.plot_overlay` / `plot_postprocess`: spline-fit simulated curve vs. experimental data, MSE-annotated overlay + convergence panel |
| `example_07_hybrid_de_nm.py` | `hybrid` | global Differential Evolution seeding a Nelder-Mead local polish, phase-tagged convergence telemetry |
| `example_08_hybrid_slurm_workflow.py` | `de-nm` (framework) | the full stage→modify→submit→wait→compare loop: `case.source_dir` staging + `file_pipeline` + generation-batched fake-sbatch dispatch + `curve_loss` scoring vs. synthetic experiment; doubles as living documentation for `config/config_hybrid_workflow.yaml` |

Common flags: `--smoke` (tiny sizes for CI), `--seed N` (reproducibility),
`--runtime slurm` (self-submits via `sbatch_template.sh` instead of running
locally — the *only* change needed to move a workflow onto the cluster).
