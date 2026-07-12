# Varify — Practical Manual

All commands are run from the repository root. Every mode accepts
`--config <path>` (default `config/config.yaml`), `--dry-run` and
`--verbose`.

---

## 1. Configure and run a grid search (legacy logic)

### 1.1 Configure

Edit `config/config.yaml`:

```yaml
scan:
  mode: grid                # full Cartesian product ("coupled" = lock-step zip)
  workspace: sweep_workspace
  template_files: [templates/input.template, templates/run_script.sh]
  output_file: stdout.log
  output_regex: 'RESULT:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'

parameters:
  - name: tau
    default: 1.0
    sweep: {start: 0.5, stop: 5.0, num: 10}     # or values: [0.5, 1.0, 2.0]
  - name: gamma
    default: 0.5
    sweep: {start: 0.1, stop: 2.0, num: 5}
  - name: kappa                                  # static, rides along with tau
    default: 1.0
    coupled_to: tau
    coupled_fn: kappa_from_tau                   # defined in config/hooks.py

slurm:
  directives: {partition: compute, time: "04:00:00", ntasks: 1}
```

A **line search** is simply a grid with one swept parameter (give only one
param a `sweep:` block). Your simulation must print `RESULT: <float>` (or
adapt `output_regex`); `@TAU@`-style tokens in the templates are substituted
per case.

### 1.2 Preview, then submit

```bash
# Prepare all case directories but do NOT call sbatch:
python main.py --mode scan --dry-run

# Fire one sbatch job per grid point:
python main.py --mode scan
```

Case folders appear under `sweep_workspace/case_<param>_<value>_.../`;
submissions are audited in `submission.log` and registered in
`sweep_workspace/jobs_registry.csv`.

---

## 2. Launch an MCMC optimization run

### 2.1 Configure priors and chain settings

In `config/config.yaml`, give every sampled parameter a prior (params
without one stay fixed at `default`), and make your simulation print
`LOG_PROB: <float>`:

```yaml
parameters:
  - name: tau
    default: 1.0
    mcmc: {prior_low: 0.0, prior_high: 10.0, init_center: 1.0, init_width: 0.5}

mcmc:
  num_walkers: 10        # ≥ 2 × number of sampled params, even
  num_iters: 500         # steps per walker after burn-in
  burnin: 100
  stretch_a: 2.0
  poll_interval: 30.0
  job_timeout: 3600.0
  chain_csv: mcmc_chain.csv
```

### 2.2 Run (blocking orchestration)

```bash
# Preview walker proposals without submitting jobs:
python main.py --mode optimize --method mcmc --dry-run

# Full run: one sbatch job per walker per step, chain checkpointed
# to mcmc_chain.csv after every step (safe to interrupt & re-run — the
# chain resumes from the last committed step):
python main.py --mode optimize --method mcmc
```

Walker case dirs are named `sweep_workspace/mcmc_step<NNNNNN>_w<WWWW>/`.

### 2.3 Gradient-free alternative (Nelder-Mead)

Uses the same `mcmc.prior_low/high` entries as box bounds and minimizes
(or maximizes) the metric parsed by `optimizer.objective_regex`
(default: the scan `output_regex`):

```bash
python main.py --mode optimize --method nelder-mead
```

Every evaluation is appended to `optimization_history.csv`. `--mode
optimize --method` now accepts three values: `mcmc` (ensemble sampling,
§2.2), `nelder-mead` (this section), and the hybrid `de-nm` method
described next in §2.4.

### 2.4 Hybrid Differential Evolution → Nelder-Mead optimizer (`de-nm`)

Global search followed by local polish, in one command. Each **generation**
of Differential Evolution candidates is built and submitted to SLURM
*concurrently*, then the whole batch is awaited in one call to
`SlurmDispatcher.wait_for_batch` (far fewer round-trips than one job at a
time). Once DE finishes — either by exhausting `de_generations` or by
stalling for `de_stall_generations` generations without improvement — scipy
Nelder-Mead refines the DE champion with whatever evaluation budget remains
under `max_evaluations`.

#### 2.4.1 Configure

Add the `de_*` keys under the existing `optimizer:` block (same search box
as MCMC/Nelder-Mead — `mcmc.prior_low`/`prior_high` per parameter):

```yaml
optimizer:
  method: de-nm
  maximize: false
  max_evaluations: 300        # hard cap across DE + NM phases combined
  tolerance: 1.0e-4
  poll_interval: 15.0         # seconds between wait_for_batch polls
  job_timeout: 3600.0
  history_csv: optimization_history.csv

  de_popsize: 15               # candidates per DE generation
  de_generations: 20           # max DE generations before NM refinement
  de_f: 0.7                    # DE differential weight (mutation scale)
  de_cr: 0.9                   # DE crossover probability
  de_stall_generations: 5      # early-stop DE after this many stalled gens
  de_seed: 42                  # seed for the DE random number generator
```

#### 2.4.2 Run

```bash
python main.py --mode optimize --method de-nm
```

Case directories carry descriptive, greppable names that embed the
candidate's parameter values, e.g.
`sweep_workspace/de_g003_c007__amplitude_2.59_decay_0.691/` (generation 3,
candidate 7; the plain Nelder-Mead/MCMC methods do the same, e.g.
`opt_eval00012__tau_1.5_gamma_0.25`). Every evaluation — DE and Nelder-Mead
alike — is appended to `optimization_history.csv`, phase-tagged so you can
tell the two apart.

See `config/config_hybrid_workflow.yaml` for a complete, heavily annotated
config that combines `de-nm` with case staging (§4 below), the file
pipeline (§5) and postprocess scoring (§6) in one end-to-end example, and
`examples/example_07_hybrid_de_nm.py` for the same algorithm driven through
the zero-dependency `runner/` layer (`strategy="hybrid"`) instead of the
config-file framework. §9 below walks through the full loop.

---

## 3. Synchronous sweeps: block until jobs finish (`--wait`)

By default `--mode scan` fires every sbatch job and returns immediately.
Pass `--wait` to block until every submitted job in the sweep reaches a
terminal state and get a completed/failed summary before the command
exits — handy for scripting a sweep directly into a following `analyze`
step without a separate watchdog process:

```bash
python main.py --mode scan --wait
```

Under the hood this uses `SlurmDispatcher.wait_for_batch`, which checks job
state via `squeue` and falls back to `sacct` (configurable via
`slurm.sacct_cmd`) once a job leaves the queue; if neither is available
(e.g. testing on a laptop with no scheduler) jobs degrade gracefully to
`COMPLETED` and the framework falls back to its usual file-poll for
completion detection.

---

## 4. Stage case files from a source directory

For simulations that need more than a couple of templated files — a whole
driver directory of scripts, config fragments and binaries — add a
top-level `case:` block. It runs **first**, before `templates/`,
`input_fn` hooks and `file_pipeline:`, for every case directory in a scan
or optimization run:

```yaml
case:
  source_dir: case_template_dir      # copied recursively into each case dir
  substitute_globs: ["*.cfg", "*.inp"]   # fnmatch patterns; "*" = substitute everything
```

`case.source_dir`'s contents are copied recursively into the case
directory; then every copied file whose name matches one of
`substitute_globs` gets `@TOKEN@` substitution applied in place (tokens are
the upper-cased parameter names, plus `@JOB_NAME@`). Files that don't match
a glob — including binaries — are copied through unchanged and are never
opened as text, so it's safe to ship compiled executables or data files
alongside your templated config in the same `source_dir`.

---

## 5. Generate or modify per-case files (`file_pipeline`)

For files that aren't `@TOKEN@` templates but need per-case values written
into a structured format (JSON/YAML/TOML/`key=value`), add a top-level
`file_pipeline:` list. It runs **last** in the per-case pipeline, after
`case:` staging, `templates/` rendering and `input_fn` hooks:

```yaml
file_pipeline:
  - action: generate            # write a brand-new file
    file: params.json           # relative to the case directory
    fmt: json                   # optional; auto-detected from the extension
    keys:
      tau: "$tau"                # "$name" is resolved from this case's own param value
      note: "generated by file_pipeline"
  - action: modify               # edit an existing file in place, comment-preservingly
    file: params.json           # runs after the "generate" entry above
    keys:
      gamma: "$gamma"            # new top-level key, appended
      tau: "$tau"                # existing key, value replaced in place
```

`generate` writes a new file; `modify` edits an existing one without
disturbing surrounding comments/formatting. The same engine is also
invocable standalone, outside the full framework, for one-off case
directories:

```bash
python -m utils.file_pipeline <case_dir> <params.json> <spec.json>
```

---

## 6. Score against an experiment (`postprocess`)

Instead of regex-scraping a single scalar from the job log, the optimizer
can align each case's simulated curve to a reference experimental dataset
and score it with a configurable loss — useful whenever "fit the
simulation to measured data" is the actual objective:

```yaml
optimizer:
  postprocess: true                  # true -> score via curve_loss instead of output_regex
  experimental_data: experiment.csv  # two-column (x, y) file; optional 3rd column = sigma
  sim_output_file: output.dat        # simulated two-column (x, y) curve, relative to case_dir
  interp: spline                     # spline | linear
  spline_k: 3                        # spline degree (auto-reduced if too few points)
  spline_s: 0.0                      # 0 = exact interpolation; >0 = smoothing spline
  loss: mse                          # mse | rmse | mae | huber | chi2
  huber_delta: 1.0                   # only used when loss: huber
  experimental_err_col: null         # column index of sigma for loss: chi2
  loss_fn: null                      # name of a hook in config/hooks.py: (y_pred, y_ref) -> float;
                                      # overrides 'loss' entirely when set
```

With `postprocess: true`, each case is scored by reading `sim_output_file`
and `experimental_data`, restricting to their overlapping x-domain,
interpolating the simulated curve onto the experimental x-grid (spline or
linear) and computing the chosen loss (lower is better; `maximize: false`
is almost always what you want here). This works with `nelder-mead` and
`de-nm` (both driven by the `optimizer:` block); see
`config/config_hybrid_workflow.yaml` for it wired up end-to-end with
`de-nm`.

> **Warning — `output_file` vs. `sim_output_file`:** `scan.output_file` is
> the file whose *existence* the framework polls to decide a case has
> finished (`ResultParser.wait_for_output`); it must be a file your
> **simulation itself** writes, e.g. `output.dat`. Never point it at a
> shell/scheduler-redirected log such as `stdout.log` — the scheduler
> creates that file the instant the job starts, long before any real
> output exists, so the wait would return immediately and every case would
> be scored on an empty or partial file. Keep `scan.output_file` and
> `optimizer.sim_output_file` pointing at the same real result file (as in
> `config/config_hybrid_workflow.yaml`, both set to `output.dat`).

---

## 7. Deploy the background watchdog agent

The watchdog polls `jobs_registry.csv` every `watchdog.poll_interval`
seconds and, per active job, detects **missing output files**, **NaN values
in the `.out`/`.err` logs** (`watchdog.nan_regex`) and **stalled file
modification times** (`watchdog.stall_timeout`). On failure it logs to the
central `sweep_workspace/status.csv`, runs `scancel <job_id>`, and — up to
`watchdog.max_resubmits` times — resubmits the case with the parameters in
`watchdog.resubmit_scaling` multiplied (e.g. a halved timestep).

```bash
# Run interactively in a login-node shell (Ctrl-C to stop):
python main.py --mode watch

# Single polling pass (useful in cron or for testing):
python main.py --mode watch --once

# Deploy as an independent sbatch background agent
# (directives from watchdog.agent_directives):
python main.py --mode watch --deploy-agent

# Stop the agent later:
scancel --name=varify_watchdog
```

---

## 8. Post-processing: parser and plotting suite

```bash
# Harvest all run folders → sweep_results.csv + results/results.db,
# generate all plots under results/plots/, run the analysis hooks:
python main.py --mode analyze

# Variants:
python main.py --mode analyze --no-plots          # data extraction only
python main.py --mode analyze --no-sqlite         # skip the SQLite export
python main.py --mode analyze --no-analysis-fns   # skip config/hooks.py fns
```

Generated automatically (depending on which artifacts exist):

| Artifact | Content |
|---|---|
| `sweep_results.csv` / `results.db:results` | one row per case: `param_*` columns + `output` |
| `results/results.db:mcmc_chain` / `:optimization_history` | MCMC chain / optimizer history tables |
| `results/plots/scan_1d.png` | 1-D sensitivity curve + finite-difference derivative |
| `results/plots/scan_2d_heatmap.png` | 2-D grid heatmap with contour overlay |
| `results/plots/scan_marginals.png` | marginal sensitivity panels (N-D / coupled) |
| `results/plots/mcmc_traces.png`, `mcmc_chi2.png`, `mcmc_corner.png`, `mcmc_convergence.png` | trace plots, χ² history, posterior corner plot, R̂/τ/ESS diagnostics |
| `results/plots/optimization_convergence.png` | objective vs. evaluation with running best |
| overlay plot(s) under `results/plots/` | (when `optimizer.postprocess: true`) experimental scatter + raw simulated curve + fitted spline + loss annotation, via `PlotSuite.plot_overlay(case_dir, experimental_path, sim_filename)` |
| postprocess convergence plot under `results/plots/` | (when `optimizer.postprocess: true`) log-scale loss convergence + running best across all evaluations, via `PlotSuite.plot_postprocess(history_df, best_case_dir)` |

The overlay and postprocess plots are generated automatically by
`--mode analyze` whenever `optimizer.postprocess` is `true` in the active
config — no extra flag needed. See `examples/example_06_postprocess_overlay.py`
for a standalone demo that drives `PlotSuite.plot_overlay` /
`plot_postprocess` directly, without running a full sweep first.

Custom analyses: add a function to `config/hooks.py` and list it under
`analysis.analysis_fns`. Functions declaring `df`/`cfg` receive the full
DataFrame once; all others are called per valid result row with scalar
kwargs (e.g. `def my_fn(tau, gamma, output): ...`).

---

## 9. Typical end-to-end sessions

Grid search + MCMC:

```bash
python main.py --mode scan                    # 1. submit the grid
python main.py --mode watch --deploy-agent    # 2. babysit it
# ... wait for jobs to finish ...
python main.py --mode analyze                 # 3. harvest + plots
python main.py --mode optimize --method mcmc  # 4. posterior sampling
python main.py --mode analyze                 # 5. corner + convergence plots
```

Full hybrid-optimizer workflow (stage → modify → submit&wait → compare →
advance), fitting simulated output to an experimental curve:

```bash
python main.py --mode optimize --method de-nm --config config/config_hybrid_workflow.yaml
python main.py --mode analyze --config config/config_hybrid_workflow.yaml
```

`config/config_hybrid_workflow.yaml` is a complete, heavily commented
reference for this loop — read it alongside this manual for the five-step
breakdown (case staging → `file_pipeline` → batch submit-and-wait →
postprocess scoring → DE/NM advancement). To see the whole thing run
locally with no cluster and no manual setup:

```bash
python examples/example_08_hybrid_slurm_workflow.py   # full framework loop, fake-sbatch, ~15s
python examples/example_07_hybrid_de_nm.py             # same algorithm via the zero-dep runner/ layer
python examples/example_06_postprocess_overlay.py      # just the overlay/postprocess plots
```

### Running the test suite

```bash
python -m pytest tests/ -q
```

86 tests cover the scanner, all three optimizer methods (including the
hybrid DE→NM path and its batch dispatch), `SlurmDispatcher` completion
waiting, case-directory source staging, the file pipeline, the loss/
interpolation registry, descriptive case naming and the postprocess plots.
