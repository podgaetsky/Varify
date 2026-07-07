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

Every evaluation is appended to `optimization_history.csv`.

---

## 3. Deploy the background watchdog agent

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

## 4. Post-processing: parser and plotting suite

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

Custom analyses: add a function to `config/hooks.py` and list it under
`analysis.analysis_fns`. Functions declaring `df`/`cfg` receive the full
DataFrame once; all others are called per valid result row with scalar
kwargs (e.g. `def my_fn(tau, gamma, output): ...`).

---

## 5. Typical end-to-end session

```bash
python main.py --mode scan                    # 1. submit the grid
python main.py --mode watch --deploy-agent    # 2. babysit it
# ... wait for jobs to finish ...
python main.py --mode analyze                 # 3. harvest + plots
python main.py --mode optimize --method mcmc  # 4. posterior sampling
python main.py --mode analyze                 # 5. corner + convergence plots
```
