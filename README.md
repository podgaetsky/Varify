# Varify

A modular Python framework for **SLURM-based parametric scans, gradient-free
optimization, Bayesian MCMC sampling, automated fault-tolerance and
streamlined data analysis** of HPC simulations.

The legacy line/grid search logic (grid enumeration, coupled parameters,
template substitution, output scraping, ensemble stretch-move MCMC) is
preserved verbatim and wrapped in an extensible architecture.

## Architecture

```
Varify/
├── main.py                     # Centralized CLI (--mode scan|optimize|watch|analyze)
├── config/
│   ├── config.yaml                    # SLURM directives, parameter bounds, statics, MCMC priors
│   ├── config_hybrid_workflow.yaml    # Annotated sample: stage → modify → submit&wait → compare → advance
│   └── hooks.py                       # User callables (input_fn / coupled_fn / analysis / loss_fn)
├── templates/                  # @TOKEN@-substituted simulation input & run script
├── src/
│   ├── common/                 # Shared core
│   │   ├── params.py           #   ParamSpec / GridPoint / MCMCStep dataclasses
│   │   ├── config.py           #   YAML loader → FrameworkConfig
│   │   └── casebuilder.py      #   Case-dir builder (case.source_dir staging + templates + input_fns)
│   ├── scanner/                # Legacy line/grid search, wrapped
│   │   ├── base.py             #   Scanner ABC (coupled machinery, case prep)
│   │   └── grid.py             #   GridScanner (Cartesian) / CoupledScanner (zip)
│   ├── optimizer/              # Parameter-set → job → metric execution loop
│   │   ├── base.py             #   BaseOptimizer (dispatch / wait / parse)
│   │   ├── gradient_free.py    #   Nelder-Mead via scipy.optimize
│   │   ├── mcmc.py             #   Ensemble stretch-move MCMC (emcee-equivalent)
│   │   └── hybrid.py           #   Hybrid DE→Nelder-Mead (method: de-nm); batch-submitted generations
│   ├── slurm/                  # Scheduler integration
│   │   ├── dispatcher.py       #   sbatch submission + job_state / wait_for_completion / wait_for_batch
│   │   ├── registry.py         #   Central jobs_registry.csv (watchdog input)
│   │   └── watchdog.py         #   Background failure monitor / sbatch agent
│   └── analysis/               # Post-processing
│       ├── parser.py           #   ResultParser → DataFrame / CSV / SQLite
│       ├── analysis_dispatcher.py  # Signature-introspecting analysis hooks
│       ├── diagnostics.py      #   Gelman-Rubin R̂, autocorrelation time
│       ├── plotting.py         #   1-D curves, 2-D heatmaps, corner/MCMC plots + overlay/postprocess plots
│       └── postprocess.py      #   Curve interp (spline/linear) + loss registry (mse/rmse/mae/huber/chi2)
├── tests/                      # 86 pytest tests: scanner, optimizer/hybrid, dispatcher-wait, casebuilder
│                                #   staging, file_pipeline, postprocess/losses, plotting, runner-hybrid
└── results/plots/              # Auto-generated figures (created on demand)
```

### Compute & automation layer (zero-dependency)

Alongside the config-driven orchestrator above, the repository ships a
standard-library-only compute layer:

```
├── sbatch_template.sh          # @TOKEN@-driven SLURM dispatch shell (cluster portability)
├── utils/
│   ├── io_handlers.py          # encoding-fallback readers, atomic writes, comment-
│   │                           #   preserving in-place JSON/YAML/TOML value mutation
│   ├── diagnostics.py          # Slurm log post-mortem: OOM/timeout/segfault/NaN flags
│   │                           #   + wall-time & throughput profiling
│   └── file_pipeline.py        # Declarative per-case generate/modify of json/yaml/toml/
│                               #   key=value files; also runnable standalone (python -m)
├── runner/                     # agnostic workflow runner
│   ├── core.py                 # RunSpec lifecycle; strategy + runtime are single strings;
│   │                           #   wait/wait_timeout/wait_poll block on self-submitted sbatch jobs
│   ├── strategies.py           # optimize / grid / mcmc / mcmc_diagnostic / benchmark / hybrid
│   ├── checkpoint.py           # SIGTERM/USR1 wall-time trap + atomic resumable state
│   ├── preflight.py            # environment/config validator (runs before allocation)
│   └── provenance.py           # immutable git/seed/env/timestamp record in every payload
└── examples/                   # 8 standalone blueprint workflows → timestamped results/
```

Every workflow declares a `RunSpec`; swapping the algorithm
(`strategy="mcmc"`) or the execution target (`runtime="slurm"`, which
self-submits via `sbatch_template.sh`) is a one-string change.  scipy /
emcee / numpy / matplotlib are optional upgrades — pure-Python fallbacks
keep the layer dependency-free.  See `examples/README.md`.

**Data flow.** Every mode reads `config/config.yaml`. Each case directory is
built by a common, per-case pipeline: (1) if a top-level `case:` block is
present, `case.source_dir` is copied recursively into the case dir first;
(2) `templates/` files are rendered with `@TOKEN@` substitution (can
overwrite same-named staged files); (3) per-parameter `input_fn` hooks run;
(4) a declarative `file_pipeline:` (generate/modify json/yaml/toml/
key=value files) applies last. *Scan* enumerates the parameter space, runs
that pipeline once per grid point and fires one SLURM job each (`--wait`
blocks until every submitted job finishes and prints a completed/failed
summary). *Optimize* closes the loop with one of three methods:
Nelder-Mead or MCMC propose parameters one job at a time (submit → block
until the output file appears → regex-parse the objective/log-probability →
next step), while the hybrid `de-nm` method runs Differential Evolution
first — each **generation's** whole population of candidates is built and
submitted to SLURM *concurrently* and awaited as one batch
(`SlurmDispatcher.wait_for_batch`) — before handing the DE champion to a
scipy Nelder-Mead local refinement for the remaining evaluation budget. Any
optimizer method can score cases either by the legacy `output_regex` scrape
or, with `optimizer.postprocess: true`, by interpolating the simulated
curve onto an experimental x-grid and computing a configurable loss
(mse/rmse/mae/huber/chi2, or a custom `loss_fn` hook). Every submission is
recorded in `<workspace>/jobs_registry.csv`; the *watch* daemon polls it,
detects missing outputs, NaN-contaminated logs and stalled files, then logs
to `status.csv`, cancels the job and optionally resubmits with scaled
parameters. *Analyze* harvests all run folders into a `pandas` DataFrame
(CSV + SQLite) and renders the full plotting suite — including the
postprocess overlay and convergence plots when `optimizer.postprocess` is
on — into `results/plots/`.

> **Gotcha:** `scan.output_file` must be the file your *simulation itself*
> writes (e.g. `output.dat`), never a shell/scheduler-redirected log such as
> `stdout.log`. Redirected logs are created by the scheduler the instant the
> job starts, long before results exist, which defeats the wait-for-output
> completion gate and scores an empty case.

## Installation

### pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate varify
```

Optional: `pip install corner` for publication-quality MCMC corner plots
(a seaborn-style fallback is built in).

## Quick start

```bash
python main.py --mode scan --dry-run     # preview case dirs, no submission
python main.py --mode scan               # submit the configured grid search
python main.py --mode scan --wait        # ...and block until every job finishes
python main.py --mode watch --deploy-agent   # fault-tolerance agent via sbatch
python main.py --mode optimize --method mcmc  # Bayesian sampling
python main.py --mode optimize --method de-nm # hybrid Differential Evolution → Nelder-Mead
python main.py --mode analyze            # CSV + SQLite + plots + hooks
```

### Full workflow example

`config/config_hybrid_workflow.yaml` is a complete, heavily annotated config
showing the hybrid optimizer's per-iteration loop end to end, and
`examples/example_08_hybrid_slurm_workflow.py` runs that exact loop locally
(no cluster required, ~15 s) with a fake-sbatch stand-in, recovering the
true synthetic parameters to within a few percent:

1. **Stage** — `case.source_dir` is copied into each case directory and
   `@TOKEN@`s are substituted.
2. **Modify** — `file_pipeline:` generates/edits extra per-case files (e.g.
   `params.json`).
3. **Submit & wait** — the hybrid optimizer submits a whole DE generation of
   candidates to SLURM concurrently and blocks on
   `dispatcher.wait_for_batch` for the batch to complete.
4. **Compare** — each case's simulated curve is interpolated onto the
   experimental x-grid and scored with a configurable loss.
5. **Advance** — DE mutates/selects the next generation until it stalls or
   exhausts its budget, then Nelder-Mead polishes the DE champion with the
   remaining evaluations.

```bash
python examples/example_08_hybrid_slurm_workflow.py
```

See **[MANUAL.md](MANUAL.md)** for copy-pasteable, task-by-task commands.
