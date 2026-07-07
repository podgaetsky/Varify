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
│   ├── config.yaml             # SLURM directives, parameter bounds, statics, MCMC priors
│   └── hooks.py                # User callables (input_fn / coupled_fn / analysis fns)
├── templates/                  # @TOKEN@-substituted simulation input & run script
├── src/
│   ├── common/                 # Shared core
│   │   ├── params.py           #   ParamSpec / GridPoint / MCMCStep dataclasses
│   │   ├── config.py           #   YAML loader → FrameworkConfig
│   │   └── casebuilder.py      #   Case-dir builder (templates + input_fns)
│   ├── scanner/                # Legacy line/grid search, wrapped
│   │   ├── base.py             #   Scanner ABC (coupled machinery, case prep)
│   │   └── grid.py             #   GridScanner (Cartesian) / CoupledScanner (zip)
│   ├── optimizer/              # Parameter-set → job → metric execution loop
│   │   ├── base.py             #   BaseOptimizer (dispatch / wait / parse)
│   │   ├── gradient_free.py    #   Nelder-Mead via scipy.optimize
│   │   └── mcmc.py             #   Ensemble stretch-move MCMC (emcee-equivalent)
│   ├── slurm/                  # Scheduler integration
│   │   ├── dispatcher.py       #   sbatch submission wrapper + job-id parsing
│   │   ├── registry.py         #   Central jobs_registry.csv (watchdog input)
│   │   └── watchdog.py         #   Background failure monitor / sbatch agent
│   └── analysis/               # Post-processing
│       ├── parser.py           #   ResultParser → DataFrame / CSV / SQLite
│       ├── analysis_dispatcher.py  # Signature-introspecting analysis hooks
│       ├── diagnostics.py      #   Gelman-Rubin R̂, autocorrelation time
│       └── plotting.py         #   1-D curves, 2-D heatmaps, corner & MCMC plots
└── results/plots/              # Auto-generated figures (created on demand)
```

**Data flow.** Every mode reads `config/config.yaml`. *Scan* enumerates the
parameter space, renders one case directory per point from `templates/` and
fires one SLURM job each. *Optimize* (Nelder-Mead or MCMC) closes the loop:
propose parameters → submit job → block until the output file appears →
regex-parse the objective / log-probability → next step. Every submission is
recorded in `<workspace>/jobs_registry.csv`; the *watch* daemon polls it,
detects missing outputs, NaN-contaminated logs and stalled files, then logs
to `status.csv`, cancels the job and optionally resubmits with scaled
parameters. *Analyze* harvests all run folders into a `pandas` DataFrame
(CSV + SQLite) and renders the full plotting suite into `results/plots/`.

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
python main.py --mode watch --deploy-agent   # fault-tolerance agent via sbatch
python main.py --mode optimize --method mcmc # Bayesian sampling
python main.py --mode analyze            # CSV + SQLite + plots + hooks
```

See **[MANUAL.md](MANUAL.md)** for copy-pasteable, task-by-task commands.
