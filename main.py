"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Parametric Sweep & Bayesian MCMC — Unified Cluster Orchestrator           ║
║                                                                              ║
║   Schedulers : SLURM (sbatch) · PBS/Torque (qsub) · LSF (bsub)             ║
║                                                                              ║
║   Sweep modes                                                                ║
║     grid    – Cartesian product of all swept params (default)                ║
║     coupled – params advance in lock-step (zip)                              ║
║     mcmc    – ensemble Metropolis-Hastings with parallel walkers             ║
║                                                                              ║
║   CLI flags                                                                  ║
║     --submit   generate dirs & fire jobs to scheduler                        ║
║     --harvest  scrape finished outputs → unified CSV                         ║
║     --plot     publication-quality figures (1-D / 2-D / MCMC)               ║
║     --analyse  run registered ANALYSIS_FNS on results CSV                    ║
║     --mcmc     run the full MCMC state machine (blocking orchestration)      ║
║     --dry-run  prepare everything but do NOT call the scheduler              ║
╚══════════════════════════════════════════════════════════════════════════════╝

MCMC integration summary
─────────────────────────
The MCMC state machine (MCMCManager) reuses ClusterDispatcher for job
submission and ResultHarvester._parse_case for output scraping, but wraps
them in a sequential, walker-parallel control loop:

  1. Propose  – Gaussian stretch-move proposal from the complementary ensemble
                (identical to emcee's GoodmanWeare sampler).
  2. Dispatch – ClusterDispatcher fires one job per walker via the standard
                submit_cmd template; each walker gets its own isolated case dir.
  3. Poll     – MCMCManager polls SLURM/PBS job status every MCMC_POLL_INTERVAL
                seconds (or uses SLURM afterok dependency chains when
                MCMC_USE_DEPENDENCY=True).  ResultHarvester detects completion
                by the presence of the output file.
  4. Harvest  – ResultHarvester._parse_case reads the log-probability value.
  5. Accept   – Metropolis-Hastings criterion; log α = log p(x') - log p(x).
  6. Persist  – Every step is flushed to MCMC_CHAIN_CSV; the chain resumes
                from the last written step after an interruption.
  7. Repeat   – until MCMC_NUM_ITERS steps per walker are accumulated.

MCMC analysis (--plot in mcmc mode) produces:
  • Trace plots  (parameter value vs. chain step, per walker)
  • χ² history   (−2 × log p, all visited + accepted)
  • Corner plot  (posterior marginals, requires the `corner` package)
  • Convergence  (τ, τ-multiples, ESS vs. chain length)

Usage examples
──────────────
  python cluster_sweep.py --submit             # grid/coupled sweep
  python cluster_sweep.py --harvest --plot
  python cluster_sweep.py --analyse
  python cluster_sweep.py --mcmc               # full MCMC orchestration
  python cluster_sweep.py --mcmc --dry-run     # preview proposals only
  python cluster_sweep.py --plot               # MCMC plots after chain done
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List,
    Optional, Sequence, Tuple,
)

import numpy as np
import pandas as pd

# ── Optional plot stack ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

try:
    import corner as _corner_mod
    _CORNER_AVAILABLE = True
except ImportError:
    _CORNER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
# ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
# ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
# ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sweep / MCMC mode ─────────────────────────────────────────────────────────
#   "grid"    – full Cartesian product of all swept params  (default)
#   "coupled" – zip all swept params together in lock-step
#   "mcmc"    – Bayesian ensemble MCMC; see MCMC CONFIG section below
SWEEP_MODE: str = "grid"

# ── Parameter definitions ────────────────────────────────────────────────────
#
#   Required keys per entry:
#     name      (str)   – placeholder token; @NAME@ will be substituted
#     default   (float) – value used when this param is NOT being swept
#
#   Sweep specification (one of):
#     values    (list / array)               – explicit list
#     start + stop + num [+ log_scale]       – linspace / logspace range
#
#   MCMC-specific keys (used when SWEEP_MODE = "mcmc"):
#     mcmc_prior_low   (float) – hard lower bound; proposals outside → -inf log p
#     mcmc_prior_high  (float) – hard upper bound; proposals outside → -inf log p
#     mcmc_init_center (float) – centre for initial walker scatter (default: default)
#     mcmc_init_width  (float) – ± scatter around init_center for initial walkers
#
#   Optional keys (all modes):
#     input_fn  (callable | None)
#               fn(case_dir: Path, value: float, **all_params) -> None
#
#     coupled_to  (str | None) – ride along with another param's grid axis
#     coupled_fn  (callable | None) – driver_val → this_val

def _tau_input_fn(case_dir: Path, value: float, **params: Any) -> None:
    """Example input_fn: write a JSON sidecar for the tau parameter."""
    sidecar = case_dir / "tau_meta.json"
    sidecar.write_text(
        json.dumps({"tau": value, "all_params": params}, indent=2),
        encoding="utf-8",
    )

PARAMS: List[Dict[str, Any]] = [
    dict(
        name             = "tau",
        default          = 1.0,
        start            = 0.5,
        stop             = 5.0,
        num              = 10,
        log_scale        = False,
        input_fn         = _tau_input_fn,
        mcmc_prior_low   = 0.0,
        mcmc_prior_high  = 10.0,
        mcmc_init_center = 1.0,
        mcmc_init_width  = 0.5,
    ),
    dict(
        name             = "gamma",
        default          = 0.5,
        start            = 0.1,
        stop             = 2.0,
        num              = 5,
        log_scale        = False,
        input_fn         = None,
        mcmc_prior_low   = 0.0,
        mcmc_prior_high  = 5.0,
        mcmc_init_center = 0.5,
        mcmc_init_width  = 0.2,
    ),
    dict(
        name             = "kappa",
        default          = 1.0,
        coupled_to       = "tau",
        coupled_fn       = lambda tau: tau / 2.0,
        input_fn         = None,
        # kappa is NOT swept in MCMC (no prior bounds supplied → fixed at default)
    ),
]

# ── MCMC configuration ────────────────────────────────────────────────────────
#   Only used when SWEEP_MODE = "mcmc".
#
#   MCMC_LOG_PROB_REGEX  regex capturing the log-probability from each job's
#                        output file (separate from OUTPUT_REGEX which is used
#                        in grid/coupled harvest).  Must have one capture group.
#
#   MCMC_NUM_WALKERS     ensemble size (≥ 2 × number of MCMC params; even number)
#   MCMC_NUM_ITERS       chain steps per walker after burn-in
#   MCMC_BURNIN          steps discarded as burn-in before statistics
#   MCMC_STRETCH_A       emcee stretch-move scale parameter (default 2.0)
#   MCMC_POLL_INTERVAL   seconds between completion polls
#   MCMC_JOB_TIMEOUT     seconds before a single walker job is declared failed
#   MCMC_USE_DEPENDENCY  if True, use sbatch --dependency=afterok:<prev_id>
#                        for sequential walker chains (SLURM only)
#   MCMC_CHAIN_CSV       persistent state file for the MCMC chain

MCMC_LOG_PROB_REGEX:  str   = r"LOG_PROB:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
MCMC_NUM_WALKERS:     int   = 10
MCMC_NUM_ITERS:       int   = 500
MCMC_BURNIN:          int   = 100
MCMC_STRETCH_A:       float = 2.0
MCMC_POLL_INTERVAL:   float = 30.0    # seconds
MCMC_JOB_TIMEOUT:     float = 3600.0  # seconds (1 hour per walker job)
MCMC_USE_DEPENDENCY:  bool  = False
MCMC_CHAIN_CSV:       str   = "mcmc_chain.csv"

# ── Analysis functions ─────────────────────────────────────────────────────────
#   Keywords available in frame mode: df, cfg, output, <param_name>
#   Keywords available in row   mode: output, <param_name>  (scalars)

def example_row_analysis(tau: float, gamma: float, output: float) -> None:
    ratio = output / tau if tau != 0 else float("nan")
    print(f"  [row]  tau={tau:.4g}  gamma={gamma:.4g}  out={output:.6g}  out/tau={ratio:.4g}")

def example_frame_analysis(df: pd.DataFrame, output: np.ndarray, cfg: Any) -> None:
    valid = output[~np.isnan(output)]
    print(f"  [frame] n_valid={len(valid)}  mean={np.nanmean(output):.6g}"
          f"  std={np.nanstd(output):.6g}  swept={cfg.swept_names}")

def example_sensitivity(df: pd.DataFrame, cfg: Any) -> None:
    for name in cfg.swept_names:
        col = f"param_{name}"
        grp = (
            df.dropna(subset=["output"])
              .groupby(col)["output"].mean()
              .reset_index().sort_values(col)
        )
        if len(grp) < 2:
            continue
        x    = grp[col].to_numpy(float)
        y    = grp["output"].to_numpy(float)
        sens = np.abs(np.diff(y) / np.where(np.diff(x) != 0, np.diff(x), np.nan))
        print(f"  [sensitivity] {name}: max|dOut/d{name}|={np.nanmax(sens):.4g}"
              f"  at {name}≈{x[np.nanargmax(sens) + 1]:.4g}")

ANALYSIS_FNS: List[Callable] = [
    example_row_analysis,
    example_frame_analysis,
    example_sensitivity,
]

# ── Template & auxiliary files ────────────────────────────────────────────────
TEMPLATE_FILES: List[str] = [
    "input.template",
    "run_script.sh",
]

# ── Cluster submission command ────────────────────────────────────────────────
#   Placeholders: {job_name}, {case_dir}, {param_<name>}
SUBMIT_CMD: str = (
    "sbatch "
    "--job-name={job_name} "
    "--output=stdout.log "
    "--error=stderr.log "
    "run_script.sh"
)

# ── Output parsing (grid / coupled harvest) ───────────────────────────────────
OUTPUT_FILE:  str = "stdout.log"
OUTPUT_REGEX: str = r"RESULT:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# ── Paths ─────────────────────────────────────────────────────────────────────
SWEEP_ROOT:     str = "sweep_workspace"
SUBMISSION_LOG: str = "submission.log"
RESULTS_CSV:    str = "sweep_results.csv"
PLOT_FILE:      str = "cluster_sweep_results.png"

# ═══════════════════════════════════════════════════════════════════════════════
#  END OF USER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_logging(verbose: bool = False) -> logging.Logger:
    level   = logging.DEBUG if verbose else logging.INFO
    fmt     = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stdout)
    return logging.getLogger("cluster_sweep")

log = _setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
#  Core data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamSpec:
    name:             str
    default:          float
    values:           Optional[np.ndarray]
    input_fn:         Optional[Callable]
    coupled_to:       Optional[str]      = None
    coupled_fn:       Optional[Callable] = None
    # MCMC fields (None → param is fixed in MCMC, swept in grid/coupled)
    mcmc_prior_low:   Optional[float]    = None
    mcmc_prior_high:  Optional[float]    = None
    mcmc_init_center: Optional[float]    = None
    mcmc_init_width:  Optional[float]    = None

    @property
    def is_swept(self) -> bool:
        return self.values is not None and len(self.values) > 0 and self.coupled_to is None

    @property
    def is_coupled(self) -> bool:
        return self.coupled_to is not None

    @property
    def is_mcmc_param(self) -> bool:
        """A param participates in MCMC only if it has a prior range."""
        return self.mcmc_prior_low is not None and self.mcmc_prior_high is not None

    @property
    def n(self) -> int:
        return len(self.values) if self.values is not None else 1


@dataclass
class GridPoint:
    params:      Dict[str, float]
    swept_names: List[str]

    @property
    def job_name(self) -> str:
        parts = ["sweep"] + [f"{n}_{self.params[n]:.4g}" for n in self.swept_names]
        return "_".join(parts).replace(".", "_").replace("/", "_")

    @property
    def case_dir_name(self) -> str:
        if not self.swept_names:
            return "case_default"
        parts = ["case"] + [f"{n}_{self.params[n]:.6g}" for n in self.swept_names]
        return "_".join(parts)

    def substitution_map(self) -> Dict[str, str]:
        m: Dict[str, str] = {"JOB_NAME": self.job_name}
        for name, val in self.params.items():
            m[name.upper()] = f"{val:.10g}"
        return m


@dataclass
class MCMCStep:
    """One accepted (or pending) step in the MCMC chain."""
    step:       int
    walker:     int
    params:     Dict[str, float]   # only MCMC params
    log_prob:   float
    accepted:   bool
    case_dir:   str
    job_id:     Optional[str] = None


@dataclass
class SweepConfig:
    param_specs:          List[ParamSpec]
    swept_specs:          List[ParamSpec]
    sweep_mode:           str
    template_files:       List[str]
    submit_cmd_template:  str
    output_file:          str
    output_regex:         str
    sweep_root:           Path
    submission_log:       Path
    results_csv:          Path
    plot_file:            Path
    analysis_fns:         List[Callable]
    # MCMC
    mcmc_log_prob_regex:  str
    mcmc_num_walkers:     int
    mcmc_num_iters:       int
    mcmc_burnin:          int
    mcmc_stretch_a:       float
    mcmc_poll_interval:   float
    mcmc_job_timeout:     float
    mcmc_use_dependency:  bool
    mcmc_chain_csv:       Path

    @property
    def swept_names(self) -> List[str]:
        return [s.name for s in self.swept_specs]

    @property
    def coupled_specs(self) -> List[ParamSpec]:
        return [s for s in self.param_specs if s.is_coupled]

    @property
    def all_names(self) -> List[str]:
        return [s.name for s in self.param_specs]

    @property
    def mcmc_specs(self) -> List[ParamSpec]:
        return [s for s in self.param_specs if s.is_mcmc_param]

    @property
    def mcmc_names(self) -> List[str]:
        return [s.name for s in self.mcmc_specs]

    @property
    def is_2d_grid(self) -> bool:
        return self.sweep_mode == "grid" and len(self.swept_specs) == 2

    @property
    def defaults(self) -> Dict[str, float]:
        return {s.name: s.default for s in self.param_specs}


# ═══════════════════════════════════════════════════════════════════════════════
#  AnalysisDispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisDispatcher:
    """
    Introspects each ANALYSIS_FNS entry and forwards exactly the kwargs it
    declares.  Two calling conventions are auto-detected:

      frame mode – fn declares 'df' or 'cfg': called once with full DataFrame.
      row   mode – all other fns: called once per non-NaN result row.
    """

    _FRAME_TRIGGERS = frozenset({"df", "cfg"})

    def __init__(self, cfg: SweepConfig) -> None:
        self.cfg  = cfg
        self._log = logging.getLogger("cluster_sweep.analysis")

    @staticmethod
    def _declared_params(fn: Callable) -> List[str]:
        sig = inspect.signature(fn)
        return [
            name for name, p in sig.parameters.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

    @staticmethod
    def _has_var_keyword(fn: Callable) -> bool:
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fn).parameters.values()
        )

    def _is_frame_mode(self, fn: Callable) -> bool:
        return bool(set(self._declared_params(fn)) & self._FRAME_TRIGGERS) \
               or self._has_var_keyword(fn)

    def _frame_pool(self, df: pd.DataFrame) -> Dict[str, Any]:
        pool: Dict[str, Any] = {
            "df": df, "cfg": self.cfg,
            "output": df["output"].to_numpy(float),
        }
        for name in self.cfg.all_names:
            col = f"param_{name}"
            if col in df.columns:
                pool[name] = df[col].to_numpy(float)
        return pool

    @staticmethod
    def _row_pool(row: pd.Series, all_names: List[str]) -> Dict[str, Any]:
        pool: Dict[str, Any] = {"output": float(row["output"])}
        for name in all_names:
            col = f"param_{name}"
            if col in row.index:
                pool[name] = float(row[col])
        return pool

    def _filter_kwargs(self, fn: Callable, pool: Dict[str, Any]) -> Dict[str, Any]:
        if self._has_var_keyword(fn):
            return pool
        declared = self._declared_params(fn)
        return {k: pool[k] for k in declared if k in pool}

    def _run_one(self, fn: Callable, df: pd.DataFrame) -> None:
        fn_name = getattr(fn, "__name__", repr(fn))
        try:
            if self._is_frame_mode(fn):
                fn(**self._filter_kwargs(fn, self._frame_pool(df)))
            else:
                for _, row in df.dropna(subset=["output"]).iterrows():
                    fn(**self._filter_kwargs(fn, self._row_pool(row, self.cfg.all_names)))
        except Exception as exc:
            self._log.error("Analysis function '%s' raised: %s", fn_name, exc, exc_info=True)

    def run_all(self, df: pd.DataFrame) -> None:
        if not self.cfg.analysis_fns:
            self._log.info("No analysis functions registered.")
            return
        self._log.info(
            "Running %d analysis function(s) on %d rows (%d valid)…",
            len(self.cfg.analysis_fns), len(df), df["output"].notna().sum(),
        )
        for fn in self.cfg.analysis_fns:
            self._log.info("  → %s", getattr(fn, "__name__", repr(fn)))
            self._run_one(fn, df)
        self._log.info("Analysis complete.")


# ═══════════════════════════════════════════════════════════════════════════════
#  GridManager
# ═══════════════════════════════════════════════════════════════════════════════

class GridManager:
    """Generates the parameter grid, prepares case directories, runs input_fns."""

    def __init__(self, cfg: SweepConfig) -> None:
        self.cfg  = cfg
        self._log = logging.getLogger("cluster_sweep.grid")

    # ── Grid iteration ────────────────────────────────────────────────────────

    def iter_grid(self) -> Iterator[GridPoint]:
        cfg   = self.cfg
        base  = cfg.defaults.copy()
        swept = cfg.swept_specs

        coupled_by_driver: Dict[str, List[ParamSpec]] = {}
        for cs in cfg.coupled_specs:
            assert cs.coupled_to is not None
            coupled_by_driver.setdefault(cs.coupled_to, []).append(cs)

        driver_names = {s.name for s in swept}
        for cs in cfg.coupled_specs:
            if cs.coupled_to in driver_names and cs.coupled_fn is None:
                driver_spec = next(s for s in swept if s.name == cs.coupled_to)
                if cs.values is None or len(cs.values) != driver_spec.n:
                    raise ValueError(
                        f"Coupled param '{cs.name}' needs coupled_fn or a values "
                        f"array of length {driver_spec.n} (driver '{cs.coupled_to}')."
                    )

        def _apply_coupled(p: Dict[str, float], driver_name: str, driver_idx: int) -> None:
            driver_val = p[driver_name]
            for cs in coupled_by_driver.get(driver_name, []):
                if cs.coupled_fn is not None:
                    p[cs.name] = float(cs.coupled_fn(driver_val))
                elif cs.values is not None:
                    p[cs.name] = float(cs.values[driver_idx])

        if not swept:
            p = base.copy()
            for cs in cfg.coupled_specs:
                dval = p.get(cs.coupled_to, base.get(cs.coupled_to, 0.0))  # type: ignore[arg-type]
                if cs.coupled_fn is not None:
                    p[cs.name] = float(cs.coupled_fn(dval))
            yield GridPoint(params=p, swept_names=[])
            return

        if cfg.sweep_mode == "coupled":
            lengths = [s.n for s in swept]
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"SWEEP_MODE='coupled' requires equal-length arrays, "
                    f"got {dict(zip(cfg.swept_names, lengths))}"
                )
            for idx, combo in enumerate(zip(*(s.values for s in swept))):  # type: ignore[arg-type]
                p = base.copy()
                for spec, val in zip(swept, combo):
                    p[spec.name] = float(val)
                for spec in swept:
                    _apply_coupled(p, spec.name, idx)
                yield GridPoint(params=p, swept_names=cfg.swept_names)
        else:  # grid
            for idx_combo, val_combo in zip(
                itertools.product(*(range(s.n) for s in swept)),
                itertools.product(*(s.values for s in swept)),  # type: ignore[arg-type]
            ):
                p = base.copy()
                for spec, val in zip(swept, val_combo):
                    p[spec.name] = float(val)
                for spec, didx in zip(swept, idx_combo):
                    _apply_coupled(p, spec.name, didx)
                yield GridPoint(params=p, swept_names=cfg.swept_names)

    def total_points(self) -> int:
        swept = self.cfg.swept_specs
        if not swept:
            return 1
        if self.cfg.sweep_mode == "coupled":
            return swept[0].n
        return int(np.prod([s.n for s in swept]))

    # ── Template substitution & dir prep ─────────────────────────────────────

    @staticmethod
    def _substitute(text: str, smap: Dict[str, str]) -> str:
        for key, val in smap.items():
            text = text.replace(f"@{key}@", val)
        return text

    def prepare_case(self, gp: GridPoint) -> Path:
        case_dir = self.cfg.sweep_root / gp.case_dir_name
        case_dir.mkdir(parents=True, exist_ok=True)
        smap = gp.substitution_map()

        for tpl_path_str in self.cfg.template_files:
            tpl_path = Path(tpl_path_str)
            if not tpl_path.exists():
                self._log.warning("Template not found, skipping: %s", tpl_path)
                continue
            raw       = tpl_path.read_text(encoding="utf-8")
            filled    = self._substitute(raw, smap)
            dest_name = tpl_path.name
            if dest_name.endswith(".template"):
                dest_name = dest_name[: -len(".template")]
            dest = case_dir / dest_name
            dest.write_text(filled, encoding="utf-8")
            if tpl_path.suffix == ".sh" or os.access(tpl_path, os.X_OK):
                dest.chmod(dest.stat().st_mode | 0o111)

        for spec in self.cfg.param_specs:
            if spec.input_fn is None:
                continue
            val = gp.params[spec.name]
            try:
                sig        = inspect.signature(spec.input_fn)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                declared_kw = [
                    n for n, p in sig.parameters.items()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ) and n not in ("case_dir", "value")
                ]
                extra = gp.params.copy() if has_var_kw else {
                    k: gp.params[k] for k in declared_kw if k in gp.params
                }
                spec.input_fn(case_dir, val, **extra)
            except Exception as exc:
                self._log.error(
                    "input_fn(%s) FAILED for %s: %s", spec.name, gp.case_dir_name, exc
                )

        self._log.debug("Prepared: %s", case_dir)
        return case_dir

    def existing_cases(self) -> List[Tuple[GridPoint, Path]]:
        results: List[Tuple[GridPoint, Path]] = []
        for gp in self.iter_grid():
            p = self.cfg.sweep_root / gp.case_dir_name
            if p.is_dir():
                results.append((gp, p))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ClusterDispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class ClusterDispatcher:
    """
    Non-blocking cluster job submission with submission audit log.
    Shared by both the grid sweep orchestrator and the MCMC state machine.
    """

    _JOB_ID_PATTERNS: List[re.Pattern] = [
        re.compile(r"Submitted batch job (\d+)", re.I),
        re.compile(r"^(\d+)(?:\.\S+)?$"),
        re.compile(r"Job <(\d+)> is submitted", re.I),
        re.compile(r"(\d{4,})"),
    ]

    def __init__(self, cfg: SweepConfig, dry_run: bool = False) -> None:
        self.cfg     = cfg
        self.dry_run = dry_run
        self._log    = logging.getLogger("cluster_sweep.dispatcher")
        self._fh: Optional[Any] = None

    def __enter__(self) -> "ClusterDispatcher":
        self._fh = open(self.cfg.submission_log, "a", encoding="utf-8")
        self._fh.write("\n# ── Submission session ──────────────────\n")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def _extract_job_id(self, stdout: str) -> Optional[str]:
        for pat in self._JOB_ID_PATTERNS:
            m = pat.search(stdout.strip())
            if m:
                return m.group(1)
        return None

    def _build_cmd(self, gp: GridPoint, case_dir: Path,
                   dependency_job_id: Optional[str] = None) -> str:
        fmt: Dict[str, Any] = {
            "job_name": gp.job_name,
            "case_dir": str(case_dir.resolve()),
            **{f"param_{n}": v for n, v in gp.params.items()},
        }
        cmd = self.cfg.submit_cmd_template.format(**fmt)
        if dependency_job_id and self.cfg.mcmc_use_dependency:
            # Insert SLURM dependency flag before the script name (last token)
            tokens   = cmd.split()
            dep_flag = f"--dependency=afterok:{dependency_job_id}"
            cmd      = " ".join(tokens[:-1] + [dep_flag, tokens[-1]])
        return cmd

    def dispatch(self, gp: GridPoint, case_dir: Path,
                 dependency_job_id: Optional[str] = None) -> Optional[str]:
        cmd = self._build_cmd(gp, case_dir, dependency_job_id)
        self._log.info("[DISPATCH] %s  →  %s", gp.case_dir_name, cmd)
        if self._fh:
            self._fh.write(f"{gp.case_dir_name}\t{cmd}\n")

        if self.dry_run:
            self._log.info("[DRY-RUN] Would execute: %s", cmd)
            return "DRY_RUN"

        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(case_dir.resolve()),
                capture_output=True, text=True, timeout=30,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode != 0:
                self._log.error(
                    "Submission failed %s (rc=%d): %s",
                    gp.case_dir_name, result.returncode, stderr or stdout,
                )
                return None
            job_id = self._extract_job_id(stdout)
            self._log.info("Submitted %s → job_id=%s", gp.case_dir_name, job_id or "?")
            if self._fh:
                self._fh.write(f"  job_id={job_id}  rc={result.returncode}\n")
            return job_id
        except subprocess.TimeoutExpired:
            self._log.error("Timeout submitting %s", gp.case_dir_name)
            return None
        except Exception as exc:
            self._log.error("Error submitting %s: %s", gp.case_dir_name, exc)
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ResultHarvester
# ═══════════════════════════════════════════════════════════════════════════════

class ResultHarvester:
    """
    Walks case directories, parses output files, assembles a Pandas DataFrame.
    Also exposes _parse_case and _parse_log_prob used by the MCMC state machine.
    """

    def __init__(self, cfg: SweepConfig, grid_mgr: Optional[GridManager] = None) -> None:
        self.cfg      = cfg
        self.grid_mgr = grid_mgr
        self._log     = logging.getLogger("cluster_sweep.harvester")

    def _parse_case(self, gp: GridPoint, case_dir: Path,
                    regex: Optional[str] = None) -> float:
        """Return the scalar captured by *regex* (defaults to OUTPUT_REGEX), or NaN."""
        out_file = case_dir / self.cfg.output_file
        pattern  = regex or self.cfg.output_regex
        try:
            if not out_file.exists():
                raise FileNotFoundError(f"{out_file} not found (job still running?)")
            text  = out_file.read_text(encoding="utf-8", errors="replace")
            match = re.search(pattern, text)
            if not match:
                raise ValueError(f"Pattern {pattern!r} not found in {out_file}")
            val = float(match.group(1))
            self._log.debug("Parsed %s → %.6g", gp.case_dir_name, val)
            return val
        except Exception as exc:
            self._log.warning("Skip %s — %s", gp.case_dir_name, exc)
            return float("nan")

    def _parse_log_prob(self, case_dir: Path, gp: GridPoint) -> float:
        """Parse the MCMC log-probability from a finished walker job."""
        return self._parse_case(gp, case_dir, regex=self.cfg.mcmc_log_prob_regex)

    def _wait_for_output(self, case_dir: Path, timeout: float) -> bool:
        """Block until stdout.log exists or *timeout* seconds have elapsed."""
        out_file = case_dir / self.cfg.output_file
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if out_file.exists():
                return True
            time.sleep(self.cfg.mcmc_poll_interval)
        return False

    def harvest(self) -> pd.DataFrame:
        assert self.grid_mgr is not None, "grid_mgr required for harvest()"
        cases = self.grid_mgr.existing_cases()
        self._log.info("Harvesting %d case directories…", len(cases))
        rows: List[Dict[str, Any]] = []
        for gp, case_dir in cases:
            row: Dict[str, Any] = {f"param_{n}": gp.params[n] for n in self.cfg.all_names}
            row["output"] = self._parse_case(gp, case_dir)
            rows.append(row)
        col_order = [f"param_{n}" for n in self.cfg.all_names] + ["output"]
        df = pd.DataFrame(rows, columns=col_order)
        sort_cols = [f"param_{n}" for n in self.cfg.swept_names] or col_order[:-1]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df.to_csv(self.cfg.results_csv, index=False)
        self._log.info(
            "Saved %d rows → %s  (NaN: %d)",
            len(df), self.cfg.results_csv, df["output"].isna().sum(),
        )
        return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MCMCManager  ── the heart of SWEEP_MODE = "mcmc"
# ═══════════════════════════════════════════════════════════════════════════════

class MCMCManager:
    """
    Ensemble Metropolis-Hastings MCMC orchestrator.

    Architecture
    ─────────────
    • Uses ClusterDispatcher to fire one cluster job per walker per step.
    • Uses ResultHarvester._parse_log_prob to read the walker's log-probability.
    • Implements the emcee stretch-move proposal: for walker k, the proposal is

          x_proposed = x_j + z * (x_k - x_j)

      where j is a randomly chosen walker from the complementary half-ensemble
      and z is drawn from g(z) ∝ 1/√z for z ∈ [1/a, a] (a = MCMC_STRETCH_A).
      The acceptance probability is min(1, z^(d-1) * p(x')/p(x)).

    • State is checkpointed to MCMC_CHAIN_CSV after every step; the chain
      resumes from the last committed step on restart.

    CSV columns
    ────────────
    step, walker, accepted, log_prob, case_dir, job_id, <param_name>, ...
    """

    def __init__(self, cfg: SweepConfig, dry_run: bool = False) -> None:
        self.cfg       = cfg
        self.dry_run   = dry_run
        self._log      = logging.getLogger("cluster_sweep.mcmc")
        self._harvester = ResultHarvester(cfg)
        self._rng       = np.random.default_rng()

    # ── Prior ─────────────────────────────────────────────────────────────────

    def _log_prior(self, params: Dict[str, float]) -> float:
        for spec in self.cfg.mcmc_specs:
            v = params[spec.name]
            lo = spec.mcmc_prior_low
            hi = spec.mcmc_prior_high
            assert lo is not None and hi is not None
            if not (lo <= v <= hi):
                return -math.inf
        return 0.0  # uniform prior within bounds

    # ── Stretch-move proposal ─────────────────────────────────────────────────

    def _stretch_move(self,
                      current: np.ndarray,
                      complement: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Return (proposed_vector, log_acceptance_correction).
        log_q = (d - 1) * log(z)  where d = dim(parameter space).
        """
        d     = len(current)
        a     = self.cfg.mcmc_stretch_a
        # Draw z ~ g(z) ∝ 1/√z on [1/a, a]  (inverse CDF method)
        u     = self._rng.uniform(0, 1)
        z     = (1.0 + u * (a - 1.0 / a) + (1.0 / a - 1)) ** 2 / a
        # Pick a random walker from the complementary ensemble
        j_idx = self._rng.integers(0, len(complement))
        x_j   = complement[j_idx]
        x_prop = x_j + z * (current - x_j)
        log_q  = (d - 1) * math.log(z)
        return x_prop, log_q

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_chain(self) -> pd.DataFrame:
        if self.cfg.mcmc_chain_csv.exists():
            try:
                df = pd.read_csv(self.cfg.mcmc_chain_csv)
                self._log.info(
                    "Resumed chain from %s (%d rows)", self.cfg.mcmc_chain_csv, len(df)
                )
                return df
            except Exception as exc:
                self._log.warning("Could not load chain CSV: %s — starting fresh", exc)
        return pd.DataFrame()

    def _append_step(self, step: MCMCStep) -> None:
        row: Dict[str, Any] = {
            "step":     step.step,
            "walker":   step.walker,
            "accepted": int(step.accepted),
            "log_prob": step.log_prob,
            "case_dir": step.case_dir,
            "job_id":   step.job_id or "",
        }
        row.update(step.params)
        df_new = pd.DataFrame([row])
        write_header = not self.cfg.mcmc_chain_csv.exists()
        df_new.to_csv(
            self.cfg.mcmc_chain_csv, mode="a", header=write_header, index=False
        )

    # ── Initial ensemble ──────────────────────────────────────────────────────

    def _initial_ensemble(self) -> np.ndarray:
        """
        Build (n_walkers × n_params) initial ensemble.
        Walkers are scattered uniformly within ± init_width around init_center.
        Clipped to [prior_low, prior_high].
        """
        specs  = self.cfg.mcmc_specs
        n_w    = self.cfg.mcmc_num_walkers
        n_p    = len(specs)
        ensemble = np.empty((n_w, n_p))
        for j, spec in enumerate(specs):
            lo    = spec.mcmc_prior_low
            hi    = spec.mcmc_prior_high
            ctr   = spec.mcmc_init_center if spec.mcmc_init_center is not None else spec.default
            width = spec.mcmc_init_width  if spec.mcmc_init_width  is not None else (hi - lo) * 0.1  # type: ignore[operator]
            vals  = ctr + self._rng.uniform(-width, width, n_w)
            ensemble[:, j] = np.clip(vals, lo, hi)
        return ensemble

    # ── Single-walker job ─────────────────────────────────────────────────────

    def _make_walker_grid_point(self, step: int, walker: int,
                                params_vec: np.ndarray) -> GridPoint:
        """Build a GridPoint for one walker evaluation."""
        full_params = self.cfg.defaults.copy()
        for spec, val in zip(self.cfg.mcmc_specs, params_vec):
            full_params[spec.name] = float(val)
        swept_names = self.cfg.mcmc_names
        # Stable dir name: mcmc_step_<step>_walker_<walker>
        gp = GridPoint(params=full_params, swept_names=swept_names)
        # Override case_dir_name via monkey-patch for MCMC naming
        object.__setattr__(gp, "_mcmc_dir", f"mcmc_step{step:06d}_w{walker:04d}")
        return gp

    def _mcmc_dir(self, step: int, walker: int) -> Path:
        return self.cfg.sweep_root / f"mcmc_step{step:06d}_w{walker:04d}"

    def _dispatch_walker(self, step: int, walker: int,
                         params_vec: np.ndarray,
                         dispatcher: ClusterDispatcher,
                         grid_mgr:   GridManager,
                         dependency_job_id: Optional[str] = None) -> Tuple[Path, Optional[str]]:
        """Prepare case dir and fire the cluster job for one walker."""
        full_params = self.cfg.defaults.copy()
        for spec, val in zip(self.cfg.mcmc_specs, params_vec):
            full_params[spec.name] = float(val)
        # Synthesise a GridPoint whose case_dir_name is the MCMC naming scheme
        gp_inner          = GridPoint(params=full_params, swept_names=self.cfg.mcmc_names)
        case_dir          = self._mcmc_dir(step, walker)
        # Re-use GridManager's template + input_fn machinery, but target our MCMC dir
        case_dir.mkdir(parents=True, exist_ok=True)
        smap = gp_inner.substitution_map()
        smap["JOB_NAME"] = f"mcmc_s{step}_w{walker}"
        for tpl_path_str in self.cfg.template_files:
            tpl_path = Path(tpl_path_str)
            if not tpl_path.exists():
                continue
            raw       = GridManager._substitute.__func__(None, tpl_path.read_text(encoding="utf-8"), smap)  # type: ignore[arg-type]
            dest_name = tpl_path.name
            if dest_name.endswith(".template"):
                dest_name = dest_name[: -len(".template")]
            dest = case_dir / dest_name
            dest.write_text(raw, encoding="utf-8")
            if tpl_path.suffix == ".sh" or os.access(tpl_path, os.X_OK):
                dest.chmod(dest.stat().st_mode | 0o111)
        for spec in self.cfg.param_specs:
            if spec.input_fn is None:
                continue
            val = full_params[spec.name]
            try:
                sig        = inspect.signature(spec.input_fn)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                declared_kw = [
                    n for n, p in sig.parameters.items()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ) and n not in ("case_dir", "value")
                ]
                extra = full_params.copy() if has_var_kw else {
                    k: full_params[k] for k in declared_kw if k in full_params
                }
                spec.input_fn(case_dir, val, **extra)
            except Exception as exc:
                self._log.error("input_fn(%s) FAILED walker %d step %d: %s",
                                spec.name, walker, step, exc)

        # Build a minimal GridPoint whose job_name matches our naming
        class _NamedGP(GridPoint):
            @property
            def job_name(self) -> str:
                return f"mcmc_s{step}_w{walker}"
            @property
            def case_dir_name(self) -> str:
                return case_dir.name

        named_gp = _NamedGP(params=full_params, swept_names=self.cfg.mcmc_names)
        job_id   = dispatcher.dispatch(named_gp, case_dir, dependency_job_id)
        return case_dir, job_id

    # ── Main MCMC loop ────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full MCMC state machine and return the chain DataFrame.
        This method blocks; it exits when MCMC_NUM_ITERS steps per walker
        have been accumulated (burn-in not counted in the returned samples).
        """
        cfg         = self.cfg
        specs       = cfg.mcmc_specs
        n_walkers   = cfg.mcmc_num_walkers
        n_iters     = cfg.mcmc_num_iters + cfg.mcmc_burnin
        n_params    = len(specs)
        param_names = cfg.mcmc_names

        if n_params == 0:
            self._log.error("No MCMC params defined. Add mcmc_prior_low/high to PARAMS.")
            return pd.DataFrame()
        if n_walkers < 2 * n_params:
            self._log.warning(
                "MCMC_NUM_WALKERS=%d < 2 × n_params=%d; emcee recommends ≥ 2×.",
                n_walkers, n_params,
            )

        chain_df   = self._load_chain()
        grid_mgr   = GridManager(cfg)

        # Determine resume state
        if not chain_df.empty:
            last_step = int(chain_df["step"].max())
            start_step = last_step + 1
            # Reconstruct current ensemble from last accepted position per walker
            ensemble = np.empty((n_walkers, n_params))
            log_probs = np.full(n_walkers, -math.inf)
            for w in range(n_walkers):
                walker_rows = chain_df[
                    (chain_df["walker"] == w) & (chain_df["accepted"] == 1)
                ]
                if walker_rows.empty:
                    # No accepted step yet for this walker — use last row anyway
                    walker_rows = chain_df[chain_df["walker"] == w]
                if walker_rows.empty:
                    # Walker hasn't started — sample from prior
                    init = self._initial_ensemble()
                    ensemble[w] = init[w]
                    log_probs[w] = -math.inf
                else:
                    last_row    = walker_rows.iloc[-1]
                    ensemble[w] = np.array([float(last_row[n]) for n in param_names])
                    log_probs[w] = float(last_row["log_prob"])
            self._log.info("Resuming from step %d", start_step)
        else:
            start_step = 0
            ensemble   = self._initial_ensemble()
            log_probs  = np.full(n_walkers, -math.inf)
            # Evaluate log-probs for the initial ensemble (step -1)
            self._log.info("Evaluating initial ensemble (%d walkers)…", n_walkers)
            with ClusterDispatcher(cfg, dry_run=self.dry_run) as dispatcher:
                init_dirs: List[Path]          = []
                init_ids:  List[Optional[str]] = []
                for w in range(n_walkers):
                    cdir, jid = self._dispatch_walker(-1, w, ensemble[w], dispatcher, grid_mgr)
                    init_dirs.append(cdir)
                    init_ids.append(jid)
                for w in range(n_walkers):
                    if not self.dry_run:
                        ok = self._harvester._wait_for_output(init_dirs[w], cfg.mcmc_job_timeout)
                        if ok:
                            dummy_gp = GridPoint(
                                params=cfg.defaults.copy(), swept_names=[]
                            )
                            log_probs[w] = self._harvester._parse_log_prob(init_dirs[w], dummy_gp)
                        else:
                            self._log.warning("Walker %d init timed out", w)
                    else:
                        log_probs[w] = 0.0  # dry-run placeholder

        n_accepted    = np.zeros(n_walkers, dtype=int)
        n_total       = np.zeros(n_walkers, dtype=int)

        with ClusterDispatcher(cfg, dry_run=self.dry_run) as dispatcher:
            for step in range(start_step, n_iters):
                self._log.info("── MCMC step %d / %d ──", step + 1, n_iters)
                proposed   = np.empty_like(ensemble)
                log_q      = np.empty(n_walkers)
                case_dirs: List[Path]          = []
                job_ids:   List[Optional[str]] = []

                # ── Propose & dispatch all walkers (two half-ensemble passes) ──
                half = n_walkers // 2
                for w in range(n_walkers):
                    complement_idx = (
                        list(range(half, n_walkers)) if w < half
                        else list(range(half))
                    )
                    complement = ensemble[complement_idx]
                    x_prop, lq = self._stretch_move(ensemble[w], complement)

                    # Clamp to prior
                    prior_ok = True
                    for j, spec in enumerate(specs):
                        lo = spec.mcmc_prior_low
                        hi = spec.mcmc_prior_high
                        assert lo is not None and hi is not None
                        if not (lo <= x_prop[j] <= hi):
                            prior_ok = False
                            break

                    if not prior_ok:
                        proposed[w] = ensemble[w]
                        log_q[w]    = -math.inf
                        case_dirs.append(self._mcmc_dir(step, w))
                        job_ids.append(None)
                    else:
                        proposed[w] = x_prop
                        log_q[w]    = lq
                        cdir, jid   = self._dispatch_walker(
                            step, w, x_prop, dispatcher, grid_mgr
                        )
                        case_dirs.append(cdir)
                        job_ids.append(jid)

                # ── Poll / harvest all walkers ─────────────────────────────────
                proposed_lps = np.full(n_walkers, -math.inf)
                dummy_gp     = GridPoint(params=cfg.defaults.copy(), swept_names=[])

                for w in range(n_walkers):
                    if log_q[w] == -math.inf:
                        # Prior violation — no job was run
                        continue
                    if self.dry_run:
                        proposed_lps[w] = 0.0
                        continue
                    ok = self._harvester._wait_for_output(case_dirs[w], cfg.mcmc_job_timeout)
                    if ok:
                        proposed_lps[w] = self._harvester._parse_log_prob(
                            case_dirs[w], dummy_gp
                        )
                    else:
                        self._log.warning("Walker %d step %d timed out", w, step)

                # ── Metropolis-Hastings acceptance ─────────────────────────────
                for w in range(n_walkers):
                    n_total[w] += 1
                    log_alpha = (
                        log_q[w]
                        + proposed_lps[w]
                        - log_probs[w]
                    )
                    accept = math.log(self._rng.uniform()) < log_alpha

                    if accept:
                        ensemble[w]  = proposed[w]
                        log_probs[w] = proposed_lps[w]
                        n_accepted[w] += 1

                    params_dict = {
                        spec.name: float(ensemble[w][j])
                        for j, spec in enumerate(specs)
                    }
                    mcmc_step = MCMCStep(
                        step     = step,
                        walker   = w,
                        params   = params_dict,
                        log_prob = log_probs[w],
                        accepted = accept,
                        case_dir = str(case_dirs[w]),
                        job_id   = job_ids[w],
                    )
                    self._append_step(mcmc_step)

                acc_rates = n_accepted / np.maximum(n_total, 1)
                self._log.info(
                    "Step %d done. Mean acceptance rate: %.3f",
                    step, float(acc_rates.mean()),
                )

        final_df = pd.read_csv(cfg.mcmc_chain_csv)
        self._log.info(
            "MCMC complete. Chain: %d rows. Acceptance rates: %s",
            len(final_df),
            [f"{r:.3f}" for r in (n_accepted / np.maximum(n_total, 1))],
        )
        return final_df

    # ── Convergence diagnostics (Gelman-Rubin) ────────────────────────────────

    @staticmethod
    def gelman_rubin(chains: np.ndarray) -> np.ndarray:
        """
        Gelman-Rubin R̂ statistic per parameter.
        chains: (n_steps, n_walkers, n_params)
        """
        n, m, d = chains.shape
        B = n * np.var(chains.mean(axis=0), axis=0, ddof=1)
        W = np.mean(np.var(chains, axis=0, ddof=1), axis=0)
        var_hat = (n - 1) / n * W + B / n
        R_hat   = np.sqrt(var_hat / np.where(W > 0, W, np.nan))
        return R_hat

    @staticmethod
    def autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
        """
        Estimate integrated autocorrelation time for 1-D chain x using
        the automated windowing procedure (Sokal 1989).
        """
        n    = len(x)
        x    = x - x.mean()
        # Full autocorrelation via FFT
        f    = np.fft.fft(x, n=2 * n)
        acf  = np.fft.ifft(f * np.conj(f)).real[:n] / (n * np.var(x) + 1e-30)
        tau  = 2.0 * np.cumsum(acf) - 1.0
        # Automated window: stop when window >= c * tau
        for M in range(1, n):
            if M >= c * tau[M]:
                return float(tau[M])
        return float(tau[-1])


# ═══════════════════════════════════════════════════════════════════════════════
#  ResultPlotter
# ═══════════════════════════════════════════════════════════════════════════════

class ResultPlotter:
    """
    Generates publication-quality figures for all three sweep modes.

    grid / coupled (1-D) → line plot + finite-difference derivative
    grid           (2-D) → heatmap + contour overlay
    coupled / grid (N-D) → marginal sensitivity panels
    mcmc                 → trace plots, χ² history, corner plot, convergence
    """

    _PALETTE = "viridis"
    _DPI     = 180

    def __init__(self, cfg: SweepConfig) -> None:
        if not _PLOT_AVAILABLE:
            raise ImportError("pip install matplotlib seaborn")
        self.cfg  = cfg
        self._log = logging.getLogger("cluster_sweep.plotter")
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _note(self, df: pd.DataFrame, fig: Any) -> None:
        pct = 100 * df["output"].notna().mean() if "output" in df.columns else 100.0
        n_valid = df["output"].notna().sum() if "output" in df.columns else len(df)
        fig.text(
            0.99, 0.01,
            f"completeness: {pct:.1f}%  ({n_valid}/{len(df)} pts)",
            ha="right", va="bottom", fontsize=8, color="grey",
        )

    def _save(self, fig: Any, path: Path) -> None:
        fig.savefig(path, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)
        self._log.info("Saved → %s", path)

    # ── Grid sweep plots (unchanged from pre-MCMC version) ────────────────────

    def _plot_1d(self, df: pd.DataFrame) -> None:
        xcol  = f"param_{self.cfg.swept_names[0]}"
        valid = df.dropna(subset=["output"])
        if valid.empty:
            self._log.warning("No valid data.")
            return
        x     = valid[xcol].to_numpy(float)
        y     = valid["output"].to_numpy(float)
        dx    = np.diff(x)
        deriv = np.where(dx != 0, np.diff(y) / dx, np.nan)
        x_mid = 0.5 * (x[:-1] + x[1:])
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(9, 7),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}, sharex=True,
        )
        ax0.plot(x, y, "o-", color="#1f77b4", lw=2, ms=6, label="output")
        nan_mask = df["output"].isna()
        if nan_mask.any():
            ax0.scatter(
                df.loc[nan_mask, xcol], np.zeros(nan_mask.sum()),
                marker="x", color="crimson", zorder=5, s=80, label="NaN/failed",
            )
        ax0.set_ylabel("Output", fontsize=12)
        ax0.set_title(f"1-D Sweep — {self.cfg.swept_names[0]}", fontsize=14, fontweight="bold")
        ax0.legend(framealpha=0.85)
        ax0.grid(True, alpha=0.4)
        ax1.step(x_mid, deriv, where="mid", color="#ff7f0e", lw=2)
        ax1.axhline(0, color="grey", lw=0.8, ls="--")
        ax1.fill_between(x_mid, deriv, 0, alpha=0.18, color="#ff7f0e", step="mid")
        ax1.set_xlabel(self.cfg.swept_names[0], fontsize=12)
        ax1.set_ylabel(r"$\Delta\,\mathrm{Output}\,/\,\Delta x$", fontsize=11)
        ax1.grid(True, alpha=0.4)
        self._note(df, fig)
        self._save(fig, self.cfg.plot_file)

    def _plot_2d(self, df: pd.DataFrame) -> None:
        xcol  = f"param_{self.cfg.swept_names[0]}"
        ycol  = f"param_{self.cfg.swept_names[1]}"
        pivot = df.pivot_table(index=ycol, columns=xcol, values="output", aggfunc="mean")
        Z     = pivot.to_numpy(float)
        X_t   = pivot.columns.to_numpy(float)
        Y_t   = pivot.index.to_numpy(float)
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.isnan(Z)
        sns.heatmap(
            pivot, ax=ax, cmap=self._PALETTE,
            annot=(Z.size <= 100), fmt=".3g", mask=mask,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Output", "shrink": 0.82},
        )
        ax2 = ax.twinx().twiny()
        ax2.set_xlim(X_t.min(), X_t.max())
        ax2.set_ylim(Y_t.min(), Y_t.max())
        ax2.set_xticks([]); ax2.set_yticks([])
        if (~mask).sum() >= 4:
            try:
                Xg, Yg = np.meshgrid(X_t, Y_t)
                cs = ax2.contour(
                    Xg, Yg, np.where(~mask, Z, np.nanmean(Z)),
                    levels=min(10, (~mask).sum() // 2),
                    colors="white", linewidths=0.9, alpha=0.7,
                )
                ax2.clabel(cs, inline=True, fontsize=7, fmt="%.3g")
            except Exception as exc:
                self._log.warning("Contour failed: %s", exc)
        ax.set_title(
            f"2-D Sweep  —  {self.cfg.swept_names[0]}  ×  {self.cfg.swept_names[1]}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel(self.cfg.swept_names[0], fontsize=12)
        ax.set_ylabel(self.cfg.swept_names[1], fontsize=12)
        self._note(df, fig)
        self._save(fig, self.cfg.plot_file)

    def _plot_nd(self, df: pd.DataFrame) -> None:
        swept = self.cfg.swept_names
        n     = len(swept)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        fig.suptitle(
            f"Marginal sensitivity  ({self.cfg.sweep_mode})", fontsize=14, fontweight="bold"
        )
        colors = cm.tab10(np.linspace(0, 0.9, n))
        for idx, (name, color) in enumerate(zip(swept, colors)):
            xcol = f"param_{name}"
            ax   = axes[0][idx]
            grp  = (
                df.dropna(subset=["output"]).groupby(xcol)["output"]
                  .agg(["mean", "std"]).reset_index()
            )
            ax.plot(grp[xcol], grp["mean"], "o-", color=color, lw=2, ms=5)
            if grp["std"].notna().any():
                ax.fill_between(
                    grp[xcol], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                    alpha=0.2, color=color,
                )
            ax.set_xlabel(name, fontsize=11)
            if idx == 0:
                ax.set_ylabel("Output (mean ± σ)", fontsize=11)
            ax.set_title(f"∂ Output / ∂ {name}", fontsize=10)
            ax.grid(True, alpha=0.4)
        self._note(df, fig)
        fig.tight_layout()
        self._save(fig, self.cfg.plot_file)

    # ── MCMC plots ────────────────────────────────────────────────────────────

    def _plot_mcmc(self, chain_df: pd.DataFrame) -> None:
        cfg         = self.cfg
        param_names = cfg.mcmc_names
        n_walkers   = cfg.mcmc_num_walkers
        burnin      = cfg.mcmc_burnin
        stem        = self.cfg.plot_file.with_suffix("")
        colors_w    = cm.rainbow(np.linspace(0, 1, n_walkers))

        # Filter to post-burn-in accepted steps for posterior samples
        post_df = chain_df[
            (chain_df["step"] >= burnin) & (chain_df["accepted"] == 1)
        ].copy()

        # 1. Trace plots ───────────────────────────────────────────────────────
        n_p = len(param_names)
        fig, axes = plt.subplots(nrows=n_p, figsize=(12, 3.5 * n_p), sharex=True)
        if n_p == 1:
            axes = [axes]
        for i, pname in enumerate(param_names):
            ax = axes[i]
            for w in range(n_walkers):
                wdf = chain_df[chain_df["walker"] == w].sort_values("step")
                if pname not in wdf.columns:
                    continue
                label = None
                if w == 0 and pname in post_df.columns and not post_df.empty:
                    post_vals = post_df[pname].to_numpy(float)
                    tau_est   = MCMCManager.autocorr_time(post_vals)
                    ess       = max(len(post_vals) / tau_est, 0)
                    label     = f"τ≈{tau_est:.1f}  ESS≈{ess:.0f}"
                ax.plot(wdf["step"], wdf[pname], alpha=0.6,
                        color=colors_w[w], lw=0.8, label=label)
            ax.axvline(burnin, color="k", ls="--", lw=1, alpha=0.5, label="burn-in" if i == 0 else None)
            ax.set_ylabel(pname, fontsize=11)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            ax.grid(True, alpha=0.35)
        axes[-1].set_xlabel("Chain step", fontsize=11)
        axes[0].set_title("MCMC Trace Plots (all walkers)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        trace_path = stem.parent / (stem.name + "_mcmc_traces.png")
        self._save(fig, trace_path)

        # 2. χ² history ────────────────────────────────────────────────────────
        all_lp   = chain_df["log_prob"].to_numpy(float)
        steps_all = chain_df["step"].to_numpy(float)
        post_lp  = post_df["log_prob"].to_numpy(float) if not post_df.empty else np.array([])
        steps_post = post_df["step"].to_numpy(float)   if not post_df.empty else np.array([])
        chi2_all  = -2.0 * all_lp
        chi2_post = -2.0 * post_lp

        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        ax = axs[0]
        ax.scatter(steps_all, chi2_all, c="k", s=4, alpha=0.2, label="all visited")
        if len(chi2_post):
            ax.scatter(steps_post, chi2_post, c="r", s=4, alpha=0.4, label="accepted (post burn-in)")
        ax.axvline(burnin, color="blue", ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Chain step"); ax.set_ylabel(r"$\chi^2 = -2\ln p$")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax = axs[1]
        finite = chi2_all[np.isfinite(chi2_all)]
        if len(finite):
            lo_p, hi_p = np.percentile(finite, [1, 99])
            rng_  = (lo_p, hi_p)
            ax.hist(finite, bins=50, range=rng_, color="k", density=True,
                    histtype="step", label="all visited")
        if len(chi2_post):
            fp = chi2_post[np.isfinite(chi2_post)]
            if len(fp):
                ax.hist(fp, bins=50, color="r", density=True, histtype="step",
                        label=f"post-burnin accepted  mean={fp.mean():.2f}")
        ax.set_xlabel(r"$\chi^2$"); ax.set_yticks([])
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.suptitle(r"$\chi^2$ exploration during MCMC", fontsize=13, fontweight="bold")
        fig.tight_layout()
        chi2_path = stem.parent / (stem.name + "_mcmc_chi2.png")
        self._save(fig, chi2_path)

        # 3. Corner plot ───────────────────────────────────────────────────────
        if not post_df.empty and all(p in post_df.columns for p in param_names):
            samples_flat = post_df[param_names].to_numpy(float)
            if _CORNER_AVAILABLE:
                fig = plt.figure(figsize=(7, 7))
                _corner_mod.corner(
                    samples_flat,
                    labels=param_names,
                    color="steelblue",
                    fig=fig,
                    bins=40,
                    show_titles=True,
                    title_fmt=".4f",
                    smooth=1.0,
                )
                fig.suptitle("Posterior parameter distribution", fontsize=13, fontweight="bold")
                fig.tight_layout()
                corner_path = stem.parent / (stem.name + "_mcmc_corner.png")
                self._save(fig, corner_path)
            else:
                # Fallback: seaborn pairplot-style
                n_p_ = len(param_names)
                fig, axes = plt.subplots(n_p_, n_p_, figsize=(4 * n_p_, 4 * n_p_))
                for i, pi in enumerate(param_names):
                    for j, pj in enumerate(param_names):
                        ax = axes[i][j]
                        if i == j:
                            ax.hist(samples_flat[:, i], bins=30, color="steelblue",
                                    density=True, edgecolor="white")
                            ax.set_xlabel(pi, fontsize=9)
                        elif i > j:
                            ax.scatter(samples_flat[:, j], samples_flat[:, i],
                                       s=3, alpha=0.3, color="steelblue")
                            if j == 0:
                                ax.set_ylabel(pi, fontsize=9)
                            if i == n_p_ - 1:
                                ax.set_xlabel(pj, fontsize=9)
                        else:
                            ax.set_visible(False)
                fig.suptitle("Posterior distribution (install `corner` for a nicer plot)",
                             fontsize=11)
                fig.tight_layout()
                corner_path = stem.parent / (stem.name + "_mcmc_corner.png")
                self._save(fig, corner_path)

        # 4. Convergence diagnostics ──────────────────────────────────────────
        all_steps_sorted = sorted(chain_df["step"].unique())
        post_steps = [s for s in all_steps_sorted if s >= burnin]
        if len(post_steps) > 10 and len(param_names) > 0:
            checkpoints = post_steps[::max(1, len(post_steps) // 10)]
            tau_history:  List[np.ndarray] = []
            gr_history:   List[np.ndarray] = []
            ess_history:  List[np.ndarray] = []
            ckpt_labels: List[int] = []
            for ckpt in checkpoints:
                sub = chain_df[
                    (chain_df["step"] >= burnin) &
                    (chain_df["step"] <= ckpt) &
                    (chain_df["accepted"] == 1)
                ]
                if sub.empty:
                    continue
                n_s   = ckpt - burnin + 1
                chains_3d = np.zeros((n_s, n_walkers, len(param_names)))
                for w in range(n_walkers):
                    wdf = sub[sub["walker"] == w].sort_values("step")
                    for pi, pname in enumerate(param_names):
                        if pname in wdf.columns:
                            vals = wdf[pname].to_numpy(float)
                            chains_3d[:len(vals), w, pi] = vals
                taus  = np.array([MCMCManager.autocorr_time(chains_3d[:, w, pi].ravel())
                                  for pi in range(len(param_names))])
                esss  = np.array([n_walkers * n_s / max(taus[pi], 1)
                                  for pi in range(len(param_names))])
                gr    = MCMCManager.gelman_rubin(chains_3d)
                tau_history.append(taus)
                ess_history.append(esss)
                gr_history.append(gr)
                ckpt_labels.append(ckpt)

            if tau_history:
                metrics   = [("τ (autocorr time)", tau_history),
                             ("ESS", ess_history),
                             ("Gelman-Rubin R̂", gr_history)]
                param_clrs = cm.tab10(np.linspace(0, 0.9, len(param_names)))
                fig, axes = plt.subplots(len(metrics), 1,
                                         figsize=(10, 4 * len(metrics)), sharex=True)
                for ax, (label, hist) in zip(axes, metrics):
                    arr = np.array(hist)  # (n_ckpts, n_params)
                    for pi, pname in enumerate(param_names):
                        ax.plot(ckpt_labels, arr[:, pi], "-o", ms=5,
                                color=param_clrs[pi], label=pname)
                    ax.set_ylabel(label, fontsize=10)
                    ax.legend(fontsize=8); ax.grid(True, alpha=0.35)
                axes[-1].set_xlabel("Chain step", fontsize=11)
                axes[0].set_title("MCMC Convergence Diagnostics", fontsize=13, fontweight="bold")
                fig.tight_layout()
                conv_path = stem.parent / (stem.name + "_mcmc_convergence.png")
                self._save(fig, conv_path)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def plot(self, df: Optional[pd.DataFrame] = None,
             chain_df: Optional[pd.DataFrame] = None) -> None:
        if self.cfg.sweep_mode == "mcmc":
            if chain_df is None:
                if self.cfg.mcmc_chain_csv.exists():
                    chain_df = pd.read_csv(self.cfg.mcmc_chain_csv)
                    self._log.info("Loaded chain from %s", self.cfg.mcmc_chain_csv)
                else:
                    self._log.error("No MCMC chain CSV found. Run --mcmc first.")
                    return
            self._plot_mcmc(chain_df)
            return

        if df is None or df.empty:
            self._log.error("Empty DataFrame.")
            return
        n = len(self.cfg.swept_names)
        if n == 0:
            self._log.warning("No swept params.")
        elif n == 1:
            self._plot_1d(df)
        elif self.cfg.is_2d_grid:
            self._plot_2d(df)
        else:
            self._plot_nd(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration factory
# ═══════════════════════════════════════════════════════════════════════════════

def _make_values(p: Dict[str, Any]) -> Optional[np.ndarray]:
    if "values" in p:
        arr = np.asarray(p["values"], dtype=float)
        return arr if len(arr) > 0 else None
    if "start" in p and "stop" in p and "num" in p:
        fn = np.logspace if p.get("log_scale") else np.linspace
        return fn(p["start"], p["stop"], p["num"])
    return None


def _validate_input_fn(name: str, fn: Any) -> Optional[Callable]:
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(f"input_fn for '{name}' must be callable, got {type(fn)}")
    sig    = inspect.signature(fn)
    n_args = len(sig.parameters)
    has_var = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if n_args < 2 and not has_var:
        raise TypeError(
            f"input_fn for '{name}' must accept at least (case_dir, value), got {n_args} args"
        )
    return fn


def _validate_analysis_fn(fn: Any) -> Callable:
    if not callable(fn):
        raise TypeError(f"ANALYSIS_FNS entry must be callable, got {type(fn)}")
    return fn


def build_config() -> SweepConfig:
    if not PARAMS:
        raise ValueError("PARAMS list is empty.")

    param_specs: List[ParamSpec] = []
    for p in PARAMS:
        name       = str(p["name"])
        default    = float(p.get("default", 0.0))
        values     = _make_values(p)
        fn         = _validate_input_fn(name, p.get("input_fn"))
        coupled_to = p.get("coupled_to")
        coupled_fn = p.get("coupled_fn")
        if coupled_to is not None:
            coupled_to = str(coupled_to)
        if coupled_fn is not None and not callable(coupled_fn):
            raise TypeError(f"coupled_fn for '{name}' must be callable.")
        # MCMC fields
        mcmc_pl = p.get("mcmc_prior_low")
        mcmc_ph = p.get("mcmc_prior_high")
        mcmc_ic = p.get("mcmc_init_center")
        mcmc_iw = p.get("mcmc_init_width")

        param_specs.append(ParamSpec(
            name             = name,
            default          = default,
            values           = values,
            input_fn         = fn,
            coupled_to       = coupled_to,
            coupled_fn       = coupled_fn,
            mcmc_prior_low   = float(mcmc_pl) if mcmc_pl is not None else None,
            mcmc_prior_high  = float(mcmc_ph) if mcmc_ph is not None else None,
            mcmc_init_center = float(mcmc_ic) if mcmc_ic is not None else None,
            mcmc_init_width  = float(mcmc_iw) if mcmc_iw is not None else None,
        ))

    all_param_names = {s.name for s in param_specs}
    for s in param_specs:
        if s.coupled_to is not None and s.coupled_to not in all_param_names:
            raise ValueError(
                f"Param '{s.name}' coupled_to='{s.coupled_to}' which is not "
                f"defined in PARAMS. Known: {sorted(all_param_names)}"
            )

    swept_specs = [s for s in param_specs if s.is_swept]

    mode = SWEEP_MODE.lower()
    if mode not in ("grid", "coupled", "mcmc"):
        raise ValueError(f"SWEEP_MODE must be 'grid', 'coupled', or 'mcmc', got {SWEEP_MODE!r}")

    analysis_fns = [_validate_analysis_fn(f) for f in ANALYSIS_FNS]
    root = Path(SWEEP_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    return SweepConfig(
        param_specs          = param_specs,
        swept_specs          = swept_specs,
        sweep_mode           = mode,
        template_files       = TEMPLATE_FILES,
        submit_cmd_template  = SUBMIT_CMD,
        output_file          = OUTPUT_FILE,
        output_regex         = OUTPUT_REGEX,
        sweep_root           = root,
        submission_log       = Path(SUBMISSION_LOG),
        results_csv          = Path(RESULTS_CSV),
        plot_file            = Path(PLOT_FILE),
        analysis_fns         = analysis_fns,
        mcmc_log_prob_regex  = MCMC_LOG_PROB_REGEX,
        mcmc_num_walkers     = MCMC_NUM_WALKERS,
        mcmc_num_iters       = MCMC_NUM_ITERS,
        mcmc_burnin          = MCMC_BURNIN,
        mcmc_stretch_a       = MCMC_STRETCH_A,
        mcmc_poll_interval   = MCMC_POLL_INTERVAL,
        mcmc_job_timeout     = MCMC_JOB_TIMEOUT,
        mcmc_use_dependency  = MCMC_USE_DEPENDENCY,
        mcmc_chain_csv       = Path(MCMC_CHAIN_CSV),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SweepOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class SweepOrchestrator:
    """High-level controller: submit → harvest → plot → analyse (grid/coupled)."""

    def __init__(self, cfg: SweepConfig) -> None:
        self.cfg      = cfg
        self.grid_mgr = GridManager(cfg)
        self._log     = logging.getLogger("cluster_sweep.orchestrator")

    def submit(self, dry_run: bool = False) -> None:
        total = self.grid_mgr.total_points()
        self._log.info(
            "Sweep: mode=%s  swept=%s  total=%d  dry_run=%s",
            self.cfg.sweep_mode, self.cfg.swept_names or ["(none)"], total, dry_run,
        )
        submitted = errors = 0
        with ClusterDispatcher(self.cfg, dry_run=dry_run) as dispatcher:
            for i, gp in enumerate(self.grid_mgr.iter_grid(), 1):
                self._log.info("[%d/%d] %s", i, total, gp.case_dir_name)
                case_dir = self.grid_mgr.prepare_case(gp)
                job_id   = dispatcher.dispatch(gp, case_dir)
                (submitted if job_id is not None else errors)  # noqa: B018
                if job_id is not None:
                    submitted += 1
                else:
                    errors += 1
        self._log.info("Done: %d submitted, %d errors", submitted, errors)

    def harvest(self) -> pd.DataFrame:
        return ResultHarvester(self.cfg, self.grid_mgr).harvest()

    def plot(self, df: Optional[pd.DataFrame] = None) -> None:
        df_ = self._load_df(df)
        if df_ is not None:
            ResultPlotter(self.cfg).plot(df_)

    def analyse(self, df: Optional[pd.DataFrame] = None) -> None:
        df_ = self._load_df(df)
        if df_ is not None:
            AnalysisDispatcher(self.cfg).run_all(df_)

    def _load_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is not None:
            return df
        if not self.cfg.results_csv.exists():
            self._log.error("No results CSV at %s. Run --harvest first.", self.cfg.results_csv)
            return None
        df = pd.read_csv(self.cfg.results_csv)
        self._log.info("Loaded %d rows from %s", len(df), self.cfg.results_csv)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parametric sweep + Bayesian MCMC cluster orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--submit",  action="store_true",
                        help="Prepare case directories and dispatch grid/coupled jobs.")
    parser.add_argument("--harvest", action="store_true",
                        help="Scrape completed output files → results CSV.")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate plots (adapts to current SWEEP_MODE).")
    parser.add_argument("--analyse", action="store_true",
                        help="Run all registered ANALYSIS_FNS on results CSV.")
    parser.add_argument("--mcmc",    action="store_true",
                        help="Run the full MCMC state machine (blocking).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare directories but do NOT call the scheduler.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not any([args.submit, args.harvest, args.plot, args.analyse, args.mcmc]):
        log.error("No action specified.  Use --submit / --harvest / --plot / --analyse / --mcmc.")
        sys.exit(1)

    cfg          = build_config()
    orchestrator = SweepOrchestrator(cfg)

    # ── MCMC mode ─────────────────────────────────────────────────────────────
    if args.mcmc:
        if cfg.sweep_mode != "mcmc":
            log.error("Set SWEEP_MODE = 'mcmc' in the config block before using --mcmc.")
            sys.exit(1)
        manager  = MCMCManager(cfg, dry_run=args.dry_run)
        chain_df = manager.run()
        if args.plot:
            ResultPlotter(cfg).plot(chain_df=chain_df)
        return

    # ── Grid / coupled sweep ───────────────────────────────────────────────────
    if args.submit:
        orchestrator.submit(dry_run=args.dry_run)

    df: Optional[pd.DataFrame] = None
    if args.harvest:
        df = orchestrator.harvest()
    if args.plot:
        # In mcmc mode without --mcmc flag, delegate to MCMC plot path
        if cfg.sweep_mode == "mcmc":
            ResultPlotter(cfg).plot()
        else:
            orchestrator.plot(df)
    if args.analyse:
        orchestrator.analyse(df)


if __name__ == "__main__":
    main()
