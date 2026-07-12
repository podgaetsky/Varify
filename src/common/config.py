"""YAML configuration loading and validation.

The framework is driven by a single ``config/config.yaml`` file.  Callables
(per-parameter ``input_fn``, ``coupled_fn`` and the registered analysis
functions) cannot live in YAML, so they are defined in a plain Python *hooks
module* (default ``config/hooks.py``) and referenced from the YAML by name.

The validation and value-grid construction logic (``_make_values``,
``_validate_input_fn``) is migrated unchanged from the legacy script.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from varify.src.common.params import ParamSpec

log = logging.getLogger("varify.config")

DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"

_DEFAULT_SUBMIT_CMD = (
    "sbatch "
    "--job-name={job_name} "
    "--output=stdout.log "
    "--error=stderr.log "
    "run_script.sh"
)


# ═════════════════════════════════════════════════════════════════════════════
#  Sub-configuration dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SlurmSettings:
    """Scheduler command templates and sbatch directives."""

    submit_cmd: str = _DEFAULT_SUBMIT_CMD
    directives: Dict[str, str] = field(default_factory=dict)
    cancel_cmd: str = "scancel {job_id}"
    status_cmd: str = "squeue -h -j {job_id} -o %T"
    sacct_cmd: str = "sacct -j {job_id} -n -o State -X"
    submit_timeout: float = 30.0

    @property
    def effective_submit_cmd(self) -> str:
        """Submit template with SLURM directives expanded.

        If the template contains a ``{directives}`` placeholder the rendered
        ``--key=value`` string replaces it; otherwise the directives are
        inserted right after the scheduler executable (first token).
        """
        rendered = " ".join(f"--{k}={v}" for k, v in self.directives.items())
        if "{directives}" in self.submit_cmd:
            return self.submit_cmd.replace("{directives}", rendered).strip()
        if not rendered:
            return self.submit_cmd
        head, _, tail = self.submit_cmd.partition(" ")
        return f"{head} {rendered} {tail}".strip()


@dataclass
class MCMCSettings:
    """Ensemble stretch-move MCMC configuration (legacy semantics preserved)."""

    log_prob_regex: str = r"LOG_PROB:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    num_walkers: int = 10
    num_iters: int = 500
    burnin: int = 100
    stretch_a: float = 2.0
    poll_interval: float = 30.0
    job_timeout: float = 3600.0
    use_dependency: bool = False
    chain_csv: Path = Path("mcmc_chain.csv")


@dataclass
class OptimizerSettings:
    """Gradient-free (Nelder-Mead) optimizer configuration."""

    method: str = "nelder-mead"
    maximize: bool = False
    max_evaluations: int = 100
    tolerance: float = 1e-4
    objective_regex: Optional[str] = None  # None -> scan output_regex
    poll_interval: float = 30.0
    job_timeout: float = 3600.0
    history_csv: Path = Path("optimization_history.csv")
    postprocess: bool = False
    experimental_data: Optional[Path] = None
    sim_output_file: str = "output.dat"
    spline_k: int = 3
    spline_s: float = 0.0
    loss: str = "mse"
    interp: str = "spline"
    huber_delta: float = 1.0
    experimental_err_col: Optional[int] = None
    loss_fn: Optional[Callable] = None
    de_popsize: int = 15
    de_generations: int = 20
    de_f: float = 0.7
    de_cr: float = 0.9
    de_stall_generations: int = 5
    de_seed: int = 42


@dataclass
class WatchdogSettings:
    """Background failure-monitor configuration."""

    poll_interval: float = 60.0
    stall_timeout: float = 1800.0
    nan_regex: str = r"(?i)\bnan\b"
    log_files: List[str] = field(default_factory=lambda: ["stdout.log", "stderr.log"])
    status_csv: Path = Path("status.csv")
    max_resubmits: int = 2
    resubmit_scaling: Dict[str, float] = field(default_factory=dict)
    agent_directives: Dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisSettings:
    """Post-processing outputs: plots directory, SQLite sink, analysis hooks."""

    plots_dir: Path = Path("results") / "plots"
    sqlite_db: Path = Path("results") / "results.db"
    sqlite_table: str = "results"
    analysis_fns: List[Callable] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
#  Top-level framework configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameworkConfig:
    """Aggregate configuration shared by scanner, optimizers and analysis.

    Property names mirror the legacy ``SweepConfig`` so migrated logic keeps
    working unchanged.
    """

    param_specs: List[ParamSpec]
    swept_specs: List[ParamSpec]
    sweep_mode: str
    template_files: List[str]
    output_file: str
    output_regex: str
    sweep_root: Path
    submission_log: Path
    results_csv: Path
    slurm: SlurmSettings
    mcmc: MCMCSettings
    optimizer: OptimizerSettings
    watchdog: WatchdogSettings
    analysis: AnalysisSettings
    file_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    case_source_dir: Optional[Path] = None
    case_substitute_globs: List[str] = field(default_factory=lambda: ["*"])

    # ── Legacy-compatible derived views ──────────────────────────────────────

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

    # ── Convenience paths ────────────────────────────────────────────────────

    @property
    def jobs_registry_csv(self) -> Path:
        return self.sweep_root / "jobs_registry.csv"

    @property
    def status_csv(self) -> Path:
        p = self.watchdog.status_csv
        return p if p.is_absolute() or p.parent != Path(".") else self.sweep_root / p

    @property
    def analysis_fns(self) -> List[Callable]:
        return self.analysis.analysis_fns

    @property
    def objective_regex(self) -> str:
        return self.optimizer.objective_regex or self.output_regex


# ═════════════════════════════════════════════════════════════════════════════
#  Hooks module loading & callable resolution
# ═════════════════════════════════════════════════════════════════════════════

def _load_hooks_module(path: Path) -> Optional[ModuleType]:
    if not path.exists():
        log.warning("Hooks module not found: %s — callables unavailable.", path)
        return None
    spec = importlib.util.spec_from_file_location("varify_hooks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import hooks module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_callable(
    hooks: Optional[ModuleType], name: Optional[str], context: str
) -> Optional[Callable]:
    if name is None:
        return None
    if hooks is None:
        raise ValueError(
            f"{context} references hook '{name}' but no hooks module is loaded."
        )
    fn = getattr(hooks, str(name), None)
    if fn is None or not callable(fn):
        raise ValueError(f"{context}: hook '{name}' not found or not callable.")
    return fn


# ═════════════════════════════════════════════════════════════════════════════
#  Legacy validators (migrated unchanged)
# ═════════════════════════════════════════════════════════════════════════════

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
    sig = inspect.signature(fn)
    n_args = len(sig.parameters)
    has_var = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if n_args < 2 and not has_var:
        raise TypeError(
            f"input_fn for '{name}' must accept at least (case_dir, value), "
            f"got {n_args} args"
        )
    return fn


def _validate_analysis_fn(fn: Any) -> Callable:
    if not callable(fn):
        raise TypeError(f"analysis_fns entry must be callable, got {type(fn)}")
    return fn


# ═════════════════════════════════════════════════════════════════════════════
#  Parameter parsing
# ═════════════════════════════════════════════════════════════════════════════

def _build_param_specs(
    raw_params: List[Dict[str, Any]], hooks: Optional[ModuleType]
) -> List[ParamSpec]:
    if not raw_params:
        raise ValueError("'parameters' list is empty in the configuration.")

    specs: List[ParamSpec] = []
    for p in raw_params:
        name = str(p["name"])
        default = float(p.get("default", 0.0))
        sweep = dict(p.get("sweep") or {})
        values = _make_values(sweep)
        input_fn = _validate_input_fn(
            name, _resolve_callable(hooks, p.get("input_fn"), f"param '{name}'")
        )
        coupled_to = p.get("coupled_to")
        coupled_fn = _resolve_callable(
            hooks, p.get("coupled_fn"), f"param '{name}' coupled_fn"
        )
        if coupled_to is not None:
            coupled_to = str(coupled_to)
        if coupled_fn is not None and not callable(coupled_fn):
            raise TypeError(f"coupled_fn for '{name}' must be callable.")

        mcmc = dict(p.get("mcmc") or {})
        mcmc_pl = mcmc.get("prior_low")
        mcmc_ph = mcmc.get("prior_high")
        mcmc_ic = mcmc.get("init_center")
        mcmc_iw = mcmc.get("init_width")

        specs.append(ParamSpec(
            name=name,
            default=default,
            values=values,
            input_fn=input_fn,
            coupled_to=coupled_to,
            coupled_fn=coupled_fn,
            mcmc_prior_low=float(mcmc_pl) if mcmc_pl is not None else None,
            mcmc_prior_high=float(mcmc_ph) if mcmc_ph is not None else None,
            mcmc_init_center=float(mcmc_ic) if mcmc_ic is not None else None,
            mcmc_init_width=float(mcmc_iw) if mcmc_iw is not None else None,
        ))

    all_names = {s.name for s in specs}
    for s in specs:
        if s.coupled_to is not None and s.coupled_to not in all_names:
            raise ValueError(
                f"Param '{s.name}' coupled_to='{s.coupled_to}' which is not "
                f"defined in parameters. Known: {sorted(all_names)}"
            )
    return specs


# ═════════════════════════════════════════════════════════════════════════════
#  Loader
# ═════════════════════════════════════════════════════════════════════════════

def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FrameworkConfig:
    """Parse ``config.yaml`` (+ hooks module) into a ``FrameworkConfig``."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    raw: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    hooks_rel = raw.get("hooks_module", "config/hooks.py")
    hooks_path = Path(hooks_rel)
    if not hooks_path.is_absolute():
        hooks_path = (cfg_path.parent.parent / hooks_path).resolve() \
            if not hooks_path.exists() else hooks_path
    hooks = _load_hooks_module(hooks_path)

    scan = dict(raw.get("scan") or {})
    mode = str(scan.get("mode", "grid")).lower()
    if mode not in ("grid", "coupled"):
        raise ValueError(f"scan.mode must be 'grid' or 'coupled', got {mode!r}")

    param_specs = _build_param_specs(list(raw.get("parameters") or []), hooks)
    swept_specs = [s for s in param_specs if s.is_swept]

    slurm_raw = dict(raw.get("slurm") or {})
    slurm = SlurmSettings(
        submit_cmd=str(slurm_raw.get("submit_cmd", _DEFAULT_SUBMIT_CMD)),
        directives={
            str(k): str(v) for k, v in (slurm_raw.get("directives") or {}).items()
        },
        cancel_cmd=str(slurm_raw.get("cancel_cmd", "scancel {job_id}")),
        status_cmd=str(slurm_raw.get("status_cmd", "squeue -h -j {job_id} -o %T")),
        sacct_cmd=str(slurm_raw.get(
            "sacct_cmd", "sacct -j {job_id} -n -o State -X")),
        submit_timeout=float(slurm_raw.get("submit_timeout", 30.0)),
    )

    mcmc_raw = dict(raw.get("mcmc") or {})
    mcmc = MCMCSettings(
        log_prob_regex=str(mcmc_raw.get(
            "log_prob_regex", MCMCSettings.log_prob_regex)),
        num_walkers=int(mcmc_raw.get("num_walkers", 10)),
        num_iters=int(mcmc_raw.get("num_iters", 500)),
        burnin=int(mcmc_raw.get("burnin", 100)),
        stretch_a=float(mcmc_raw.get("stretch_a", 2.0)),
        poll_interval=float(mcmc_raw.get("poll_interval", 30.0)),
        job_timeout=float(mcmc_raw.get("job_timeout", 3600.0)),
        use_dependency=bool(mcmc_raw.get("use_dependency", False)),
        chain_csv=Path(mcmc_raw.get("chain_csv", "mcmc_chain.csv")),
    )

    opt_raw = dict(raw.get("optimizer") or {})
    optimizer = OptimizerSettings(
        method=str(opt_raw.get("method", "nelder-mead")).lower(),
        maximize=bool(opt_raw.get("maximize", False)),
        max_evaluations=int(opt_raw.get("max_evaluations", 100)),
        tolerance=float(opt_raw.get("tolerance", 1e-4)),
        objective_regex=opt_raw.get("objective_regex"),
        poll_interval=float(opt_raw.get("poll_interval", 30.0)),
        job_timeout=float(opt_raw.get("job_timeout", 3600.0)),
        history_csv=Path(opt_raw.get("history_csv", "optimization_history.csv")),
        postprocess=bool(opt_raw.get("postprocess", False)),
        experimental_data=(
            Path(opt_raw["experimental_data"])
            if opt_raw.get("experimental_data") is not None else None
        ),
        sim_output_file=str(opt_raw.get("sim_output_file", "output.dat")),
        spline_k=int(opt_raw.get("spline_k", 3)),
        spline_s=float(opt_raw.get("spline_s", 0.0)),
        loss=str(opt_raw.get("loss", "mse")),
        interp=str(opt_raw.get("interp", "spline")),
        huber_delta=float(opt_raw.get("huber_delta", 1.0)),
        experimental_err_col=(
            int(opt_raw["experimental_err_col"])
            if opt_raw.get("experimental_err_col") is not None else None
        ),
        loss_fn=_resolve_callable(hooks, opt_raw.get("loss_fn"), "optimizer.loss_fn"),
        de_popsize=int(opt_raw.get("de_popsize", 15)),
        de_generations=int(opt_raw.get("de_generations", 20)),
        de_f=float(opt_raw.get("de_f", 0.7)),
        de_cr=float(opt_raw.get("de_cr", 0.9)),
        de_stall_generations=int(opt_raw.get("de_stall_generations", 5)),
        de_seed=int(opt_raw.get("de_seed", 42)),
    )

    wd_raw = dict(raw.get("watchdog") or {})
    watchdog = WatchdogSettings(
        poll_interval=float(wd_raw.get("poll_interval", 60.0)),
        stall_timeout=float(wd_raw.get("stall_timeout", 1800.0)),
        nan_regex=str(wd_raw.get("nan_regex", r"(?i)\bnan\b")),
        log_files=[str(f) for f in (wd_raw.get("log_files")
                                    or ["stdout.log", "stderr.log"])],
        status_csv=Path(wd_raw.get("status_csv", "status.csv")),
        max_resubmits=int(wd_raw.get("max_resubmits", 2)),
        resubmit_scaling={
            str(k): float(v)
            for k, v in (wd_raw.get("resubmit_scaling") or {}).items()
        },
        agent_directives={
            str(k): str(v)
            for k, v in (wd_raw.get("agent_directives") or {}).items()
        },
    )

    ana_raw = dict(raw.get("analysis") or {})
    analysis_fns = [
        _validate_analysis_fn(_resolve_callable(hooks, fn_name, "analysis_fns"))
        for fn_name in (ana_raw.get("analysis_fns") or [])
    ]
    analysis = AnalysisSettings(
        plots_dir=Path(ana_raw.get("plots_dir", Path("results") / "plots")),
        sqlite_db=Path(ana_raw.get("sqlite_db", Path("results") / "results.db")),
        sqlite_table=str(ana_raw.get("sqlite_table", "results")),
        analysis_fns=analysis_fns,
    )

    root = Path(scan.get("workspace", "sweep_workspace"))
    root.mkdir(parents=True, exist_ok=True)

    file_pipeline = [dict(entry) for entry in (raw.get("file_pipeline") or [])]

    case_raw = dict(raw.get("case") or {})
    case_source_dir = (
        Path(case_raw["source_dir"]) if case_raw.get("source_dir") is not None
        else None
    )
    case_substitute_globs = [
        str(g) for g in (case_raw.get("substitute_globs") or ["*"])
    ]

    return FrameworkConfig(
        param_specs=param_specs,
        swept_specs=swept_specs,
        sweep_mode=mode,
        template_files=[str(t) for t in (scan.get("template_files")
                                         or ["input.template", "run_script.sh"])],
        output_file=str(scan.get("output_file", "stdout.log")),
        output_regex=str(scan.get(
            "output_regex",
            r"RESULT:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")),
        sweep_root=root,
        submission_log=Path(scan.get("submission_log", "submission.log")),
        results_csv=Path(scan.get("results_csv", "sweep_results.csv")),
        slurm=slurm,
        mcmc=mcmc,
        optimizer=optimizer,
        watchdog=watchdog,
        analysis=analysis,
        file_pipeline=file_pipeline,
        case_source_dir=case_source_dir,
        case_substitute_globs=case_substitute_globs,
    )
