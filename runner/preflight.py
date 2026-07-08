"""Comprehensive pre-flight environment validator (standard library only).

Runs *before* any cluster allocation is consumed and verifies:

* interpreter version, availability of optional scientific packages;
* dependency paths (``sbatch``/``squeue``/``scancel`` and the sbatch
  template when the run targets SLURM; git for provenance);
* hardware & permissions: results directory writability, free disk space,
  requested worker count vs. available cores, model picklability for
  multiprocessing strategies;
* structural config constraints: bounds ordering, grid axis sizes, MCMC
  walker/dimension rules, benchmark worker lists.

Every check yields PASS / WARN / FAIL; any FAIL aborts the run with a
``PreflightError`` unless explicitly overridden.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from varify.runner.core import RunSpec

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

_MIN_PYTHON: Tuple[int, int] = (3, 10)
_MIN_FREE_MB: int = 200
_OPTIONAL_PACKAGES = ("numpy", "scipy", "emcee", "matplotlib")


class PreflightError(RuntimeError):
    """Raised when one or more FAIL-level checks block the run."""


@dataclass
class CheckResult:
    name: str
    level: str
    message: str


@dataclass
class PreflightReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.level == FAIL for c in self.checks)

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.level == FAIL]

    def add(self, name: str, level: str, message: str) -> None:
        self.checks.append(CheckResult(name, level, message))

    def render(self) -> str:
        width = max((len(c.name) for c in self.checks), default=10)
        lines = [
            f"  [{c.level:4s}] {c.name:<{width}s}  {c.message}"
            for c in self.checks
        ]
        verdict = "READY" if self.ok else "BLOCKED"
        return "Pre-flight report — " + verdict + "\n" + "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "level": c.level, "message": c.message}
                for c in self.checks
            ],
        }


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_python(report: PreflightReport) -> None:
    ver = sys.version_info[:2]
    level = PASS if ver >= _MIN_PYTHON else FAIL
    report.add("python_version", level,
               f"{sys.version.split()[0]} (need ≥ {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]})")


def _check_packages(report: PreflightReport) -> None:
    missing = [
        name for name in _OPTIONAL_PACKAGES
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        report.add("optional_packages", WARN,
                   f"missing {missing} — pure-Python fallbacks will be used")
    else:
        report.add("optional_packages", PASS, "full scientific stack available")


def _check_results_writable(report: PreflightReport, results_root: Path) -> None:
    try:
        results_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=results_root, prefix=".preflight_"):
            pass
        report.add("results_writable", PASS, str(results_root.resolve()))
    except OSError as exc:
        report.add("results_writable", FAIL, f"{results_root}: {exc}")


def _check_disk_space(report: PreflightReport, results_root: Path) -> None:
    try:
        free_mb = shutil.disk_usage(results_root).free / 1e6
    except OSError as exc:
        report.add("disk_space", WARN, f"could not stat: {exc}")
        return
    level = PASS if free_mb >= _MIN_FREE_MB else FAIL
    report.add("disk_space", level,
               f"{free_mb:.0f} MB free (need ≥ {_MIN_FREE_MB} MB)")


def _check_workers(report: PreflightReport, workers: int) -> None:
    cores = os.cpu_count() or 1
    if workers < 1:
        report.add("workers", FAIL, f"requested {workers} (< 1)")
    elif workers > cores:
        report.add("workers", WARN,
                   f"requested {workers} > {cores} available cores")
    else:
        report.add("workers", PASS, f"{workers} of {cores} cores")


def _check_picklable_model(report: PreflightReport, spec: "RunSpec") -> None:
    if spec.model is None:
        report.add("model", FAIL, "no model callable supplied")
        return
    if not callable(spec.model):
        report.add("model", FAIL, f"model is not callable: {type(spec.model)}")
        return
    workers = int(spec.options.get("workers", 1))
    counts = [int(w) for w in spec.options.get("worker_counts", [])]
    needs_pickle = (
        (spec.strategy == "grid" and workers > 1)
        or (spec.strategy == "benchmark" and any(c > 1 for c in counts))
    )
    if needs_pickle:
        try:
            pickle.dumps(spec.model)
            report.add("model_picklable", PASS,
                       "model survives pickling (multiprocessing OK)")
        except Exception as exc:
            report.add("model_picklable", FAIL,
                       f"model not picklable for multiprocessing: {exc}")
    else:
        report.add("model", PASS, getattr(spec.model, "__name__", repr(spec.model)))


def _check_bounds(report: PreflightReport, spec: "RunSpec") -> None:
    if not spec.bounds:
        if spec.strategy in ("optimize", "mcmc", "mcmc_diagnostic"):
            report.add("bounds", FAIL,
                       f"strategy {spec.strategy!r} requires parameter bounds")
        return
    bad = [
        name for name, (lo, hi) in spec.bounds.items() if not lo < hi
    ]
    if bad:
        report.add("bounds", FAIL, f"lo >= hi for {bad}")
    else:
        report.add("bounds", PASS,
                   f"{len(spec.bounds)} parameter(s), all lo < hi")


def _check_strategy_constraints(report: PreflightReport, spec: "RunSpec") -> None:
    opts = spec.options
    if spec.strategy in ("mcmc", "mcmc_diagnostic"):
        ndim = len(spec.bounds)
        walkers = int(opts.get("walkers", max(2 * ndim, 4)))
        if walkers < 2 * ndim or walkers % 2:
            report.add("mcmc_walkers", FAIL,
                       f"walkers={walkers} must be even and ≥ 2×ndim={2 * ndim}")
        else:
            report.add("mcmc_walkers", PASS, f"walkers={walkers}, ndim={ndim}")
    if spec.strategy == "grid":
        axes = opts.get("axes", {})
        empty = [k for k, v in axes.items() if not list(v)]
        if not axes or empty:
            report.add("grid_axes", FAIL,
                       f"axes missing or empty: {empty or 'no axes defined'}")
        else:
            points = 1
            for v in axes.values():
                points *= len(list(v))
            report.add("grid_axes", PASS,
                       f"{len(axes)} axes, {points} total points")
    if spec.strategy == "benchmark":
        counts = [int(w) for w in opts.get("worker_counts", [])]
        if not counts or any(w < 1 for w in counts):
            report.add("benchmark_workers", FAIL,
                       f"invalid worker_counts: {counts}")
        else:
            report.add("benchmark_workers", PASS, f"{counts}")


def _check_slurm(report: PreflightReport, spec: "RunSpec") -> None:
    if spec.runtime != "slurm":
        return
    for exe in ("sbatch", "squeue", "scancel"):
        path = shutil.which(exe)
        report.add(f"slurm_{exe}", PASS if path else FAIL,
                   path or "not found on PATH")
    template = spec.sbatch_template()
    report.add("sbatch_template", PASS if template.exists() else FAIL,
               str(template.resolve()))


def _check_git(report: PreflightReport) -> None:
    path = shutil.which("git")
    report.add("git", PASS if path else WARN,
               path or "not found — provenance will lack commit info")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_preflight(spec: "RunSpec", strict: bool = True) -> PreflightReport:
    """Validate the environment and *spec*; raise ``PreflightError`` on FAIL."""
    report = PreflightReport()
    _check_python(report)
    _check_packages(report)
    _check_results_writable(report, spec.results_root)
    _check_disk_space(report, spec.results_root)
    _check_workers(report, int(spec.options.get("workers", 1)))
    _check_picklable_model(report, spec)
    _check_bounds(report, spec)
    _check_strategy_constraints(report, spec)
    _check_slurm(report, spec)
    _check_git(report)
    if strict and not report.ok:
        raise PreflightError(
            "Pre-flight validation failed:\n" + report.render()
        )
    return report
