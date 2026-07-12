"""Core parameter data structures.

These classes are migrated verbatim (logic-wise) from the legacy
``cluster_sweep``-style script; the grid/coupled/MCMC semantics they encode
must not change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._+-]")


def descriptive_name(
    prefix: str,
    params: Dict[str, float],
    names: Optional[List[str]] = None,
    max_params: int = 6,
) -> str:
    """Build a filesystem-safe, human-readable case/job name.

    Produces ``f"{prefix}__{n1}_{v1:.6g}_{n2}_{v2:.6g}..."`` using only
    *names* (or every key in *params*, insertion order, if *names* is
    ``None``), capped at *max_params* entries. Mirrors the formatting
    convention of :attr:`GridPoint.case_dir_name` (``%.6g`` values, ``/``
    replaced with ``_``), plus a sweep of any remaining filesystem-unsafe
    characters (whitespace or anything outside ``[A-Za-z0-9._+-]``) to
    ``_`` so the result is always a safe single path component.
    """
    keys = list(names) if names is not None else list(params.keys())
    keys = keys[:max_params]
    rest = "_".join(f"{n}_{params[n]:.6g}" for n in keys if n in params)
    name = f"{prefix}__{rest}" if rest else prefix
    name = name.replace("/", "_")
    return _UNSAFE_CHARS_RE.sub("_", name)


@dataclass
class ParamSpec:
    """One simulation parameter: sweep axis, coupling and/or MCMC prior."""

    name: str
    default: float
    values: Optional[np.ndarray]
    input_fn: Optional[Callable]
    coupled_to: Optional[str] = None
    coupled_fn: Optional[Callable] = None
    # MCMC fields (None -> param is fixed in MCMC, swept in grid/coupled)
    mcmc_prior_low: Optional[float] = None
    mcmc_prior_high: Optional[float] = None
    mcmc_init_center: Optional[float] = None
    mcmc_init_width: Optional[float] = None

    @property
    def is_swept(self) -> bool:
        return (
            self.values is not None
            and len(self.values) > 0
            and self.coupled_to is None
        )

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
    """One fully-resolved parameter combination (a single simulation case)."""

    params: Dict[str, float]
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

    step: int
    walker: int
    params: Dict[str, float]  # only MCMC params
    log_prob: float
    accepted: bool
    case_dir: str
    job_id: Optional[str] = None
