"""Shared optimizer execution loop.

Both the MCMC sampler and the gradient-free solver follow the same cycle:

    generate parameter set → build case dir → submit SLURM job →
    wait for the output file → regex-parse the objective metric →
    decide the next step / compute the posterior.

``BaseOptimizer`` provides the SLURM-facing half of that loop (case
building, dispatch, blocking wait, metric parsing); subclasses implement
the parameter-generation strategy in ``run``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.common.casebuilder import CaseBuilder
from src.common.config import FrameworkConfig
from src.analysis.parser import ResultParser
from src.slurm.dispatcher import SlurmDispatcher


class BaseOptimizer(ABC):
    """Common SLURM-evaluation machinery for all optimizers."""

    def __init__(self, cfg: FrameworkConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.builder = CaseBuilder(cfg)
        self.parser = ResultParser(cfg)
        self._log = logging.getLogger(f"varify.optimizer.{type(self).__name__}")

    # ── SLURM-facing evaluation primitives ────────────────────────────────────

    def _dispatch_case(
        self,
        case_name: str,
        job_name: str,
        params: Dict[str, float],
        dispatcher: SlurmDispatcher,
        dependency_job_id: Optional[str] = None,
    ) -> Tuple[Path, Optional[str]]:
        """Prepare the case directory and fire one scheduler job."""
        case_dir = self.cfg.sweep_root / case_name
        self.builder.build(case_dir, params, job_name)
        job_id = dispatcher.dispatch(job_name, case_dir, params, dependency_job_id)
        return case_dir, job_id

    def _wait_and_parse(
        self,
        case_dir: Path,
        regex: str,
        timeout: float,
        poll_interval: float,
    ) -> float:
        """Block until the output file appears, then parse the metric (NaN on
        timeout or parse failure)."""
        ok = self.parser.wait_for_output(case_dir, timeout, poll_interval)
        if not ok:
            self._log.warning("Timed out waiting for output in %s", case_dir.name)
            return float("nan")
        return self.parser.parse_case(case_dir, regex=regex)

    # ── Strategy (subclass responsibility) ────────────────────────────────────

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute the full optimization; return the history/chain DataFrame."""
