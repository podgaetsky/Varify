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

from varify.src.common.casebuilder import CaseBuilder
from varify.src.common.config import FrameworkConfig
from varify.src.analysis.parser import ResultParser
from varify.src.analysis.postprocess import curve_loss
from varify.src.slurm.dispatcher import SlurmDispatcher


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
        job_id: Optional[str] = None,
        dispatcher: Optional[SlurmDispatcher] = None,
    ) -> float:
        """Block until the output file appears, then parse the metric (NaN on
        timeout or parse failure).

        If both *job_id* and *dispatcher* are given, first block on the
        scheduler's own completion state via
        :meth:`SlurmDispatcher.wait_for_completion`; a non-COMPLETED
        terminal state (or scheduler timeout) short-circuits to NaN without
        bothering to poll for the output file. Omitting either argument
        preserves the previous output-file-only behavior.

        When ``cfg.optimizer.postprocess`` is enabled, the metric is instead
        the interpolated-curve loss against the configured experimental
        dataset (see :meth:`_postprocess_case`) rather than a regex-parsed
        scalar. That metric is an error — lower is better — which composes
        naturally with the ``maximize: false`` default.
        """
        if job_id is not None and dispatcher is not None:
            ok = dispatcher.wait_for_completion(job_id, timeout, poll_interval)
            if not ok:
                self._log.warning(
                    "Job %s did not complete for %s", job_id, case_dir.name,
                )
                return float("nan")
        ok = self.parser.wait_for_output(case_dir, timeout, poll_interval)
        if not ok:
            self._log.warning("Timed out waiting for output in %s", case_dir.name)
            return float("nan")
        if self.cfg.optimizer.postprocess:
            return self._postprocess_case(case_dir)
        return self.parser.parse_case(case_dir, regex=regex)

    def _postprocess_case(self, case_dir: Path) -> float:
        """Score *case_dir* via interpolated-curve loss against experiment.

        The interpolation (spline/linear) and loss (mse/rmse/mae/huber/chi2,
        or a user-supplied ``loss_fn`` hook) are selected from
        ``cfg.optimizer``; see :func:`~src.analysis.postprocess.curve_loss`.

        Returns NaN (with a logged error) if no ``experimental_data`` path
        is configured.
        """
        opt = self.cfg.optimizer
        if opt.experimental_data is None:
            self._log.error(
                "postprocess is enabled but optimizer.experimental_data is "
                "not set; returning NaN for %s", case_dir.name,
            )
            return float("nan")
        loss = opt.loss_fn or opt.loss
        loss_kwargs = {"delta": opt.huber_delta} if loss == "huber" else None
        return curve_loss(
            case_dir,
            opt.experimental_data,
            opt.sim_output_file,
            loss=loss,
            interp=opt.interp,
            k=opt.spline_k,
            s=opt.spline_s,
            y_err_col=opt.experimental_err_col,
            loss_kwargs=loss_kwargs,
        )

    # ── Strategy (subclass responsibility) ────────────────────────────────────

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute the full optimization; return the history/chain DataFrame."""
