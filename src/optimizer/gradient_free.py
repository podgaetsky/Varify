"""Gradient-free optimization (Nelder-Mead simplex via ``scipy.optimize``).

The optimizer treats the SLURM-executed simulation as a black-box objective:
each function evaluation builds a case directory, submits one job, blocks
until the output file appears, and regex-parses the objective metric
(``optimizer.objective_regex``, defaulting to the scan ``output_regex``).

Optimization variables are the parameters that carry MCMC prior bounds
(``mcmc.prior_low`` / ``prior_high``), which double as hard box constraints:
out-of-bounds simplex points receive a large penalty instead of a job
submission.  Every evaluation is appended to ``optimizer.history_csv``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from varify.src.common.config import FrameworkConfig
from varify.src.common.params import descriptive_name
from varify.src.optimizer.base import BaseOptimizer
from varify.src.slurm.dispatcher import SlurmDispatcher

_PENALTY: float = 1.0e30


class NelderMeadOptimizer(BaseOptimizer):
    """Sequential Nelder-Mead over SLURM-evaluated simulation objectives."""

    def __init__(self, cfg: FrameworkConfig, dry_run: bool = False) -> None:
        super().__init__(cfg, dry_run=dry_run)
        self._n_evals: int = 0
        self._sign: float = -1.0 if cfg.optimizer.maximize else 1.0

    # ── Search-space definition ───────────────────────────────────────────────

    def _opt_specs(self) -> List[Any]:
        specs = self.cfg.mcmc_specs  # params with prior bounds = search space
        if not specs:
            raise ValueError(
                "No optimization parameters defined: give at least one param "
                "mcmc.prior_low / prior_high bounds in config.yaml."
            )
        return specs

    def _x0(self) -> np.ndarray:
        return np.array([
            s.mcmc_init_center if s.mcmc_init_center is not None else s.default
            for s in self._opt_specs()
        ])

    def _in_bounds(self, x: np.ndarray) -> bool:
        for xi, spec in zip(x, self._opt_specs()):
            assert spec.mcmc_prior_low is not None
            assert spec.mcmc_prior_high is not None
            if not (spec.mcmc_prior_low <= xi <= spec.mcmc_prior_high):
                return False
        return True

    # ── History persistence ───────────────────────────────────────────────────

    def _append_history(
        self,
        params: Dict[str, float],
        objective: float,
        case_dir: str,
        job_id: Optional[str],
    ) -> None:
        row: Dict[str, Any] = {
            "eval": self._n_evals,
            "objective": objective,
            "case_dir": case_dir,
            "job_id": job_id or "",
            "timestamp": time.time(),
        }
        row.update(params)
        path = self.cfg.optimizer.history_csv
        pd.DataFrame([row]).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )

    # ── Objective (one SLURM job per call) ────────────────────────────────────

    def _evaluate(self, x: np.ndarray, dispatcher: SlurmDispatcher) -> float:
        self._n_evals += 1
        specs = self._opt_specs()
        params = self.cfg.defaults.copy()
        for spec, val in zip(specs, x):
            params[spec.name] = float(val)

        if not self._in_bounds(x):
            self._log.info(
                "[eval %04d] out of bounds → penalty", self._n_evals
            )
            self._append_history(params, float("nan"), "", None)
            return _PENALTY

        case_name = descriptive_name(
            f"opt_eval{self._n_evals:05d}", params, [s.name for s in specs]
        )
        job_name = case_name
        case_dir, job_id = self._dispatch_case(
            case_name, job_name, params, dispatcher
        )

        if self.dry_run:
            self._append_history(params, 0.0, str(case_dir), job_id)
            return 0.0

        metric = self._wait_and_parse(
            case_dir,
            self.cfg.objective_regex,
            self.cfg.optimizer.job_timeout,
            self.cfg.optimizer.poll_interval,
        )
        self._append_history(params, metric, str(case_dir), job_id)
        if not np.isfinite(metric):
            self._log.warning(
                "[eval %04d] non-finite metric → penalty", self._n_evals
            )
            return _PENALTY
        value = self._sign * metric
        self._log.info(
            "[eval %04d] %s → objective=%.6g",
            self._n_evals,
            {s.name: f"{v:.5g}" for s, v in zip(specs, x)},
            metric,
        )
        return value

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run the Nelder-Mead search; return the evaluation-history DataFrame."""
        specs = self._opt_specs()
        x0 = self._x0()
        self._log.info(
            "Nelder-Mead over %s  (x0=%s, maximize=%s, max_evals=%d)",
            [s.name for s in specs], x0.tolist(),
            self.cfg.optimizer.maximize, self.cfg.optimizer.max_evaluations,
        )

        with SlurmDispatcher(self.cfg, dry_run=self.dry_run) as dispatcher:
            result = optimize.minimize(
                self._evaluate,
                x0,
                args=(dispatcher,),
                method="Nelder-Mead",
                options={
                    "maxfev": self.cfg.optimizer.max_evaluations,
                    "xatol": self.cfg.optimizer.tolerance,
                    "fatol": self.cfg.optimizer.tolerance,
                    "adaptive": True,
                },
            )

        best_params = {
            spec.name: float(val) for spec, val in zip(specs, result.x)
        }
        best_val = self._sign * float(result.fun)
        self._log.info(
            "Optimization finished after %d evaluations: best=%s  objective=%.6g "
            "(%s)",
            self._n_evals, best_params, best_val, result.message,
        )

        if self.cfg.optimizer.history_csv.exists():
            return pd.read_csv(self.cfg.optimizer.history_csv)
        return pd.DataFrame()
