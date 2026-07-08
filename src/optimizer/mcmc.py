"""Ensemble Metropolis-Hastings MCMC orchestrator (SLURM-executed likelihood).

Migrated from the legacy ``MCMCManager`` with the sampling mathematics
preserved verbatim:

* emcee-style Goodman-Weare stretch-move proposal: for walker *k* the
  proposal is ``x' = x_j + z * (x_k - x_j)`` with *j* drawn from the
  complementary half-ensemble and ``z ~ g(z) ∝ 1/√z`` on ``[1/a, a]``;
  acceptance probability ``min(1, z^(d-1) * p(x')/p(x))``.
* Uniform (top-hat) priors from each parameter's ``mcmc.prior_low/high``;
  proposals outside the prior are rejected without running a job.
* One SLURM job per walker per step; the log-probability is regex-parsed
  from the finished job's output file.
* Every step is flushed to ``mcmc.chain_csv``; the chain resumes from the
  last committed step after an interruption.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from varify.src.analysis.diagnostics import autocorr_time, gelman_rubin
from varify.src.common.config import FrameworkConfig
from varify.src.common.params import MCMCStep
from varify.src.optimizer.base import BaseOptimizer
from varify.src.slurm.dispatcher import SlurmDispatcher


class MCMCOptimizer(BaseOptimizer):
    """Walker-parallel stretch-move MCMC over SLURM-evaluated log-probabilities."""

    # Re-exported diagnostics (legacy API compatibility)
    gelman_rubin = staticmethod(gelman_rubin)
    autocorr_time = staticmethod(autocorr_time)

    def __init__(self, cfg: FrameworkConfig, dry_run: bool = False) -> None:
        super().__init__(cfg, dry_run=dry_run)
        self._rng = np.random.default_rng()

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

    # ── Stretch-move proposal (legacy-verbatim) ───────────────────────────────

    def _stretch_move(
        self, current: np.ndarray, complement: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Return (proposed_vector, log_acceptance_correction).

        log_q = (d - 1) * log(z)  where d = dim(parameter space).
        """
        d = len(current)
        a = self.cfg.mcmc.stretch_a
        # Draw z ~ g(z) ∝ 1/√z on [1/a, a]  (inverse CDF method)
        u = self._rng.uniform(0, 1)
        z = (1.0 + u * (a - 1.0 / a) + (1.0 / a - 1)) ** 2 / a
        # Pick a random walker from the complementary ensemble
        j_idx = self._rng.integers(0, len(complement))
        x_j = complement[j_idx]
        x_prop = x_j + z * (current - x_j)
        log_q = (d - 1) * math.log(z)
        return x_prop, log_q

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_chain(self) -> pd.DataFrame:
        if self.cfg.mcmc.chain_csv.exists():
            try:
                df = pd.read_csv(self.cfg.mcmc.chain_csv)
                self._log.info(
                    "Resumed chain from %s (%d rows)",
                    self.cfg.mcmc.chain_csv, len(df),
                )
                return df
            except Exception as exc:
                self._log.warning(
                    "Could not load chain CSV: %s — starting fresh", exc
                )
        return pd.DataFrame()

    def _append_step(self, step: MCMCStep) -> None:
        row: Dict[str, Any] = {
            "step": step.step,
            "walker": step.walker,
            "accepted": int(step.accepted),
            "log_prob": step.log_prob,
            "case_dir": step.case_dir,
            "job_id": step.job_id or "",
        }
        row.update(step.params)
        df_new = pd.DataFrame([row])
        write_header = not self.cfg.mcmc.chain_csv.exists()
        df_new.to_csv(
            self.cfg.mcmc.chain_csv, mode="a", header=write_header, index=False
        )

    # ── Initial ensemble ──────────────────────────────────────────────────────

    def _initial_ensemble(self) -> np.ndarray:
        """Build (n_walkers × n_params) initial ensemble.

        Walkers are scattered uniformly within ± init_width around
        init_center, clipped to [prior_low, prior_high].
        """
        specs = self.cfg.mcmc_specs
        n_w = self.cfg.mcmc.num_walkers
        n_p = len(specs)
        ensemble = np.empty((n_w, n_p))
        for j, spec in enumerate(specs):
            lo = spec.mcmc_prior_low
            hi = spec.mcmc_prior_high
            ctr = spec.mcmc_init_center if spec.mcmc_init_center is not None \
                else spec.default
            width = spec.mcmc_init_width if spec.mcmc_init_width is not None \
                else (hi - lo) * 0.1  # type: ignore[operator]
            vals = ctr + self._rng.uniform(-width, width, n_w)
            ensemble[:, j] = np.clip(vals, lo, hi)
        return ensemble

    # ── Single-walker job ─────────────────────────────────────────────────────

    def _mcmc_dir(self, step: int, walker: int) -> Path:
        return self.cfg.sweep_root / f"mcmc_step{step:06d}_w{walker:04d}"

    def _dispatch_walker(
        self,
        step: int,
        walker: int,
        params_vec: np.ndarray,
        dispatcher: SlurmDispatcher,
        dependency_job_id: Optional[str] = None,
    ) -> Tuple[Path, Optional[str]]:
        """Prepare case dir and fire the cluster job for one walker."""
        full_params = self.cfg.defaults.copy()
        for spec, val in zip(self.cfg.mcmc_specs, params_vec):
            full_params[spec.name] = float(val)
        case_dir = self._mcmc_dir(step, walker)
        job_name = f"mcmc_s{step}_w{walker}"
        self.builder.build(case_dir, full_params, job_name)
        job_id = dispatcher.dispatch(
            job_name, case_dir, full_params, dependency_job_id
        )
        return case_dir, job_id

    # ── Main MCMC loop (legacy-verbatim control flow) ─────────────────────────

    def run(self) -> pd.DataFrame:
        """Execute the full MCMC state machine and return the chain DataFrame.

        This method blocks; it exits when ``mcmc.num_iters`` steps per walker
        (plus burn-in) have been accumulated.
        """
        cfg = self.cfg
        specs = cfg.mcmc_specs
        n_walkers = cfg.mcmc.num_walkers
        n_iters = cfg.mcmc.num_iters + cfg.mcmc.burnin
        n_params = len(specs)
        param_names = cfg.mcmc_names
        poll = cfg.mcmc.poll_interval

        if n_params == 0:
            self._log.error(
                "No MCMC params defined. Add mcmc.prior_low/prior_high to "
                "parameters in config.yaml."
            )
            return pd.DataFrame()
        if n_walkers < 2 * n_params:
            self._log.warning(
                "mcmc.num_walkers=%d < 2 × n_params=%d; emcee recommends ≥ 2×.",
                n_walkers, n_params,
            )

        chain_df = self._load_chain()

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
                    last_row = walker_rows.iloc[-1]
                    ensemble[w] = np.array(
                        [float(last_row[n]) for n in param_names]
                    )
                    log_probs[w] = float(last_row["log_prob"])
            self._log.info("Resuming from step %d", start_step)
        else:
            start_step = 0
            ensemble = self._initial_ensemble()
            log_probs = np.full(n_walkers, -math.inf)
            # Evaluate log-probs for the initial ensemble (step -1)
            self._log.info(
                "Evaluating initial ensemble (%d walkers)…", n_walkers
            )
            with SlurmDispatcher(cfg, dry_run=self.dry_run) as dispatcher:
                init_dirs: List[Path] = []
                init_ids: List[Optional[str]] = []
                for w in range(n_walkers):
                    cdir, jid = self._dispatch_walker(
                        -1, w, ensemble[w], dispatcher
                    )
                    init_dirs.append(cdir)
                    init_ids.append(jid)
                for w in range(n_walkers):
                    if not self.dry_run:
                        ok = self.parser.wait_for_output(
                            init_dirs[w], cfg.mcmc.job_timeout, poll
                        )
                        if ok:
                            log_probs[w] = self.parser.parse_log_prob(init_dirs[w])
                        else:
                            self._log.warning("Walker %d init timed out", w)
                    else:
                        log_probs[w] = 0.0  # dry-run placeholder

        n_accepted = np.zeros(n_walkers, dtype=int)
        n_total = np.zeros(n_walkers, dtype=int)

        with SlurmDispatcher(cfg, dry_run=self.dry_run) as dispatcher:
            for step in range(start_step, n_iters):
                self._log.info("── MCMC step %d / %d ──", step + 1, n_iters)
                proposed = np.empty_like(ensemble)
                log_q = np.empty(n_walkers)
                case_dirs: List[Path] = []
                job_ids: List[Optional[str]] = []

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
                        log_q[w] = -math.inf
                        case_dirs.append(self._mcmc_dir(step, w))
                        job_ids.append(None)
                    else:
                        proposed[w] = x_prop
                        log_q[w] = lq
                        cdir, jid = self._dispatch_walker(
                            step, w, x_prop, dispatcher
                        )
                        case_dirs.append(cdir)
                        job_ids.append(jid)

                # ── Poll / harvest all walkers ─────────────────────────────────
                proposed_lps = np.full(n_walkers, -math.inf)

                for w in range(n_walkers):
                    if log_q[w] == -math.inf:
                        # Prior violation — no job was run
                        continue
                    if self.dry_run:
                        proposed_lps[w] = 0.0
                        continue
                    ok = self.parser.wait_for_output(
                        case_dirs[w], cfg.mcmc.job_timeout, poll
                    )
                    if ok:
                        proposed_lps[w] = self.parser.parse_log_prob(case_dirs[w])
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
                        ensemble[w] = proposed[w]
                        log_probs[w] = proposed_lps[w]
                        n_accepted[w] += 1

                    params_dict = {
                        spec.name: float(ensemble[w][j])
                        for j, spec in enumerate(specs)
                    }
                    mcmc_step = MCMCStep(
                        step=step,
                        walker=w,
                        params=params_dict,
                        log_prob=log_probs[w],
                        accepted=accept,
                        case_dir=str(case_dirs[w]),
                        job_id=job_ids[w],
                    )
                    self._append_step(mcmc_step)

                acc_rates = n_accepted / np.maximum(n_total, 1)
                self._log.info(
                    "Step %d done. Mean acceptance rate: %.3f",
                    step, float(acc_rates.mean()),
                )

        final_df = pd.read_csv(cfg.mcmc.chain_csv)
        self._log.info(
            "MCMC complete. Chain: %d rows. Acceptance rates: %s",
            len(final_df),
            [f"{r:.3f}" for r in (n_accepted / np.maximum(n_total, 1))],
        )
        return final_df
