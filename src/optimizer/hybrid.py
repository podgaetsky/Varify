"""Hybrid global/local optimizer: Differential Evolution → Nelder-Mead.

Differential Evolution (rand/1/bin) explores the prior box broadly; each DE
generation's trial vectors are submitted to SLURM *concurrently* and waited
on as one batch (never one blocking job at a time — see
:meth:`HybridDEOptimizer._evaluate_batch`).  Once DE stalls or exhausts its
generation budget, a sequential Nelder-Mead refinement (identical to
:class:`~src.optimizer.gradient_free.NelderMeadOptimizer`) polishes the DE
champion using whatever evaluation budget remains.

Search-space definition, bounds, ``x0``, the history CSV schema and the
``maximize`` sign convention are all inherited unchanged from
``NelderMeadOptimizer`` so the two optimizers stay comparable and share one
history/analysis pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from varify.src.common.config import FrameworkConfig
from varify.src.optimizer.gradient_free import NelderMeadOptimizer, _PENALTY
from varify.src.slurm.dispatcher import SlurmDispatcher


class HybridDEOptimizer(NelderMeadOptimizer):
    """Differential Evolution exploration + Nelder-Mead local refinement."""

    def __init__(self, cfg: FrameworkConfig, dry_run: bool = False) -> None:
        super().__init__(cfg, dry_run=dry_run)
        self._rng = np.random.default_rng(cfg.optimizer.de_seed)

    # ── Search-space bounds (arrays, for vectorized DE ops) ───────────────────

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        specs = self._opt_specs()
        low = np.array([s.mcmc_prior_low for s in specs], dtype=float)
        high = np.array([s.mcmc_prior_high for s in specs], dtype=float)
        return low, high

    # ── Batch objective (one SLURM submission wave per call) ──────────────────

    def _evaluate_batch(
        self, xs: List[np.ndarray], dispatcher: SlurmDispatcher, tag: str,
    ) -> List[float]:
        """Evaluate every candidate in *xs* as one concurrently-submitted
        SLURM batch, never blocking on one job before submitting the next.
        """
        specs = self._opt_specs()
        results: List[Optional[float]] = [None] * len(xs)
        pending: List[Tuple[int, "object", Optional[str], dict]] = []

        for i, x in enumerate(xs):
            self._n_evals += 1
            params = self.cfg.defaults.copy()
            for spec, val in zip(specs, x):
                params[spec.name] = float(val)

            if not self._in_bounds(x):
                self._log.info(
                    "[eval %04d] %s_c%03d out of bounds → penalty",
                    self._n_evals, tag, i,
                )
                self._append_history(params, float("nan"), "", None)
                results[i] = _PENALTY
                continue

            case_name = f"{tag}_c{i:03d}"
            case_dir, job_id = self._dispatch_case(
                case_name, case_name, params, dispatcher
            )

            if self.dry_run:
                self._append_history(params, 0.0, str(case_dir), job_id)
                results[i] = 0.0
                continue

            pending.append((i, case_dir, job_id, params))

        if pending:
            waitable = [jid for _, _, jid, _ in pending if jid is not None]
            batch_status = {}
            if waitable:
                batch_status = dispatcher.wait_for_batch(
                    waitable,
                    timeout=self.cfg.optimizer.job_timeout,
                    poll_interval=self.cfg.optimizer.poll_interval,
                )
            for i, case_dir, job_id, params in pending:
                if job_id is None:
                    self._log.warning(
                        "[eval %04d] submission failed for %s", self._n_evals, case_dir,
                    )
                    metric = float("nan")
                elif not batch_status.get(job_id, False):
                    self._log.warning(
                        "Job %s did not complete for %s", job_id, case_dir,
                    )
                    metric = float("nan")
                elif self.cfg.optimizer.postprocess:
                    metric = self._postprocess_case(case_dir)
                else:
                    metric = self.parser.parse_case(
                        case_dir, regex=self.cfg.objective_regex
                    )

                self._append_history(params, metric, str(case_dir), job_id)
                if not np.isfinite(metric):
                    results[i] = _PENALTY
                else:
                    results[i] = self._sign * metric

        assert all(r is not None for r in results)
        return [float(r) for r in results]  # type: ignore[arg-type]

    # ── DE phase (rand/1/bin, batched per generation) ─────────────────────────

    def _de_phase(self, dispatcher: SlurmDispatcher) -> Tuple[np.ndarray, float]:
        specs = self._opt_specs()
        dim = len(specs)
        low, high = self._bounds()
        popsize = self.cfg.optimizer.de_popsize
        f = self.cfg.optimizer.de_f
        cr = self.cfg.optimizer.de_cr
        max_evals = self.cfg.optimizer.max_evaluations
        stall_limit = self.cfg.optimizer.de_stall_generations

        pop = self._rng.uniform(low, high, size=(popsize, dim))
        pop[0] = self._x0()

        values = np.array(self._evaluate_batch(list(pop), dispatcher, tag="de_init"))
        best_idx = int(np.argmin(values))
        best_x = pop[best_idx].copy()
        best_val = float(values[best_idx])
        self._log.info(
            "[DE init] popsize=%d best=%.6g (evals=%d)",
            popsize, self._sign * best_val, self._n_evals,
        )

        stall_count = 0
        for g in range(1, self.cfg.optimizer.de_generations + 1):
            if self._n_evals >= max_evals:
                self._log.info(
                    "DE stopping before gen %03d: evaluation budget exhausted (%d)",
                    g, self._n_evals,
                )
                break

            trials = np.empty_like(pop)
            for i in range(popsize):
                idxs = [j for j in range(popsize) if j != i]
                r1, r2, r3 = self._rng.choice(idxs, size=3, replace=False)
                mutant = pop[r1] + f * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, low, high)

                trial = pop[i].copy()
                cross_mask = self._rng.random(dim) < cr
                j_rand = int(self._rng.integers(dim))
                cross_mask[j_rand] = True
                trial[cross_mask] = mutant[cross_mask]
                trials[i] = trial

            trial_values = np.array(
                self._evaluate_batch(list(trials), dispatcher, tag=f"de_g{g:03d}")
            )

            improved = trial_values < values
            pop[improved] = trials[improved]
            values[improved] = trial_values[improved]

            gen_best_idx = int(np.argmin(values))
            gen_best_val = float(values[gen_best_idx])
            if gen_best_val < best_val - self.cfg.optimizer.tolerance:
                best_val = gen_best_val
                best_x = pop[gen_best_idx].copy()
                stall_count = 0
            else:
                stall_count += 1

            self._log.info(
                "[DE gen %03d] best=%.6g stall=%d/%d (evals=%d)",
                g, self._sign * best_val, stall_count, stall_limit, self._n_evals,
            )

            if stall_count >= stall_limit:
                self._log.info(
                    "DE stopping after gen %03d: stalled for %d generations",
                    g, stall_count,
                )
                break
            if self._n_evals >= max_evals:
                self._log.info(
                    "DE stopping after gen %03d: evaluation budget exhausted (%d)",
                    g, self._n_evals,
                )
                break

        return best_x, best_val

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Run DE exploration followed by Nelder-Mead refinement."""
        specs = self._opt_specs()
        self._log.info(
            "Hybrid DE→Nelder-Mead over %s  (maximize=%s, de_popsize=%d, "
            "de_generations=%d, max_evals=%d)",
            [s.name for s in specs], self.cfg.optimizer.maximize,
            self.cfg.optimizer.de_popsize, self.cfg.optimizer.de_generations,
            self.cfg.optimizer.max_evaluations,
        )

        with SlurmDispatcher(self.cfg, dry_run=self.dry_run) as dispatcher:
            de_best_x, de_best_val = self._de_phase(dispatcher)

            remaining = self.cfg.optimizer.max_evaluations - self._n_evals
            if remaining <= 0:
                self._log.info(
                    "Skipping Nelder-Mead refinement: evaluation budget "
                    "exhausted after DE (%d evals used).", self._n_evals,
                )
                final_x, final_val = de_best_x, de_best_val
            else:
                result = optimize.minimize(
                    self._evaluate,
                    de_best_x,
                    args=(dispatcher,),
                    method="Nelder-Mead",
                    options={
                        "maxfev": remaining,
                        "xatol": self.cfg.optimizer.tolerance,
                        "fatol": self.cfg.optimizer.tolerance,
                        "adaptive": True,
                    },
                )
                final_x, final_val = result.x, float(result.fun)

        de_best_params = {s.name: float(v) for s, v in zip(specs, de_best_x)}
        final_params = {s.name: float(v) for s, v in zip(specs, final_x)}
        self._log.info(
            "DE phase best: %s  objective=%.6g",
            de_best_params, self._sign * de_best_val,
        )
        self._log.info(
            "Hybrid optimization finished after %d evaluations: "
            "DE best objective=%.6g → NM-refined objective=%.6g  final=%s",
            self._n_evals, self._sign * de_best_val, self._sign * final_val,
            final_params,
        )

        if self.cfg.optimizer.history_csv.exists():
            return pd.read_csv(self.cfg.optimizer.history_csv)
        return pd.DataFrame()
