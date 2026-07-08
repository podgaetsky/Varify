"""Tests for the DE step logic of ``HybridDEOptimizer``.

``_evaluate_batch`` is monkeypatched to score 2-D Rosenbrock directly (no
SLURM, no scipy, no real dispatcher — a ``MagicMock`` stands in for
``SlurmDispatcher``) so these tests exercise only the DE mutation /
crossover / selection / stall-early-stop logic in
``src.optimizer.hybrid.HybridDEOptimizer._de_phase``, plus the
out-of-bounds short-circuit in the real (unpatched) ``_evaluate_batch``.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from varify.src.common.params import ParamSpec
from varify.src.optimizer.hybrid import HybridDEOptimizer, _PENALTY

_LOW, _HIGH = -2.0, 2.0


def _rosenbrock(x: np.ndarray) -> float:
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def _make_specs() -> list:
    return [
        ParamSpec(
            name="x", default=-1.5, values=None, input_fn=None,
            mcmc_prior_low=_LOW, mcmc_prior_high=_HIGH, mcmc_init_center=-1.5,
        ),
        ParamSpec(
            name="y", default=-1.5, values=None, input_fn=None,
            mcmc_prior_low=_LOW, mcmc_prior_high=_HIGH, mcmc_init_center=-1.5,
        ),
    ]


def _make_cfg(tmp_path: Path, **opt_overrides) -> types.SimpleNamespace:
    """Duck-typed FrameworkConfig stand-in with only the attributes touched
    by HybridDEOptimizer / NelderMeadOptimizer / BaseOptimizer construction
    and the DE phase (CaseBuilder/ResultParser only store cfg, no I/O at
    construction time).
    """
    optimizer = types.SimpleNamespace(
        maximize=False,
        max_evaluations=5000,
        tolerance=1e-6,
        objective_regex=None,
        poll_interval=0.0,
        job_timeout=0.0,
        history_csv=tmp_path / "optimization_history.csv",
        postprocess=False,
        de_popsize=8,
        de_generations=25,
        de_f=0.7,
        de_cr=0.9,
        de_stall_generations=6,
        de_seed=42,
    )
    for k, v in opt_overrides.items():
        setattr(optimizer, k, v)
    specs = _make_specs()
    return types.SimpleNamespace(
        mcmc_specs=specs,
        param_specs=specs,
        template_files=[],
        file_pipeline=[],
        defaults={"x": -1.5, "y": -1.5},
        optimizer=optimizer,
        sweep_root=tmp_path / "sweep",
    )


def _make_optimizer(tmp_path: Path, **opt_overrides) -> HybridDEOptimizer:
    cfg = _make_cfg(tmp_path, **opt_overrides)
    return HybridDEOptimizer(cfg, dry_run=False)


# ── DE step logic (mutation/crossover/selection) via Rosenbrock ───────────────

def test_de_phase_improves_objective_and_respects_bounds(tmp_path):
    # de_seed=0 gives a init population whose best member is far from the
    # Rosenbrock optimum, so DE has visible room to improve within the
    # generation/stall budget below (verified empirically across seeds).
    opt = _make_optimizer(tmp_path, de_seed=0, de_stall_generations=8)
    batches: list = []

    def fake_evaluate_batch(xs, dispatcher, tag):
        batches.append([x.copy() for x in xs])
        return [_rosenbrock(x) for x in xs]

    opt._evaluate_batch = fake_evaluate_batch  # type: ignore[method-assign]

    best_x, best_val = opt._de_phase(mock.MagicMock())

    assert len(batches) >= 2  # init population + at least one generation
    init_best = min(_rosenbrock(x) for x in batches[0])

    # DE must never do worse than the initial population's best member.
    assert best_val <= init_best
    # And on Rosenbrock from a poor start it should actually improve.
    assert best_val < init_best

    # Every candidate ever evaluated (init population + all trial vectors)
    # must respect the prior box bounds.
    for batch in batches:
        for x in batch:
            assert np.all(x >= _LOW - 1e-9) and np.all(x <= _HIGH + 1e-9)

    assert best_x.shape == (2,)


def test_de_phase_stops_early_on_stall(tmp_path):
    stall_limit = 4
    opt = _make_optimizer(
        tmp_path, de_generations=50, de_stall_generations=stall_limit,
        tolerance=1e-6,
    )
    call_count = {"n": 0}

    def flat_evaluate_batch(xs, dispatcher, tag):
        call_count["n"] += 1
        # First (init) batch has real spread; every generation after that
        # returns an identical, non-improving objective so DE cannot find
        # any improvement and must trigger the stall-based early stop.
        if call_count["n"] == 1:
            return [_rosenbrock(x) for x in xs]
        return [1.0e6 for _ in xs]

    opt._evaluate_batch = flat_evaluate_batch  # type: ignore[method-assign]
    opt._de_phase(mock.MagicMock())

    generations_run = call_count["n"] - 1  # exclude the init-population call
    assert generations_run == stall_limit
    assert generations_run < 50


def test_de_phase_stops_on_evaluation_budget(tmp_path):
    # popsize=8 → after the init batch the budget is already exhausted, so
    # no generation should run at all.
    opt = _make_optimizer(tmp_path, max_evaluations=8, de_generations=50)
    call_count = {"n": 0}

    def fake_evaluate_batch(xs, dispatcher, tag):
        call_count["n"] += 1
        opt._n_evals += len(xs)
        return [_rosenbrock(x) for x in xs]

    opt._evaluate_batch = fake_evaluate_batch  # type: ignore[method-assign]
    opt._de_phase(mock.MagicMock())

    assert call_count["n"] == 1  # only the initial population was evaluated


# ── Out-of-bounds short-circuit on the real _evaluate_batch ──────────────────

def test_out_of_bounds_candidate_gets_penalty_without_dispatch(tmp_path):
    opt = _make_optimizer(tmp_path)
    dispatcher = mock.MagicMock()

    out_of_bounds = np.array([-10.0, 10.0])  # well outside [-2, 2]
    results = opt._evaluate_batch([out_of_bounds], dispatcher, tag="test_g000")

    assert results == [_PENALTY]
    dispatcher.dispatch.assert_not_called()

    history_path = opt.cfg.optimizer.history_csv
    assert history_path.exists()
    import pandas as pd
    df = pd.read_csv(history_path)
    assert len(df) == 1
    assert np.isnan(df.loc[0, "objective"])
    assert df.loc[0, "case_dir"] == "" or pd.isna(df.loc[0, "case_dir"])


def test_mixed_batch_only_dispatches_in_bounds_candidates(tmp_path):
    opt = _make_optimizer(tmp_path)
    dispatcher = mock.MagicMock()
    dispatcher.dispatch.return_value = "DRY_RUN"

    xs = [np.array([-10.0, -10.0]), np.array([0.0, 0.0])]
    # dry_run=True on the optimizer avoids needing wait_for_batch/parser wiring.
    opt.dry_run = True
    results = opt._evaluate_batch(xs, dispatcher, tag="test_g001")

    assert results[0] == _PENALTY  # out of bounds, no dispatch
    assert results[1] == 0.0       # in bounds, dry-run score
    assert dispatcher.dispatch.call_count == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
