"""Tests for the ``hybrid`` runner strategy (DE + Nelder-Mead) and the
SLURM completion-waiting helper in ``runner.core``.

The strategy tests build a minimal ``RunContext`` directly (a
``CheckpointManager`` pointed at a tmp path, never entered as a context
manager, plus a bare ``{"seed": ...}`` provenance dict — the only key
``RunContext.__init__`` reads) rather than going through the full
``WorkflowRunner.run()`` lifecycle, so they exercise only the DE
mutation/crossover/selection + Nelder-Mead hand-off logic on a 2-D
Rosenbrock objective. No SLURM, no scipy, no multiprocessing.

The ``_wait_for_slurm_job`` tests mock ``subprocess.run``/``time.sleep`` to
drive squeue -> sacct transitions without touching a real scheduler.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest import mock

import pytest

from runner import strategies  # noqa: F401  (registers built-in strategies)
from runner.checkpoint import CheckpointManager
from runner.core import RunContext, RunSpec, _wait_for_slurm_job

_LOW_X, _HIGH_X = -2.0, 2.0
_LOW_Y, _HIGH_Y = -1.0, 3.0


def _rosenbrock(x: float, y: float) -> float:
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def _make_ctx(tmp_path: Path, **opt_overrides) -> RunContext:
    options = {
        "max_evaluations": 300,
        "de_popsize": 10,
        "de_generations": 15,
        "de_f": 0.7,
        "de_cr": 0.9,
        "de_stall": 6,
        "xatol": 1e-6,
        "fatol": 1e-10,
    }
    options.update(opt_overrides)
    spec = RunSpec(
        name="hybrid_test",
        strategy="hybrid",
        model=_rosenbrock,
        bounds={"x": (_LOW_X, _HIGH_X), "y": (_LOW_Y, _HIGH_Y)},
        options=options,
        seed=42,
        results_root=tmp_path,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = CheckpointManager(run_dir / "checkpoint.json")
    return RunContext(spec, run_dir, checkpoint, provenance={"seed": 42})


# ── DE + NM hybrid strategy logic ─────────────────────────────────────────────

def test_hybrid_improves_over_initial_population_and_respects_bounds(tmp_path):
    ctx = _make_ctx(tmp_path)
    result = strategies.hybrid_strategy(ctx)

    with open(ctx.run_dir / "history.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    popsize = int(ctx.spec.options["de_popsize"])
    de_rows = [r for r in rows if r["phase"] == "de"]
    nm_rows = [r for r in rows if r["phase"] == "nm"]
    assert de_rows and nm_rows  # both phases actually ran

    init_best = min(float(r["value"]) for r in de_rows[:popsize])
    assert result["best_value"] <= init_best
    assert result["best_value"] < init_best  # real improvement on Rosenbrock

    # Every evaluated candidate (DE population/trials + NM simplex) must
    # respect the box bounds.
    for row in rows:
        x, y = float(row["x"]), float(row["y"])
        assert _LOW_X - 1e-9 <= x <= _HIGH_X + 1e-9
        assert _LOW_Y - 1e-9 <= y <= _HIGH_Y + 1e-9

    # Globally numbered history: eval column is a contiguous 1..N sequence.
    assert [int(r["eval"]) for r in rows] == list(range(1, len(rows) + 1))

    # Should land close to the (1, 1) optimum within a few hundred evals.
    assert abs(result["best_params"]["x"] - 1.0) < 0.05
    assert abs(result["best_params"]["y"] - 1.0) < 0.05

    # Payload shape mirrors optimize_strategy's core keys.
    for key in ("best_params", "best_value", "n_evaluations", "converged",
                "backend"):
        assert key in result
    assert result["n_evaluations"] == len(rows)
    assert result["de_evaluations"] == len(de_rows)
    assert result["nm_evaluations"] == len(nm_rows)


def test_hybrid_respects_evaluation_budget(tmp_path):
    ctx = _make_ctx(tmp_path, max_evaluations=25, de_popsize=10,
                     de_generations=50)
    result = strategies.hybrid_strategy(ctx)
    assert result["n_evaluations"] <= 25


def test_hybrid_stops_early_on_stall(tmp_path):
    # A pathologically flat model: DE can never improve, so it must trigger
    # the stall-based early stop well before de_generations is exhausted.
    def flat(x: float, y: float) -> float:
        return 1.0

    spec = RunSpec(
        name="stall_test", strategy="hybrid", model=flat,
        bounds={"x": (_LOW_X, _HIGH_X), "y": (_LOW_Y, _HIGH_Y)},
        options={
            "max_evaluations": 2000, "de_popsize": 8,
            "de_generations": 50, "de_stall": 4,
        },
        seed=7, results_root=tmp_path,
    )
    run_dir = tmp_path / "run_flat"
    run_dir.mkdir()
    ctx = RunContext(spec, run_dir, CheckpointManager(run_dir / "checkpoint.json"),
                      provenance={"seed": 7})
    result = strategies.hybrid_strategy(ctx)
    assert result["de_generations"] == 4
    assert result["de_generations"] < 50


# ── _wait_for_slurm_job() ─────────────────────────────────────────────────────

class _Result:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_wait_for_slurm_job_running_then_gone_then_sacct_completed():
    calls = {"squeue": 0, "sacct": 0}

    def _run(cmd, capture_output=True, text=True, timeout=None):
        if cmd[0] == "squeue":
            calls["squeue"] += 1
            if calls["squeue"] == 1:
                return _Result(0, "RUNNING\n")
            return _Result(0, "")
        calls["sacct"] += 1
        return _Result(0, "COMPLETED\n")

    with mock.patch("runner.core.subprocess.run", side_effect=_run), \
         mock.patch("runner.core.time.sleep", return_value=None):
        ok = _wait_for_slurm_job("123", timeout=10.0, poll_interval=0.01)

    assert ok is True
    assert calls == {"squeue": 2, "sacct": 1}


def test_wait_for_slurm_job_failed():
    def _run(cmd, capture_output=True, text=True, timeout=None):
        if cmd[0] == "squeue":
            return _Result(0, "")
        return _Result(0, "FAILED\n")

    with mock.patch("runner.core.subprocess.run", side_effect=_run), \
         mock.patch("runner.core.time.sleep", return_value=None):
        ok = _wait_for_slurm_job("456", timeout=10.0, poll_interval=0.01)

    assert ok is False


def test_wait_for_slurm_job_sacct_unavailable_nonzero_rc_is_completed():
    def _run(cmd, capture_output=True, text=True, timeout=None):
        if cmd[0] == "squeue":
            return _Result(0, "")
        return _Result(127, "", "sacct: command not found")

    with mock.patch("runner.core.subprocess.run", side_effect=_run), \
         mock.patch("runner.core.time.sleep", return_value=None):
        ok = _wait_for_slurm_job("789", timeout=10.0, poll_interval=0.01)

    assert ok is True


def test_wait_for_slurm_job_sacct_filenotfound_is_completed():
    def _run(cmd, capture_output=True, text=True, timeout=None):
        if cmd[0] == "squeue":
            return _Result(0, "")
        raise FileNotFoundError("sacct not found")

    with mock.patch("runner.core.subprocess.run", side_effect=_run), \
         mock.patch("runner.core.time.sleep", return_value=None):
        ok = _wait_for_slurm_job("999", timeout=10.0, poll_interval=0.01)

    assert ok is True


def test_wait_for_slurm_job_timeout_returns_false():
    def _run(cmd, capture_output=True, text=True, timeout=None):
        assert cmd[0] == "squeue"
        return _Result(0, "PENDING\n")

    with mock.patch("runner.core.subprocess.run", side_effect=_run), \
         mock.patch("runner.core.time.sleep", return_value=None):
        ok = _wait_for_slurm_job("111", timeout=0.0, poll_interval=0.01)

    assert ok is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
