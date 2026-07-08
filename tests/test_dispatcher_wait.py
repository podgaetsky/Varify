"""Tests for SlurmDispatcher's native completion-waiting.

Mocks ``subprocess.run`` to drive job_state()/wait_for_completion()/
wait_for_batch() through squeue → sacct transitions without touching a
real scheduler. Plain pytest-style ``test_*`` functions; also runnable
directly via ``python tests/test_dispatcher_wait.py`` (see __main__ block)
when pytest is not installed.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest import mock

from src.slurm.dispatcher import SlurmDispatcher


class _Result:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_cfg() -> types.SimpleNamespace:
    """A duck-typed FrameworkConfig stand-in with only what SlurmDispatcher
    touches: cfg.slurm.{status_cmd,sacct_cmd,submit_timeout} and
    cfg.jobs_registry_csv (JobRegistry just stores the path, no I/O here).
    """
    slurm = types.SimpleNamespace(
        status_cmd="squeue -h -j {job_id} -o %T",
        sacct_cmd="sacct -j {job_id} -n -o State -X",
        submit_timeout=30.0,
    )
    return types.SimpleNamespace(
        slurm=slurm,
        jobs_registry_csv=Path("/tmp/varify_test_jobs_registry.csv"),
    )


def _make_dispatcher() -> SlurmDispatcher:
    return SlurmDispatcher(_make_cfg(), dry_run=False)


def _run_with(scenario):
    """Return a subprocess.run stand-in dispatching to *scenario(kind, n)*
    where kind is 'squeue' or 'sacct' and n is the 1-based call count for
    that kind."""
    calls = {"squeue": 0, "sacct": 0}

    def _run(cmd, shell=True, capture_output=True, text=True, timeout=None):
        kind = "squeue" if "squeue" in cmd else "sacct"
        calls[kind] += 1
        return scenario(kind, calls[kind])

    return _run, calls


# ── job_state() ────────────────────────────────────────────────────────────

def test_job_state_running_from_squeue():
    def scenario(kind, n):
        assert kind == "squeue"
        return _Result(0, "RUNNING\n")

    run_fn, _ = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn):
        assert d.job_state("123") == "RUNNING"


def test_job_state_squeue_empty_then_sacct_completed():
    def scenario(kind, n):
        if kind == "squeue":
            return _Result(0, "")
        return _Result(0, "COMPLETED\n")

    run_fn, calls = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn):
        assert d.job_state("123") == "COMPLETED"
    assert calls == {"squeue": 1, "sacct": 1}


def test_job_state_squeue_empty_then_sacct_failed():
    def scenario(kind, n):
        if kind == "squeue":
            return _Result(0, "")
        return _Result(0, "FAILED\n")

    run_fn, _ = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn):
        assert d.job_state("123") == "FAILED"


def test_job_state_sacct_unavailable_rc_nonzero_is_completed():
    def scenario(kind, n):
        if kind == "squeue":
            return _Result(0, "")
        return _Result(127, "", "sacct: command not found")

    run_fn, _ = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn):
        assert d.job_state("123") == "COMPLETED"


def test_job_state_sacct_filenotfound_is_completed():
    def _run(cmd, shell=True, capture_output=True, text=True, timeout=None):
        if "squeue" in cmd:
            return _Result(0, "")
        raise FileNotFoundError("sacct not found")

    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=_run):
        assert d.job_state("123") == "COMPLETED"


def test_job_state_dry_run_short_circuits():
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run") as run_mock:
        assert d.job_state("DRY_RUN") == "COMPLETED"
        run_mock.assert_not_called()


# ── wait_for_completion() ───────────────────────────────────────────────────

def test_wait_for_completion_running_then_sacct_completed():
    # squeue: RUNNING on the first poll, empty afterwards; sacct: COMPLETED.
    def scenario(kind, n):
        if kind == "squeue":
            return _Result(0, "RUNNING\n") if n == 1 else _Result(0, "")
        return _Result(0, "COMPLETED\n")

    run_fn, calls = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn), \
         mock.patch("src.slurm.dispatcher.time.sleep", return_value=None):
        ok = d.wait_for_completion("123", timeout=5.0, poll_interval=0.01)
    assert ok is True
    assert calls["squeue"] == 2
    assert calls["sacct"] == 1


def test_wait_for_completion_failed_path():
    def scenario(kind, n):
        if kind == "squeue":
            return _Result(0, "")
        return _Result(0, "FAILED\n")

    run_fn, _ = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn), \
         mock.patch("src.slurm.dispatcher.time.sleep", return_value=None):
        ok = d.wait_for_completion("456", timeout=5.0, poll_interval=0.01)
    assert ok is False


def test_wait_for_completion_sacct_unavailable_is_completed():
    def scenario(kind, n):
        if kind == "squeue":
            return _Result(0, "")
        return _Result(1, "", "no sacct here")

    run_fn, _ = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn), \
         mock.patch("src.slurm.dispatcher.time.sleep", return_value=None):
        ok = d.wait_for_completion("789", timeout=5.0, poll_interval=0.01)
    assert ok is True


def test_wait_for_completion_dry_run():
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run") as run_mock:
        ok = d.wait_for_completion("DRY_RUN", timeout=5.0, poll_interval=0.01)
    assert ok is True
    run_mock.assert_not_called()


def test_wait_for_completion_timeout_returns_false():
    # Always PENDING; never reaches a terminal state → timeout path.
    def scenario(kind, n):
        assert kind == "squeue"
        return _Result(0, "PENDING\n")

    run_fn, _ = _run_with(scenario)
    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=run_fn), \
         mock.patch("src.slurm.dispatcher.time.sleep", return_value=None):
        ok = d.wait_for_completion("999", timeout=0.0, poll_interval=0.01)
    assert ok is False


# ── wait_for_batch() ─────────────────────────────────────────────────────────

def test_wait_for_batch_mixed_outcomes():
    """Job 'A' completes via squeue→sacct; job 'B' fails; a 'DRY_RUN' id
    resolves without any subprocess call."""

    state = {"A_squeue_calls": 0}

    def _run(cmd, shell=True, capture_output=True, text=True, timeout=None):
        # cmd is e.g. "squeue -h -j A -o %T" or "sacct -j A -n -o State -X".
        job_id = cmd.split()[3] if "squeue" in cmd else cmd.split()[2]
        if "squeue" in cmd:
            if job_id == "A":
                state["A_squeue_calls"] += 1
                return _Result(0, "RUNNING\n") if state["A_squeue_calls"] == 1 \
                    else _Result(0, "")
            if job_id == "B":
                return _Result(0, "")
            raise AssertionError(f"unexpected squeue cmd: {cmd}")
        # sacct
        if job_id == "A":
            return _Result(0, "COMPLETED\n")
        if job_id == "B":
            return _Result(0, "FAILED\n")
        raise AssertionError(f"unexpected sacct cmd: {cmd}")

    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=_run), \
         mock.patch("src.slurm.dispatcher.time.sleep", return_value=None):
        results = d.wait_for_batch(
            ["A", "B", "DRY_RUN"], timeout=5.0, poll_interval=0.01,
        )

    assert results == {"A": True, "B": False, "DRY_RUN": True}


def test_wait_for_batch_timeout():
    def _run(cmd, shell=True, capture_output=True, text=True, timeout=None):
        assert "squeue" in cmd
        return _Result(0, "PENDING\n")

    d = _make_dispatcher()
    with mock.patch("src.slurm.dispatcher.subprocess.run", side_effect=_run), \
         mock.patch("src.slurm.dispatcher.time.sleep", return_value=None):
        results = d.wait_for_batch(["X", "Y"], timeout=0.0, poll_interval=0.01)
    assert results == {"X": False, "Y": False}


if __name__ == "__main__":
    # Ad-hoc runner for environments without pytest installed.
    failures = 0
    tests = {name: fn for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)}
    for name, fn in tests.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
