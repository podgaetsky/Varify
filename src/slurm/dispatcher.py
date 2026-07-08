r"""Non-blocking scheduler job submission (sbatch/qsub/bsub compatible).

Migrated from the legacy ``ClusterDispatcher``: identical job-id extraction
patterns, submission-log auditing and error handling.  New here: every
successful submission is recorded in the ``JobRegistry`` so the watchdog
daemon can monitor it, and native completion-waiting so optimizers/grid
sweeps can block on a job (or a batch of jobs) without shelling out to
``sbatch --wait``.

This module is the canonical reference for talking to sbatch from Python.
The minimal round trip — submit, parse the job id, poll to completion — is
just two ``subprocess.run`` calls and a regex::

    import re
    import subprocess
    import time

    submit = subprocess.run(
        ["sbatch", "run_script.sh"], capture_output=True, text=True,
    )
    job_id = re.search(r"Submitted batch job (\d+)", submit.stdout).group(1)

    while True:
        status = subprocess.run(
            f"squeue -h -j {job_id} -o %T", shell=True,
            capture_output=True, text=True,
        )
        if not status.stdout.strip():
            break  # left the queue; check sacct/output file for the outcome
        time.sleep(30)

``SlurmDispatcher.dispatch`` / ``.job_state`` / ``.wait_for_completion`` /
``.wait_for_batch`` below wrap exactly this pattern with config-driven
commands, state normalization and a sacct fallback.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.config import FrameworkConfig
from src.slurm.registry import JobRegistry


class SlurmDispatcher:
    """Fires one scheduler job per case directory; context-managed audit log."""

    _JOB_ID_PATTERNS: List[re.Pattern] = [
        re.compile(r"Submitted batch job (\d+)", re.I),
        re.compile(r"^(\d+)(?:\.\S+)?$"),
        re.compile(r"Job <(\d+)> is submitted", re.I),
        re.compile(r"(\d{4,})"),
    ]

    # Normalized terminal states returned by job_state(); anything else
    # (PENDING/RUNNING/UNKNOWN) is considered still in-flight.
    _TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})

    # squeue/sacct state tokens → normalized bucket.
    _STATE_MAP: Dict[str, str] = {
        "COMPLETED": "COMPLETED",
        "FAILED": "FAILED",
        "NODE_FAIL": "FAILED",
        "BOOT_FAIL": "FAILED",
        "OUT_OF_MEMORY": "FAILED",
        "OOM": "FAILED",
        "PREEMPTED": "FAILED",
        "TIMEOUT": "TIMEOUT",
        "DEADLINE": "TIMEOUT",
        "PENDING": "PENDING",
        "REQUEUED": "PENDING",
        "REQUEUE_HOLD": "PENDING",
        "REQUEUE_FED": "PENDING",
        "RUNNING": "RUNNING",
        "COMPLETING": "RUNNING",
        "CONFIGURING": "RUNNING",
        "RESIZING": "RUNNING",
        "SUSPENDED": "RUNNING",
        "STAGE_OUT": "RUNNING",
        "SIGNALING": "RUNNING",
        "REVOKED": "CANCELLED",
    }

    def __init__(self, cfg: FrameworkConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.registry = JobRegistry(cfg.jobs_registry_csv)
        self._log = logging.getLogger("varify.dispatcher")
        self._fh: Optional[Any] = None

    def __enter__(self) -> "SlurmDispatcher":
        self._fh = open(self.cfg.submission_log, "a", encoding="utf-8")
        self._fh.write("\n# ── Submission session ──────────────────\n")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def _extract_job_id(self, stdout: str) -> Optional[str]:
        for pat in self._JOB_ID_PATTERNS:
            m = pat.search(stdout.strip())
            if m:
                return m.group(1)
        return None

    def _build_cmd(
        self,
        job_name: str,
        case_dir: Path,
        params: Dict[str, float],
        dependency_job_id: Optional[str] = None,
    ) -> str:
        fmt: Dict[str, Any] = {
            "job_name": job_name,
            "case_dir": str(case_dir.resolve()),
            **{f"param_{n}": v for n, v in params.items()},
        }
        cmd = self.cfg.slurm.effective_submit_cmd.format(**fmt)
        if dependency_job_id and self.cfg.mcmc.use_dependency:
            # Insert SLURM dependency flag before the script name (last token)
            tokens = cmd.split()
            dep_flag = f"--dependency=afterok:{dependency_job_id}"
            cmd = " ".join(tokens[:-1] + [dep_flag, tokens[-1]])
        return cmd

    def dispatch(
        self,
        job_name: str,
        case_dir: Path,
        params: Dict[str, float],
        dependency_job_id: Optional[str] = None,
        resubmits: int = 0,
    ) -> Optional[str]:
        """Submit one job; return the scheduler job id (or None on failure)."""
        cmd = self._build_cmd(job_name, case_dir, params, dependency_job_id)
        self._log.info("[DISPATCH] %s  →  %s", case_dir.name, cmd)
        if self._fh:
            self._fh.write(f"{case_dir.name}\t{cmd}\n")

        if self.dry_run:
            self._log.info("[DRY-RUN] Would execute: %s", cmd)
            return "DRY_RUN"

        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(case_dir.resolve()),
                capture_output=True, text=True,
                timeout=self.cfg.slurm.submit_timeout,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode != 0:
                self._log.error(
                    "Submission failed %s (rc=%d): %s",
                    case_dir.name, result.returncode, stderr or stdout,
                )
                return None
            job_id = self._extract_job_id(stdout)
            self._log.info("Submitted %s → job_id=%s", case_dir.name, job_id or "?")
            if self._fh:
                self._fh.write(f"  job_id={job_id}  rc={result.returncode}\n")
            if job_id is not None:
                self.registry.add(job_id, job_name, case_dir, params, resubmits)
            return job_id
        except subprocess.TimeoutExpired:
            self._log.error("Timeout submitting %s", case_dir.name)
            return None
        except Exception as exc:
            self._log.error("Error submitting %s: %s", case_dir.name, exc)
            return None

    # ── Job control (used by the watchdog) ───────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduler job via the configured cancel command."""
        cmd = self.cfg.slurm.cancel_cmd.format(job_id=job_id)
        if self.dry_run:
            self._log.info("[DRY-RUN] Would cancel: %s", cmd)
            return True
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            ok = result.returncode == 0
            self._log.log(
                logging.INFO if ok else logging.ERROR,
                "Cancel job %s (rc=%d)", job_id, result.returncode,
            )
            return ok
        except Exception as exc:
            self._log.error("Error cancelling job %s: %s", job_id, exc)
            return False

    def is_queued_or_running(self, job_id: str) -> bool:
        """True if the scheduler still reports the job (queued or running)."""
        cmd = self.cfg.slurm.status_cmd.format(job_id=job_id)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            return bool(result.stdout.strip())
        except Exception as exc:
            self._log.warning("Status query failed for job %s: %s", job_id, exc)
            return False

    # ── Completion-waiting (used by optimizers & synchronous grid sweeps) ────

    @classmethod
    def _normalize_state(cls, raw: str) -> str:
        """Map a raw squeue/sacct state token to one of the normalized
        buckets: PENDING/RUNNING/COMPLETED/FAILED/CANCELLED/TIMEOUT/UNKNOWN.
        """
        token = raw.strip().upper()
        if not token:
            return "UNKNOWN"
        # sacct sometimes appends detail, e.g. "CANCELLED by 1000".
        token = token.split()[0]
        if token.startswith("CANCELLED"):
            return "CANCELLED"
        return cls._STATE_MAP.get(token, "UNKNOWN")

    def job_state(self, job_id: str) -> str:
        """Query the scheduler for *job_id*'s current normalized state.

        Tries ``squeue`` (``cfg.slurm.status_cmd``) first, since it is cheap
        and authoritative while the job is still queued/running.  Once the
        job has left the queue, squeue returns nothing, so we fall back to
        ``sacct`` (``cfg.slurm.sacct_cmd``) to learn the terminal state.  If
        sacct itself is unavailable (non-zero rc or missing binary), the
        queue-absence is treated as COMPLETED — the downstream output-file
        wait remains the ground truth for whether the run actually
        succeeded.
        """
        if job_id == "DRY_RUN":
            return "COMPLETED"

        cmd = self.cfg.slurm.status_cmd.format(job_id=job_id)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            stdout = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            self._log.warning("squeue query failed for job %s: %s", job_id, exc)
            stdout = ""

        if stdout:
            return self._normalize_state(stdout.splitlines()[0])

        # Job is no longer in the queue; ask sacct for the terminal state.
        sacct_cmd = self.cfg.slurm.sacct_cmd.format(job_id=job_id)
        try:
            result = subprocess.run(
                sacct_cmd, shell=True, capture_output=True, text=True,
                timeout=30,
            )
        except FileNotFoundError:
            self._log.debug(
                "sacct unavailable for job %s; treating queue-absence as "
                "COMPLETED", job_id,
            )
            return "COMPLETED"
        except Exception as exc:
            self._log.warning("sacct query failed for job %s: %s", job_id, exc)
            return "COMPLETED"

        if result.returncode != 0:
            self._log.debug(
                "sacct rc=%d for job %s; treating queue-absence as "
                "COMPLETED", result.returncode, job_id,
            )
            return "COMPLETED"

        sacct_out = result.stdout.strip()
        if not sacct_out:
            return "COMPLETED"
        return self._normalize_state(sacct_out.splitlines()[0])

    def wait_for_completion(
        self, job_id: str, timeout: float, poll_interval: float,
    ) -> bool:
        """Block until *job_id* reaches a terminal state or *timeout*
        elapses. Returns True iff the terminal state is COMPLETED.

        Dry-run ids (``"DRY_RUN"``) resolve immediately without polling.
        """
        if job_id == "DRY_RUN":
            return True

        deadline = time.monotonic() + timeout
        while True:
            state = self.job_state(job_id)
            if state in self._TERMINAL_STATES:
                return state == "COMPLETED"
            if time.monotonic() >= deadline:
                self._log.warning(
                    "Timeout waiting for job %s (last state=%s)",
                    job_id, state,
                )
                return False
            time.sleep(poll_interval)

    def wait_for_batch(
        self, job_ids: List[str], timeout: float, poll_interval: float,
    ) -> Dict[str, bool]:
        """Block until every id in *job_ids* reaches a terminal state or the
        shared *timeout* elapses; return ``{job_id: completed_ok}``.

        Polls the set of still-pending ids once per *poll_interval* (one
        status query per pending id, not a per-job sleep) against a single
        deadline for the whole batch. Dry-run ids resolve immediately.
        """
        results: Dict[str, bool] = {}
        pending = set(job_ids)
        for jid in list(pending):
            if jid == "DRY_RUN":
                results[jid] = True
                pending.discard(jid)

        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            for jid in list(pending):
                state = self.job_state(jid)
                if state in self._TERMINAL_STATES:
                    results[jid] = (state == "COMPLETED")
                    pending.discard(jid)
            if pending:
                time.sleep(poll_interval)

        for jid in pending:
            self._log.warning("Timeout waiting for job %s in batch", jid)
            results[jid] = False

        return results
