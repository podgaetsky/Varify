"""Non-blocking scheduler job submission (sbatch/qsub/bsub compatible).

Migrated from the legacy ``ClusterDispatcher``: identical job-id extraction
patterns, submission-log auditing and error handling.  New here: every
successful submission is recorded in the ``JobRegistry`` so the watchdog
daemon can monitor it.
"""

from __future__ import annotations

import logging
import re
import subprocess
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
