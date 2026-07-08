"""SLURM watchdog daemon: automated fault detection and intervention.

A standalone background monitor, designed to run either in a login-node
shell or as an independent ``sbatch`` agent (see ``deploy_as_agent``).

Every ``watchdog.poll_interval`` seconds it iterates over the active jobs in
the central ``JobRegistry`` and checks each case directory for:

* **missing output**   — the job has left the scheduler queue but the
  expected output file was never produced (or contains no parsable metric);
* **NaN contamination** — ``watchdog.nan_regex`` matches in any monitored
  ``.out`` / ``.err`` log file;
* **stall**            — no monitored file has been modified for longer
  than ``watchdog.stall_timeout`` seconds while the job is still running.

On failure the watchdog (1) appends a row to the central ``status.csv``,
(2) cancels the hanging scheduler job, and (3) optionally resubmits the
case with modified parameters (each parameter listed in
``watchdog.resubmit_scaling`` is multiplied by its factor — e.g. halving a
timestep) up to ``watchdog.max_resubmits`` times.

Run standalone:  ``python main.py --mode watch``  or
``python -m src.slurm.watchdog --config config/config.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from varify.src.common.casebuilder import CaseBuilder
from varify.src.common.config import FrameworkConfig, load_config
from varify.src.slurm.dispatcher import SlurmDispatcher
from varify.src.slurm.registry import JobRegistry

_STATUS_COLUMNS = [
    "timestamp", "iso_time", "job_id", "job_name",
    "case_dir", "reason", "action", "resubmits",
]


class Watchdog:
    """Polling failure monitor over the central job registry."""

    def __init__(self, cfg: FrameworkConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.registry = JobRegistry(cfg.jobs_registry_csv)
        self.builder = CaseBuilder(cfg)
        self._nan_re = re.compile(cfg.watchdog.nan_regex)
        self._output_re = re.compile(cfg.output_regex)
        self._log_prob_re = re.compile(cfg.mcmc.log_prob_regex)
        self._log = logging.getLogger("varify.watchdog")

    # ── Status log ────────────────────────────────────────────────────────────

    def _log_status(
        self,
        job_id: str,
        job_name: str,
        case_dir: Path,
        reason: str,
        action: str,
        resubmits: int,
    ) -> None:
        path = self.cfg.status_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        now = time.time()
        with open(path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(_STATUS_COLUMNS)
            writer.writerow([
                f"{now:.1f}",
                _dt.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                job_id, job_name, str(case_dir), reason, action, resubmits,
            ])
        self._log.info(
            "[STATUS] job=%s case=%s reason=%s action=%s",
            job_id, case_dir.name, reason, action,
        )

    # ── Per-job checks ────────────────────────────────────────────────────────

    def _has_metric(self, case_dir: Path) -> bool:
        """True if the output file exists and contains a parsable metric."""
        out_file = case_dir / self.cfg.output_file
        if not out_file.exists():
            return False
        text = out_file.read_text(encoding="utf-8", errors="replace")
        return bool(self._output_re.search(text) or self._log_prob_re.search(text))

    def _nan_detected(self, case_dir: Path) -> Optional[str]:
        for log_name in self.cfg.watchdog.log_files:
            log_path = case_dir / log_name
            if not log_path.exists():
                continue
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if self._nan_re.search(text):
                return f"nan_in_{log_name}"
        return None

    def _newest_mtime(self, case_dir: Path) -> Optional[float]:
        candidates: List[Path] = [
            case_dir / f
            for f in [*self.cfg.watchdog.log_files, self.cfg.output_file]
        ]
        mtimes = [p.stat().st_mtime for p in candidates if p.exists()]
        return max(mtimes) if mtimes else None

    def check_job(
        self, row: pd.Series, dispatcher: SlurmDispatcher
    ) -> Optional[str]:
        """Return a failure reason string, 'DONE' for completion, else None."""
        case_dir = Path(str(row["case_dir"]))
        job_id = str(row["job_id"])

        if not case_dir.is_dir():
            return "case_dir_missing"

        nan_reason = self._nan_detected(case_dir)
        if nan_reason is not None:
            return nan_reason

        in_queue = dispatcher.is_queued_or_running(job_id)
        if not in_queue:
            # Job left the queue: either finished cleanly or died silently.
            return "DONE" if self._has_metric(case_dir) else "missing_output"

        # Job still running: check for a stalled simulation.
        newest = self._newest_mtime(case_dir)
        baseline = newest if newest is not None else float(row["submit_time"])
        if time.time() - baseline > self.cfg.watchdog.stall_timeout:
            return "stalled"
        return None

    # ── Intervention ──────────────────────────────────────────────────────────

    def _resubmit(self, row: pd.Series, dispatcher: SlurmDispatcher) -> str:
        """Resubmit the case with scaled parameters; return action string."""
        case_dir = Path(str(row["case_dir"]))
        params = JobRegistry.params_of(row)
        resubmits = JobRegistry.resubmits_of(row)

        if resubmits >= self.cfg.watchdog.max_resubmits or not params:
            self.registry.update_state(str(row["job_id"]), "abandoned")
            return "abandoned"

        new_params: Dict[str, float] = dict(params)
        for pname, factor in self.cfg.watchdog.resubmit_scaling.items():
            if pname in new_params:
                new_params[pname] = new_params[pname] * factor

        # Remove stale logs/output so the failure is not immediately re-detected.
        for stale in [*self.cfg.watchdog.log_files, self.cfg.output_file]:
            stale_path = case_dir / stale
            if stale_path.exists():
                stale_path.unlink()

        job_name = f"{row['job_name']}_r{resubmits + 1}"
        self.builder.build(case_dir, new_params, job_name)
        new_id = dispatcher.dispatch(
            job_name, case_dir, new_params, resubmits=resubmits + 1
        )
        self.registry.update_state(str(row["job_id"]), "resubmitted")
        if new_id is None:
            return "resubmit_failed"
        return f"resubmitted_as_{new_id}"

    def intervene(
        self, row: pd.Series, dispatcher: SlurmDispatcher, reason: str
    ) -> None:
        job_id = str(row["job_id"])
        case_dir = Path(str(row["case_dir"]))

        dispatcher.cancel(job_id)
        action = self._resubmit(row, dispatcher)
        if action == "resubmit_failed":
            self.registry.update_state(job_id, "failed")
        self._log_status(
            job_id, str(row["job_name"]), case_dir, reason, action,
            JobRegistry.resubmits_of(row),
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_cycle(self) -> int:
        """One polling pass over all active jobs; returns #active remaining."""
        dispatcher = SlurmDispatcher(self.cfg, dry_run=self.dry_run)
        active = self.registry.active_jobs()
        self._log.info("Watchdog cycle: %d active job(s).", len(active))
        for _, row in active.iterrows():
            verdict = self.check_job(row, dispatcher)
            if verdict is None:
                continue
            if verdict == "DONE":
                self.registry.update_state(str(row["job_id"]), "done")
                self._log.info(
                    "Job %s (%s) completed.", row["job_id"],
                    Path(str(row["case_dir"])).name,
                )
            else:
                self._log.warning(
                    "Job %s (%s) FAILED: %s", row["job_id"],
                    Path(str(row["case_dir"])).name, verdict,
                )
                self.intervene(row, dispatcher, verdict)
        return len(self.registry.active_jobs())

    def run(self, once: bool = False) -> None:
        """Poll forever (or a single pass with ``once=True``)."""
        self._log.info(
            "Watchdog started: poll=%.0fs stall_timeout=%.0fs max_resubmits=%d",
            self.cfg.watchdog.poll_interval,
            self.cfg.watchdog.stall_timeout,
            self.cfg.watchdog.max_resubmits,
        )
        while True:
            self.run_cycle()
            if once:
                return
            time.sleep(self.cfg.watchdog.poll_interval)


# ═════════════════════════════════════════════════════════════════════════════
#  sbatch agent deployment
# ═════════════════════════════════════════════════════════════════════════════

def deploy_as_agent(
    cfg: FrameworkConfig,
    config_path: Path,
    dry_run: bool = False,
) -> Optional[str]:
    """Submit the watchdog itself as an independent sbatch background agent."""
    log = logging.getLogger("varify.watchdog")
    script = cfg.sweep_root / "watchdog_agent.sh"
    directives = "\n".join(
        f"#SBATCH --{k}={v}" for k, v in cfg.watchdog.agent_directives.items()
    )
    script.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=varify_watchdog\n"
        "#SBATCH --output=watchdog_agent.log\n"
        "#SBATCH --error=watchdog_agent.log\n"
        f"{directives}\n\n"
        f"cd {Path.cwd().resolve()}\n"
        f"exec python main.py --mode watch --config {config_path.resolve()}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | 0o111)
    cmd = f"sbatch {script.resolve()}"
    if dry_run:
        log.info("[DRY-RUN] Would deploy watchdog agent: %s", cmd)
        return "DRY_RUN"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.error("Agent submission failed: %s", result.stderr.strip())
        return None
    log.info("Watchdog agent submitted: %s", result.stdout.strip())
    return result.stdout.strip()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Varify SLURM watchdog daemon")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--once", action="store_true",
                        help="Run a single polling pass and exit.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    Watchdog(load_config(args.config), dry_run=args.dry_run).run(once=args.once)


if __name__ == "__main__":
    main()
