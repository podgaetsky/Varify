"""Agnostic workflow runner: one lifecycle for every scientific workflow.

Decoupling contract
───────────────────
A workflow is a declarative :class:`RunSpec`: a model callable plus plain
data (bounds, options) plus **two strings** —

* ``strategy`` selects the mathematical algorithm from a registry
  (``optimize`` / ``grid`` / ``mcmc`` / ``mcmc_diagnostic`` / ``benchmark``
  built in; user code adds more via :func:`register_strategy`);
* ``runtime`` selects where it executes: ``"local"`` runs in-process,
  ``"slurm"`` renders ``sbatch_template.sh`` and resubmits the *calling
  script itself* to the scheduler, re-entering in local mode inside the
  allocation (guarded by the ``VARIFY_INSIDE_SLURM`` environment variable).

Changing algorithm or execution target therefore touches exactly one string;
cluster-specific values live only in the template and the ``slurm`` mapping.

Lifecycle (innovation layer built in)
─────────────────────────────────────
pre-flight validation → provenance capture (seeded, immutable) →
timestamped run directory (or resume of an interrupted one) →
checkpoint-guarded strategy execution → telemetry/matrix export → summary.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from runner.checkpoint import CheckpointManager
from runner.preflight import PreflightReport, run_preflight
from runner.provenance import capture_provenance, write_provenance
from utils.io_handlers import read_text_safe, write_atomic

log = logging.getLogger("varify.runner")

Bounds = Dict[str, Tuple[float, float]]
StrategyFn = Callable[["RunContext"], Dict[str, Any]]

_REGISTRY: Dict[str, StrategyFn] = {}

_INSIDE_SLURM_ENV = "VARIFY_INSIDE_SLURM"
_STATUS_FILE = "STATUS"


def register_strategy(name: str) -> Callable[[StrategyFn], StrategyFn]:
    """Register a strategy under *name* (used as ``RunSpec.strategy``)."""

    def _decorator(fn: StrategyFn) -> StrategyFn:
        _REGISTRY[name] = fn
        return fn

    return _decorator


def available_strategies() -> List[str]:
    return sorted(_REGISTRY)


# ═════════════════════════════════════════════════════════════════════════════
#  Declarative run specification
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RunSpec:
    """Everything the runner needs, as data + one model callable."""

    name: str
    strategy: str
    model: Optional[Callable[..., float]] = None
    bounds: Bounds = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    runtime: str = "local"                      # "local" | "slurm"
    results_root: Path = Path("results")
    slurm: Dict[str, Any] = field(default_factory=dict)

    def sbatch_template(self) -> Path:
        """The dispatch template: explicit override or the repo-root default."""
        default = Path(__file__).resolve().parents[1] / "sbatch_template.sh"
        return Path(self.slurm.get("template", default))

    def meta(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "model": getattr(self.model, "__name__", repr(self.model)),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "options": {
                k: v for k, v in self.options.items() if _is_jsonable(v)
            },
            "runtime": self.runtime,
        }


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  Run context handed to strategies
# ═════════════════════════════════════════════════════════════════════════════

class RunContext:
    """Per-run services: RNG, checkpointing, telemetry & matrix export."""

    def __init__(
        self,
        spec: RunSpec,
        run_dir: Path,
        checkpoint: CheckpointManager,
        provenance: Dict[str, Any],
    ) -> None:
        import random

        self.spec = spec
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.provenance = provenance
        self.rng = random.Random(provenance["seed"])
        self.log = logging.getLogger(f"varify.runner.{spec.name}")

    # ── Export helpers ────────────────────────────────────────────────────────

    def save_json(self, name: str, payload: Dict[str, Any]) -> Path:
        return write_atomic(
            self.run_dir / name, json.dumps(payload, indent=2, default=str)
        )

    def save_rows(
        self, name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> Path:
        path = self.run_dir / name
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(list(header))
            writer.writerows(rows)
        return path

    def save_matrix(
        self,
        name: str,
        matrix: Sequence[Sequence[float]],
        row_labels: Optional[Sequence[Any]] = None,
        col_labels: Optional[Sequence[Any]] = None,
    ) -> Path:
        """Export a labelled 2-D visualization matrix as CSV."""
        path = self.run_dir / name
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if col_labels is not None:
                writer.writerow([""] + list(col_labels))
            for i, row in enumerate(matrix):
                label = [row_labels[i]] if row_labels is not None else []
                writer.writerow(label + list(row))
        return path


# ═════════════════════════════════════════════════════════════════════════════
#  Runner
# ═════════════════════════════════════════════════════════════════════════════

class WorkflowRunner:
    """Executes one :class:`RunSpec` through the full resilient lifecycle."""

    def __init__(self, spec: RunSpec) -> None:
        if spec.strategy not in _REGISTRY:
            raise ValueError(
                f"Unknown strategy {spec.strategy!r}; "
                f"available: {available_strategies()}"
            )
        self.spec = spec

    # ── Run-directory management ──────────────────────────────────────────────

    def _new_run_dir(self) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        d = self.spec.results_root / f"{stamp}__{self.spec.name}"
        suffix = 0
        while d.exists():
            suffix += 1
            d = self.spec.results_root / f"{stamp}__{self.spec.name}_{suffix}"
        d.mkdir(parents=True)
        return d

    def _resumable_run_dir(self) -> Optional[Path]:
        root = self.spec.results_root
        if not root.exists():
            return None
        candidates = sorted(
            d for d in root.iterdir()
            if d.is_dir() and d.name.endswith(f"__{self.spec.name}")
            and (d / "checkpoint.json").exists()
            and (d / _STATUS_FILE).exists()
            and read_text_safe(d / _STATUS_FILE).strip() == "interrupted"
        )
        return candidates[-1] if candidates else None

    # ── SLURM self-dispatch ───────────────────────────────────────────────────

    def _submit_to_slurm(self) -> Dict[str, Any]:
        """Render the sbatch template and resubmit the calling script."""
        slurm = dict(self.spec.slurm)
        slurm.pop("template", None)
        template = self.spec.sbatch_template()
        run_dir = self._new_run_dir()
        tokens: Dict[str, str] = {
            "JOB_NAME": slurm.pop("job_name", self.spec.name),
            "PARTITION": str(slurm.pop("partition", "compute")),
            "TIME": str(slurm.pop("time", "01:00:00")),
            "NODES": str(slurm.pop("nodes", 1)),
            "NTASKS": str(slurm.pop("ntasks", 1)),
            "CPUS_PER_TASK": str(
                slurm.pop("cpus_per_task",
                          self.spec.options.get("workers", 1))),
            "MEM": str(slurm.pop("mem", "2G")),
            "EXTRA_DIRECTIVES": "\n".join(
                f"#SBATCH --{k}={v}" for k, v in slurm.items()
            ),
            "PYTHON_BIN": sys.executable,
            "SCRIPT": str(Path(sys.argv[0]).resolve()),
            "SCRIPT_ARGS": " ".join(sys.argv[1:]),
            "WORKDIR": str(Path.cwd().resolve()),
            "RUN_DIR": str(run_dir.resolve()),
        }
        rendered = read_text_safe(template)
        for key, val in tokens.items():
            rendered = rendered.replace(f"@{key}@", val)
        script = run_dir / "job.sbatch"
        write_atomic(script, rendered)
        script.chmod(script.stat().st_mode | 0o111)

        result = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, timeout=30,
        )
        submission = {
            "mode": "slurm_submission",
            "run_dir": str(run_dir),
            "script": str(script),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        write_atomic(run_dir / "submission.json",
                     json.dumps(submission, indent=2))
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
        log.info("Submitted to SLURM: %s", result.stdout.strip())
        return submission

    # ── Main lifecycle ────────────────────────────────────────────────────────

    def run(
        self,
        resume: bool = True,
        strict_preflight: bool = True,
    ) -> Dict[str, Any]:
        spec = self.spec

        # 1. Pre-flight validation (before any allocation is consumed).
        report: PreflightReport = run_preflight(spec, strict=strict_preflight)
        log.info("\n%s", report.render())

        # 2. Runtime dispatch: hand off to SLURM unless already inside it.
        if spec.runtime == "slurm" and not os.environ.get(_INSIDE_SLURM_ENV):
            return self._submit_to_slurm()

        # 3. Run directory: resume an interrupted run or start a new one.
        run_dir = (self._resumable_run_dir() if resume else None)
        resumed = run_dir is not None
        if run_dir is None:
            run_dir = self._new_run_dir()
        log.info("%s run dir: %s", "Resuming" if resumed else "New", run_dir)
        write_atomic(run_dir / _STATUS_FILE, "running")

        # 4. Provenance capture (seeds the global RNGs).
        provenance = capture_provenance(spec.seed, extra={"resumed": resumed})
        write_provenance(provenance, run_dir / "provenance.json")
        write_atomic(run_dir / "preflight.json",
                     json.dumps(report.to_dict(), indent=2))

        # 5. Checkpoint-guarded strategy execution.
        checkpoint = CheckpointManager(run_dir / "checkpoint.json")
        ctx = RunContext(spec, run_dir, checkpoint, provenance)
        t0 = time.monotonic()
        with checkpoint:
            result = _REGISTRY[spec.strategy](ctx)
        elapsed = time.monotonic() - t0
        interrupted = checkpoint.stop_requested

        # 6. Telemetry export: results payload embeds the provenance record.
        telemetry: Dict[str, Any] = {
            "spec": spec.meta(),
            "result": result,
            "perf": {
                "elapsed_s": round(elapsed, 3),
                "interrupted": interrupted,
                "trapped_signal": checkpoint.trapped_signal,
            },
            "provenance": provenance,
        }
        ctx.save_json("telemetry.json", telemetry)
        write_atomic(run_dir / _STATUS_FILE,
                     "interrupted" if interrupted else "completed")
        if not interrupted:
            checkpoint.clear()

        # 7. Human-readable summary.
        summary = [
            f"run       : {spec.name} [{spec.strategy}]",
            f"status    : {'INTERRUPTED (resumable)' if interrupted else 'completed'}",
            f"elapsed   : {elapsed:.2f}s",
            f"seed      : {provenance['seed']}",
            f"git commit: {provenance['git'].get('commit', 'n/a')}",
            f"artifacts : {', '.join(sorted(p.name for p in run_dir.iterdir()))}",
        ]
        write_atomic(run_dir / "summary.txt", "\n".join(summary) + "\n")
        log.info("Run finished (%s) → %s",
                 "interrupted" if interrupted else "completed", run_dir)
        telemetry["run_dir"] = str(run_dir)
        return telemetry
