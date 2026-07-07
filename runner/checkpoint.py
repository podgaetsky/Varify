"""Automated checkpoint & resumable run context (standard library only).

Designed around SLURM wall-time limits: SLURM delivers ``SIGTERM`` shortly
before killing a job (and ``SIGUSR1`` when submitted with
``--signal=USR1@60``).  ``CheckpointManager`` traps both, flips a
``stop_requested`` flag that long-running strategies poll between units of
work, and guarantees the last state snapshot is on disk before the
allocation dies.  Snapshots are JSON, written atomically, so a checkpoint is
either the complete previous state or the complete new one — never a torn
file.  On restart, the strategy reloads the snapshot and continues exactly
where it stopped.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, Dict, Optional, Type

from utils.io_handlers import read_text_safe, write_atomic

_TRAPPED_SIGNALS = (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)


class CheckpointManager:
    """Atomic JSON checkpoints + wall-time signal trapping.

    Usage::

        ckpt = CheckpointManager(run_dir / "checkpoint.json")
        state = ckpt.load() or {"next_index": 0, "results": []}
        with ckpt:                        # installs SIGTERM/SIGUSR1/SIGINT trap
            for i in range(state["next_index"], n):
                state["results"].append(work(i))
                state["next_index"] = i + 1
                ckpt.maybe_save(state)    # rate-limited autosave
                if ckpt.stop_requested:   # wall-time approaching
                    ckpt.save(state)      # final flush
                    break
    """

    def __init__(
        self,
        path: Path,
        autosave_interval: float = 15.0,
    ) -> None:
        self.path = Path(path)
        self.autosave_interval = autosave_interval
        self.stop_requested: bool = False
        self.trapped_signal: Optional[int] = None
        self._last_save: float = 0.0
        self._saves: int = 0
        self._prev_handlers: Dict[int, Any] = {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Optional[Dict[str, Any]]:
        """Return the last committed state, or None if no checkpoint exists."""
        if not self.path.exists():
            return None
        try:
            state = json.loads(read_text_safe(self.path))
        except (ValueError, OSError):
            return None
        return state.get("state") if isinstance(state, dict) else None

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically commit *state* (temp file + rename)."""
        self._saves += 1
        payload = {
            "_meta": {
                "schema": 1,
                "saved_at_unix": time.time(),
                "save_count": self._saves,
                "interrupted": self.stop_requested,
            },
            "state": state,
        }
        write_atomic(self.path, json.dumps(payload, indent=2, default=str))
        self._last_save = time.monotonic()

    def maybe_save(self, state: Dict[str, Any]) -> bool:
        """Save if the autosave interval elapsed (or a stop was requested)."""
        due = (time.monotonic() - self._last_save) >= self.autosave_interval
        if due or self.stop_requested:
            self.save(state)
            return True
        return False

    def clear(self) -> None:
        """Delete the checkpoint (call after a run completes cleanly)."""
        if self.path.exists():
            self.path.unlink()

    # ── Signal trapping ───────────────────────────────────────────────────────

    def _handler(self, signum: int, _frame: Optional[FrameType]) -> None:
        self.stop_requested = True
        self.trapped_signal = signum

    def install(self) -> None:
        for sig in _TRAPPED_SIGNALS:
            try:
                self._prev_handlers[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass  # not the main thread / unsupported platform

    def restore(self) -> None:
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers.clear()

    def __enter__(self) -> "CheckpointManager":
        self.install()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.restore()
