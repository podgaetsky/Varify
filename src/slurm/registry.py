"""Central registry of submitted jobs (consumed by the watchdog daemon).

Every dispatched job is appended to ``<workspace>/jobs_registry.csv`` with
its scheduler job id, case directory, parameter set (JSON) and lifecycle
state (``active`` → ``done`` / ``failed`` / ``resubmitted`` / ``abandoned``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_COLUMNS = [
    "job_id", "job_name", "case_dir", "params_json",
    "submit_time", "resubmits", "state",
]


class JobRegistry:
    """Append/update interface over the jobs CSV."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._log = logging.getLogger("varify.registry")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        df = pd.read_csv(self.path, dtype={"job_id": str})
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = "" if col != "resubmits" else 0
        return df

    def _save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path, index=False)

    def add(
        self,
        job_id: str,
        job_name: str,
        case_dir: Path,
        params: Dict[str, float],
        resubmits: int = 0,
    ) -> None:
        df = self.load()
        row = {
            "job_id": str(job_id),
            "job_name": job_name,
            "case_dir": str(case_dir.resolve()),
            "params_json": json.dumps(params),
            "submit_time": time.time(),
            "resubmits": resubmits,
            "state": "active",
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._save(df)
        self._log.debug("Registered job %s (%s)", job_id, case_dir.name)

    def update_state(self, job_id: str, state: str) -> None:
        df = self.load()
        mask = df["job_id"].astype(str) == str(job_id)
        if not mask.any():
            self._log.warning("Job %s not found in registry.", job_id)
            return
        df.loc[mask, "state"] = state
        self._save(df)

    def active_jobs(self) -> pd.DataFrame:
        df = self.load()
        return df[df["state"] == "active"].copy()

    @staticmethod
    def params_of(row: pd.Series) -> Dict[str, float]:
        try:
            raw = json.loads(row.get("params_json") or "{}")
            return {str(k): float(v) for k, v in raw.items()}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def resubmits_of(row: pd.Series) -> int:
        try:
            return int(row.get("resubmits", 0))
        except (TypeError, ValueError):
            return 0

    def find_by_case(self, case_dir: Path) -> Optional[pd.Series]:
        df = self.load()
        mask = df["case_dir"] == str(case_dir.resolve())
        if not mask.any():
            return None
        return df[mask].iloc[-1]
