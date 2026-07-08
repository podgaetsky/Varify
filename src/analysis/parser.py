"""Automated result extraction from run folders.

``ResultParser`` migrates the legacy ``ResultHarvester`` unchanged: it walks
case directories, regex-parses the scalar metric from each output file (NaN
on any failure) and assembles a single sorted ``pandas`` DataFrame saved to
CSV.  New here: the compiled frame can additionally be persisted to a SQLite
table, and the parser exposes generic loaders for the scan results and the
MCMC chain.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from varify.src.common.config import FrameworkConfig
from varify.src.scanner.base import Scanner


class ResultParser:
    """Walks run folders, parses output files, compiles DataFrame/SQLite."""

    def __init__(
        self, cfg: FrameworkConfig, scanner: Optional[Scanner] = None
    ) -> None:
        self.cfg = cfg
        self.scanner = scanner
        self._log = logging.getLogger("varify.parser")

    # ── Single-case parsing (legacy-verbatim semantics) ──────────────────────

    def parse_case(self, case_dir: Path, regex: Optional[str] = None) -> float:
        """Return the scalar captured by *regex* (default: output_regex), or NaN."""
        out_file = case_dir / self.cfg.output_file
        pattern = regex or self.cfg.output_regex
        try:
            if not out_file.exists():
                raise FileNotFoundError(f"{out_file} not found (job still running?)")
            text = out_file.read_text(encoding="utf-8", errors="replace")
            match = re.search(pattern, text)
            if not match:
                raise ValueError(f"Pattern {pattern!r} not found in {out_file}")
            val = float(match.group(1))
            self._log.debug("Parsed %s → %.6g", case_dir.name, val)
            return val
        except Exception as exc:
            self._log.warning("Skip %s — %s", case_dir.name, exc)
            return float("nan")

    def parse_log_prob(self, case_dir: Path) -> float:
        """Parse the MCMC log-probability from a finished walker job."""
        return self.parse_case(case_dir, regex=self.cfg.mcmc.log_prob_regex)

    def wait_for_output(
        self, case_dir: Path, timeout: float, poll_interval: float
    ) -> bool:
        """Block until the output file exists or *timeout* seconds elapsed."""
        out_file = case_dir / self.cfg.output_file
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if out_file.exists():
                return True
            time.sleep(poll_interval)
        return False

    # ── Full-scan harvesting ──────────────────────────────────────────────────

    def harvest(self) -> pd.DataFrame:
        """Compile every existing case of the configured scan into a DataFrame."""
        assert self.scanner is not None, "scanner required for harvest()"
        cases = self.scanner.existing_cases()
        self._log.info("Harvesting %d case directories…", len(cases))
        rows: List[Dict[str, Any]] = []
        for gp, case_dir in cases:
            row: Dict[str, Any] = {
                f"param_{n}": gp.params[n] for n in self.cfg.all_names
            }
            row["output"] = self.parse_case(case_dir)
            rows.append(row)
        col_order = [f"param_{n}" for n in self.cfg.all_names] + ["output"]
        df = pd.DataFrame(rows, columns=col_order)
        sort_cols = [f"param_{n}" for n in self.cfg.swept_names] or col_order[:-1]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df.to_csv(self.cfg.results_csv, index=False)
        self._log.info(
            "Saved %d rows → %s  (NaN: %d)",
            len(df), self.cfg.results_csv, df["output"].isna().sum(),
        )
        return df

    # ── Persistence & loading helpers ─────────────────────────────────────────

    def to_sqlite(self, df: pd.DataFrame, table: Optional[str] = None) -> Path:
        """Write *df* to the configured SQLite database (table replaced)."""
        db_path = self.cfg.analysis.sqlite_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        table_name = table or self.cfg.analysis.sqlite_table
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        self._log.info("Saved %d rows → %s (table '%s')", len(df), db_path, table_name)
        return db_path

    def load_results(self) -> Optional[pd.DataFrame]:
        if not self.cfg.results_csv.exists():
            self._log.error(
                "No results CSV at %s. Run '--mode analyze' after the scan "
                "finished (or check the workspace).", self.cfg.results_csv,
            )
            return None
        df = pd.read_csv(self.cfg.results_csv)
        self._log.info("Loaded %d rows from %s", len(df), self.cfg.results_csv)
        return df

    def load_chain(self) -> Optional[pd.DataFrame]:
        if not self.cfg.mcmc.chain_csv.exists():
            return None
        df = pd.read_csv(self.cfg.mcmc.chain_csv)
        self._log.info("Loaded chain from %s (%d rows)", self.cfg.mcmc.chain_csv, len(df))
        return df

    def load_optimization_history(self) -> Optional[pd.DataFrame]:
        if not self.cfg.optimizer.history_csv.exists():
            return None
        df = pd.read_csv(self.cfg.optimizer.history_csv)
        self._log.info(
            "Loaded optimization history from %s (%d rows)",
            self.cfg.optimizer.history_csv, len(df),
        )
        return df
