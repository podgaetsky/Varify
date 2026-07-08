"""Signature-introspecting dispatcher for user analysis functions.

Migrated verbatim from the legacy ``AnalysisDispatcher``:

* frame mode — fn declares ``df`` or ``cfg`` (or ``**kwargs``): called once
  with the full DataFrame-derived keyword pool;
* row mode   — all other fns: called once per non-NaN result row with scalar
  keyword arguments.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List

import pandas as pd

from varify.src.common.config import FrameworkConfig


class AnalysisDispatcher:
    """Forwards exactly the kwargs each registered analysis fn declares."""

    _FRAME_TRIGGERS = frozenset({"df", "cfg"})

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self._log = logging.getLogger("varify.analysis")

    @staticmethod
    def _declared_params(fn: Callable) -> List[str]:
        sig = inspect.signature(fn)
        return [
            name for name, p in sig.parameters.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

    @staticmethod
    def _has_var_keyword(fn: Callable) -> bool:
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fn).parameters.values()
        )

    def _is_frame_mode(self, fn: Callable) -> bool:
        return bool(set(self._declared_params(fn)) & self._FRAME_TRIGGERS) \
               or self._has_var_keyword(fn)

    def _frame_pool(self, df: pd.DataFrame) -> Dict[str, Any]:
        pool: Dict[str, Any] = {
            "df": df, "cfg": self.cfg,
            "output": df["output"].to_numpy(float),
        }
        for name in self.cfg.all_names:
            col = f"param_{name}"
            if col in df.columns:
                pool[name] = df[col].to_numpy(float)
        return pool

    @staticmethod
    def _row_pool(row: pd.Series, all_names: List[str]) -> Dict[str, Any]:
        pool: Dict[str, Any] = {"output": float(row["output"])}
        for name in all_names:
            col = f"param_{name}"
            if col in row.index:
                pool[name] = float(row[col])
        return pool

    def _filter_kwargs(self, fn: Callable, pool: Dict[str, Any]) -> Dict[str, Any]:
        if self._has_var_keyword(fn):
            return pool
        declared = self._declared_params(fn)
        return {k: pool[k] for k in declared if k in pool}

    def _run_one(self, fn: Callable, df: pd.DataFrame) -> None:
        fn_name = getattr(fn, "__name__", repr(fn))
        try:
            if self._is_frame_mode(fn):
                fn(**self._filter_kwargs(fn, self._frame_pool(df)))
            else:
                for _, row in df.dropna(subset=["output"]).iterrows():
                    fn(**self._filter_kwargs(
                        fn, self._row_pool(row, self.cfg.all_names)))
        except Exception as exc:
            self._log.error(
                "Analysis function '%s' raised: %s", fn_name, exc, exc_info=True
            )

    def run_all(self, df: pd.DataFrame) -> None:
        if not self.cfg.analysis_fns:
            self._log.info("No analysis functions registered.")
            return
        self._log.info(
            "Running %d analysis function(s) on %d rows (%d valid)…",
            len(self.cfg.analysis_fns), len(df), df["output"].notna().sum(),
        )
        for fn in self.cfg.analysis_fns:
            self._log.info("  → %s", getattr(fn, "__name__", repr(fn)))
            self._run_one(fn, df)
        self._log.info("Analysis complete.")
