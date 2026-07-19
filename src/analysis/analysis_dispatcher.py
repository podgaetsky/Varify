"""Signature-introspecting dispatchers for user analysis functions.

Migrated verbatim from the legacy ``AnalysisDispatcher``:

* frame mode — fn declares ``df`` or ``cfg`` (or ``**kwargs``): called once
  with the full DataFrame-derived keyword pool;
* row mode   — all other fns: called once per non-NaN result row with scalar
  keyword arguments.

:class:`PostJobDispatcher` (below) shares the same signature-introspection
approach but fires per case directory right after that case's job is
confirmed complete, rather than once over the harvested DataFrame — see its
docstring.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from varify.src.common.config import FrameworkConfig

_FRAME_TRIGGERS = frozenset({"df", "cfg"})


def _declared_params(fn: Callable) -> List[str]:
    sig = inspect.signature(fn)
    return [
        name for name, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


def _has_var_keyword(fn: Callable) -> bool:
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(fn).parameters.values()
    )


def _filter_kwargs(fn: Callable, pool: Dict[str, Any]) -> Dict[str, Any]:
    """Forward exactly the kwargs *fn* declares (all of *pool* if it takes
    ``**kwargs``)."""
    if _has_var_keyword(fn):
        return pool
    declared = _declared_params(fn)
    return {k: pool[k] for k in declared if k in pool}


class AnalysisDispatcher:
    """Forwards exactly the kwargs each registered analysis fn declares."""

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self._log = logging.getLogger("varify.analysis")

    def _is_frame_mode(self, fn: Callable) -> bool:
        return bool(set(_declared_params(fn)) & _FRAME_TRIGGERS) \
               or _has_var_keyword(fn)

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

    def _run_one(self, fn: Callable, df: pd.DataFrame) -> None:
        fn_name = getattr(fn, "__name__", repr(fn))
        try:
            if self._is_frame_mode(fn):
                fn(**_filter_kwargs(fn, self._frame_pool(df)))
            else:
                for _, row in df.dropna(subset=["output"]).iterrows():
                    fn(**_filter_kwargs(
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


class PostJobDispatcher:
    """Runs per-case analysis hooks right after that case's job completes.

    Registered under ``analysis.post_job_fns`` in config.yaml — each entry
    names a hook function plus an optional static ``kwargs`` mapping (e.g.
    which raw file to read, which columns to use). A typical use is
    reducing a raw simulation log/file into the two-column ``(x, y)`` curve
    file that ``optimizer.postprocess``/:func:`~src.analysis.postprocess.
    curve_loss` expects at ``case_dir / sim_output_file`` — see
    :func:`~src.analysis.postprocess.write_xy` and the ``config/hooks.py``
    example.

    Every hook is called once per finished case with a pool of keyword
    arguments: ``case_dir``, ``job_id``, ``output_file`` (``case_dir /
    cfg.output_file``, the raw file the scan/optimizer regex-scrapes),
    ``cfg`` and one scalar entry per case parameter (e.g. ``tau``,
    ``gamma``). The function's signature is introspected exactly like
    :class:`AnalysisDispatcher`: only declared parameter names are
    forwarded, unless the function takes ``**kwargs``. Each hook's static
    ``kwargs`` (from config) are merged into the pool first and take
    precedence over same-named pool entries, so config can pin e.g. a
    source filename or column index per hook.

    A hook raising is logged and does not stop the remaining hooks or the
    calling scan/optimizer loop.
    """

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self._log = logging.getLogger("varify.analysis.post_job")

    def run_case(
        self,
        case_dir: Path,
        params: Dict[str, float],
        job_id: Optional[str] = None,
    ) -> None:
        hooks: List[Tuple[Callable, Dict[str, Any]]] = self.cfg.post_job_fns
        if not hooks:
            return
        pool: Dict[str, Any] = {
            "case_dir": Path(case_dir),
            "job_id": job_id,
            "output_file": Path(case_dir) / self.cfg.output_file,
            "cfg": self.cfg,
            **{name: float(val) for name, val in params.items()},
        }
        for fn, static_kwargs in hooks:
            fn_name = getattr(fn, "__name__", repr(fn))
            merged = {**pool, **static_kwargs}
            try:
                fn(**_filter_kwargs(fn, merged))
            except Exception as exc:
                self._log.error(
                    "post_job_fn '%s' raised for %s: %s",
                    fn_name, Path(case_dir).name, exc, exc_info=True,
                )
