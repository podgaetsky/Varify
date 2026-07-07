"""User hook functions referenced by name from ``config/config.yaml``.

Three kinds of hooks are supported:

* ``input_fn(case_dir, value, **params)`` — extra per-case input generation
  for one parameter (runs after template substitution);
* ``coupled_fn(driver_value) -> value``  — derives a coupled parameter from
  its driver parameter;
* analysis functions — registered under ``analysis.analysis_fns``; the
  dispatcher inspects the signature: functions declaring ``df``/``cfg`` (or
  ``**kwargs``) are called once with the full DataFrame pool, all others are
  called once per valid result row with scalar kwargs.

These examples are migrated unchanged from the legacy script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── input_fn hooks ────────────────────────────────────────────────────────────

def tau_input_fn(case_dir: Path, value: float, **params: Any) -> None:
    """Example input_fn: write a JSON sidecar for the tau parameter."""
    sidecar = case_dir / "tau_meta.json"
    sidecar.write_text(
        json.dumps({"tau": value, "all_params": params}, indent=2),
        encoding="utf-8",
    )


# ── coupled_fn hooks ─────────────────────────────────────────────────────────

def kappa_from_tau(tau: float) -> float:
    """kappa rides along with tau at half its value."""
    return tau / 2.0


# ── analysis functions ────────────────────────────────────────────────────────

def example_row_analysis(tau: float, gamma: float, output: float) -> None:
    ratio = output / tau if tau != 0 else float("nan")
    print(
        f"  [row]  tau={tau:.4g}  gamma={gamma:.4g}  "
        f"out={output:.6g}  out/tau={ratio:.4g}"
    )


def example_frame_analysis(
    df: pd.DataFrame, output: np.ndarray, cfg: Any
) -> None:
    valid = output[~np.isnan(output)]
    print(
        f"  [frame] n_valid={len(valid)}  mean={np.nanmean(output):.6g}"
        f"  std={np.nanstd(output):.6g}  swept={cfg.swept_names}"
    )


def example_sensitivity(df: pd.DataFrame, cfg: Any) -> None:
    for name in cfg.swept_names:
        col = f"param_{name}"
        grp = (
            df.dropna(subset=["output"])
            .groupby(col)["output"].mean()
            .reset_index().sort_values(col)
        )
        if len(grp) < 2:
            continue
        x = grp[col].to_numpy(float)
        y = grp["output"].to_numpy(float)
        sens = np.abs(np.diff(y) / np.where(np.diff(x) != 0, np.diff(x), np.nan))
        print(
            f"  [sensitivity] {name}: max|dOut/d{name}|={np.nanmax(sens):.4g}"
            f"  at {name}≈{x[np.nanargmax(sens) + 1]:.4g}"
        )
