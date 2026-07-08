"""Spline-based post-processing of simulated output against experimental data.

Provides an alternative to the regex-scalar objective used by
:class:`~src.optimizer.base.BaseOptimizer`: a simulated two-column curve is
fit with a spline and evaluated against an experimental two-column dataset on
the overlap of their x-domains, yielding a mean-squared-error metric that
optimizers can minimize.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

log = logging.getLogger("varify.postprocess")


def load_xy(path: Path, x_col: int = 0, y_col: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Read a two-column dataset, tolerant of CSV or whitespace delimiting.

    Comment lines starting with ``#`` and blank lines are skipped. The
    delimiter (comma vs. whitespace) is auto-detected per line. Rows are
    sorted by x and duplicate x values are dropped (splines require a
    strictly increasing x-grid).
    """
    xs: list = []
    ys: list = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            else:
                parts = line.split()
            if len(parts) <= max(x_col, y_col):
                continue
            try:
                x_val = float(parts[x_col])
                y_val = float(parts[y_col])
            except ValueError:
                continue
            xs.append(x_val)
            ys.append(y_val)

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    # Drop duplicate x values, keeping the first occurrence.
    if x_arr.size > 0:
        keep = np.ones(x_arr.size, dtype=bool)
        keep[1:] = np.diff(x_arr) != 0
        x_arr = x_arr[keep]
        y_arr = y_arr[keep]

    return x_arr, y_arr


def fit_spline(
    x: np.ndarray, y: np.ndarray, k: int = 3, s: float = 0.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit and return a callable spline through (*x*, *y*).

    Uses :func:`scipy.interpolate.make_interp_spline` when *s* == 0 (exact
    interpolation), otherwise :class:`scipy.interpolate.UnivariateSpline`
    (smoothing). The spline degree *k* is reduced automatically (down to 1)
    when there are fewer than ``k + 1`` points available.

    Public because the plotting suite reuses it to overlay fitted curves.
    """
    from scipy.interpolate import UnivariateSpline, make_interp_spline

    n_points = len(x)
    eff_k = k
    while eff_k > 1 and n_points < eff_k + 1:
        eff_k -= 1

    if s == 0.0:
        return make_interp_spline(x, y, k=eff_k)
    return UnivariateSpline(x, y, k=eff_k, s=s)


def spline_mse(
    case_dir: Path,
    experimental_path: Path,
    sim_filename: str,
    k: int = 3,
    s: float = 0.0,
    x_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Fit a spline to the simulated curve and score it against experiment.

    Reads the simulated curve from ``case_dir / sim_filename`` and the
    experimental curve from *experimental_path*, fits a spline to the
    simulated data, evaluates it on the experimental x-grid restricted to
    the overlap of both x-domains (further clipped by *x_range* if given),
    and returns the mean squared error against the experimental y-values.

    Never raises: any failure (missing file, too few points, no domain
    overlap, scipy error) is logged as a warning and ``float("nan")`` is
    returned, which already routes to the penalty path in the optimizers.
    """
    try:
        sim_path = Path(case_dir) / sim_filename
        if not sim_path.exists():
            raise FileNotFoundError(f"simulated output {sim_path} not found")

        x_sim, y_sim = load_xy(sim_path)
        x_exp, y_exp = load_xy(Path(experimental_path))

        if x_sim.size < 2 or x_exp.size < 1:
            raise ValueError(
                f"insufficient points (sim={x_sim.size}, exp={x_exp.size})"
            )

        spline = fit_spline(x_sim, y_sim, k=k, s=s)

        lo = max(x_sim.min(), x_exp.min())
        hi = min(x_sim.max(), x_exp.max())
        if x_range is not None:
            lo = max(lo, x_range[0])
            hi = min(hi, x_range[1])

        if not (hi > lo):
            raise ValueError(
                f"no domain overlap between sim [{x_sim.min()}, {x_sim.max()}] "
                f"and exp [{x_exp.min()}, {x_exp.max()}]"
                + (f" clipped to {x_range}" if x_range is not None else "")
            )

        mask = (x_exp >= lo) & (x_exp <= hi)
        x_eval = x_exp[mask]
        y_eval = y_exp[mask]
        if x_eval.size == 0:
            raise ValueError("no experimental points fall within the overlap range")

        y_pred = spline(x_eval)
        return float(np.mean((y_pred - y_eval) ** 2))
    except Exception as exc:
        log.warning("spline_mse failed for %s — %s", case_dir, exc)
        return float("nan")
