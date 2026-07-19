"""Post-processing of simulated output against experimental data.

Provides an alternative to the regex-scalar objective used by
:class:`~src.optimizer.base.BaseOptimizer`: a simulated two-column curve is
interpolated (spline or linear) and evaluated against an experimental
two-column dataset on the overlap of their x-domains, yielding a scalar loss
(MSE, RMSE, MAE, Huber, reduced chi-squared, or a user-supplied callable)
that optimizers can minimize. See :func:`curve_loss` for the generalized
entry point and :func:`spline_mse` for the original spline+MSE special case,
kept as a thin backward-compatible wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

log = logging.getLogger("varify.postprocess")


def load_xy(
    path: Path, x_col: int = 0, y_col: int = 1, err_col: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read a two- (or three-) column dataset, tolerant of CSV or whitespace
    delimiting.

    Comment lines starting with ``#`` and blank lines are skipped. The
    delimiter (comma vs. whitespace) is auto-detected per line. Rows are
    sorted by x and duplicate x values are dropped (splines require a
    strictly increasing x-grid).

    When *err_col* is given, a third column is also read (e.g. an
    experimental uncertainty sigma for :func:`chi2`) and returned as a third
    array, aligned row-for-row with *x* and *y* (rows missing any of the
    three columns are skipped entirely, so the arrays never drift out of
    alignment). Omitting *err_col* preserves the original two-array return.
    """
    xs: list = []
    ys: list = []
    es: list = []
    needed = max(x_col, y_col, err_col if err_col is not None else -1)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            else:
                parts = line.split()
            if len(parts) <= needed:
                continue
            try:
                x_val = float(parts[x_col])
                y_val = float(parts[y_col])
                e_val = float(parts[err_col]) if err_col is not None else None
            except ValueError:
                continue
            xs.append(x_val)
            ys.append(y_val)
            if err_col is not None:
                es.append(e_val)

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    e_arr = np.asarray(es, dtype=float) if err_col is not None else None

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if e_arr is not None:
        e_arr = e_arr[order]

    # Drop duplicate x values, keeping the first occurrence.
    if x_arr.size > 0:
        keep = np.ones(x_arr.size, dtype=bool)
        keep[1:] = np.diff(x_arr) != 0
        x_arr = x_arr[keep]
        y_arr = y_arr[keep]
        if e_arr is not None:
            e_arr = e_arr[keep]

    if err_col is not None:
        return x_arr, y_arr, e_arr
    return x_arr, y_arr


def write_xy(
    path: Path,
    x: Iterable[float],
    y: Iterable[float],
    header: Optional[str] = None,
) -> Path:
    """Write a two-column whitespace-delimited ``(x, y)`` file, round-trip
    readable by :func:`load_xy`.

    Meant for post-job analysis hooks (``analysis.post_job_fns``) that
    reduce a raw per-case simulation file into the simulated curve
    ``optimizer.postprocess``/:func:`curve_loss` compares against
    ``optimizer.experimental_data``. Rows are written in the given order
    (sorted by x is the caller's responsibility, same as *load_xy* expects
    for spline fitting); parent directories are created if missing.
    """
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"x and y must be the same length (got {x_arr.size} vs {y_arr.size})"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {header}"] if header else []
    lines.extend(f"{xi:.10g}\t{yi:.10g}" for xi, yi in zip(x_arr, y_arr))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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



# ═════════════════════════════════════════════════════════════════════════════
#  Loss primitives
# ═════════════════════════════════════════════════════════════════════════════
#
# All operate on two (optionally three, for chi2) aligned 1-D arrays and are
# NaN-safe: pairs where either y_pred or y_ref is non-finite are dropped
# before reduction; if nothing remains, NaN is returned rather than raising.

def _finite_pairs(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Drop indices where any of *arrays* is non-finite; return the rest."""
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in arrs)


def mse(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """Mean squared error."""
    yp, yr = _finite_pairs(y_pred, y_ref)
    if yp.size == 0:
        return float("nan")
    return float(np.mean((yp - yr) ** 2))


def rmse(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """Root mean squared error."""
    m = mse(y_pred, y_ref)
    return float(np.sqrt(m)) if m == m else float("nan")


def mae(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """Mean absolute error."""
    yp, yr = _finite_pairs(y_pred, y_ref)
    if yp.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yp - yr)))


def huber(y_pred: np.ndarray, y_ref: np.ndarray, delta: float = 1.0) -> float:
    """Mean Huber loss: quadratic within *delta* of the residual, linear
    beyond it (robust to outliers relative to plain MSE)."""
    yp, yr = _finite_pairs(y_pred, y_ref)
    if yp.size == 0:
        return float("nan")
    abs_resid = np.abs(yp - yr)
    quad = np.minimum(abs_resid, delta)
    lin = abs_resid - quad
    return float(np.mean(0.5 * quad ** 2 + delta * lin))


def chi2(
    y_pred: np.ndarray, y_ref: np.ndarray, y_err: Optional[np.ndarray]
) -> float:
    """Reduced chi-squared: mean of ``((pred - ref) / err) ** 2``.

    Zero or NaN entries in *y_err* are treated as 1.0 (with a logged
    warning) rather than propagating a division-by-zero or NaN. *y_err* of
    ``None`` is equivalent to an all-ones error array (i.e. reduces to MSE).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    y_err = np.ones_like(y_pred) if y_err is None else np.asarray(y_err, dtype=float)

    mask = np.isfinite(y_pred) & np.isfinite(y_ref)
    yp, yr, ye = y_pred[mask], y_ref[mask], y_err[mask]
    if yp.size == 0:
        return float("nan")

    bad_err = ~np.isfinite(ye) | (ye == 0)
    if np.any(bad_err):
        log.warning(
            "chi2: %d zero/NaN error value(s) replaced with 1.0",
            int(np.sum(bad_err)),
        )
        ye = np.where(bad_err, 1.0, ye)

    return float(np.mean(((yp - yr) / ye) ** 2))


LOSSES: Dict[str, Callable[..., float]] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "huber": huber,
    "chi2": chi2,
}


LossSpec = Union[str, Callable[[np.ndarray, np.ndarray], float]]


def compute_loss(
    y_pred: np.ndarray,
    y_ref: np.ndarray,
    loss: LossSpec = "mse",
    y_err: Optional[np.ndarray] = None,
    **kw,
) -> float:
    """Dispatch to a named loss in :data:`LOSSES`, or call *loss* directly.

    *loss* may be one of the registered names (``"mse"``, ``"rmse"``,
    ``"mae"``, ``"huber"``, ``"chi2"``) or an arbitrary callable taken as-is
    with signature ``loss(y_pred, y_ref) -> float``. Extra keyword
    arguments are forwarded to the named loss (currently only ``delta`` for
    ``"huber"`` is consumed); *y_err* is forwarded only to ``"chi2"``.
    """
    if callable(loss):
        return float(loss(y_pred, y_ref))

    name = str(loss).lower()
    fn = LOSSES.get(name)
    if fn is None:
        raise ValueError(f"Unknown loss {loss!r}. Available: {sorted(LOSSES)}")

    if name == "chi2":
        return fn(y_pred, y_ref, y_err)
    if name == "huber":
        return fn(y_pred, y_ref, delta=kw.get("delta", 1.0))
    return fn(y_pred, y_ref)


# ═════════════════════════════════════════════════════════════════════════════
#  Curve alignment + scoring
# ═════════════════════════════════════════════════════════════════════════════

def curve_loss(
    case_dir: Path,
    experimental_path: Path,
    sim_filename: str,
    loss: LossSpec = "mse",
    interp: str = "spline",
    k: int = 3,
    s: float = 0.0,
    x_range: Optional[Tuple[float, float]] = None,
    y_err_col: Optional[int] = None,
    loss_kwargs: Optional[Dict] = None,
) -> float:
    """Interpolate the simulated curve onto the experimental x-grid and
    score it against experiment with a pluggable loss.

    Reads the simulated curve from ``case_dir / sim_filename`` and the
    experimental curve from *experimental_path*, aligns the simulated curve
    onto the experimental x-grid over the domain overlap (further clipped
    by *x_range* if given) using either a spline fit (``interp="spline"``,
    via :func:`fit_spline` with degree *k* and smoothing *s*) or piecewise
    linear interpolation (``interp="linear"``, via ``numpy.interp``), and
    returns :func:`compute_loss` of the result against the experimental
    y-values.

    When *y_err_col* is given, that column of *experimental_path* is read
    as the per-point experimental uncertainty and passed through to the
    loss (relevant for ``loss="chi2"``). *loss_kwargs* are forwarded to
    :func:`compute_loss` as extra keyword arguments (e.g.
    ``{"delta": 2.0}`` for ``loss="huber"``).

    Never raises: any failure (missing file, too few points, no domain
    overlap, unknown interp/loss, scipy error) is logged as a warning and
    ``float("nan")`` is returned, which already routes to the penalty path
    in the optimizers.
    """
    loss_kwargs = dict(loss_kwargs or {})
    try:
        sim_path = Path(case_dir) / sim_filename
        if not sim_path.exists():
            raise FileNotFoundError(f"simulated output {sim_path} not found")

        x_sim, y_sim = load_xy(sim_path)
        if y_err_col is not None:
            x_exp, y_exp, y_err_exp = load_xy(
                Path(experimental_path), err_col=y_err_col
            )
        else:
            x_exp, y_exp = load_xy(Path(experimental_path))
            y_err_exp = None

        if x_sim.size < 2 or x_exp.size < 1:
            raise ValueError(
                f"insufficient points (sim={x_sim.size}, exp={x_exp.size})"
            )

        if interp == "spline":
            predictor: Callable[[np.ndarray], np.ndarray] = fit_spline(
                x_sim, y_sim, k=k, s=s
            )
        elif interp == "linear":
            predictor = lambda x: np.interp(x, x_sim, y_sim)  # noqa: E731
        else:
            raise ValueError(
                f"Unknown interp {interp!r} (expected 'spline' or 'linear')"
            )

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

        y_pred = predictor(x_eval)
        y_err_eval = y_err_exp[mask] if y_err_exp is not None else None

        return compute_loss(
            y_pred, y_eval, loss=loss, y_err=y_err_eval, **loss_kwargs
        )
    except Exception as exc:
        log.warning("curve_loss failed for %s — %s", case_dir, exc)
        return float("nan")


def spline_mse(
    case_dir: Path,
    experimental_path: Path,
    sim_filename: str,
    k: int = 3,
    s: float = 0.0,
    x_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Fit a spline to the simulated curve and score it against experiment
    with MSE.

    Back-compat wrapper over :func:`curve_loss` with
    ``loss="mse", interp="spline"``; identical signature and behavior to
    the original implementation. Never raises (see :func:`curve_loss`).
    """
    return curve_loss(
        case_dir,
        experimental_path,
        sim_filename,
        loss="mse",
        interp="spline",
        k=k,
        s=s,
        x_range=x_range,
    )
