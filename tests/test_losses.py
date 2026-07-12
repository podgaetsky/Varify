"""Tests for src.analysis.postprocess: loss primitives, compute_loss, and
curve_loss (the generalized interp+loss engine behind spline_mse).

Plain pytest-style ``test_*`` functions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from varify.src.analysis.postprocess import (
    chi2,
    compute_loss,
    curve_loss,
    huber,
    mae,
    mse,
    rmse,
)


def _write_xy(path: Path, xs, ys, delimiter: str = " ", header: bool = True) -> None:
    lines = []
    if header:
        lines.append("# x y  (comment line)")
        lines.append("")
    for x, y in zip(xs, ys):
        lines.append(f"{x}{delimiter}{y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_xye(path: Path, xs, ys, es, delimiter: str = " ") -> None:
    lines = ["# x y err", ""]
    for x, y, e in zip(xs, ys, es):
        lines.append(f"{x}{delimiter}{y}{delimiter}{e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dense_sine(n: int = 200, lo: float = 0.0, hi: float = 10.0):
    xs = np.linspace(lo, hi, n)
    ys = np.sin(xs)
    return xs, ys


# ── loss primitives: hand-computed values ──────────────────────────────────────

def test_mse_hand_computed() -> None:
    y_pred = np.array([1.0, 2.0, 3.0])
    y_ref = np.array([1.0, 2.0, 5.0])
    # squared errors: 0, 0, 4 -> mean = 4/3
    assert mse(y_pred, y_ref) == pytest.approx(4.0 / 3.0)


def test_rmse_hand_computed() -> None:
    y_pred = np.array([0.0, 0.0])
    y_ref = np.array([3.0, 4.0])
    # mse = (9+16)/2 = 12.5 -> rmse = sqrt(12.5)
    assert rmse(y_pred, y_ref) == pytest.approx(math.sqrt(12.5))


def test_mae_hand_computed() -> None:
    y_pred = np.array([1.0, 2.0, 3.0])
    y_ref = np.array([2.0, 2.0, 6.0])
    # abs errors: 1, 0, 3 -> mean = 4/3
    assert mae(y_pred, y_ref) == pytest.approx(4.0 / 3.0)


def test_huber_hand_computed_small_residual() -> None:
    # residual 0.5 < delta=1.0 -> quadratic branch: 0.5 * 0.5^2 = 0.125
    y_pred = np.array([0.5])
    y_ref = np.array([0.0])
    assert huber(y_pred, y_ref, delta=1.0) == pytest.approx(0.125)


def test_huber_hand_computed_large_residual() -> None:
    # residual 3.0 > delta=1.0 -> linear branch: 0.5*1^2 + 1*(3-1) = 0.5 + 2 = 2.5
    y_pred = np.array([3.0])
    y_ref = np.array([0.0])
    assert huber(y_pred, y_ref, delta=1.0) == pytest.approx(2.5)


def test_chi2_hand_computed() -> None:
    y_pred = np.array([1.0, 2.0])
    y_ref = np.array([0.0, 0.0])
    y_err = np.array([1.0, 2.0])
    # ((1-0)/1)^2 = 1 ; ((2-0)/2)^2 = 1 -> mean = 1
    assert chi2(y_pred, y_ref, y_err) == pytest.approx(1.0)


def test_chi2_zero_err_fallback(caplog) -> None:
    y_pred = np.array([2.0, 2.0])
    y_ref = np.array([0.0, 0.0])
    y_err = np.array([0.0, 2.0])
    # first point: err 0 -> treated as 1.0 -> ((2-0)/1)^2 = 4
    # second point: ((2-0)/2)^2 = 1
    # mean = 2.5
    with caplog.at_level("WARNING"):
        result = chi2(y_pred, y_ref, y_err)
    assert result == pytest.approx(2.5)
    assert any("zero/NaN" in rec.message for rec in caplog.records)


def test_chi2_nan_err_fallback() -> None:
    y_pred = np.array([1.0])
    y_ref = np.array([0.0])
    y_err = np.array([float("nan")])
    # NaN err -> treated as 1.0 -> ((1-0)/1)^2 = 1
    assert chi2(y_pred, y_ref, y_err) == pytest.approx(1.0)


def test_chi2_none_err_defaults_to_ones() -> None:
    y_pred = np.array([1.0, 2.0])
    y_ref = np.array([0.0, 0.0])
    assert chi2(y_pred, y_ref, None) == pytest.approx(mse(y_pred, y_ref))


def test_losses_nan_safe_drops_nonfinite_pairs() -> None:
    y_pred = np.array([1.0, float("nan"), 3.0])
    y_ref = np.array([1.0, 2.0, 3.0])
    assert mse(y_pred, y_ref) == pytest.approx(0.0)


def test_losses_all_nonfinite_returns_nan() -> None:
    y_pred = np.array([float("nan"), float("inf")])
    y_ref = np.array([1.0, 2.0])
    assert math.isnan(mse(y_pred, y_ref))
    assert math.isnan(mae(y_pred, y_ref))
    assert math.isnan(rmse(y_pred, y_ref))
    assert math.isnan(huber(y_pred, y_ref))
    assert math.isnan(chi2(y_pred, y_ref, None))


# ── compute_loss dispatch ───────────────────────────────────────────────────────

def test_compute_loss_by_name() -> None:
    y_pred = np.array([1.0, 2.0, 3.0])
    y_ref = np.array([1.0, 2.0, 5.0])
    assert compute_loss(y_pred, y_ref, loss="mse") == pytest.approx(4.0 / 3.0)
    assert compute_loss(y_pred, y_ref, loss="MSE") == pytest.approx(4.0 / 3.0)


def test_compute_loss_huber_kwarg() -> None:
    y_pred = np.array([3.0])
    y_ref = np.array([0.0])
    assert compute_loss(
        y_pred, y_ref, loss="huber", delta=1.0
    ) == pytest.approx(2.5)


def test_compute_loss_chi2_uses_y_err() -> None:
    y_pred = np.array([1.0, 2.0])
    y_ref = np.array([0.0, 0.0])
    y_err = np.array([1.0, 2.0])
    assert compute_loss(
        y_pred, y_ref, loss="chi2", y_err=y_err
    ) == pytest.approx(1.0)


def test_compute_loss_callable() -> None:
    y_pred = np.array([1.0, 2.0])
    y_ref = np.array([1.0, 2.0])

    def custom(y_pred, y_ref):
        return float(np.sum(np.abs(y_pred - y_ref)) + 42.0)

    assert compute_loss(y_pred, y_ref, loss=custom) == pytest.approx(42.0)


def test_compute_loss_unknown_name_raises() -> None:
    y_pred = np.array([1.0])
    y_ref = np.array([1.0])
    with pytest.raises(ValueError):
        compute_loss(y_pred, y_ref, loss="not-a-real-loss")


# ── curve_loss: interpolation modes ─────────────────────────────────────────────

def test_curve_loss_linear_identical_curves_near_zero(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case_linear"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys)

    loss = curve_loss(case_dir, exp_path, "output.dat", loss="mse", interp="linear")
    assert loss == loss  # not NaN
    assert loss < 1e-6


def test_curve_loss_spline_identical_curves_near_zero(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case_spline"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys)

    loss = curve_loss(case_dir, exp_path, "output.dat", loss="mse", interp="spline")
    assert loss == loss  # not NaN
    assert loss < 1e-10


def test_curve_loss_mae_constant_offset(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case_offset"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys + 0.5)

    loss = curve_loss(
        case_dir, exp_path, "output.dat", loss="mae", interp="linear"
    )
    assert loss == pytest.approx(0.5, abs=1e-6)


def test_curve_loss_unknown_interp_returns_nan(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case_bad_interp"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys)

    loss = curve_loss(case_dir, exp_path, "output.dat", interp="quadratic")
    assert math.isnan(loss)


def test_curve_loss_y_err_col_chi2(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case_err"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    errs = np.full_like(xs, 2.0)
    _write_xye(exp_path, xs, ys, errs)

    loss = curve_loss(
        case_dir, exp_path, "output.dat",
        loss="chi2", interp="linear", y_err_col=2,
    )
    assert loss == loss  # not NaN
    assert loss < 1e-6


def test_curve_loss_huber_loss_kwargs(tmp_path: Path) -> None:
    xs, ys = _dense_sine()
    case_dir = tmp_path / "case_huber"
    case_dir.mkdir()
    _write_xy(case_dir / "output.dat", xs, ys)

    exp_path = tmp_path / "experiment.dat"
    _write_xy(exp_path, xs, ys + 0.1)

    loss = curve_loss(
        case_dir, exp_path, "output.dat",
        loss="huber", interp="linear", loss_kwargs={"delta": 0.5},
    )
    assert loss == pytest.approx(0.5 * 0.1 ** 2, abs=1e-4)


def test_curve_loss_never_raises_on_garbage_input(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_garbage"
    case_dir.mkdir()
    (case_dir / "output.dat").write_text("not a number, at all\n$$$ ---\n")

    exp_path = tmp_path / "experiment.dat"
    exp_path.write_text("also garbage\n")

    loss = curve_loss(case_dir, exp_path, "output.dat")
    assert math.isnan(loss)


def test_curve_loss_never_raises_missing_sim_file(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_missing"
    case_dir.mkdir()

    exp_path = tmp_path / "experiment.dat"
    xs, ys = _dense_sine()
    _write_xy(exp_path, xs, ys)

    loss = curve_loss(case_dir, exp_path, "output.dat", loss="rmse")
    assert math.isnan(loss)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
