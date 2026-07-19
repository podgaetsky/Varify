"""Tests for the post-job analysis hook feature:

* ``src.analysis.postprocess.write_xy`` (round-trips through ``load_xy``)
* ``src.analysis.analysis_dispatcher.PostJobDispatcher`` (signature-filtered
  dispatch, static kwargs precedence, per-hook error isolation)
* ``src.common.config._build_post_job_fns`` (bare-name and {fn, kwargs} YAML
  entries)

Uses a small duck-typed ``SimpleNamespace`` config, matching the style of
``tests/test_plot_postprocess.py``.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from varify.src.analysis.analysis_dispatcher import PostJobDispatcher
from varify.src.analysis.postprocess import load_xy, write_xy
from varify.src.common.config import _build_post_job_fns


def _make_cfg(post_job_fns, output_file: str = "output.dat"):
    return types.SimpleNamespace(output_file=output_file, post_job_fns=post_job_fns)


# ═════════════════════════════════════════════════════════════════════════════
#  write_xy
# ═════════════════════════════════════════════════════════════════════════════

def test_write_xy_round_trips_through_load_xy(tmp_path: Path) -> None:
    x = np.linspace(0.0, 5.0, 11)
    y = np.sin(x)
    path = write_xy(tmp_path / "curve.dat", x, y)

    x_back, y_back = load_xy(path)
    assert np.allclose(x_back, x, atol=1e-9)
    assert np.allclose(y_back, y, atol=1e-9)


def test_write_xy_creates_parent_dirs(tmp_path: Path) -> None:
    path = write_xy(tmp_path / "nested" / "dir" / "curve.dat", [1.0, 2.0], [3.0, 4.0])
    assert path.exists()


def test_write_xy_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_xy(tmp_path / "curve.dat", [1.0, 2.0], [3.0])


# ═════════════════════════════════════════════════════════════════════════════
#  PostJobDispatcher
# ═════════════════════════════════════════════════════════════════════════════

def test_post_job_dispatcher_forwards_declared_params_and_writes_xy(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_tau_1.5"
    case_dir.mkdir()
    (case_dir / "raw_output.dat").write_text("0.0 0.0\n1.0 2.0\n2.0 4.0\n")

    calls = []

    def reduce_to_xy(case_dir: Path, tau: float, x_col: int = 0, y_col: int = 1):
        calls.append((case_dir, tau))
        x, y = load_xy(case_dir / "raw_output.dat", x_col=x_col, y_col=y_col)
        write_xy(case_dir / "sim_curve.dat", x, y * tau)

    cfg = _make_cfg(post_job_fns=[(reduce_to_xy, {"y_col": 1})])
    dispatcher = PostJobDispatcher(cfg)
    dispatcher.run_case(case_dir, params={"tau": 2.0, "gamma": 0.5}, job_id="123")

    assert calls == [(case_dir, 2.0)]
    x_out, y_out = load_xy(case_dir / "sim_curve.dat")
    assert np.allclose(y_out, [0.0, 4.0, 8.0])


def test_post_job_dispatcher_var_keyword_gets_full_pool(tmp_path: Path) -> None:
    case_dir = tmp_path / "case0"
    case_dir.mkdir()
    seen = {}

    def catch_all(**kwargs):
        seen.update(kwargs)

    cfg = _make_cfg(post_job_fns=[(catch_all, {"extra": "pinned"})])
    PostJobDispatcher(cfg).run_case(case_dir, params={"tau": 1.0}, job_id="7")

    assert seen["case_dir"] == case_dir
    assert seen["job_id"] == "7"
    assert seen["output_file"] == case_dir / "output.dat"
    assert seen["tau"] == 1.0
    assert seen["extra"] == "pinned"


def test_post_job_dispatcher_static_kwargs_override_pool(tmp_path: Path) -> None:
    case_dir = tmp_path / "case0"
    case_dir.mkdir()
    received = {}

    def fn(tau: float):
        received["tau"] = tau

    # static kwarg for "tau" should win over the per-case param value
    cfg = _make_cfg(post_job_fns=[(fn, {"tau": 999.0})])
    PostJobDispatcher(cfg).run_case(case_dir, params={"tau": 1.0})

    assert received["tau"] == 999.0


def test_post_job_dispatcher_one_hook_raising_does_not_block_others(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case0"
    case_dir.mkdir()
    ran = []

    def bad_hook(case_dir: Path):
        raise RuntimeError("boom")

    def good_hook(case_dir: Path):
        ran.append(case_dir)

    cfg = _make_cfg(post_job_fns=[(bad_hook, {}), (good_hook, {})])
    PostJobDispatcher(cfg).run_case(case_dir, params={})

    assert ran == [case_dir]


def test_post_job_dispatcher_no_hooks_is_a_noop(tmp_path: Path) -> None:
    cfg = _make_cfg(post_job_fns=[])
    PostJobDispatcher(cfg).run_case(tmp_path, params={"tau": 1.0})  # must not raise


# ═════════════════════════════════════════════════════════════════════════════
#  config.py YAML entry parsing
# ═════════════════════════════════════════════════════════════════════════════

def test_build_post_job_fns_bare_name_and_kwargs_form() -> None:
    def hook_a(case_dir):
        pass

    def hook_b(case_dir, x_col):
        pass

    hooks = types.SimpleNamespace(hook_a=hook_a, hook_b=hook_b)
    raw = ["hook_a", {"fn": "hook_b", "kwargs": {"x_col": 2}}]

    result = _build_post_job_fns(raw, hooks)

    assert result == [(hook_a, {}), (hook_b, {"x_col": 2})]


def test_build_post_job_fns_unknown_name_raises() -> None:
    hooks = types.SimpleNamespace()
    with pytest.raises(ValueError):
        _build_post_job_fns(["missing_fn"], hooks)
