"""Tests for CaseBuilder source-directory staging (``_stage_source_dir``).

Uses a small duck-typed ``SimpleNamespace`` config (matching the attributes
CaseBuilder.build actually touches) rather than a full FrameworkConfig /
load_config round-trip, consistent with the style of the other optimizer
tests in this suite.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from varify.src.common.casebuilder import CaseBuilder


def _make_cfg(
    tmp_path: Path,
    source_dir=None,
    substitute_globs=None,
    template_files=None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        template_files=template_files or [],
        param_specs=[],
        file_pipeline=[],
        case_source_dir=source_dir,
        case_substitute_globs=substitute_globs if substitute_globs is not None else ["*"],
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Basic staging: recursive copy + substitution
# ═════════════════════════════════════════════════════════════════════════════

def test_source_dir_copied_recursively(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "top.txt").write_text("top", encoding="utf-8")
    (src / "sub" / "nested.txt").write_text("nested", encoding="utf-8")

    case_dir = tmp_path / "case_001"
    cfg = _make_cfg(tmp_path, source_dir=src)
    builder = CaseBuilder(cfg)
    builder.build(case_dir, {}, "job1")

    assert (case_dir / "top.txt").read_text(encoding="utf-8") == "top"
    assert (case_dir / "sub" / "nested.txt").read_text(encoding="utf-8") == "nested"


def test_token_substituted_in_matching_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "input.dat").write_text("value=@TAU@ end", encoding="utf-8")

    case_dir = tmp_path / "case_002"
    cfg = _make_cfg(tmp_path, source_dir=src)
    builder = CaseBuilder(cfg)
    builder.build(case_dir, {"tau": 1.5}, "job2")

    text = (case_dir / "input.dat").read_text(encoding="utf-8")
    assert text == "value=1.5 end"


# ═════════════════════════════════════════════════════════════════════════════
#  Glob filtering
# ═════════════════════════════════════════════════════════════════════════════

def test_glob_filtering_skips_non_matching_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "input.inp").write_text("value=@TAU@", encoding="utf-8")
    (src / "notes.txt").write_text("value=@TAU@", encoding="utf-8")

    case_dir = tmp_path / "case_003"
    cfg = _make_cfg(tmp_path, source_dir=src, substitute_globs=["*.inp"])
    builder = CaseBuilder(cfg)
    builder.build(case_dir, {"tau": 2.0}, "job3")

    assert (case_dir / "input.inp").read_text(encoding="utf-8") == "value=2"
    # notes.txt does not match *.inp -> left untouched
    assert (case_dir / "notes.txt").read_text(encoding="utf-8") == "value=@TAU@"


# ═════════════════════════════════════════════════════════════════════════════
#  Binary files
# ═════════════════════════════════════════════════════════════════════════════

def test_binary_file_survives_untouched(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    binary_payload = bytes(range(256))
    (src / "data.bin").write_bytes(binary_payload)

    case_dir = tmp_path / "case_004"
    cfg = _make_cfg(tmp_path, source_dir=src)
    builder = CaseBuilder(cfg)
    builder.build(case_dir, {"tau": 1.0}, "job4")

    assert (case_dir / "data.bin").read_bytes() == binary_payload


# ═════════════════════════════════════════════════════════════════════════════
#  Missing source_dir
# ═════════════════════════════════════════════════════════════════════════════

def test_missing_source_dir_logs_warning_and_continues(
    tmp_path: Path, caplog
) -> None:
    missing = tmp_path / "does_not_exist"
    case_dir = tmp_path / "case_005"
    cfg = _make_cfg(tmp_path, source_dir=missing)
    builder = CaseBuilder(cfg)

    with caplog.at_level(logging.WARNING, logger="varify.casebuilder"):
        result = builder.build(case_dir, {}, "job5")

    assert result == case_dir
    assert case_dir.exists()  # build() still creates the (now-empty) case dir
    assert any("case_source_dir" in rec.message for rec in caplog.records)


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline order: templates render after staging and may overwrite
# ═════════════════════════════════════════════════════════════════════════════

def test_template_overwrites_staged_file_of_same_name(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "run_script.sh").write_text("staged @TAU@", encoding="utf-8")

    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    tpl = tpl_dir / "run_script.sh"
    tpl.write_text("rendered @TAU@", encoding="utf-8")

    case_dir = tmp_path / "case_006"
    cfg = _make_cfg(
        tmp_path, source_dir=src, template_files=[str(tpl)],
    )
    builder = CaseBuilder(cfg)
    builder.build(case_dir, {"tau": 3.0}, "job6")

    # The template (rendered after staging) wins.
    assert (case_dir / "run_script.sh").read_text(encoding="utf-8") == "rendered 3"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
