"""Tests for utils.file_pipeline: generate/modify/apply_pipeline + CLI.

Plain pytest-style ``test_*`` functions using ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from varify.utils.file_pipeline import (
    apply_pipeline,
    generate_config_file,
    main,
    modify_config_file,
)


# ═════════════════════════════════════════════════════════════════════════════
#  generate_config_file
# ═════════════════════════════════════════════════════════════════════════════

def test_generate_json(tmp_path: Path) -> None:
    dest = tmp_path / "params.json"
    generate_config_file(dest, {"tau": 1.5, "name": "case1"}, fmt="json")
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data == {"tau": 1.5, "name": "case1"}


def test_generate_keyvalue(tmp_path: Path) -> None:
    dest = tmp_path / "params.conf"
    generate_config_file(dest, {"tau": 1.5, "gamma": 0.5}, fmt="keyvalue")
    text = dest.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert "tau = 1.5" in lines
    assert "gamma = 0.5" in lines


def test_generate_auto_detects_format(tmp_path: Path) -> None:
    dest_json = tmp_path / "a.json"
    dest_conf = tmp_path / "a.conf"
    generate_config_file(dest_json, {"x": 1}, fmt="auto")
    generate_config_file(dest_conf, {"x": 1}, fmt="auto")
    assert json.loads(dest_json.read_text(encoding="utf-8")) == {"x": 1}
    assert dest_conf.read_text(encoding="utf-8").strip() == "x = 1"


# ═════════════════════════════════════════════════════════════════════════════
#  modify_config_file
# ═════════════════════════════════════════════════════════════════════════════

def test_modify_existing_json_preserves_format(tmp_path: Path) -> None:
    dest = tmp_path / "cfg.json"
    dest.write_text(
        '{\n  // a comment\n  "tau": 1.0,\n  "gamma": 0.5\n}\n', encoding="utf-8"
    )
    modify_config_file(dest, {"tau": 2.5})
    text = dest.read_text(encoding="utf-8")
    assert "// a comment" in text  # comment preserved
    assert '"tau": 2.5' in text
    assert '"gamma": 0.5' in text  # untouched key preserved


def test_modify_appends_missing_single_segment_key_json(tmp_path: Path) -> None:
    dest = tmp_path / "cfg.json"
    dest.write_text('{"tau": 1.0}\n', encoding="utf-8")
    modify_config_file(dest, {"gamma": 0.75})
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data == {"tau": 1.0, "gamma": 0.75}


def test_modify_appends_missing_key_keyvalue(tmp_path: Path) -> None:
    dest = tmp_path / "cfg.conf"
    dest.write_text("tau = 1.0\n", encoding="utf-8")
    modify_config_file(dest, {"gamma": 0.5})
    text = dest.read_text(encoding="utf-8")
    assert "tau = 1.0" in text
    assert "gamma = 0.5" in text


def test_modify_nested_missing_path_raises(tmp_path: Path) -> None:
    dest = tmp_path / "cfg.json"
    dest.write_text('{"a": {"b": 1}}\n', encoding="utf-8")
    try:
        modify_config_file(dest, {"a.c.d": 5})
        assert False, "expected KeyError"
    except KeyError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  apply_pipeline
# ═════════════════════════════════════════════════════════════════════════════

def test_apply_pipeline_generate_and_modify(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_001"
    case_dir.mkdir()
    params = {"tau": 1.25, "gamma": 0.5}
    spec = [
        {
            "action": "generate",
            "file": "params.json",
            "keys": {"tau": "$tau", "note": "static"},
        },
        {
            "action": "modify",
            "file": "params.json",
            "keys": {"gamma": "$gamma"},
        },
    ]
    apply_pipeline(case_dir, params, spec)
    data = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
    assert data == {"tau": 1.25, "note": "static", "gamma": 0.5}


def test_apply_pipeline_missing_param_logs_error_and_continues(
    tmp_path: Path, caplog
) -> None:
    case_dir = tmp_path / "case_002"
    case_dir.mkdir()
    params = {"tau": 1.0}
    spec = [
        {
            "action": "generate",
            "file": "bad.json",
            "keys": {"gamma": "$gamma"},  # gamma not in params -> KeyError
        },
        {
            "action": "generate",
            "file": "good.json",
            "keys": {"tau": "$tau"},
        },
    ]
    with caplog.at_level(logging.ERROR, logger="varify.file_pipeline"):
        apply_pipeline(case_dir, params, spec)

    assert not (case_dir / "bad.json").exists()
    assert (case_dir / "good.json").exists()
    assert any("FAILED" in rec.message for rec in caplog.records)


def test_apply_pipeline_literal_value_not_resolved(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_003"
    case_dir.mkdir()
    params = {"tau": 1.0}
    spec = [
        {
            "action": "generate",
            "file": "out.json",
            "keys": {"mode": "production", "tau": "$tau"},
        }
    ]
    apply_pipeline(case_dir, params, spec)
    data = json.loads((case_dir / "out.json").read_text(encoding="utf-8"))
    assert data == {"mode": "production", "tau": 1.0}


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def test_cli_main_function(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_cli"
    case_dir.mkdir()
    params_json = tmp_path / "params.json"
    spec_json = tmp_path / "spec.json"
    params_json.write_text(json.dumps({"tau": 3.0}), encoding="utf-8")
    spec_json.write_text(
        json.dumps([
            {"action": "generate", "file": "out.json", "keys": {"tau": "$tau"}}
        ]),
        encoding="utf-8",
    )
    rc = main([str(case_dir), str(params_json), str(spec_json)])
    assert rc == 0
    data = json.loads((case_dir / "out.json").read_text(encoding="utf-8"))
    assert data == {"tau": 3.0}


def test_cli_subprocess_invocation(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_cli2"
    case_dir.mkdir()
    params_json = tmp_path / "params.json"
    spec_json = tmp_path / "spec.json"
    params_json.write_text(json.dumps({"gamma": 0.25}), encoding="utf-8")
    spec_json.write_text(
        json.dumps([
            {"action": "generate", "file": "out.json", "keys": {"gamma": "$gamma"}}
        ]),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "utils.file_pipeline",
         str(case_dir), str(params_json), str(spec_json)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((case_dir / "out.json").read_text(encoding="utf-8"))
    assert data == {"gamma": 0.25}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
