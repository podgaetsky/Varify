"""Parameter-driven file generation/modification pipeline (standard library
only at import time; ``yaml``/``toml`` writers lazy-import their package).

Two primitives and a declarative driver on top of them:

* ``generate_config_file(dest, params, fmt="auto")`` — serialize a flat
  ``params`` dict to a brand-new file. Supported formats: ``json``,
  ``keyvalue`` (``key = value`` lines, plain text), ``yaml``, ``toml``.
  ``fmt="auto"`` picks the format from ``dest``'s extension (``.json`` ->
  json, ``.yaml``/``.yml`` -> yaml, ``.toml`` -> toml, anything else ->
  keyvalue). Writes are atomic via ``utils.io_handlers.write_atomic``.

* ``modify_config_file(path, updates)`` — apply a flat ``{dotted.key.path:
  value}`` dict of edits to an *existing* file via
  ``utils.io_handlers.update_config_value`` (comment/format preserving).
  A single-segment key missing from a flat JSON or keyvalue file is
  appended rather than treated as an error; multi-segment paths that don't
  exist raise ``KeyError`` with a clarified message.

* ``apply_pipeline(case_dir, params, spec)`` — runs a list of declarative
  actions against a case directory once it has been built. Each entry in
  *spec* is a dict::

      {
          "action": "generate" | "modify",
          "file": "relative/path/inside/case_dir",
          "keys": {"dotted.key.or.field": value_or_"$param_name"},
      }

  ``keys`` values that are strings of the form ``"$name"`` are resolved
  against *params* (``KeyError`` if *name* is absent); every other value
  (including strings not starting with ``$``) is used literally. For
  ``"generate"`` the resolved ``keys`` dict becomes the new file's content;
  for ``"modify"`` it is passed straight to ``modify_config_file`` as the
  dotted-path update map. Optional per-entry keys:

      "fmt"      — forwarded to generate_config_file (default "auto")

  Each action is logged at INFO before it runs. A failing entry logs an
  ERROR (with traceback text) and processing continues with the next
  entry — mirroring the error tolerance of
  ``CaseBuilder._run_input_fns``, so one bad pipeline step never aborts
  case construction.

CLI: ``python -m utils.file_pipeline <case_dir> <params.json> <spec.json>``
loads the two JSON files and runs ``apply_pipeline`` against *case_dir*.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from utils.io_handlers import update_config_value, write_atomic

log = logging.getLogger("varify.file_pipeline")

Scalar = Union[str, int, float, bool, None]


# ═════════════════════════════════════════════════════════════════════════════
#  Serialization helpers
# ═════════════════════════════════════════════════════════════════════════════

def _detect_fmt(dest: Path) -> str:
    ext = dest.suffix.lower()
    if ext in (".json", ".jsonc"):
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext == ".toml":
        return "toml"
    return "keyvalue"


def _dump_keyvalue(params: Dict[str, Any]) -> str:
    lines = [f"{k} = {v}" for k, v in params.items()]
    return "\n".join(lines) + ("\n" if lines else "")


def _dump_yaml(params: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "generate_config_file(fmt='yaml') requires the 'PyYAML' package "
            "(pip install pyyaml)."
        ) from exc
    return yaml.safe_dump(params, sort_keys=False)


def _dump_toml(params: Dict[str, Any]) -> str:
    try:
        import toml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "generate_config_file(fmt='toml') requires the 'toml' package "
            "(pip install toml)."
        ) from exc
    return toml.dumps(params)


def generate_config_file(
    dest: Path, params: Dict[str, Any], fmt: str = "auto"
) -> Path:
    """Serialize *params* to a new file at *dest*, returning *dest*.

    *fmt* is one of ``"json"``, ``"yaml"``, ``"toml"``, ``"keyvalue"`` or
    ``"auto"`` (detect from ``dest``'s extension; see module docstring).
    """
    dest = Path(dest)
    resolved = _detect_fmt(dest) if fmt == "auto" else fmt

    if resolved == "json":
        text = json.dumps(params, indent=2) + "\n"
    elif resolved == "keyvalue":
        text = _dump_keyvalue(params)
    elif resolved == "yaml":
        text = _dump_yaml(params)
    elif resolved == "toml":
        text = _dump_toml(params)
    else:
        raise ValueError(f"Unsupported fmt: {fmt!r}")

    return write_atomic(dest, text)


# ═════════════════════════════════════════════════════════════════════════════
#  In-place modification
# ═════════════════════════════════════════════════════════════════════════════

def _keyvalue_set(path: Path, key: str, value: Scalar) -> Path:
    """Replace ``key = value`` in-place if present, else append the line.

    ``update_config_value`` only understands JSON/YAML/TOML, so keyvalue
    (``key = value``) files are edited directly here.
    """
    import re as _re

    from utils.io_handlers import read_text_safe

    text = read_text_safe(path)
    line_re = _re.compile(
        r"^(\s*)" + _re.escape(key) + r"(\s*=\s*)(.*?)(\s*)$", _re.MULTILINE
    )
    literal = str(value)
    new_text, n = line_re.subn(
        lambda m: f"{m.group(1)}{key}{m.group(2)}{literal}{m.group(4)}",
        text,
        count=1,
    )
    if n == 0:
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"
        new_text += f"{key} = {literal}\n"
    return write_atomic(path, new_text)


def _append_json_top_level(path: Path, key: str, value: Scalar) -> Path:
    from utils.io_handlers import read_text_safe

    data = json.loads(read_text_safe(path))
    if not isinstance(data, dict):
        raise KeyError(
            f"Cannot append key {key!r}: top-level JSON value in {path} "
            "is not an object."
        )
    data[key] = value
    return write_atomic(path, json.dumps(data, indent=2) + "\n")


def modify_config_file(path: Path, updates: Dict[str, Any]) -> Path:
    """Apply *updates* (``{dotted.key.path: value}``) to an existing file.

    JSON/YAML/TOML edits go through ``update_config_value`` (comment/format
    preserving in-place edit). Keyvalue (``key = value``) files — anything
    with an extension ``update_config_value`` doesn't recognize — are
    edited directly since that helper only understands JSON/YAML/TOML.

    If a *single-segment* key path is missing from a keyvalue-format file,
    the ``key = value`` line is appended; if missing from a JSON file
    whose top level is a flat object, the key is inserted via a full
    parse-modify-dump (only in this fallback path — normal edits never
    re-serialize the document). Missing multi-segment paths re-raise
    ``KeyError`` with a clarified message; nothing to append to for those.
    """
    path = Path(path)
    fmt = path.suffix.lower()
    is_keyvalue = fmt not in (".json", ".jsonc", ".yaml", ".yml", ".toml")

    for key_path, value in updates.items():
        segments = key_path.split(".")
        if is_keyvalue:
            if len(segments) != 1:
                raise KeyError(
                    f"Key path {key_path!r} not found in {path}: keyvalue "
                    "files only support single-segment (flat) keys."
                )
            _keyvalue_set(path, key_path, value)
            continue
        try:
            update_config_value(path, key_path, value)
        except KeyError:
            if len(segments) != 1:
                raise KeyError(
                    f"Key path {key_path!r} not found in {path} and cannot "
                    "be auto-created (only single-segment top-level keys "
                    "are appended)."
                )
            if fmt in (".json", ".jsonc"):
                _append_json_top_level(path, key_path, value)
            else:
                raise KeyError(
                    f"Key path {key_path!r} not found in {path}; YAML/TOML "
                    "files require the key to already exist."
                )
    return path


# ═════════════════════════════════════════════════════════════════════════════
#  Declarative pipeline driver
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_keys(keys: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for k, v in keys.items():
        if isinstance(v, str) and v.startswith("$"):
            name = v[1:]
            if name not in params:
                raise KeyError(
                    f"file_pipeline: param {name!r} referenced as {v!r} "
                    "not found in params."
                )
            resolved[k] = params[name]
        else:
            resolved[k] = v
    return resolved


def apply_pipeline(
    case_dir: Path, params: Dict[str, Any], spec: List[Dict[str, Any]]
) -> None:
    """Run each declarative *spec* entry against *case_dir*.

    See the module docstring for the entry schema. A failing entry is
    logged at ERROR and processing continues with the next entry.
    """
    case_dir = Path(case_dir)
    for entry in spec:
        action = entry.get("action")
        file_rel = entry.get("file")
        keys = entry.get("keys") or {}
        try:
            if action not in ("generate", "modify"):
                raise ValueError(f"Unknown action {action!r}")
            if not file_rel:
                raise ValueError("Missing 'file' in file_pipeline entry.")
            target = case_dir / file_rel
            resolved = _resolve_keys(keys, params)
            if action == "generate":
                fmt = entry.get("fmt", "auto")
                log.info("file_pipeline: generating %s (fmt=%s)", target, fmt)
                generate_config_file(target, resolved, fmt=fmt)
            else:
                log.info("file_pipeline: modifying %s (keys=%s)",
                          target, list(resolved))
                modify_config_file(target, resolved)
        except Exception as exc:
            log.error(
                "file_pipeline: entry FAILED (action=%s, file=%s): %s",
                action, file_rel, exc,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m utils.file_pipeline",
        description="Run a declarative file-pipeline spec against a case directory.",
    )
    parser.add_argument("case_dir", type=Path, help="Target case directory.")
    parser.add_argument("params_json", type=Path, help="Path to params.json.")
    parser.add_argument("spec_json", type=Path, help="Path to spec.json (a list).")
    args = parser.parse_args(argv)

    params = json.loads(args.params_json.read_text(encoding="utf-8"))
    spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
    apply_pipeline(args.case_dir, params, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
