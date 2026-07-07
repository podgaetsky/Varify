"""Safe file-handling primitives (standard library only).

Three capabilities:

* ``read_text_safe``      — multi-encoding fallback reader (UTF-8 → UTF-8-BOM
  → UTF-16 → Latin-1, then lossy UTF-8) so a stray encoding never kills a run.
* ``write_atomic``        — write-to-temp + ``os.replace`` so readers never
  observe a half-written file, even across a SIGKILL.
* ``update_config_value`` — **non-destructive streaming in-place mutation** of
  one nested value in a JSON/JSONC, YAML or TOML file.  The document is never
  re-serialized: a format-aware scanner locates the exact character span of
  the targeted value and splices in the new literal, so native comments,
  key ordering, quoting style and whitespace all survive untouched.  The
  edit is verified by re-parsing (``json`` / ``tomllib``; ``yaml`` when
  installed) before the original file is atomically replaced.

Limitations (documented, by design): keys are addressed by dotted path with
integer segments indexing JSON arrays; YAML support covers block mappings
(list items are opaque); values spanning multiple lines (YAML block scalars,
multi-line TOML arrays) cannot be replaced.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

Scalar = Union[str, int, float, bool, None]

_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "utf-16", "latin-1")


# ═════════════════════════════════════════════════════════════════════════════
#  Safe reading / atomic writing
# ═════════════════════════════════════════════════════════════════════════════

def read_text_safe(
    path: Union[str, Path],
    encodings: Sequence[str] = _ENCODINGS,
) -> str:
    """Read text trying each encoding in turn; final fallback is lossy UTF-8.

    Raises ``FileNotFoundError`` for missing files — encoding trouble is
    recoverable, a missing file is not.
    """
    p = Path(path)
    data = p.read_bytes()
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def write_atomic(
    path: Union[str, Path],
    text: str,
    encoding: str = "utf-8",
) -> Path:
    """Write *text* to *path* atomically (temp file + ``os.replace``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


# ═════════════════════════════════════════════════════════════════════════════
#  Scalar serialization per format
# ═════════════════════════════════════════════════════════════════════════════

_YAML_BARE_RE = re.compile(r"^[A-Za-z0-9_./+-]+$")
_YAML_RESERVED = frozenset({
    "true", "false", "null", "yes", "no", "on", "off", "~", "none",
})


def _fmt_scalar(value: Scalar, fmt: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        if fmt == "toml":
            raise ValueError("TOML has no null literal.")
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if fmt == "yaml" and _YAML_BARE_RE.match(value) \
                and value.lower() not in _YAML_RESERVED:
            return value
        return json.dumps(value)
    raise TypeError(f"Unsupported scalar type: {type(value)!r}")


# ═════════════════════════════════════════════════════════════════════════════
#  JSON / JSONC value-span scanner
# ═════════════════════════════════════════════════════════════════════════════

def _json_find_span(text: str, key_path: Tuple[str, ...]) -> Tuple[int, int]:
    """Return (start, end) of the value at *key_path* in JSON/JSONC text."""
    n = len(text)
    found: List[Tuple[int, int]] = []

    def skip(i: int) -> int:
        while i < n:
            c = text[i]
            if c in " \t\r\n":
                i += 1
            elif text.startswith("//", i):
                j = text.find("\n", i)
                i = n if j < 0 else j + 1
            elif text.startswith("/*", i):
                j = text.find("*/", i + 2)
                if j < 0:
                    raise ValueError("Unterminated /* comment")
                i = j + 2
            else:
                break
        return i

    def string_end(i: int) -> int:
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
            elif text[j] == '"':
                return j + 1
            else:
                j += 1
        raise ValueError("Unterminated string")

    def parse_value(i: int, path: Tuple[str, ...]) -> int:
        i = skip(i)
        if i >= n:
            raise ValueError("Unexpected end of document")
        c = text[i]
        if c == "{":
            return parse_object(i, path)
        if c == "[":
            return parse_array(i, path)
        if c == '"':
            end = string_end(i)
        else:
            j = i
            while j < n and text[j] not in ",]}" and text[j] not in " \t\r\n" \
                    and not text.startswith("//", j) \
                    and not text.startswith("/*", j):
                j += 1
            end = j
        if path == key_path:
            found.append((i, end))
        return end

    def parse_object(i: int, path: Tuple[str, ...]) -> int:
        i += 1  # past '{'
        while True:
            i = skip(i)
            if i >= n:
                raise ValueError("Unterminated object")
            if text[i] == "}":
                return i + 1
            if text[i] == ",":
                i += 1
                continue
            if text[i] != '"':
                raise ValueError(f"Expected key string at offset {i}")
            kend = string_end(i)
            key = json.loads(text[i:kend])
            i = skip(kend)
            if i >= n or text[i] != ":":
                raise ValueError(f"Expected ':' at offset {i}")
            i = parse_value(i + 1, path + (key,))

    def parse_array(i: int, path: Tuple[str, ...]) -> int:
        i += 1  # past '['
        idx = 0
        while True:
            i = skip(i)
            if i >= n:
                raise ValueError("Unterminated array")
            if text[i] == "]":
                return i + 1
            if text[i] == ",":
                i += 1
                continue
            i = parse_value(i, path + (str(idx),))
            idx += 1

    parse_value(0, ())
    if not found:
        raise KeyError(f"Key path {'.'.join(key_path)!r} not found")
    return found[0]


def _strip_jsonc(text: str) -> str:
    """Blank out // and /* */ comments (preserving offsets) for validation."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            out.append(text[i:j])
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2) + 2
            out.append(" " * (j - i))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ═════════════════════════════════════════════════════════════════════════════
#  YAML block-mapping line walker
# ═════════════════════════════════════════════════════════════════════════════

_YAML_KEY_RE = re.compile(r"^(\s*)([^\s#\-][^:]*?)\s*:(.*)$")


def _split_inline_comment(value: str, fmt: str) -> Tuple[str, str]:
    """Split ``value`` into (value_part, comment_part) respecting quotes."""
    quote: Optional[str] = None
    depth = 0
    for i, c in enumerate(value):
        if quote:
            if fmt == "toml" and quote == '"' and c == "\\":
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == "#" and depth == 0:
            if fmt == "toml" or i == 0 or value[i - 1] in " \t":
                return value[:i], value[i:]
    return value, ""


def _value_padding(value_part: str) -> Tuple[str, str]:
    """Return the whitespace surrounding the scalar inside *value_part*."""
    stripped = value_part.strip()
    lead_n = value_part.index(stripped) if stripped else len(value_part)
    return value_part[:lead_n], value_part[lead_n + len(stripped):]


def _yaml_edit(text: str, key_path: Tuple[str, ...], literal: str) -> str:
    lines = text.splitlines(keepends=True)
    stack: List[Tuple[int, str]] = []  # (indent, key)
    for li, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        m = _YAML_KEY_RE.match(line.rstrip("\r\n"))
        if not m:
            continue
        indent = len(m.group(1))
        key = m.group(2).strip().strip("\"'")
        rest = m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(k for _, k in stack) + (key,)
        if path == key_path:
            value_part, comment = _split_inline_comment(rest, "yaml")
            if not value_part.strip() or value_part.strip() in ("|", ">"):
                raise ValueError(
                    f"Key {'.'.join(key_path)!r} holds a nested/block value; "
                    "only scalar values can be replaced in place."
                )
            lead, trail = _value_padding(value_part)
            eol = line[len(line.rstrip('\r\n')):]
            head = line[: line.index(":", len(m.group(1))) + 1]
            lines[li] = f"{head}{lead}{literal}{trail}{comment}{eol}"
            return "".join(lines)
        stack.append((indent, key))
    raise KeyError(f"Key path {'.'.join(key_path)!r} not found")


# ═════════════════════════════════════════════════════════════════════════════
#  TOML section/key line walker
# ═════════════════════════════════════════════════════════════════════════════

_TOML_SECTION_RE = re.compile(r"^\s*\[{1,2}\s*([^\]]+?)\s*\]{1,2}\s*(#.*)?$")
_TOML_KEY_RE = re.compile(
    r"""^(\s*)((?:[A-Za-z0-9_-]+|"[^"]*"|'[^']*')"""
    r"""(?:\s*\.\s*(?:[A-Za-z0-9_-]+|"[^"]*"|'[^']*'))*)\s*=(.*)$"""
)


def _toml_key_parts(raw: str) -> Tuple[str, ...]:
    return tuple(
        part.strip().strip("\"'") for part in raw.split(".")
    )


def _toml_edit(text: str, key_path: Tuple[str, ...], literal: str) -> str:
    lines = text.splitlines(keepends=True)
    section: Tuple[str, ...] = ()
    for li, line in enumerate(lines):
        body = line.rstrip("\r\n")
        sec = _TOML_SECTION_RE.match(body)
        if sec:
            section = _toml_key_parts(sec.group(1))
            continue
        m = _TOML_KEY_RE.match(body)
        if not m:
            continue
        path = section + _toml_key_parts(m.group(2))
        if path != key_path:
            continue
        rest = m.group(3)
        value_part, comment = _split_inline_comment(rest, "toml")
        if not value_part.strip():
            raise ValueError(f"Key {'.'.join(key_path)!r} has no inline value.")
        lead, trail = _value_padding(value_part)
        eq_pos = line.index("=", len(m.group(1)))
        eol = line[len(body):]
        lines[li] = f"{line[:eq_pos + 1]}{lead}{literal}{trail}{comment}{eol}"
        return "".join(lines)
    raise KeyError(f"Key path {'.'.join(key_path)!r} not found")


# ═════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".json", ".jsonc"):
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext == ".toml":
        return "toml"
    raise ValueError(f"Unsupported config format: {path.name!r}")


def _verify(fmt: str, text: str) -> None:
    """Re-parse the edited document; raise if the splice broke it."""
    if fmt == "json":
        json.loads(_strip_jsonc(text))
    elif fmt == "toml":
        import tomllib
        tomllib.loads(text)
    elif fmt == "yaml":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            return  # optional dependency: structural checks already applied
        yaml.safe_load(text)


def update_config_value(
    path: Union[str, Path],
    key_path: Union[str, Sequence[str]],
    value: Scalar,
    verify: bool = True,
) -> Path:
    """Replace one nested scalar in a JSON/JSONC, YAML or TOML file in place.

    *key_path* is a dotted string (``"slurm.directives.time"``) or a sequence
    of segments; integer segments index JSON arrays.  Comments, ordering and
    formatting are preserved verbatim; only the value's character span is
    rewritten.  The result is re-parse-verified, then atomically replaces the
    original file.  Raises ``KeyError`` if the path does not exist.
    """
    p = Path(path)
    keys: Tuple[str, ...] = (
        tuple(key_path.split(".")) if isinstance(key_path, str)
        else tuple(str(k) for k in key_path)
    )
    if not keys or any(not k for k in keys):
        raise ValueError(f"Invalid key path: {key_path!r}")
    fmt = _detect_format(p)
    text = read_text_safe(p)
    literal = _fmt_scalar(value, fmt)

    if fmt == "json":
        start, end = _json_find_span(text, keys)
        new_text = text[:start] + literal + text[end:]
    elif fmt == "yaml":
        new_text = _yaml_edit(text, keys, literal)
    else:
        new_text = _toml_edit(text, keys, literal)

    if verify:
        _verify(fmt, new_text)
    return write_atomic(p, new_text)
