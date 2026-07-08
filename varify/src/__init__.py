"""Bridge package that exposes the legacy top-level src/ tree as varify.src."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_legacy_src = Path(__file__).resolve().parents[2] / "src"
_legacy_src_str = str(_legacy_src)
if _legacy_src_str not in __path__:
    __path__.append(_legacy_src_str)
