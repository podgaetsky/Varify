"""Bridge package that exposes the legacy top-level utils/ tree as varify.utils."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_legacy_utils = Path(__file__).resolve().parents[2] / "utils"
_legacy_utils_str = str(_legacy_utils)
if _legacy_utils_str not in __path__:
    __path__.append(_legacy_utils_str)
