"""Bridge package that exposes the legacy top-level config/ tree as varify.config."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_legacy_config = Path(__file__).resolve().parents[2] / "config"
_legacy_config_str = str(_legacy_config)
if _legacy_config_str not in __path__:
    __path__.append(_legacy_config_str)
