"""Bridge package that exposes the legacy top-level runner/ tree as varify.runner."""

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_legacy_runner = Path(__file__).resolve().parents[2] / "runner"
_legacy_runner_str = str(_legacy_runner)
if _legacy_runner_str not in __path__:
    __path__.append(_legacy_runner_str)
