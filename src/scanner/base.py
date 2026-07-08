"""Extensible ``Scanner`` base class.

Wraps the legacy line/grid search machinery: coupled-parameter resolution,
case-directory preparation and job submission.  Concrete subclasses only
define how the parameter space is enumerated (``iter_points``).

The coupled-parameter logic (``_apply_coupled`` and the driver/values-length
validation) is migrated verbatim from the legacy ``GridManager``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from varify.src.common.casebuilder import CaseBuilder
from varify.src.common.config import FrameworkConfig
from varify.src.common.params import GridPoint, ParamSpec


class Scanner(ABC):
    """Base class for parameter-space scans (line, grid, coupled, ...)."""

    def __init__(self, cfg: FrameworkConfig) -> None:
        self.cfg = cfg
        self.builder = CaseBuilder(cfg)
        self._log = logging.getLogger(f"varify.scanner.{type(self).__name__}")

    # ── Point enumeration (subclass responsibility) ──────────────────────────

    @abstractmethod
    def iter_points(self) -> Iterator[GridPoint]:
        """Yield every fully-resolved parameter combination of the scan."""

    @abstractmethod
    def total_points(self) -> int:
        """Number of points ``iter_points`` will yield."""

    # ── Coupled-parameter machinery (shared, legacy-verbatim) ────────────────

    def _coupled_by_driver(self) -> Dict[str, List[ParamSpec]]:
        coupled_by_driver: Dict[str, List[ParamSpec]] = {}
        for cs in self.cfg.coupled_specs:
            assert cs.coupled_to is not None
            coupled_by_driver.setdefault(cs.coupled_to, []).append(cs)
        return coupled_by_driver

    def _validate_coupled(self) -> None:
        swept = self.cfg.swept_specs
        driver_names = {s.name for s in swept}
        for cs in self.cfg.coupled_specs:
            if cs.coupled_to in driver_names and cs.coupled_fn is None:
                driver_spec = next(s for s in swept if s.name == cs.coupled_to)
                if cs.values is None or len(cs.values) != driver_spec.n:
                    raise ValueError(
                        f"Coupled param '{cs.name}' needs coupled_fn or a values "
                        f"array of length {driver_spec.n} "
                        f"(driver '{cs.coupled_to}')."
                    )

    def _apply_coupled(
        self,
        p: Dict[str, float],
        driver_name: str,
        driver_idx: int,
        coupled_by_driver: Dict[str, List[ParamSpec]],
    ) -> None:
        driver_val = p[driver_name]
        for cs in coupled_by_driver.get(driver_name, []):
            if cs.coupled_fn is not None:
                p[cs.name] = float(cs.coupled_fn(driver_val))
            elif cs.values is not None:
                p[cs.name] = float(cs.values[driver_idx])

    def _default_point(self) -> GridPoint:
        """Single point at parameter defaults (no swept params)."""
        base = self.cfg.defaults.copy()
        p = base.copy()
        for cs in self.cfg.coupled_specs:
            dval = p.get(cs.coupled_to, base.get(cs.coupled_to, 0.0))  # type: ignore[arg-type]
            if cs.coupled_fn is not None:
                p[cs.name] = float(cs.coupled_fn(dval))
        return GridPoint(params=p, swept_names=[])

    # ── Case preparation & discovery ─────────────────────────────────────────

    def prepare_case(self, gp: GridPoint) -> Path:
        case_dir = self.cfg.sweep_root / gp.case_dir_name
        return self.builder.build(case_dir, gp.params, gp.job_name)

    def existing_cases(self) -> List[Tuple[GridPoint, Path]]:
        results: List[Tuple[GridPoint, Path]] = []
        for gp in self.iter_points():
            p = self.cfg.sweep_root / gp.case_dir_name
            if p.is_dir():
                results.append((gp, p))
        return results
