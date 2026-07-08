"""Concrete scanners: Cartesian grid and lock-step (coupled) line search.

The enumeration math is migrated verbatim from the legacy
``GridManager.iter_grid`` / ``GridManager.total_points``:

* ``GridScanner``    — full Cartesian product of all swept parameter axes.
                       A 1-D grid (single swept param) is the legacy line
                       search.
* ``CoupledScanner`` — all swept parameter arrays advance in lock-step
                       (``zip``); arrays must have equal length.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np

from varify.src.common.config import FrameworkConfig
from varify.src.common.params import GridPoint
from varify.src.scanner.base import Scanner


class GridScanner(Scanner):
    """Cartesian-product scan over every swept parameter axis."""

    def iter_points(self) -> Iterator[GridPoint]:
        cfg = self.cfg
        base = cfg.defaults.copy()
        swept = cfg.swept_specs
        coupled_by_driver = self._coupled_by_driver()
        self._validate_coupled()

        if not swept:
            yield self._default_point()
            return

        for idx_combo, val_combo in zip(
            itertools.product(*(range(s.n) for s in swept)),
            itertools.product(*(s.values for s in swept)),  # type: ignore[arg-type]
        ):
            p = base.copy()
            for spec, val in zip(swept, val_combo):
                p[spec.name] = float(val)
            for spec, didx in zip(swept, idx_combo):
                self._apply_coupled(p, spec.name, didx, coupled_by_driver)
            yield GridPoint(params=p, swept_names=cfg.swept_names)

    def total_points(self) -> int:
        swept = self.cfg.swept_specs
        if not swept:
            return 1
        return int(np.prod([s.n for s in swept]))


class CoupledScanner(Scanner):
    """Lock-step scan: all swept parameter arrays advance together (zip)."""

    def iter_points(self) -> Iterator[GridPoint]:
        cfg = self.cfg
        base = cfg.defaults.copy()
        swept = cfg.swept_specs
        coupled_by_driver = self._coupled_by_driver()
        self._validate_coupled()

        if not swept:
            yield self._default_point()
            return

        lengths = [s.n for s in swept]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"scan.mode='coupled' requires equal-length arrays, "
                f"got {dict(zip(cfg.swept_names, lengths))}"
            )
        for idx, combo in enumerate(zip(*(s.values for s in swept))):  # type: ignore[arg-type]
            p = base.copy()
            for spec, val in zip(swept, combo):
                p[spec.name] = float(val)
            for spec in swept:
                self._apply_coupled(p, spec.name, idx, coupled_by_driver)
            yield GridPoint(params=p, swept_names=cfg.swept_names)

    def total_points(self) -> int:
        swept = self.cfg.swept_specs
        if not swept:
            return 1
        return swept[0].n


def make_scanner(cfg: FrameworkConfig) -> Scanner:
    """Instantiate the scanner matching ``cfg.sweep_mode``."""
    if cfg.sweep_mode == "coupled":
        return CoupledScanner(cfg)
    if cfg.sweep_mode == "grid":
        return GridScanner(cfg)
    raise ValueError(f"Unknown scan mode: {cfg.sweep_mode!r}")
