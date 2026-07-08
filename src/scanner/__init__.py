"""Parametric scan engines (legacy line/grid search logic, preserved)."""

from varify.src.scanner.base import Scanner
from varify.src.scanner.grid import CoupledScanner, GridScanner, make_scanner

__all__ = ["Scanner", "GridScanner", "CoupledScanner", "make_scanner"]
