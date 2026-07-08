"""Gradient-free optimization and MCMC sampling over SLURM-executed simulations."""

from varify.src.optimizer.base import BaseOptimizer
from varify.src.optimizer.gradient_free import NelderMeadOptimizer
from varify.src.optimizer.hybrid import HybridDEOptimizer
from varify.src.optimizer.mcmc import MCMCOptimizer

__all__ = [
    "BaseOptimizer", "NelderMeadOptimizer", "HybridDEOptimizer", "MCMCOptimizer",
]


def make_optimizer(cfg, method: str, dry_run: bool = False) -> BaseOptimizer:
    """Factory for the optimizer selected on the CLI / in the config."""
    method = method.lower()
    if method == "mcmc":
        return MCMCOptimizer(cfg, dry_run=dry_run)
    if method in ("nelder-mead", "neldermead", "nm"):
        return NelderMeadOptimizer(cfg, dry_run=dry_run)
    if method in ("de-nm", "de_nm", "hybrid"):
        return HybridDEOptimizer(cfg, dry_run=dry_run)
    raise ValueError(f"Unknown optimizer method: {method!r}")
