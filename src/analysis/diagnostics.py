"""MCMC convergence diagnostics (migrated verbatim from the legacy script)."""

from __future__ import annotations

import numpy as np


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """Gelman-Rubin R-hat statistic per parameter.

    chains: (n_steps, n_walkers, n_params)
    """
    n, m, d = chains.shape
    B = n * np.var(chains.mean(axis=0), axis=0, ddof=1)
    W = np.mean(np.var(chains, axis=0, ddof=1), axis=0)
    var_hat = (n - 1) / n * W + B / n
    R_hat = np.sqrt(var_hat / np.where(W > 0, W, np.nan))
    return R_hat


def autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time for a 1-D chain *x* using the
    automated windowing procedure (Sokal 1989)."""
    n = len(x)
    x = x - x.mean()
    # Full autocorrelation via FFT
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n] / (n * np.var(x) + 1e-30)
    tau = 2.0 * np.cumsum(acf) - 1.0
    # Automated window: stop when window >= c * tau
    for M in range(1, n):
        if M >= c * tau[M]:
            return float(tau[M])
    return float(tau[-1])
