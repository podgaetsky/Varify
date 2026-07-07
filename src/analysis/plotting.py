"""Publication-quality plotting suite.

Migrated from the legacy ``ResultPlotter`` with the plotting mathematics
unchanged; all figures are now saved into the dynamically-created
``results/plots/`` directory (``analysis.plots_dir``):

* 1-D scans  → sensitivity curve + finite-difference derivative panel;
* 2-D grids  → heatmap with contour overlay;
* N-D scans  → marginal sensitivity panels;
* MCMC       → trace plots, chi-square history, corner plot (with seaborn
  fallback when ``corner`` is unavailable) and convergence diagnostics;
* Optimizer  → objective convergence history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

try:
    import corner as _corner_mod
    _CORNER_AVAILABLE = True
except ImportError:
    _CORNER_AVAILABLE = False

from src.analysis.diagnostics import autocorr_time, gelman_rubin
from src.common.config import FrameworkConfig


class PlotSuite:
    """Generates and saves all standard figures for scans, MCMC and optimizer."""

    _PALETTE = "viridis"
    _DPI = 180

    def __init__(self, cfg: FrameworkConfig) -> None:
        if not _PLOT_AVAILABLE:
            raise ImportError("pip install matplotlib seaborn")
        self.cfg = cfg
        self._log = logging.getLogger("varify.plotter")
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def plots_dir(self) -> Path:
        d = self.cfg.analysis.plots_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _note(self, df: pd.DataFrame, fig: Any) -> None:
        pct = 100 * df["output"].notna().mean() if "output" in df.columns else 100.0
        n_valid = df["output"].notna().sum() if "output" in df.columns else len(df)
        fig.text(
            0.99, 0.01,
            f"completeness: {pct:.1f}%  ({n_valid}/{len(df)} pts)",
            ha="right", va="bottom", fontsize=8, color="grey",
        )

    def _save(self, fig: Any, name: str) -> None:
        path = self.plots_dir / name
        fig.savefig(path, dpi=self._DPI, bbox_inches="tight")
        plt.close(fig)
        self._log.info("Saved → %s", path)

    # ── Scan plots (legacy math preserved) ────────────────────────────────────

    def plot_1d(self, df: pd.DataFrame) -> None:
        """1-D sensitivity curve + finite-difference derivative."""
        xcol = f"param_{self.cfg.swept_names[0]}"
        valid = df.dropna(subset=["output"])
        if valid.empty:
            self._log.warning("No valid data.")
            return
        x = valid[xcol].to_numpy(float)
        y = valid["output"].to_numpy(float)
        dx = np.diff(x)
        deriv = np.where(dx != 0, np.diff(y) / dx, np.nan)
        x_mid = 0.5 * (x[:-1] + x[1:])
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(9, 7),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}, sharex=True,
        )
        ax0.plot(x, y, "o-", color="#1f77b4", lw=2, ms=6, label="output")
        nan_mask = df["output"].isna()
        if nan_mask.any():
            ax0.scatter(
                df.loc[nan_mask, xcol], np.zeros(nan_mask.sum()),
                marker="x", color="crimson", zorder=5, s=80, label="NaN/failed",
            )
        ax0.set_ylabel("Output", fontsize=12)
        ax0.set_title(
            f"1-D Sweep — {self.cfg.swept_names[0]}",
            fontsize=14, fontweight="bold",
        )
        ax0.legend(framealpha=0.85)
        ax0.grid(True, alpha=0.4)
        ax1.step(x_mid, deriv, where="mid", color="#ff7f0e", lw=2)
        ax1.axhline(0, color="grey", lw=0.8, ls="--")
        ax1.fill_between(x_mid, deriv, 0, alpha=0.18, color="#ff7f0e", step="mid")
        ax1.set_xlabel(self.cfg.swept_names[0], fontsize=12)
        ax1.set_ylabel(r"$\Delta\,\mathrm{Output}\,/\,\Delta x$", fontsize=11)
        ax1.grid(True, alpha=0.4)
        self._note(df, fig)
        self._save(fig, "scan_1d.png")

    def plot_2d(self, df: pd.DataFrame) -> None:
        """2-D grid heatmap with contour overlay."""
        xcol = f"param_{self.cfg.swept_names[0]}"
        ycol = f"param_{self.cfg.swept_names[1]}"
        pivot = df.pivot_table(index=ycol, columns=xcol, values="output",
                               aggfunc="mean")
        Z = pivot.to_numpy(float)
        X_t = pivot.columns.to_numpy(float)
        Y_t = pivot.index.to_numpy(float)
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.isnan(Z)
        sns.heatmap(
            pivot, ax=ax, cmap=self._PALETTE,
            annot=(Z.size <= 100), fmt=".3g", mask=mask,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Output", "shrink": 0.82},
        )
        ax2 = ax.twinx().twiny()
        ax2.set_xlim(X_t.min(), X_t.max())
        ax2.set_ylim(Y_t.min(), Y_t.max())
        ax2.set_xticks([])
        ax2.set_yticks([])
        if (~mask).sum() >= 4:
            try:
                Xg, Yg = np.meshgrid(X_t, Y_t)
                cs = ax2.contour(
                    Xg, Yg, np.where(~mask, Z, np.nanmean(Z)),
                    levels=min(10, (~mask).sum() // 2),
                    colors="white", linewidths=0.9, alpha=0.7,
                )
                ax2.clabel(cs, inline=True, fontsize=7, fmt="%.3g")
            except Exception as exc:
                self._log.warning("Contour failed: %s", exc)
        ax.set_title(
            f"2-D Sweep  —  {self.cfg.swept_names[0]}  ×  {self.cfg.swept_names[1]}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel(self.cfg.swept_names[0], fontsize=12)
        ax.set_ylabel(self.cfg.swept_names[1], fontsize=12)
        self._note(df, fig)
        self._save(fig, "scan_2d_heatmap.png")

    def plot_nd(self, df: pd.DataFrame) -> None:
        """Marginal sensitivity panels for N-D / coupled scans."""
        swept = self.cfg.swept_names
        n = len(swept)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        fig.suptitle(
            f"Marginal sensitivity  ({self.cfg.sweep_mode})",
            fontsize=14, fontweight="bold",
        )
        colors = cm.tab10(np.linspace(0, 0.9, n))
        for idx, (name, color) in enumerate(zip(swept, colors)):
            xcol = f"param_{name}"
            ax = axes[0][idx]
            grp = (
                df.dropna(subset=["output"]).groupby(xcol)["output"]
                .agg(["mean", "std"]).reset_index()
            )
            ax.plot(grp[xcol], grp["mean"], "o-", color=color, lw=2, ms=5)
            if grp["std"].notna().any():
                ax.fill_between(
                    grp[xcol], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                    alpha=0.2, color=color,
                )
            ax.set_xlabel(name, fontsize=11)
            if idx == 0:
                ax.set_ylabel("Output (mean ± σ)", fontsize=11)
            ax.set_title(f"∂ Output / ∂ {name}", fontsize=10)
            ax.grid(True, alpha=0.4)
        self._note(df, fig)
        fig.tight_layout()
        self._save(fig, "scan_marginals.png")

    # ── MCMC plots (legacy math preserved) ────────────────────────────────────

    def plot_mcmc(self, chain_df: pd.DataFrame) -> None:
        cfg = self.cfg
        param_names = cfg.mcmc_names
        n_walkers = cfg.mcmc.num_walkers
        burnin = cfg.mcmc.burnin
        colors_w = cm.rainbow(np.linspace(0, 1, n_walkers))

        # Filter to post-burn-in accepted steps for posterior samples
        post_df = chain_df[
            (chain_df["step"] >= burnin) & (chain_df["accepted"] == 1)
        ].copy()

        # 1. Trace plots ───────────────────────────────────────────────────────
        n_p = len(param_names)
        fig, axes = plt.subplots(nrows=n_p, figsize=(12, 3.5 * n_p), sharex=True)
        if n_p == 1:
            axes = [axes]
        for i, pname in enumerate(param_names):
            ax = axes[i]
            for w in range(n_walkers):
                wdf = chain_df[chain_df["walker"] == w].sort_values("step")
                if pname not in wdf.columns:
                    continue
                label = None
                if w == 0 and pname in post_df.columns and not post_df.empty:
                    post_vals = post_df[pname].to_numpy(float)
                    tau_est = autocorr_time(post_vals)
                    ess = max(len(post_vals) / tau_est, 0)
                    label = f"τ≈{tau_est:.1f}  ESS≈{ess:.0f}"
                ax.plot(wdf["step"], wdf[pname], alpha=0.6,
                        color=colors_w[w], lw=0.8, label=label)
            ax.axvline(burnin, color="k", ls="--", lw=1, alpha=0.5,
                       label="burn-in" if i == 0 else None)
            ax.set_ylabel(pname, fontsize=11)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
            ax.grid(True, alpha=0.35)
        axes[-1].set_xlabel("Chain step", fontsize=11)
        axes[0].set_title("MCMC Trace Plots (all walkers)",
                          fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "mcmc_traces.png")

        # 2. χ² history ────────────────────────────────────────────────────────
        all_lp = chain_df["log_prob"].to_numpy(float)
        steps_all = chain_df["step"].to_numpy(float)
        post_lp = post_df["log_prob"].to_numpy(float) if not post_df.empty \
            else np.array([])
        steps_post = post_df["step"].to_numpy(float) if not post_df.empty \
            else np.array([])
        chi2_all = -2.0 * all_lp
        chi2_post = -2.0 * post_lp

        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        ax = axs[0]
        ax.scatter(steps_all, chi2_all, c="k", s=4, alpha=0.2, label="all visited")
        if len(chi2_post):
            ax.scatter(steps_post, chi2_post, c="r", s=4, alpha=0.4,
                       label="accepted (post burn-in)")
        ax.axvline(burnin, color="blue", ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Chain step")
        ax.set_ylabel(r"$\chi^2 = -2\ln p$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax = axs[1]
        finite = chi2_all[np.isfinite(chi2_all)]
        if len(finite):
            lo_p, hi_p = np.percentile(finite, [1, 99])
            rng_ = (lo_p, hi_p)
            ax.hist(finite, bins=50, range=rng_, color="k", density=True,
                    histtype="step", label="all visited")
        if len(chi2_post):
            fp = chi2_post[np.isfinite(chi2_post)]
            if len(fp):
                ax.hist(fp, bins=50, color="r", density=True, histtype="step",
                        label=f"post-burnin accepted  mean={fp.mean():.2f}")
        ax.set_xlabel(r"$\chi^2$")
        ax.set_yticks([])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.suptitle(r"$\chi^2$ exploration during MCMC",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "mcmc_chi2.png")

        # 3. Corner plot ───────────────────────────────────────────────────────
        if not post_df.empty and all(p in post_df.columns for p in param_names):
            samples_flat = post_df[param_names].to_numpy(float)
            if _CORNER_AVAILABLE:
                fig = plt.figure(figsize=(7, 7))
                _corner_mod.corner(
                    samples_flat,
                    labels=param_names,
                    color="steelblue",
                    fig=fig,
                    bins=40,
                    show_titles=True,
                    title_fmt=".4f",
                    smooth=1.0,
                )
                fig.suptitle("Posterior parameter distribution",
                             fontsize=13, fontweight="bold")
                fig.tight_layout()
                self._save(fig, "mcmc_corner.png")
            else:
                # Fallback: seaborn pairplot-style
                n_p_ = len(param_names)
                fig, axes = plt.subplots(n_p_, n_p_, figsize=(4 * n_p_, 4 * n_p_),
                                         squeeze=False)
                for i, pi in enumerate(param_names):
                    for j, pj in enumerate(param_names):
                        ax = axes[i][j]
                        if i == j:
                            ax.hist(samples_flat[:, i], bins=30, color="steelblue",
                                    density=True, edgecolor="white")
                            ax.set_xlabel(pi, fontsize=9)
                        elif i > j:
                            ax.scatter(samples_flat[:, j], samples_flat[:, i],
                                       s=3, alpha=0.3, color="steelblue")
                            if j == 0:
                                ax.set_ylabel(pi, fontsize=9)
                            if i == n_p_ - 1:
                                ax.set_xlabel(pj, fontsize=9)
                        else:
                            ax.set_visible(False)
                fig.suptitle(
                    "Posterior distribution (install `corner` for a nicer plot)",
                    fontsize=11,
                )
                fig.tight_layout()
                self._save(fig, "mcmc_corner.png")

        # 4. Convergence diagnostics ──────────────────────────────────────────
        all_steps_sorted = sorted(chain_df["step"].unique())
        post_steps = [s for s in all_steps_sorted if s >= burnin]
        if len(post_steps) > 10 and len(param_names) > 0:
            checkpoints = post_steps[::max(1, len(post_steps) // 10)]
            tau_history: List[np.ndarray] = []
            gr_history: List[np.ndarray] = []
            ess_history: List[np.ndarray] = []
            ckpt_labels: List[int] = []
            for ckpt in checkpoints:
                sub = chain_df[
                    (chain_df["step"] >= burnin) &
                    (chain_df["step"] <= ckpt) &
                    (chain_df["accepted"] == 1)
                ]
                if sub.empty:
                    continue
                n_s = ckpt - burnin + 1
                chains_3d = np.zeros((n_s, n_walkers, len(param_names)))
                for w in range(n_walkers):
                    wdf = sub[sub["walker"] == w].sort_values("step")
                    for pi, pname in enumerate(param_names):
                        if pname in wdf.columns:
                            vals = wdf[pname].to_numpy(float)
                            chains_3d[:len(vals), w, pi] = vals
                taus = np.array([
                    autocorr_time(chains_3d[:, w, pi].ravel())
                    for pi in range(len(param_names))
                ])
                esss = np.array([
                    n_walkers * n_s / max(taus[pi], 1)
                    for pi in range(len(param_names))
                ])
                gr = gelman_rubin(chains_3d)
                tau_history.append(taus)
                ess_history.append(esss)
                gr_history.append(gr)
                ckpt_labels.append(ckpt)

            if tau_history:
                metrics = [("τ (autocorr time)", tau_history),
                           ("ESS", ess_history),
                           ("Gelman-Rubin R̂", gr_history)]
                param_clrs = cm.tab10(np.linspace(0, 0.9, len(param_names)))
                fig, axes = plt.subplots(len(metrics), 1,
                                         figsize=(10, 4 * len(metrics)),
                                         sharex=True)
                for ax, (label, hist) in zip(axes, metrics):
                    arr = np.array(hist)  # (n_ckpts, n_params)
                    for pi, pname in enumerate(param_names):
                        ax.plot(ckpt_labels, arr[:, pi], "-o", ms=5,
                                color=param_clrs[pi], label=pname)
                    ax.set_ylabel(label, fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.35)
                axes[-1].set_xlabel("Chain step", fontsize=11)
                axes[0].set_title("MCMC Convergence Diagnostics",
                                  fontsize=13, fontweight="bold")
                fig.tight_layout()
                self._save(fig, "mcmc_convergence.png")

    # ── Optimizer convergence ─────────────────────────────────────────────────

    def plot_optimization(self, history_df: pd.DataFrame) -> None:
        """Objective value vs. evaluation number, with running best."""
        if history_df.empty or "objective" not in history_df.columns:
            self._log.warning("No optimization history to plot.")
            return
        y = history_df["objective"].to_numpy(float)
        x = np.arange(1, len(y) + 1)
        best = (np.fmax.accumulate(y) if self.cfg.optimizer.maximize
                else np.fmin.accumulate(y))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(x, y, "o", ms=4, alpha=0.5, color="#1f77b4", label="evaluations")
        ax.plot(x, best, "-", lw=2, color="#d62728", label="running best")
        ax.set_xlabel("Evaluation #", fontsize=12)
        ax.set_ylabel("Objective", fontsize=12)
        ax.set_title(
            f"Gradient-free optimization ({self.cfg.optimizer.method})",
            fontsize=14, fontweight="bold",
        )
        ax.legend(framealpha=0.85)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        self._save(fig, "optimization_convergence.png")

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def plot_scan(self, df: Optional[pd.DataFrame]) -> None:
        if df is None or df.empty:
            self._log.error("Empty DataFrame.")
            return
        n = len(self.cfg.swept_names)
        if n == 0:
            self._log.warning("No swept params.")
        elif n == 1:
            self.plot_1d(df)
        elif self.cfg.is_2d_grid:
            self.plot_2d(df)
        else:
            self.plot_nd(df)
