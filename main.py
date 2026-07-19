"""Varify — centralized CLI entry point.

Modes
─────
  scan      Prepare case directories for the configured line/grid search and
            dispatch one SLURM job per point (legacy sweep logic preserved).
  optimize  Run a blocking optimization loop (--method mcmc | nelder-mead):
            propose parameters → submit job → wait → parse metric → iterate.
  watch     Run the fault-tolerance watchdog daemon (optionally deploy it as
            an independent sbatch agent with --deploy-agent).
  analyze   Harvest all run folders into CSV + SQLite, generate the plotting
            suite under results/plots/ and run the registered analysis hooks.

Examples
────────
  python main.py --mode scan
  python main.py --mode scan --dry-run
  python main.py --mode optimize --method mcmc
  python main.py --mode optimize --method nelder-mead
  python main.py --mode watch --deploy-agent
  python main.py --mode analyze
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from varify.src.analysis import AnalysisDispatcher, PlotSuite, PostJobDispatcher, ResultParser
from varify.src.common.config import FrameworkConfig, load_config
from varify.src.optimizer import make_optimizer
from varify.src.scanner import make_scanner
from varify.src.slurm.dispatcher import SlurmDispatcher
from varify.src.slurm.watchdog import Watchdog, deploy_as_agent


def _setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        stream=sys.stdout)
    return logging.getLogger("varify")


# ═════════════════════════════════════════════════════════════════════════════
#  Mode handlers
# ═════════════════════════════════════════════════════════════════════════════

def run_scan(
    cfg: FrameworkConfig, dry_run: bool, wait: bool, log: logging.Logger,
) -> None:
    scanner = make_scanner(cfg)
    total = scanner.total_points()
    log.info(
        "Scan: mode=%s  swept=%s  total=%d  dry_run=%s  wait=%s",
        cfg.sweep_mode, cfg.swept_names or ["(none)"], total, dry_run, wait,
    )
    submitted = errors = 0
    jobs: List[Tuple[str, Path, Dict[str, float]]] = []  # (job_id, case_dir, params)
    with SlurmDispatcher(cfg, dry_run=dry_run) as dispatcher:
        for i, gp in enumerate(scanner.iter_points(), 1):
            log.info("[%d/%d] %s", i, total, gp.case_dir_name)
            case_dir = scanner.prepare_case(gp)
            job_id = dispatcher.dispatch(gp.job_name, case_dir, gp.params)
            if job_id is not None:
                submitted += 1
                jobs.append((job_id, case_dir, gp.params))
            else:
                errors += 1

        if wait:
            waitable = [jid for jid, _, _ in jobs if jid != "DRY_RUN"]
            if waitable:
                log.info("Waiting for %d submitted job(s) to complete…",
                          len(waitable))
                results = dispatcher.wait_for_batch(
                    waitable,
                    timeout=cfg.optimizer.job_timeout,
                    poll_interval=cfg.optimizer.poll_interval,
                )
                completed = sum(1 for ok in results.values() if ok)
                failed = len(results) - completed
                log.info("Batch complete: %d completed, %d failed/timed out",
                          completed, failed)

                if cfg.post_job_fns:
                    post_job = PostJobDispatcher(cfg)
                    for job_id, case_dir, params in jobs:
                        if job_id != "DRY_RUN" and results.get(job_id, False):
                            post_job.run_case(case_dir, params, job_id)
            else:
                log.info("Nothing to wait for (dry-run or no jobs submitted).")

    log.info("Done: %d submitted, %d errors", submitted, errors)


def run_optimize(
    cfg: FrameworkConfig, method: Optional[str], dry_run: bool,
    log: logging.Logger,
) -> None:
    chosen = (method or cfg.optimizer.method).lower()
    optimizer = make_optimizer(cfg, chosen, dry_run=dry_run)
    log.info("Optimizer: %s (dry_run=%s)", type(optimizer).__name__, dry_run)
    optimizer.run()


def run_watch(
    cfg: FrameworkConfig, config_path: Path, once: bool, deploy: bool,
    dry_run: bool, log: logging.Logger,
) -> None:
    if deploy:
        job = deploy_as_agent(cfg, config_path, dry_run=dry_run)
        if job is None:
            sys.exit(1)
        return
    Watchdog(cfg, dry_run=dry_run).run(once=once)


def run_analyze(
    cfg: FrameworkConfig, no_plots: bool, no_sqlite: bool, no_hooks: bool,
    log: logging.Logger,
) -> None:
    scanner = make_scanner(cfg)
    parser = ResultParser(cfg, scanner)
    plotter = PlotSuite(cfg) if not no_plots else None

    # ── Scan results: harvest → CSV/SQLite → plots → analysis hooks ─────────
    df = parser.harvest()
    if not df.empty:
        if not no_sqlite:
            parser.to_sqlite(df)
        if plotter is not None:
            plotter.plot_scan(df)
        if not no_hooks:
            AnalysisDispatcher(cfg).run_all(df)
    else:
        log.warning("No scan case directories found — skipping scan analysis.")

    # ── MCMC chain (if present) ──────────────────────────────────────────────
    chain_df = parser.load_chain()
    if chain_df is not None and not chain_df.empty:
        if not no_sqlite:
            parser.to_sqlite(chain_df, table="mcmc_chain")
        if plotter is not None:
            plotter.plot_mcmc(chain_df)

    # ── Optimizer history (if present) ───────────────────────────────────────
    hist_df = parser.load_optimization_history()
    if hist_df is not None and not hist_df.empty:
        if not no_sqlite:
            parser.to_sqlite(hist_df, table="optimization_history")
        if plotter is not None:
            plotter.plot_optimization(hist_df)
            if cfg.optimizer.postprocess:
                valid_obj = hist_df["objective"].dropna()
                if not valid_obj.empty:
                    best_row = hist_df.loc[valid_obj.astype(float).idxmin()]
                    case_dir_val = best_row.get("case_dir", "")
                    best_case_dir = (
                        Path(case_dir_val)
                        if isinstance(case_dir_val, str) and case_dir_val
                        else None
                    )
                    plotter.plot_postprocess(hist_df, best_case_dir=best_case_dir)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Varify — SLURM parametric scan, optimization & analysis "
                    "framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["scan", "optimize", "watch", "analyze"],
        help="Operation to perform.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare everything but do NOT call the scheduler.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    # scan
    parser.add_argument("--wait", action="store_true",
                        help="[scan] block until all submitted jobs reach a "
                             "terminal state before exiting.")
    # optimize
    parser.add_argument("--method", choices=["mcmc", "nelder-mead", "de-nm"],
                        default=None,
                        help="[optimize] override optimizer.method from config.")
    # watch
    parser.add_argument("--once", action="store_true",
                        help="[watch] run a single polling pass and exit.")
    parser.add_argument("--deploy-agent", action="store_true",
                        help="[watch] submit the watchdog itself as an sbatch "
                             "background agent.")
    # analyze
    parser.add_argument("--no-plots", action="store_true",
                        help="[analyze] skip figure generation.")
    parser.add_argument("--no-sqlite", action="store_true",
                        help="[analyze] skip the SQLite export.")
    parser.add_argument("--no-analysis-fns", action="store_true",
                        help="[analyze] skip the registered analysis hooks.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    log = _setup_logging(args.verbose)
    cfg = load_config(args.config)

    if args.mode == "scan":
        run_scan(cfg, args.dry_run, args.wait, log)
    elif args.mode == "optimize":
        run_optimize(cfg, args.method, args.dry_run, log)
    elif args.mode == "watch":
        run_watch(cfg, args.config, args.once, args.deploy_agent,
                  args.dry_run, log)
    elif args.mode == "analyze":
        run_analyze(cfg, args.no_plots, args.no_sqlite,
                    args.no_analysis_fns, log)


if __name__ == "__main__":
    main()
