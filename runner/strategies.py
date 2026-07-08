"""Built-in workflow strategies for the agnostic runner.

Every strategy is pure standard library (``math``/``random``/
``multiprocessing``/``statistics``) and upgrades transparently to
scipy/emcee when installed and requested via ``options["backend"]``:

* ``optimize``        — bound-constrained Nelder-Mead simplex (checkpointed);
* ``hybrid``          — global Differential Evolution seeding a Nelder-Mead
  local polish (checkpointed, globally-numbered two-phase history);
* ``grid``            — concurrent Cartesian parameter sweep
  (``multiprocessing.Pool``, chunk-level checkpoint/resume);
* ``mcmc``            — affine-invariant ensemble sampler
  (Goodman & Weare stretch move; emcee-compatible);
* ``mcmc_diagnostic`` — the same sampler with integrated split-chain
  Gelman-Rubin R̂ / effective-sample-size monitoring and auto-stopping;
* ``benchmark``       — hardware scaling run across worker counts with
  speedup/efficiency profiling.

Models are plain callables taking keyword parameters and returning a float
(objective value or log-probability).  All strategies poll
``ctx.checkpoint.stop_requested`` between units of work, so SLURM wall-time
signals pause the run resumably instead of killing it.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from varify.runner.core import RunContext, register_strategy

_PENALTY = 1.0e300


def _call_kwargs(job: Tuple[Callable[..., float], Dict[str, float]]) -> float:
    """Top-level worker shim (must be picklable for multiprocessing)."""
    model, kwargs = job
    return float(model(**kwargs))


def _bounded(x: Sequence[float], bounds: List[Tuple[float, float]]) -> bool:
    return all(lo <= xi <= hi for xi, (lo, hi) in zip(x, bounds))


# ═════════════════════════════════════════════════════════════════════════════
#  1. Bound-constrained optimization (Nelder-Mead, checkpointed)
# ═════════════════════════════════════════════════════════════════════════════

@register_strategy("optimize")
def optimize_strategy(ctx: RunContext) -> Dict[str, Any]:
    """Minimize ``model(**params)`` inside box bounds.

    Pure-Python Nelder-Mead by default; ``options["backend"]="scipy"`` uses
    ``scipy.optimize.minimize`` when installed.  Out-of-bounds vertices get a
    barrier penalty.  On interruption the evaluation history and incumbent
    are checkpointed; resuming restarts the simplex at the incumbent with
    the remaining evaluation budget.
    """
    spec, opts = ctx.spec, ctx.spec.options
    names = list(spec.bounds)
    bounds = [tuple(map(float, spec.bounds[n])) for n in names]
    maximize = bool(opts.get("maximize", False))
    max_evals = int(opts.get("max_evaluations", 200))
    xatol = float(opts.get("xatol", 1e-6))
    fatol = float(opts.get("fatol", 1e-8))
    sign = -1.0 if maximize else 1.0

    state = ctx.checkpoint.load() or {"history": [], "best": None}
    history: List[Dict[str, Any]] = state["history"]

    def objective(x: List[float]) -> float:
        if not _bounded(x, bounds):
            return _PENALTY
        assert spec.model is not None
        value = sign * float(spec.model(**dict(zip(names, x))))
        history.append({"x": list(x), "value": sign * value})
        if state["best"] is None or value < sign * float(state["best"]["value"]):
            state["best"] = {"x": list(x), "value": sign * value}
        ctx.checkpoint.maybe_save(state)
        return value

    def budget_left() -> int:
        return max_evals - len(history)

    # Starting point: resume incumbent, else box centre.
    if state["best"] is not None:
        x0 = list(state["best"]["x"])
        ctx.log.info("Resuming from incumbent %s (%d evals spent)",
                     x0, len(history))
    else:
        x0 = [0.5 * (lo + hi) for lo, hi in bounds]

    backend = str(opts.get("backend", "builtin"))
    converged = False
    if backend == "scipy":
        try:
            from scipy import optimize as _sopt  # type: ignore[import-untyped]

            res = _sopt.minimize(
                lambda x: objective(list(x)), x0, method="Nelder-Mead",
                bounds=bounds,
                options={"maxfev": budget_left(), "xatol": xatol,
                         "fatol": fatol},
            )
            converged = bool(res.success)
        except ImportError:
            ctx.log.warning("scipy unavailable — using built-in Nelder-Mead")
            backend = "builtin"
    if backend != "scipy":
        converged = _nelder_mead(
            objective, x0, bounds, budget_left, xatol, fatol,
            stop=lambda: ctx.checkpoint.stop_requested,
        )

    ctx.checkpoint.save(state)
    ctx.save_rows(
        "history.csv",
        ["eval", *names, "value"],
        [[i + 1, *h["x"], h["value"]] for i, h in enumerate(history)],
    )
    best = state["best"] or {"x": x0, "value": float("nan")}
    return {
        "best_params": dict(zip(names, best["x"])),
        "best_value": best["value"],
        "n_evaluations": len(history),
        "converged": converged and not ctx.checkpoint.stop_requested,
        "backend": backend,
    }


def _nelder_mead(
    fn: Callable[[List[float]], float],
    x0: List[float],
    bounds: List[Tuple[float, float]],
    budget_left: Callable[[], int],
    xatol: float,
    fatol: float,
    stop: Callable[[], bool],
) -> bool:
    """Standard Nelder-Mead simplex; returns True when tolerance was met."""
    ndim = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    simplex: List[List[float]] = [list(x0)]
    for i in range(ndim):
        v = list(x0)
        span = 0.05 * (bounds[i][1] - bounds[i][0])
        v[i] = min(v[i] + (span if span > 0 else 0.05), bounds[i][1])
        simplex.append(v)
    fvals = [fn(v) for v in simplex]

    def centroid(exclude: int) -> List[float]:
        pts = [v for k, v in enumerate(simplex) if k != exclude]
        return [sum(p[d] for p in pts) / len(pts) for d in range(ndim)]

    def combine(a: List[float], b: List[float], t: float) -> List[float]:
        return [ai + t * (bi - ai) for ai, bi in zip(a, b)]

    while budget_left() > 0 and not stop():
        order = sorted(range(ndim + 1), key=lambda k: fvals[k])
        simplex = [simplex[k] for k in order]
        fvals = [fvals[k] for k in order]
        spread_f = abs(fvals[-1] - fvals[0])
        spread_x = max(
            abs(simplex[-1][d] - simplex[0][d]) for d in range(ndim)
        )
        if spread_f <= fatol and spread_x <= xatol:
            return True

        cen = centroid(ndim)
        xr = combine(cen, simplex[-1], -alpha)
        fr = fn(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = combine(cen, simplex[-1], -gamma)
            fe = fn(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            xc = combine(cen, simplex[-1], rho)
            fc = fn(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:  # shrink toward the best vertex
                for k in range(1, ndim + 1):
                    simplex[k] = combine(simplex[0], simplex[k], sigma)
                    fvals[k] = fn(simplex[k])
    return False


# ═════════════════════════════════════════════════════════════════════════════
#  1b. Hybrid global/local optimization (Differential Evolution → Nelder-Mead)
# ═════════════════════════════════════════════════════════════════════════════

@register_strategy("hybrid")
def hybrid_strategy(ctx: RunContext) -> Dict[str, Any]:
    """Minimize ``model(**params)`` with global DE seeding a local NM polish.

    Pure-Python rand/1/bin Differential Evolution explores the box first;
    its champion hands off to the existing :func:`_nelder_mead` simplex for
    the remaining evaluation budget.  Both phases share one evaluation
    history, globally numbered, with a ``phase`` column ("de"/"nm")
    distinguishing them.  Checkpointing mirrors :func:`optimize_strategy`:
    the population/fitness/generation/stall counters and the running
    incumbent are snapshotted every autosave interval, so an interrupted
    run resumes mid-population-init, mid-generation (redoing at most the
    in-flight generation) or mid-simplex.
    """
    spec, opts, rng = ctx.spec, ctx.spec.options, ctx.rng
    names = list(spec.bounds)
    bounds = [tuple(map(float, spec.bounds[n])) for n in names]
    ndim = len(names)
    maximize = bool(opts.get("maximize", False))
    max_evals = int(opts.get("max_evaluations", 200))
    xatol = float(opts.get("xatol", 1e-6))
    fatol = float(opts.get("fatol", 1e-8))
    sign = -1.0 if maximize else 1.0

    popsize = int(opts.get("de_popsize", 15))
    generations = int(opts.get("de_generations", 20))
    f_scale = float(opts.get("de_f", 0.7))
    cr = float(opts.get("de_cr", 0.9))
    stall_limit = int(opts.get("de_stall", 5))

    assert spec.model is not None
    state = ctx.checkpoint.load() or {
        "history": [], "best": None, "phase": "de",
        "population": None, "fitness": None,
        "generation": 0, "stall": 0,
    }
    history: List[Dict[str, Any]] = state["history"]

    def budget_left() -> int:
        return max_evals - len(history)

    def can_eval() -> bool:
        return budget_left() > 0 and not ctx.checkpoint.stop_requested

    def objective(x: List[float], phase: str) -> float:
        if not _bounded(x, bounds):
            return _PENALTY
        value = sign * float(spec.model(**dict(zip(names, x))))
        history.append({"x": list(x), "value": sign * value, "phase": phase})
        if state["best"] is None or value < sign * float(state["best"]["value"]):
            state["best"] = {"x": list(x), "value": sign * value}
        ctx.checkpoint.maybe_save(state)
        return value

    # ── Phase 1: rand/1/bin Differential Evolution ────────────────────────────
    if state["phase"] == "de":
        if state["population"] is None:
            population: List[List[float]] = []
            x0_opt = opts.get("x0")
            if x0_opt is not None:
                population.append([float(x0_opt[n]) for n in names])
            while len(population) < popsize:
                population.append(
                    [rng.uniform(lo, hi) for lo, hi in bounds]
                )
            state["population"] = population
            state["fitness"] = []
            ctx.checkpoint.maybe_save(state)

        population = state["population"]
        fitness = state["fitness"]
        while len(fitness) < len(population) and can_eval():
            fitness.append(objective(population[len(fitness)], "de"))
        state["fitness"] = fitness

        generation = state["generation"]
        stall = state["stall"]
        while (
            len(fitness) == len(population)
            and generation < generations
            and stall < stall_limit
            and can_eval()
        ):
            prev_best = min(fitness)
            for i in range(len(population)):
                if not can_eval():
                    break
                idxs = [j for j in range(len(population)) if j != i]
                if len(idxs) >= 3:
                    r1, r2, r3 = rng.sample(idxs, 3)
                else:
                    r1, r2, r3 = (rng.choice(idxs) for _ in range(3))
                mutant = [
                    population[r1][d]
                    + f_scale * (population[r2][d] - population[r3][d])
                    for d in range(ndim)
                ]
                j_rand = rng.randrange(ndim)
                trial = [
                    mutant[d] if (rng.random() < cr or d == j_rand)
                    else population[i][d]
                    for d in range(ndim)
                ]
                trial = [
                    min(max(v, lo), hi) for v, (lo, hi) in zip(trial, bounds)
                ]
                trial_f = objective(trial, "de")
                if trial_f <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f
            generation += 1
            best_now = min(fitness)
            stall = stall + 1 if prev_best - best_now < fatol else 0
            state["population"], state["fitness"] = population, fitness
            state["generation"], state["stall"] = generation, stall
            ctx.checkpoint.maybe_save(state)

        state["phase"] = "nm"
        ctx.checkpoint.maybe_save(state)

    # ── Phase 2: Nelder-Mead local polish, seeded by the DE champion ─────────
    converged = False
    if state["best"] is not None and can_eval():
        x0_nm = list(state["best"]["x"])
        converged = _nelder_mead(
            lambda x: objective(x, "nm"), x0_nm, bounds, budget_left,
            xatol, fatol, stop=lambda: ctx.checkpoint.stop_requested,
        )

    ctx.checkpoint.save(state)
    ctx.save_rows(
        "history.csv",
        ["eval", *names, "value", "phase"],
        [[i + 1, *h["x"], h["value"], h["phase"]] for i, h in enumerate(history)],
    )
    best = state["best"] or {
        "x": [0.5 * (lo + hi) for lo, hi in bounds], "value": float("nan"),
    }
    return {
        "best_params": dict(zip(names, best["x"])),
        "best_value": best["value"],
        "n_evaluations": len(history),
        "converged": converged and not ctx.checkpoint.stop_requested,
        "backend": "builtin",
        "de_generations": state["generation"],
        "de_evaluations": sum(1 for h in history if h["phase"] == "de"),
        "nm_evaluations": sum(1 for h in history if h["phase"] == "nm"),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  2. Concurrent grid sweep (multiprocessing, chunk-checkpointed)
# ═════════════════════════════════════════════════════════════════════════════

@register_strategy("grid")
def grid_strategy(ctx: RunContext) -> Dict[str, Any]:
    """Evaluate ``model(**params)`` over the Cartesian product of
    ``options["axes"]`` using a worker pool; completed points are
    checkpointed per chunk, so interrupted sweeps resume with no rework."""
    spec, opts = ctx.spec, ctx.spec.options
    axes: Dict[str, List[float]] = {
        k: [float(v) for v in vals] for k, vals in opts["axes"].items()
    }
    names = list(axes)
    workers = int(opts.get("workers", 1))
    points = list(itertools.product(*axes.values()))
    chunk_size = int(opts.get("chunk_size", max(1, len(points) // 20)))

    state = ctx.checkpoint.load() or {"completed": {}}
    completed: Dict[str, float] = state["completed"]
    pending = [
        (i, pt) for i, pt in enumerate(points) if str(i) not in completed
    ]
    if len(pending) < len(points):
        ctx.log.info("Resuming grid: %d/%d points already done",
                     len(points) - len(pending), len(points))

    assert spec.model is not None
    while pending and not ctx.checkpoint.stop_requested:
        chunk, pending = pending[:chunk_size], pending[chunk_size:]
        jobs = [
            (spec.model, dict(zip(names, pt))) for _, pt in chunk
        ]
        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                values = pool.map(_call_kwargs, jobs)
        else:
            values = [_call_kwargs(job) for job in jobs]
        for (i, _pt), val in zip(chunk, values):
            completed[str(i)] = val
        ctx.checkpoint.save(state)
        ctx.log.info("Grid progress: %d/%d", len(completed), len(points))

    ctx.save_rows(
        "grid_points.csv",
        [*names, "value"],
        [
            [*points[i], completed[str(i)]]
            for i in range(len(points)) if str(i) in completed
        ],
    )
    if len(names) == 2 and len(completed) == len(points):
        xs, ys = axes[names[0]], axes[names[1]]
        matrix = [
            [completed[str(ix * len(ys) + iy)] for iy in range(len(ys))]
            for ix in range(len(xs))
        ]
        ctx.save_matrix(
            "grid_matrix.csv", matrix, row_labels=xs, col_labels=ys,
        )
    return {
        "n_points": len(points),
        "n_completed": len(completed),
        "workers": workers,
        "complete": len(completed) == len(points),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  3+4. Affine-invariant ensemble MCMC (+ diagnostics / auto-stop)
# ═════════════════════════════════════════════════════════════════════════════

def _log_prob_wrapper(
    model: Callable[..., float],
    names: List[str],
    bounds: List[Tuple[float, float]],
) -> Callable[[List[float]], float]:
    def log_prob(x: List[float]) -> float:
        if not _bounded(x, bounds):
            return -math.inf
        return float(model(**dict(zip(names, x))))

    return log_prob


def _stretch_sampler(
    ctx: RunContext,
    steps: int,
    on_step: Optional[
        Callable[[int, List[List[List[float]]]], bool]
    ] = None,
) -> Dict[str, Any]:
    """Goodman-Weare stretch-move ensemble sampler (pure Python).

    ``on_step(step, chain)`` may return True to request early termination
    (used by the diagnostic auto-stopper).  Chain state lives in the
    checkpoint, so interrupted chains resume seamlessly.
    """
    spec, opts, rng = ctx.spec, ctx.spec.options, ctx.rng
    names = list(spec.bounds)
    bounds = [tuple(map(float, spec.bounds[n])) for n in names]
    ndim = len(names)
    walkers = int(opts.get("walkers", max(2 * ndim, 4)))
    a = float(opts.get("stretch_a", 2.0))
    assert spec.model is not None
    log_prob = _log_prob_wrapper(spec.model, names, bounds)

    state = ctx.checkpoint.load() or {"chain": [], "log_probs": [], "accepts": 0}
    chain: List[List[List[float]]] = state["chain"]

    if chain:
        ensemble = [list(w) for w in chain[-1]]
        lps = [float(v) for v in state["log_probs"]]
        ctx.log.info("Resuming chain at step %d", len(chain))
    else:
        ensemble = [
            [rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(walkers)
        ]
        lps = [log_prob(w) for w in ensemble]

    accepts = int(state.get("accepts", 0))
    half = walkers // 2
    stopped_early = False

    while len(chain) < steps and not ctx.checkpoint.stop_requested:
        for w in range(walkers):
            comp = range(half, walkers) if w < half else range(half)
            xj = ensemble[rng.choice(list(comp))]
            z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
            proposal = [
                xj[d] + z * (ensemble[w][d] - xj[d]) for d in range(ndim)
            ]
            lp_new = log_prob(proposal)
            log_alpha = (ndim - 1) * math.log(z) + lp_new - lps[w]
            if math.log(rng.random() or 1e-300) < log_alpha:
                ensemble[w] = proposal
                lps[w] = lp_new
                accepts += 1
        chain.append([list(w) for w in ensemble])
        state["log_probs"] = list(lps)
        state["accepts"] = accepts
        ctx.checkpoint.maybe_save(state)
        if on_step is not None and on_step(len(chain), chain):
            stopped_early = True
            break

    ctx.checkpoint.save(state)
    total_proposals = max(len(chain) * walkers, 1)
    return {
        "chain": chain,
        "names": names,
        "walkers": walkers,
        "acceptance_rate": accepts / total_proposals,
        "stopped_early": stopped_early,
        "interrupted": ctx.checkpoint.stop_requested,
    }


def _flat_samples(
    chain: List[List[List[float]]], burn_in: int, dim: int
) -> List[float]:
    return [w[dim] for step in chain[burn_in:] for w in step]


def _export_chain(ctx: RunContext, run: Dict[str, Any], burn_in: int) -> None:
    names = run["names"]
    rows = [
        [s, w, *walker]
        for s, step in enumerate(run["chain"])
        for w, walker in enumerate(step)
    ]
    ctx.save_rows("chain.csv", ["step", "walker", *names], rows)
    # Posterior summary + a marginal histogram matrix for visualization.
    summary: Dict[str, Any] = {}
    hist_matrix: List[List[float]] = []
    for d, name in enumerate(names):
        samples = _flat_samples(run["chain"], burn_in, d)
        if not samples:
            continue
        summary[name] = {
            "mean": statistics.fmean(samples),
            "std": statistics.pstdev(samples),
            "n": len(samples),
        }
        hist_matrix.append(_histogram(samples, bins=30))
    if hist_matrix:
        ctx.save_matrix(
            "posterior_hist_matrix.csv", hist_matrix,
            row_labels=names, col_labels=list(range(30)),
        )
    run["posterior"] = summary


def _histogram(samples: List[float], bins: int) -> List[float]:
    lo, hi = min(samples), max(samples)
    width = (hi - lo) or 1.0
    counts = [0.0] * bins
    for s in samples:
        idx = min(int((s - lo) / width * bins), bins - 1)
        counts[idx] += 1.0
    return counts


@register_strategy("mcmc")
def mcmc_strategy(ctx: RunContext) -> Dict[str, Any]:
    """Baseline affine-invariant ensemble sampling (uniform prior = bounds).

    ``options["backend"]="emcee"`` delegates to emcee when installed;
    otherwise the built-in emcee-compatible stretch-move sampler runs."""
    opts = ctx.spec.options
    steps = int(opts.get("steps", 500))
    burn_in = int(opts.get("burn_in", steps // 5))

    backend = str(opts.get("backend", "builtin"))
    if backend in ("emcee", "auto"):
        try:
            return _emcee_backend(ctx, steps, burn_in)
        except ImportError:
            if backend == "emcee":
                ctx.log.warning("emcee unavailable — using built-in sampler")

    run = _stretch_sampler(ctx, steps)
    _export_chain(ctx, run, burn_in)
    run.pop("chain")
    run["backend"] = "builtin"
    run["steps"] = steps
    run["burn_in"] = burn_in
    return run


def _emcee_backend(ctx: RunContext, steps: int, burn_in: int) -> Dict[str, Any]:
    import emcee  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]

    spec = ctx.spec
    names = list(spec.bounds)
    bounds = [tuple(map(float, spec.bounds[n])) for n in names]
    ndim = len(names)
    walkers = int(spec.options.get("walkers", max(2 * ndim, 4)))
    assert spec.model is not None
    log_prob = _log_prob_wrapper(spec.model, names, bounds)

    p0 = np.array([
        [ctx.rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(walkers)
    ])
    sampler = emcee.EnsembleSampler(
        walkers, ndim, lambda x: log_prob(list(x))
    )
    sampler.run_mcmc(p0, steps, progress=False)
    chain = sampler.get_chain()  # (steps, walkers, ndim)
    run: Dict[str, Any] = {
        "chain": [[list(w) for w in step] for step in chain],
        "names": names,
        "walkers": walkers,
        "acceptance_rate": float(np.mean(sampler.acceptance_fraction)),
        "stopped_early": False,
        "interrupted": False,
    }
    _export_chain(ctx, run, burn_in)
    run.pop("chain")
    run.update({"backend": "emcee", "steps": steps, "burn_in": burn_in})
    return run


# ── Convergence diagnostics (pure Python) ─────────────────────────────────────

def split_rhat(chains: List[List[float]]) -> float:
    """Split-chain Gelman-Rubin R̂ for one parameter.

    *chains* is a list of per-walker sample sequences; each is split in half
    to detect intra-chain drift as well as inter-chain disagreement.
    """
    split: List[List[float]] = []
    for c in chains:
        h = len(c) // 2
        if h >= 2:
            split.extend([c[:h], c[h:2 * h]])
    if len(split) < 2:
        return float("inf")
    n = len(split[0])
    means = [statistics.fmean(c) for c in split]
    variances = [statistics.variance(c) for c in split]
    w = statistics.fmean(variances)
    b = n * statistics.variance(means)
    if w <= 0:
        return float("inf")
    var_hat = (n - 1) / n * w + b / n
    return math.sqrt(var_hat / w)


def autocorr_tau(samples: List[float], max_lag: Optional[int] = None) -> float:
    """Integrated autocorrelation time via Geyer's initial positive sequence."""
    n = len(samples)
    if n < 4:
        return float("inf")
    mean = statistics.fmean(samples)
    dev = [s - mean for s in samples]
    c0 = sum(d * d for d in dev) / n
    if c0 <= 0:
        return 1.0
    tau = 1.0
    for lag in range(1, max_lag or n // 2):
        rho = sum(dev[i] * dev[i + lag] for i in range(n - lag)) / (n * c0)
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau


@register_strategy("mcmc_diagnostic")
def mcmc_diagnostic_strategy(ctx: RunContext) -> Dict[str, Any]:
    """Ensemble MCMC with integrated convergence monitoring & auto-stop.

    Every ``check_every`` steps the split-chain R̂ and effective sample size
    are computed per parameter; sampling stops as soon as every parameter
    satisfies ``rhat_target`` and ``ess_target`` (or at ``steps``)."""
    opts = ctx.spec.options
    steps = int(opts.get("steps", 2000))
    burn_frac = float(opts.get("burn_fraction", 0.3))
    check_every = int(opts.get("check_every", 100))
    min_steps = int(opts.get("min_steps", 4 * check_every))
    rhat_target = float(opts.get("rhat_target", 1.02))
    ess_target = float(opts.get("ess_target", 200.0))
    names = list(ctx.spec.bounds)
    diag_history: List[List[float]] = []

    def diagnostics(chain: List[List[List[float]]]) -> Tuple[List[float], List[float]]:
        burn = int(len(chain) * burn_frac)
        rhats: List[float] = []
        esss: List[float] = []
        walkers = len(chain[0])
        for d in range(len(names)):
            per_walker = [
                [chain[s][w][d] for s in range(burn, len(chain))]
                for w in range(walkers)
            ]
            rhats.append(split_rhat(per_walker))
            pooled = [v for c in per_walker for v in c]
            tau = autocorr_tau(pooled)
            esss.append(len(pooled) / tau if math.isfinite(tau) else 0.0)
        return rhats, esss

    def on_step(step: int, chain: List[List[List[float]]]) -> bool:
        if step % check_every or step < min_steps:
            return False
        rhats, esss = diagnostics(chain)
        diag_history.append([step, *rhats, *esss])
        ctx.log.info(
            "step %d  R̂=%s  ESS=%s", step,
            [f"{r:.4f}" for r in rhats], [f"{e:.0f}" for e in esss],
        )
        return all(r < rhat_target for r in rhats) and \
            all(e >= ess_target for e in esss)

    run = _stretch_sampler(ctx, steps, on_step=on_step)
    burn_in = int(len(run["chain"]) * burn_frac)
    _export_chain(ctx, run, burn_in)
    ctx.save_matrix(
        "diagnostics_history.csv",
        diag_history,
        col_labels=["step", *(f"rhat_{n}" for n in names),
                    *(f"ess_{n}" for n in names)],
    )
    final = diag_history[-1] if diag_history else []
    steps_run = len(run["chain"])
    run.pop("chain")
    run.update({
        "backend": "builtin",
        "steps_run": steps_run,
        "burn_in": burn_in,
        "converged": bool(run["stopped_early"]),
        "final_diagnostics": final,
        "rhat_target": rhat_target,
        "ess_target": ess_target,
    })
    return run


# ═════════════════════════════════════════════════════════════════════════════
#  5. Hardware scaling benchmark
# ═════════════════════════════════════════════════════════════════════════════

@register_strategy("benchmark")
def benchmark_strategy(ctx: RunContext) -> Dict[str, Any]:
    """Profile ``model`` throughput across worker counts.

    Runs ``task_count`` independent kernel invocations under a pool of each
    size in ``options["worker_counts"]``, ``repeats`` times, and reports
    wall time, speedup vs. single-worker and parallel efficiency."""
    spec, opts = ctx.spec, ctx.spec.options
    worker_counts = [int(w) for w in opts["worker_counts"]]
    task_count = int(opts.get("task_count", 16))
    repeats = int(opts.get("repeats", 3))
    task_arg = opts.get("task_arg", {})
    assert spec.model is not None
    jobs = [(spec.model, dict(task_arg)) for _ in range(task_count)]

    state = ctx.checkpoint.load() or {"timings": {}}
    timings: Dict[str, List[float]] = state["timings"]

    for n in worker_counts:
        key = str(n)
        runs = timings.setdefault(key, [])
        while len(runs) < repeats and not ctx.checkpoint.stop_requested:
            t0 = time.perf_counter()
            if n > 1:
                with mp.Pool(processes=n) as pool:
                    pool.map(_call_kwargs, jobs)
            else:
                for job in jobs:
                    _call_kwargs(job)
            runs.append(time.perf_counter() - t0)
            ctx.checkpoint.save(state)
            ctx.log.info("workers=%d repeat %d/%d: %.3fs",
                         n, len(runs), repeats, runs[-1])
        if ctx.checkpoint.stop_requested:
            break

    baseline = statistics.fmean(timings.get("1", [float("nan")]))
    matrix: List[List[float]] = []
    table: Dict[str, Dict[str, float]] = {}
    for n in worker_counts:
        runs = timings.get(str(n), [])
        if not runs:
            continue
        mean = statistics.fmean(runs)
        std = statistics.pstdev(runs) if len(runs) > 1 else 0.0
        speedup = baseline / mean if mean > 0 else float("nan")
        eff = speedup / n
        matrix.append([mean, std, speedup, eff])
        table[str(n)] = {
            "mean_s": mean, "std_s": std,
            "speedup": speedup, "efficiency": eff,
        }
    ctx.save_matrix(
        "scaling_matrix.csv", matrix,
        row_labels=[n for n in worker_counts if str(n) in table],
        col_labels=["mean_s", "std_s", "speedup", "efficiency"],
    )
    return {
        "task_count": task_count,
        "repeats": repeats,
        "scaling": table,
        "interrupted": ctx.checkpoint.stop_requested,
    }
