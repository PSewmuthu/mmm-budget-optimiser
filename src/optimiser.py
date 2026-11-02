"""M6: Budget optimiser.

Maximise predicted incremental revenue subject to
    sum(spend) == budget   and   low <= spend <= high  (per channel).

With a fixed budget, maximising revenue is the same as maximising ROI,
because the denominator (total spend) does not change.

Only the public M3 interface is used (`response_curve`), so any refitted
model, or a list of bootstrap models, can be passed in.

Spend is per month (LKR) and represents a constant steady-state monthly
budget, matching M3's `response_curve` definition.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.config import BOUND_HIGH, BOUND_LOW, CHANNELS, SEED

# Work in LKR millions inside the solvers for numerical stability.
UNIT = 1e6

# Bounds, feasibility, projection


def channel_bounds(mean_spend, low_frac=BOUND_LOW, high_frac=BOUND_HIGH):
    """Per-channel bounds as fractions of historical mean monthly spend."""
    mean_spend = np.asarray(mean_spend, dtype=float)
    if low_frac < 0 or high_frac < low_frac:
        raise ValueError("Need 0 <= low_frac <= high_frac")
    return mean_spend * low_frac, mean_spend * high_frac


def is_feasible(budget, low, high):
    return float(np.sum(low)) - 1e-6 <= budget <= float(np.sum(high)) + 1e-6


def _check_inputs(budget, low, high):
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.shape != high.shape or low.shape != (len(CHANNELS),):
        raise ValueError(f"low/high must have length {len(CHANNELS)}")
    if np.any(low > high):
        raise ValueError("low must not exceed high")
    if not is_feasible(budget, low, high):
        raise ValueError(
            f"Budget {budget:,.0f} is outside the feasible range "
            f"[{low.sum():,.0f}, {high.sum():,.0f}]"
        )
    return low, high


def project(x, budget, low, high):
    """Euclidean projection onto {sum(s) = budget, low <= s <= high}.

    Solves s = clip(x - lam, low, high) for the shift `lam` that makes the
    sum equal the budget (bisection). Always returns a feasible allocation.
    """
    x = np.asarray(x, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    lo_lam = float(np.min(x - high))
    hi_lam = float(np.max(x - low))
    for _ in range(200):
        lam = 0.5 * (lo_lam + hi_lam)
        total = np.clip(x - lam, low, high).sum()
        if total > budget:
            lo_lam = lam
        else:
            hi_lam = lam
        if abs(total - budget) < 1e-9 * max(1.0, abs(budget)):
            break
    return np.clip(x - lam, low, high)


# Objective and diagnostics
def _as_list(model):
    return list(model) if isinstance(model, (list, tuple)) else [model]


def channel_responses(model, spend):
    """Incremental revenue (LKR/month) per channel for one model."""
    return np.array([
        float(np.asarray(model.response_curve(ch, [s]))[0])
        for ch, s in zip(CHANNELS, spend)
    ])


def total_response(model, spend):
    """Total incremental revenue (LKR/month). A list of models is averaged."""
    models = _as_list(model)
    return float(np.mean([channel_responses(m, spend).sum() for m in models]))


def marginal_roi(model, spend, rel_step=0.01, min_step=1e4):
    """Revenue gained per extra LKR on each channel (central difference)."""
    models = _as_list(model)
    spend = np.asarray(spend, dtype=float)
    out = np.zeros(len(CHANNELS))
    for j, ch in enumerate(CHANNELS):
        step = max(rel_step * spend[j], min_step)
        lo = max(spend[j] - step, 0.0)
        hi = spend[j] + step
        vals = [
            (float(np.asarray(m.response_curve(ch, [hi]))[0])
             - float(np.asarray(m.response_curve(ch, [lo]))[0])) / (hi - lo)
            for m in models
        ]
        out[j] = np.mean(vals)
    return out


def bound_status(spend, low, high, tol=1e-4):
    """'at_lower', 'at_upper' or 'interior' per channel."""
    out = []
    for s, lo, hi in zip(spend, low, high):
        scale = max(hi, 1.0)
        if s <= lo + tol * scale:
            out.append("at_lower")
        elif s >= hi - tol * scale:
            out.append("at_upper")
        else:
            out.append("interior")
    return out


def kkt_check(mroi, status, rel_tol=0.03):
    """Equimarginal (KKT) check at the optimum.

    Interior channels must share one marginal ROI (lambda); channels at their
    upper bound must have marginal ROI >= lambda; at lower bound <= lambda.
    Returns (ok, lambda or nan).
    """
    interior = [m for m, s in zip(mroi, status) if s == "interior"]
    if not interior:
        return True, float("nan")
    lam = float(np.mean(interior))
    slack = rel_tol * abs(lam)
    ok = all(abs(m - lam) <= slack for m in interior)
    for m, s in zip(mroi, status):
        if s == "at_upper" and m < lam - slack:
            ok = False
        if s == "at_lower" and m > lam + slack:
            ok = False
    return bool(ok), lam


# Solvers
def _start_points(budget, low, high, current, n_starts, rng):
    starts = [project(current, budget, low, high),
              project(np.full(len(low), budget / len(low)), budget, low, high)]
    for _ in range(max(n_starts - len(starts), 0)):
        weights = rng.dirichlet(np.ones(len(low)))
        starts.append(project(weights * budget, budget, low, high))
    return starts[:max(n_starts, 1)]


def optimise_slsqp(model, budget, low, high, n_starts=12, seed=SEED,
                   current=None):
    """Multi-start SLSQP. Returns the best feasible allocation found."""
    low, high = _check_inputs(budget, low, high)
    rng = np.random.default_rng(seed)
    if current is None:
        current = 0.5 * (low + high)

    def neg(x_m):
        return -total_response(model, x_m * UNIT) / UNIT

    best, best_val = None, -np.inf
    for x0 in _start_points(budget, low, high, current, n_starts, rng):
        res = minimize(
            neg,
            x0 / UNIT,
            method="SLSQP",
            bounds=list(zip(low / UNIT, high / UNIT)),
            constraints=[{"type": "eq",
                          "fun": lambda x: x.sum() - budget / UNIT}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        cand = project(res.x * UNIT, budget, low, high)
        val = total_response(model, cand)
        if val > best_val:
            best, best_val = cand, val
    return best


def optimise_de(model, budget, low, high, seed=SEED, popsize=20,
                maxiter=300):
    """Differential evolution cross-check.

    Every candidate is projected onto the feasible set, so the budget
    equality and bounds hold exactly without penalties.
    """
    low, high = _check_inputs(budget, low, high)

    def neg(x_m):
        s = project(x_m * UNIT, budget, low, high)
        return -total_response(model, s) / UNIT

    res = differential_evolution(
        neg,
        bounds=list(zip(low / UNIT, high / UNIT)),
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-9,
        polish=False,
    )
    return project(res.x * UNIT, budget, low, high)


def optimise_full(model, budget, low, high, current=None, n_starts=12,
                  seed=SEED, run_de=True, agree_tol=1e-3):
    """Run SLSQP (primary) and DE (cross-check); return a result dict."""
    low, high = _check_inputs(budget, low, high)
    slsqp = optimise_slsqp(model, budget, low, high, n_starts, seed, current)
    slsqp_val = total_response(model, slsqp)

    result = {
        "allocation": slsqp,
        "revenue": slsqp_val,
        "solver": "slsqp",
        "de_allocation": None,
        "de_revenue": float("nan"),
        "rel_gap_de": float("nan"),
        "solvers_agree": None,
    }

    if run_de:
        de = optimise_de(model, budget, low, high, seed)
        de_val = total_response(model, de)
        gap = abs(de_val - slsqp_val) / max(abs(slsqp_val), 1.0)
        result.update(
            de_allocation=de,
            de_revenue=de_val,
            rel_gap_de=gap,
            solvers_agree=bool(gap <= agree_tol),
        )
        if de_val > slsqp_val:  # keep whichever solver found the better point
            result.update(allocation=de, revenue=de_val, solver="de")

    alloc = result["allocation"]
    mroi = marginal_roi(model, alloc)
    status = bound_status(alloc, low, high)
    kkt_ok, lam = kkt_check(mroi, status)
    result.update(marginal_roi=mroi, bound_status=status,
                  kkt_ok=kkt_ok, lambda_mroi=lam)
    return result


def optimise(model, budget: float, low: np.ndarray, high: np.ndarray):
    """Contract entry point: best feasible allocation (LKR/month per channel)."""
    return optimise_full(model, budget, low, high)["allocation"]
