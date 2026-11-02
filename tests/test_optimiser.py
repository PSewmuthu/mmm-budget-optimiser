"""Tests for M6 (optimiser + bootstrap helpers)."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.config import CHANNELS
from src.optimiser import (
    channel_bounds,
    is_feasible,
    kkt_check,
    marginal_roi,
    optimise,
    optimise_de,
    optimise_full,
    optimise_slsqp,
    project,
    total_response
)

A = np.array([3.0, 2.0, 5.0, 4.0, 1.0, 2.5]) * 1e6      # response scale
K = np.array([2.0, 2.5, 1.5, 1.8, 2.0, 2.2]) * 1e6      # saturation point


class LogModel:
    """Stub model with the M3 interface: a_j * ln(1 + s / k_j)."""

    def __init__(self, a=A, k=K):
        self.a, self.k = a, k

    def response_curve(self, channel, grid):
        j = CHANNELS.index(channel)
        return self.a[j] * np.log1p(np.asarray(grid, dtype=float) / self.k[j])


MEAN = np.array([4.0, 4.0, 3.0, 2.0, 1.5, 2.5]) * 1e6
BUDGET = float(MEAN.sum())
LOW, HIGH = channel_bounds(MEAN, 0.0, 3.0)   # loose bounds: interior optimum


def test_project_is_feasible_and_idempotent():
    low, high = channel_bounds(MEAN, 0.5, 1.5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(0, 2 * MEAN.max(), len(MEAN))
        s = project(x, BUDGET, low, high)
        assert np.isclose(s.sum(), BUDGET, rtol=1e-9)
        assert np.all(s >= low - 1e-6) and np.all(s <= high + 1e-6)
        assert np.allclose(project(s, BUDGET, low, high), s, rtol=1e-8)


def test_infeasible_budget_raises():
    low, high = channel_bounds(MEAN, 0.8, 1.2)
    assert not is_feasible(BUDGET * 1.5, low, high)
    with pytest.raises(ValueError):
        optimise(LogModel(), BUDGET * 1.5, low, high)


def test_matches_analytic_solution_for_log_response():
    """Interior optimum of sum a ln(1+s/k): s_j = a_j/lam - k_j."""
    model = LogModel()
    lam_lo, lam_hi = 1e-9, 10.0
    for _ in range(200):
        lam = 0.5 * (lam_lo + lam_hi)
        s = np.maximum(A / lam - K, 0.0)
        if s.sum() > BUDGET:
            lam_lo = lam
        else:
            lam_hi = lam
    expected = np.maximum(A / lam - K, 0.0)

    got = optimise(model, BUDGET, LOW, HIGH)
    assert np.isclose(got.sum(), BUDGET, rtol=1e-6)
    assert np.allclose(got, expected, rtol=2e-3, atol=2e4)


def test_constraints_bounds_and_improvement():
    model = LogModel()
    low, high = channel_bounds(MEAN, 0.5, 1.5)
    res = optimise_full(model, BUDGET, low, high, current=MEAN)
    alloc = res["allocation"]
    assert np.isclose(alloc.sum(), BUDGET, rtol=1e-8)
    assert np.all(alloc >= low - 1e-3) and np.all(alloc <= high + 1e-3)
    assert res["revenue"] >= total_response(model, MEAN) - 1e-6
    equal = project(np.full(6, BUDGET / 6), BUDGET, low, high)
    assert res["revenue"] >= total_response(model, equal) - 1e-6


def test_slsqp_and_de_agree():
    model = LogModel()
    low, high = channel_bounds(MEAN, 0.5, 1.5)
    a = optimise_slsqp(model, BUDGET, low, high)
    b = optimise_de(model, BUDGET, low, high)
    ra, rb = total_response(model, a), total_response(model, b)
    assert abs(ra - rb) / ra < 1e-3


def test_marginal_roi_equalised_at_optimum():
    model = LogModel()
    res = optimise_full(model, BUDGET, LOW, HIGH, current=MEAN, run_de=False)
    interior = [m for m, s in zip(res["marginal_roi"], res["bound_status"])
                if s == "interior"]
    assert len(interior) >= 2
    assert np.ptp(interior) / np.mean(interior) < 0.03
    assert res["kkt_ok"]


def test_kkt_detects_a_bad_allocation():
    model = LogModel()
    bad = np.array([8.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 1e6 * BUDGET / 13e6
    mroi = marginal_roi(model, bad)
    assert not kkt_check(mroi, ["interior"] * 6)[0]


def test_ensemble_of_models_is_averaged():
    m1, m2 = LogModel(A), LogModel(A * 2)
    spend = MEAN
    both = total_response([m1, m2], spend)
    assert np.isclose(both, 0.5 * (total_response(m1, spend)
                                   + total_response(m2, spend)))


# ----------------------- tests against the real M3 model -------------------
MODEL_PATH = config.MODEL_DIR / "mmm_model.joblib"


@pytest.mark.skipif(not (MODEL_PATH.exists() and config.CLEAN_CSV.exists()),
                    reason="M3 model or clean data not available")
def test_real_model_optimisation():
    import joblib

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(config.CLEAN_CSV)
    current = df[CHANNELS].mean().to_numpy(dtype=float)
    budget = float(current.sum())
    low, high = channel_bounds(current, 0.7, 1.3)

    res = optimise_full(model, budget, low, high, current=current,
                        n_starts=6)
    assert np.isclose(res["allocation"].sum(), budget, rtol=1e-8)
    assert np.all(res["allocation"] >= low - 1e-3)
    assert np.all(res["allocation"] <= high + 1e-3)
    assert res["revenue"] >= total_response(model, current) - 1e-3
    assert res["solvers_agree"]
    assert res["kkt_ok"]


@pytest.mark.skipif(not (MODEL_PATH.exists() and config.CLEAN_CSV.exists()),
                    reason="M3 model or clean data not available")
def test_bootstrap_draws_are_centred_on_point_estimate():
    import joblib

    from src.m6_uncertainty import bootstrap_models

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(config.CLEAN_CSV, parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)
    boots = bootstrap_models(model, df, n_draws=20, block=3, seed=1)
    k = len(CHANNELS)
    point = model.coef_[:k] * model.revenue_scale_
    mean = np.mean([m.coef_[:k] * m.revenue_scale_ for m in boots], axis=0)
    assert np.allclose(mean, point, rtol=0.15)
    assert all((m.coef_[:k] >= 0).all() for m in boots)
