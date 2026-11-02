"""M6: bootstrap uncertainty for the optimiser.

Residual block bootstrap of the fitted MMM. Transform settings (decay,
saturation) stay fixed at M3's values; only the regression coefficients are
re-estimated on each draw. Intervals therefore describe coefficient
uncertainty, and understate total model uncertainty. Residuals are inflated
by sqrt(n / (n - p)) because in-sample residuals are too small.

Ridge shrinks coefficients, and refitting ridge on data generated from an
already-shrunk fit shrinks them a second time. Draws are therefore re-centred
on the point-estimate channel coefficients (bias-corrected bootstrap), so the
spread comes from the bootstrap but the centre stays at M3's estimate.
"""

import copy

import numpy as np

from src.config import CHANNELS, CONTROLS, SEED, TARGET
from src.optimiser import channel_responses


def bootstrap_models(model, df, n_draws=200, block=3, seed=SEED,
                     recentre=True):
    """Return `n_draws` refitted copies of `model` (moving-block residuals)."""
    n = len(df)
    p = len(CHANNELS) + len(CONTROLS) + 1
    inflate = float(np.sqrt(n / max(n - p, 1)))

    fitted = model.predict(df)
    resid = df[TARGET].to_numpy(dtype=float) - fitted
    resid = (resid - resid.mean()) * inflate

    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_draws):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block + 1))
            idx.extend(range(start, start + block))
        idx = np.array(idx[:n])

        draw = df.copy()
        draw[TARGET] = np.maximum(fitted + resid[idx], 1.0)

        member = copy.deepcopy(model)
        member.refit(draw)
        models.append(member)

    if recentre:
        k = len(CHANNELS)
        point = model.coef_[:k] * model.revenue_scale_  # LKR per unit response
        draws = np.array([m.coef_[:k] * m.revenue_scale_ for m in models])
        shifted = np.maximum(point + (draws - draws.mean(axis=0)), 0.0)
        for m, coef_lkr in zip(models, shifted):
            m.coef_ = m.coef_.copy()
            m.coef_[:k] = coef_lkr / m.revenue_scale_
    return models


def uplift_distribution(models, optimal, baseline):
    """Incremental revenue of `optimal` minus `baseline` for each draw."""
    return np.array([
        channel_responses(m, optimal).sum() -
        channel_responses(m, baseline).sum()
        for m in models
    ])


def summarise_uplift(draws):
    """p05 / p50 / p95 and the share of draws where uplift is positive."""
    draws = np.asarray(draws, dtype=float)
    return {
        "uplift_p05_lkr": float(np.percentile(draws, 5)),
        "uplift_p50_lkr": float(np.percentile(draws, 50)),
        "uplift_p95_lkr": float(np.percentile(draws, 95)),
        "prob_uplift_positive": float(np.mean(draws > 0)),
    }
