
"""Tests for M3 Primary Marketing Mix Model."""

import numpy as np
import pandas as pd
import pytest

from src.config import CHANNELS, CONTROLS, TARGET
from src.mmm import MMMModel, metrics


@pytest.fixture
def sample_data():
    """Create synthetic data for testing."""

    rng = np.random.default_rng(42)
    n = 36

    df = pd.DataFrame({
        "month": pd.date_range(
            "2023-01-01",
            periods=n,
            freq="MS",
        )
    })

    # Simulated advertising spending.
    for i, channel in enumerate(CHANNELS):

        df[channel] = (
            1e6 * (i + 1)
            + rng.uniform(0, 1e6, n)
        )

    # Simulated external controls.
    for column in CONTROLS:

        df[column] = rng.normal(
            size=n
        )

    # Simulated positive sales revenue.
    df[TARGET] = (
        5e7
        + 4 * df[CHANNELS[0]]
        + 2 * df[CHANNELS[1]]
        + rng.normal(0, 2e6, n)
    )

    return df


def test_training_and_prediction(sample_data):
    """Check model fitting and revenue prediction."""

    model = MMMModel(
        search_iterations=1
    )

    model.fit(
        sample_data.iloc[:30]
    )

    prediction = model.predict(
        sample_data.iloc[30:],
        continue_history=True,
    )

    assert isinstance(
        prediction,
        np.ndarray,
    )

    assert len(prediction) == 6

    assert np.isfinite(
        prediction
    ).all()

    # Channel coefficients must not be negative.
    assert (
        model.coef_[:len(CHANNELS)]
        >= -1e-8
    ).all()

    # Three expanding-window CV folds.
    assert len(
        model.cv_results_
    ) == 3


def test_response_curve(sample_data):
    """Check M5/M6 response curve interface."""

    model = MMMModel(
        search_iterations=1
    )

    model.fit(
        sample_data.iloc[:30]
    )

    grid = np.array([
        0,
        1e6,
        2e6,
    ])

    curve = model.response_curve(
        CHANNELS[0],
        grid,
    )

    assert isinstance(
        curve,
        np.ndarray,
    )

    assert curve.shape == grid.shape

    # No media spend gives no incremental media revenue.
    assert curve[0] == pytest.approx(0)

    # Response cannot decrease as spend increases.
    assert np.all(
        np.diff(curve) >= -1e-8
    )


def test_metrics():
    """Check evaluation metrics."""

    result = metrics(
        [100, 200],
        [100, 220],
    )

    assert result[
        "mape_pct"
    ] == pytest.approx(5)

    assert result[
        "nrmse_pct"
    ] > 0


def test_contributions_match_predictions(sample_data):
    """Contributions must reconstruct the prediction."""

    model = MMMModel(
        search_iterations=1
    )

    model.fit(
        sample_data.iloc[:30]
    )

    data = sample_data.iloc[:30]

    contributions = model.contributions(
        data
    )

    component_columns = [
        col
        for col in contributions.columns
        if col.endswith("_contribution_lkr")
    ]

    reconstructed = (
        contributions["baseline_lkr"]
        + contributions[component_columns].sum(axis=1)
    )

    np.testing.assert_allclose(
        reconstructed.to_numpy(),
        model.predict(data),
        rtol=1e-6,
    )
