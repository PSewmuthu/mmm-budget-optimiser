
import numpy as np
import pandas as pd
import pytest

from src import m4_benchmark as m4


@pytest.fixture
def model_data():
    """Prepare M4 training and testing data."""
    df = m4.load_dataset()
    train_df, test_df = m4.prepare_model_data(df)
    return df, train_df, test_df


def test_dataset_loading():
    """Check that the required dataset is available."""

    df = m4.load_dataset()

    assert not df.empty
    assert len(df) == 36

    required_columns = (
        m4.CHANNEL_COLUMNS
        + m4.CONTROL_COLUMNS
        + ["sales_revenue_lkr"]
    )

    assert all(col in df.columns for col in required_columns)


def test_time_based_split(model_data):
    """Verify chronological training and testing split."""

    df, train_df, test_df = model_data

    assert len(train_df) == 30
    assert len(test_df) == 6

    assert train_df["month"].max() < test_df["month"].min()


def test_benchmark_model(model_data):
    """Check benchmark coefficients and HAC covariance."""

    _, train_df, _ = model_data

    results = m4.fit_benchmark_model(train_df)

    assert results.nobs == 30
    assert results.cov_type == "HAC"
    assert np.isfinite(results.params.to_numpy()).all()

    for channel in m4.CHANNEL_COLUMNS:
        assert f"log_{channel}" in results.params.index


def test_external_factor_analysis(model_data, tmp_path, monkeypatch):
    """Check all external factors and confidence intervals."""

    _, train_df, _ = model_data

    monkeypatch.setattr(m4, "PROJECT_ROOT", tmp_path)

    results = m4.fit_benchmark_model(train_df)
    factors = m4.analyse_external_factors(results)

    assert len(factors) == 8
    assert factors["ci_lower"].le(factors["ci_upper"]).all()

    assert (
        tmp_path / "outputs" / "tables" / "m4"
        / "external_factor_effects.csv"
    ).exists()


def test_ablation_analysis(model_data, tmp_path, monkeypatch):
    """Check both ablation models and their metrics."""

    _, train_df, test_df = model_data

    monkeypatch.setattr(m4, "PROJECT_ROOT", tmp_path)

    results = m4.fit_benchmark_model(train_df)

    coefficients, metrics = m4.run_ablation_analysis(
        train_df,
        test_df,
        results
    )

    assert len(coefficients) == 6
    assert len(metrics) == 2

    assert set(metrics["model"]) == {
        "with_controls",
        "without_controls",
    }

    assert np.isfinite(
        metrics[["holdout_mape_pct", "holdout_nrmse", "holdout_r2"]]
        .to_numpy()
    ).all()


def test_distributed_lag_model(model_data, tmp_path, monkeypatch):
    """Check distributed-lag model results."""

    df, _, _ = model_data

    monkeypatch.setattr(m4, "PROJECT_ROOT", tmp_path)

    results = m4.fit_distributed_lag_model(df)

    assert results.nobs == 28
    assert results.cov_type == "HAC"

    for lag in range(3):
        assert f"log_spend_lag{lag}" in results.params.index

    assert (
        tmp_path / "outputs" / "tables" / "m4"
        / "distributed_lag_results.csv"
    ).exists()


def test_m3_m4_comparison(tmp_path, monkeypatch):
    """
    Verify M3 and M4 comparison outputs.
    """

    from src import m4_m3_comparison as comparison

    monkeypatch.setattr(
        comparison,
        "OUTPUT_DIR",
        tmp_path
    )

    model_metrics, predictions, channels = (
        comparison.compare_models()
    )

    # Two models must be compared
    assert len(model_metrics) == 2

    assert set(model_metrics["model"]) == {
        "M3_Primary_MMM",
        "M4_Statsmodels",
    }

    # Six holdout months
    assert len(predictions) == 6

    # Six marketing channels
    assert len(channels) == 6

    # Predictions must be finite
    assert np.isfinite(
        predictions[
            [
                "actual_revenue_lkr",
                "m3_predicted_revenue_lkr",
                "m4_predicted_revenue_lkr",
            ]
        ].to_numpy()
    ).all()

    # All comparison CSV files exist
    for filename in [
        "m3_m4_model_comparison.csv",
        "m3_m4_predictions.csv",
        "m3_m4_channel_comparison.csv",
    ]:
        assert (tmp_path / filename).exists()
