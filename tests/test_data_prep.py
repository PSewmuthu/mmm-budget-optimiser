"""M1 tests for data validation and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_prep import clean, load_raw, validate_raw


@pytest.fixture
def raw():
    """Provide a fresh dataset for each test."""
    return load_raw()


def test_clean_sorts_dates_and_preserves_input(raw):
    shuffled = raw.sample(
        frac=1,
        random_state=config.SEED,
    ).reset_index(drop=True)
    original = shuffled.copy(deep=True)

    result = clean(shuffled)

    pd.testing.assert_frame_equal(shuffled, original)
    assert result["month"].is_monotonic_increasing
    assert result["month"].is_unique
    assert len(result) == 36
    assert result["t"].tolist() == list(range(36))

    assert result["year"].tolist() == (
        [2023] * 12 + [2024] * 12 + [2025] * 12
    )
    assert result["month_num"].tolist() == list(range(1, 13)) * 3


def test_clean_preserves_original_measurements(raw):
    result = clean(raw)

    expected = raw.copy()
    expected["month"] = pd.to_datetime(expected["month"])
    expected = expected.sort_values("month").reset_index(drop=True)
    expected = expected.drop(columns=config.CONSTANT_COLUMNS)

    pd.testing.assert_frame_equal(
        result[expected.columns],
        expected,
    )

    for column in config.CONSTANT_COLUMNS:
        assert column not in result.columns


def test_known_spend_shares_and_log_revenue(raw):
    # Set known values for the earliest month.
    first_index = pd.to_datetime(raw["month"]).idxmin()

    for channel, spend in zip(
        config.CHANNELS,
        [100, 200, 300, 400, 500, 600],
    ):
        raw.loc[first_index, channel] = spend

    # An allowed LKR 2 discrepancy must not affect the denominator.
    raw.loc[first_index, config.TOTAL_SPEND] = 2102
    raw.loc[first_index, config.TARGET] = 1

    result = clean(raw)
    first = result.iloc[0]

    expected_shares = [1 / 21, 2 / 21, 3 / 21, 4 / 21, 5 / 21, 6 / 21]

    for channel, expected in zip(config.CHANNELS, expected_shares):
        assert first[f"{channel}_share"] == pytest.approx(expected)

    share_columns = [
        f"{channel}_share" for channel in config.CHANNELS
    ]
    np.testing.assert_allclose(
        result[share_columns].sum(axis=1),
        1.0,
    )

    assert first["log_revenue"] == pytest.approx(0.0)
    assert first[config.TOTAL_SPEND] == 2102


def test_missing_required_column_is_rejected(raw):
    invalid = raw.drop(columns=[config.CHANNELS[0]])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_raw(invalid)


def test_missing_value_is_rejected(raw):
    raw.loc[0, config.CONTROLS[0]] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_raw(raw)


def test_duplicate_month_is_rejected(raw):
    raw.loc[1, "month"] = raw.loc[0, "month"]

    with pytest.raises(ValueError, match="Duplicate months"):
        validate_raw(raw)


def test_gap_in_months_is_rejected(raw):
    raw["month"] = pd.to_datetime(raw["month"])
    last_index = raw["month"].idxmax()
    raw.loc[last_index, "month"] += pd.DateOffset(months=1)

    with pytest.raises(ValueError, match="consecutive"):
        validate_raw(raw)


def test_negative_spend_is_rejected(raw):
    raw.loc[0, config.CHANNELS[0]] = -1

    with pytest.raises(ValueError, match="Negative"):
        validate_raw(raw)


@pytest.mark.parametrize("difference", [-5, 5])
def test_spend_difference_at_tolerance_is_allowed(raw, difference):
    channel_sum = raw.loc[0, config.CHANNELS].sum()
    raw.loc[0, config.TOTAL_SPEND] = channel_sum + difference

    result = validate_raw(raw)

    assert result["status"] == "PASS"


@pytest.mark.parametrize("difference", [-6, 6])
def test_spend_difference_above_tolerance_is_rejected(raw, difference):
    channel_sum = raw.loc[0, config.CHANNELS].sum()
    raw.loc[0, config.TOTAL_SPEND] = channel_sum + difference

    with pytest.raises(ValueError, match="more than LKR 5"):
        validate_raw(raw)


def test_zero_revenue_is_rejected(raw):
    raw.loc[0, config.TARGET] = 0

    with pytest.raises(ValueError, match="Revenue must be positive"):
        validate_raw(raw)


def test_zero_total_channel_spend_is_rejected(raw):
    raw.loc[0, config.CHANNELS] = 0
    raw.loc[0, config.TOTAL_SPEND] = 0

    with pytest.raises(ValueError, match="positive total channel spend"):
        clean(raw)