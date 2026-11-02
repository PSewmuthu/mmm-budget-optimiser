
"""Unit tests for M5 Channel Effectiveness and ROI."""

import numpy as np
import pandas as pd
import pytest

from src.m5_channel_effectiveness import (
    safe_ratio,
    analyse_channels,
    analyse_response_curves,
)
from src.config import CHANNELS


def test_safe_ratio():
    assert safe_ratio(200, 100) == 2.0
    assert safe_ratio(0, 100) == 0.0
    assert np.isnan(safe_ratio(100, 0))


class FakeMMM:
    """Simple deterministic model for testing M5."""

    def contributions(self, df):
        result = pd.DataFrame(index=df.index)

        for channel in CHANNELS:
            result[f"{channel}_contribution_lkr"] = (
                df[channel].to_numpy(dtype=float) * 2.0
            )

        return result

    def response_curve(self, channel, grid):
        grid = np.asarray(grid, dtype=float)
        return 3.0 * grid / (1.0 + grid / 1000.0)


@pytest.fixture
def sample_data():
    data = {
        "month": pd.date_range(
            "2024-01-01", periods=3, freq="MS"
        )
    }

    for channel in CHANNELS:
        data[channel] = [100.0, 200.0, 300.0]

    return pd.DataFrame(data)


def test_channel_summary(sample_data):
    summary, monthly = analyse_channels(
        sample_data, FakeMMM()
    )

    assert len(summary) == len(CHANNELS)
    assert len(monthly) == len(sample_data)

    assert np.allclose(
        summary["attributed_roas"], 2.0
    )

    assert np.allclose(
        summary["attributed_roi_pct"], 100.0
    )

    assert np.isclose(
        summary["contribution_share_pct"].sum(),
        100.0
    )

    assert sorted(summary["effectiveness_rank"]) == list(
        range(1, len(CHANNELS) + 1)
    )


def test_response_curves(sample_data):
    curves, marginal = analyse_response_curves(
        sample_data, FakeMMM()
    )

    assert len(curves) == len(CHANNELS) * 101
    assert len(marginal) == len(CHANNELS)

    assert np.isfinite(
        curves["steady_state_revenue_lkr"]
    ).all()

    assert (
        curves["marginal_roas"] >= 0
    ).all()

    assert (
        marginal["marginal_roas_at_average_spend"] > 0
    ).all()


def test_zero_spend_channel(sample_data):
    channel = CHANNELS[0]
    sample_data[channel] = 0.0

    summary, _ = analyse_channels(
        sample_data, FakeMMM()
    )

    row = summary.loc[
        summary["channel"] == channel
    ].iloc[0]

    assert np.isnan(row["attributed_roas"])
    assert np.isnan(row["attributed_roi_pct"])
