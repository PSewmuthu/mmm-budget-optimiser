"""Activates once M1 merges data/processed/mmm_clean.csv."""
import pandas as pd
import pytest

from src import config

pytestmark = pytest.mark.skipif(not config.CLEAN_CSV.exists(), reason="M1 not merged yet")


def test_clean_contract():
    df = pd.read_csv(config.CLEAN_CSV, parse_dates=["month"])
    assert len(df) == 36 and df["month"].is_monotonic_increasing
    assert not df.isna().any().any()
    for c in config.CHANNELS + config.CONTROLS + [config.TARGET]:
        assert c in df.columns
    for c in config.CONSTANT_COLUMNS:
        assert c not in df.columns
