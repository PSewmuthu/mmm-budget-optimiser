import pandas as pd

from src import config


def test_raw_columns_match_config():
    df = pd.read_excel(config.RAW_XLSX, sheet_name="MMM_Data")
    expected = config.CHANNELS + config.CONTROLS + config.LEAKAGE_COLUMNS + config.CONSTANT_COLUMNS
    expected += [config.TARGET, config.TOTAL_SPEND, "month"]
    assert set(expected) == set(df.columns)


def test_total_spend_matches_channel_sum():
    df = pd.read_excel(config.RAW_XLSX, sheet_name="MMM_Data")
    assert (df[config.CHANNELS].sum(axis=1) - df[config.TOTAL_SPEND]).abs().max() <= 5
