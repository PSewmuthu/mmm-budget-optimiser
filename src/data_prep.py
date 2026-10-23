"""M1: data loading, validation, cleaning, feature engineering.

Contract: write data/processed/mmm_clean.csv
  - one row per month, `month` as datetime, snake_case columns as in the raw data
  - constant columns removed (listed in the README)
  - extra features: t (0..35), year, month_num, <channel>_share, log_revenue
"""
import pandas as pd

from src import config


def load_raw() -> pd.DataFrame:
    return pd.read_excel(config.RAW_XLSX, sheet_name="MMM_Data", parse_dates=["month"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("M1: implement validation + cleaning + features")


if __name__ == "__main__":
    out = clean(load_raw())
    out.to_csv(config.CLEAN_CSV, index=False)
    print(f"wrote {config.CLEAN_CSV} {out.shape}")
