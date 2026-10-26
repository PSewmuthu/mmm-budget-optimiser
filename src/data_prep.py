"""M1: data loading, validation, cleaning, feature engineering.

Contract: write data/processed/mmm_clean.csv
  - one row per month, `month` as datetime before CSV export
  - snake_case columns as in the raw data
  - constant columns removed and preserved in metadata
  - extra features: t (0..35), year, month_num,
    <channel>_share, log_revenue

Year-on-year growth is calculated separately in
notebooks/01_m1_data_eda.ipynb and exported to
outputs/tables/m1/revenue_yoy_growth_train.csv.

Its first 12 months are undefined. Keeping this output separate
preserves the clean CSV contract requiring no missing values.
"""

import json

import numpy as np
import pandas as pd

from src import config


def load_raw() -> pd.DataFrame:
    """Load the original monthly dataset without modifying it."""
    return pd.read_excel(
        config.RAW_XLSX,
        sheet_name="MMM_Data",
    )


def load_dictionary() -> pd.DataFrame:
    """Load the column descriptions and units."""
    return pd.read_excel(
        config.RAW_XLSX,
        sheet_name="Data_Dictionary",
    )


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate raw data without modifying the input DataFrame."""
    required = (
        ["month", config.TARGET, config.TOTAL_SPEND]
        + config.CHANNELS
        + config.CONTROLS
        + config.LEAKAGE_COLUMNS
        + config.CONSTANT_COLUMNS
    )

    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if len(df) != 36:
        raise ValueError(f"Expected 36 rows, found {len(df)}")

    if df[required].isna().any().any():
        raise ValueError("Raw data contains missing values")

    dates = pd.to_datetime(
        df["month"],
        format="%Y-%m-%d",
        errors="raise",
    )

    if dates.isna().any():
        raise ValueError("Missing or invalid dates")

    if dates.dt.to_period("M").duplicated().any():
        raise ValueError("Duplicate months found")

    ordered_dates = pd.DatetimeIndex(dates.sort_values())
    expected_dates = pd.date_range(
        start=ordered_dates.min(),
        periods=36,
        freq="MS",
    )

    if not ordered_dates.equals(expected_dates):
        raise ValueError(
            "Expected 36 consecutive month-start dates"
        )

    numeric_columns = (
        config.CHANNELS
        + config.CONTROLS
        + config.LEAKAGE_COLUMNS
        + [config.TARGET, config.TOTAL_SPEND]
    )

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be numeric")

        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains non-finite values"
            )

    spend_columns = config.CHANNELS + [config.TOTAL_SPEND]
    if (df[spend_columns] < 0).any().any():
        raise ValueError("Negative advertising spend found")

    if (df[config.TARGET] <= 0).any():
        raise ValueError(
            "Revenue must be positive for log-revenue features"
        )

    channel_sum = df[config.CHANNELS].sum(axis=1)
    spend_difference = (
        df[config.TOTAL_SPEND] - channel_sum
    ).abs()

    if (spend_difference > 5).any():
        raise ValueError(
            "Total spend differs from channel sum "
            "by more than LKR 5"
        )

    for column in config.CONSTANT_COLUMNS:
        if df[column].nunique(dropna=False) != 1:
            raise ValueError(f"{column} is not constant")

    if not df["data_type"].eq("SYNTHETIC").all():
        raise ValueError("Expected SYNTHETIC data flag")

    return {
        "rows": len(df),
        "start_month": ordered_dates.min().strftime("%Y-%m-%d"),
        "end_month": ordered_dates.max().strftime("%Y-%m-%d"),
        "missing_values": int(
            df[required].isna().sum().sum()
        ),
        "max_spend_difference_lkr": float(
            spend_difference.max()
        ),
        "data_type": df["data_type"].iloc[0],
        "status": "PASS",
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort and create the agreed cleaned dataset."""
    validate_raw(df)

    # Work on a copy to preserve the original DataFrame.
    cleaned = df.copy()

    # Convert dates and sort chronologically.
    cleaned["month"] = pd.to_datetime(
        cleaned["month"],
        format="%Y-%m-%d",
        errors="raise",
    )
    cleaned = cleaned.sort_values("month").reset_index(drop=True)

    # Constant values are saved separately in metadata.
    cleaned = cleaned.drop(columns=config.CONSTANT_COLUMNS)

    # Preserve the feature names from the starter contract.
    cleaned["t"] = np.arange(len(cleaned))
    cleaned["year"] = cleaned["month"].dt.year
    cleaned["month_num"] = cleaned["month"].dt.month

    # Use the channel sum so the six shares sum to 1.
    channel_total = cleaned[config.CHANNELS].sum(axis=1)

    if (channel_total <= 0).any():
        raise ValueError(
            "Spend shares require positive total channel spend"
        )

    for channel in config.CHANNELS:
        cleaned[f"{channel}_share"] = (
            cleaned[channel] / channel_total
        )

    # Target transformation, not a predictor of revenue.
    cleaned["log_revenue"] = np.log(cleaned[config.TARGET])

    return cleaned


def main() -> None:
    """Generate the cleaned CSV and supporting metadata."""
    raw = load_raw()
    dictionary = load_dictionary()

    validation = validate_raw(raw)
    cleaned = clean(raw)

    config.CLEAN_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        config.CLEAN_CSV,
        index=False,
        date_format="%Y-%m-%d",
    )

    metadata = {
        "source_file": config.RAW_XLSX.relative_to(
            config.ROOT
        ).as_posix(),
        "source_sheet": "MMM_Data",
        "dictionary_sheet": "Data_Dictionary",
        "dictionary_rows": len(dictionary),
        "constants": {
            column: raw[column].iloc[0]
            for column in config.CONSTANT_COLUMNS
        },
        "validation": validation,
        "spend_share_denominator": "sum of six channel spends",
        "log_revenue_definition": (
            "natural logarithm of sales_revenue_lkr"
        ),
                "notes": [
            "Synthetic data; findings are illustrative.",
            "Original spend and revenue values are preserved.",
            "No rows were removed or imputed.",
            "Spend discrepancies within LKR 5 are preserved.",
            (
                "Exploratory analysis is implemented separately in "
                "notebooks/01_m1_data_eda.ipynb; run that notebook "
                "to generate EDA tables and figures."
            ),
            (
                "EDA uses January 2023 to June 2025 (30 months). "
                "July to December 2025 is reserved for holdout "
                "evaluation. The cleaned CSV retains all 36 months."
            ),
            (
                "Outlier thresholds and review decisions are "
                "recorded separately in outputs/tables/m1/"
                "outlier_thresholds_train.csv and "
                "outputs/tables/m1/outlier_flags_train.csv."
            ),
            (
                "Year-on-year revenue growth is exported separately "
                "to outputs/tables/m1/revenue_yoy_growth_train.csv. "
                "The first 12 growth values are undefined and "
                "are not replaced with zero."
            ),
            (
                "Log revenue and year-on-year revenue growth must "
                "not be used as contemporaneous revenue predictors."
            ),
        ],
    }

    metadata_path = (
        config.CLEAN_CSV.parent / "mmm_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Loaded MMM_Data: {raw.shape}")
    print(f"Loaded Data_Dictionary: {dictionary.shape}")
    print(f"Cleaned dataset: {cleaned.shape}")
    print(
        f"Saved: {config.CLEAN_CSV.relative_to(config.ROOT)}"
    )
    print(f"Saved: {metadata_path.relative_to(config.ROOT)}")
    print(f"Validation: {validation['status']}")


if __name__ == "__main__":
    main()