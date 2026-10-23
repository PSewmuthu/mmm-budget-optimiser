"""Shared configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "Dataset.xlsx"
CLEAN_CSV = ROOT / "data" / "processed" / "mmm_clean.csv"
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
MODEL_DIR = OUT / "model"
POWERBI_DIR = OUT / "powerbi"

SEED = 42

TARGET = "sales_revenue_lkr"

# Channel spend columns (LKR per month), as named in the raw data.
CHANNELS = [
    "tv_spend_lkr",
    "facebook_instagram_spend_lkr",
    "google_search_spend_lkr",
    "youtube_spend_lkr",
    "radio_spend_lkr",
    "outdoor_spend_lkr",
]
TOTAL_SPEND = "total_ad_spend_lkr"

# External / control variables allowed as model inputs.
CONTROLS = [
    "average_price_index",
    "promotion_discount_pct",
    "competitor_pressure_index_0_100",
    "economic_sentiment_index_0_100",
    "lkr_usd_exchange_rate",
    "inflation_proxy_index",
    "seasonal_demand_index",
    "public_holiday_count",
]

# Outcomes of spend, NOT drivers: never use as predictors (leakage).
LEAKAGE_COLUMNS = ["impressions", "clicks", "conversions", "units_sold"]

# Constant columns moved to metadata during cleaning.
CONSTANT_COLUMNS = ["country", "brand_category", "region_scope", "data_type"]

# Default optimiser bounds: fraction of each channel's historical mean spend.
BOUND_LOW, BOUND_HIGH = 0.5, 1.5
