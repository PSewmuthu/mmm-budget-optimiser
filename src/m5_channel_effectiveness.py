
"""M5: Channel Effectiveness and ROI Analysis.

Run from project root:
    python -m src.m5_channel_effectiveness

Uses the M3 deployed model without retraining it.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CHANNELS, CLEAN_CSV, MODEL_DIR
from src.mmm import MMMModel


TABLE_DIR = Path("outputs/tables/m5")
FIGURE_DIR = Path("outputs/figures/m5")


def safe_ratio(numerator, denominator):
    """Return NaN when the denominator is zero."""
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def analyse_channels(df, model):
    """Calculate historical model-attributed channel effectiveness."""

    contributions = model.contributions(df.reset_index(drop=True))
    monthly = pd.DataFrame({
        "month": df["month"].to_numpy()
    })

    rows = []

    for channel in CHANNELS:
        spend = df[channel].to_numpy(dtype=float)
        revenue = contributions[
            f"{channel}_contribution_lkr"
        ].to_numpy(dtype=float)

        total_spend = float(spend.sum())
        attributed_revenue = float(revenue.sum())

        roas = safe_ratio(attributed_revenue, total_spend)
        roi = (roas - 1) * 100 if np.isfinite(roas) else np.nan

        monthly[f"{channel}_spend_lkr"] = spend
        monthly[f"{channel}_contribution_lkr"] = revenue

        rows.append({
            "channel": channel,
            "total_spend_lkr": total_spend,
            "attributed_revenue_lkr": attributed_revenue,
            "attributed_roas": roas,
            "attributed_roi_pct": roi,
        })

    summary = pd.DataFrame(rows)

    total_contribution = summary[
        "attributed_revenue_lkr"
    ].sum()

    summary["contribution_share_pct"] = (
        summary["attributed_revenue_lkr"]
        / total_contribution * 100
        if total_contribution > 0 else np.nan
    )

    summary = summary.sort_values(
        "attributed_roas",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    summary.insert(
        0, "effectiveness_rank",
        np.arange(1, len(summary) + 1)
    )

    return summary, monthly


def analyse_response_curves(df, model):
    """Calculate steady-state response and marginal ROAS."""

    all_curves = []
    marginal_rows = []

    for channel in CHANNELS:
        average_spend = float(df[channel].mean())

        if average_spend <= 0:
            continue

        grid = np.linspace(
            0, average_spend * 2, 101
        )

        response = np.asarray(
            model.response_curve(channel, grid),
            dtype=float
        )

        marginal_roas = np.gradient(response, grid)

        all_curves.append(pd.DataFrame({
            "channel": channel,
            "monthly_spend_lkr": grid,
            "steady_state_revenue_lkr": response,
            "marginal_roas": marginal_roas,
        }))

        current_marginal_roas = float(
            np.interp(
                average_spend,
                grid,
                marginal_roas
            )
        )

        marginal_rows.append({
            "channel": channel,
            "average_monthly_spend_lkr": average_spend,
            "marginal_roas_at_average_spend": current_marginal_roas,
        })

    if not all_curves:
        raise ValueError("No channels with positive spend")

    return (
        pd.concat(all_curves, ignore_index=True),
        pd.DataFrame(marginal_rows)
    )


def save_charts(summary, curves):
    """Generate presentation-ready charts."""

    plt.figure(figsize=(10, 6))
    plt.barh(
        summary["channel"],
        summary["attributed_roas"]
    )
    plt.axvline(1, color="gray", linestyle="--")
    plt.xlabel("Model-attributed ROAS (x)")
    plt.title("M5: Channel Effectiveness Ranking")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "channel_roas_ranking.png",
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(
        summary["channel"],
        summary["attributed_revenue_lkr"] / 1_000_000
    )
    plt.ylabel("Attributed Revenue (LKR millions)")
    plt.title("M5: Channel Revenue Contributions")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "channel_contributions.png",
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for channel, group in curves.groupby("channel"):
        plt.plot(
            group["monthly_spend_lkr"] / 1_000_000,
            group["steady_state_revenue_lkr"] / 1_000_000,
            label=channel
        )

    plt.xlabel("Monthly Spend (LKR millions)")
    plt.ylabel("Steady-state Revenue (LKR millions)")
    plt.title("M5: Channel Response Curves")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "channel_response_curves.png",
        dpi=200
    )
    plt.close()


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEAN_CSV, parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)

    model_path = Path(MODEL_DIR) / "mmm_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "M3 model missing. Ask M3 member for the saved model."
        )

    print("Loading M3 trained model...")
    model = joblib.load(model_path)

    if not isinstance(model, MMMModel) or not model.fitted_:
        raise ValueError("Expected a fitted M3 MMMModel")

    print("Calculating channel contributions...")
    summary, monthly = analyse_channels(df, model)

    print("Calculating marginal effectiveness...")
    curves, marginal = analyse_response_curves(df, model)

    summary = summary.merge(
        marginal,
        on="channel",
        how="left"
    )

    summary.to_csv(
        TABLE_DIR / "channel_effectiveness_summary.csv",
        index=False
    )

    monthly.to_csv(
        TABLE_DIR / "monthly_channel_contributions.csv",
        index=False
    )

    curves.to_csv(
        TABLE_DIR / "channel_response_curves.csv",
        index=False
    )

    print("Generating graphs...")
    save_charts(summary, curves)

    holdout_path = Path(MODEL_DIR) / "holdout_metrics.json"
    holdout_metrics = {}

    if holdout_path.exists():
        with open(holdout_path, encoding="utf-8") as file:
            holdout_metrics = json.load(file)

    metadata = {
        "analysis": "M5 Channel Effectiveness and ROI",
        "model": "M3 refitted primary MMM",
        "months_analysed": len(df),
        "holdout_metrics": holdout_metrics,
        "limitations": [
            "Attribution is model-based, not causal proof.",
            "ROI uses attributed revenue, not profit.",
            "Marginal ROAS is a steady-state model estimate.",
            "Historical ROI and marginal ROAS are not directly comparable.",
            "Low predictive accuracy reduces confidence in rankings."
        ]
    }

    with open(
        TABLE_DIR / "analysis_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nM5 CHANNEL EFFECTIVENESS RESULTS")
    print(summary.round(3).to_string(index=False))
    print("\nM5 analysis completed successfully.")


if __name__ == "__main__":
    main()
