
"""M2 analysis: adstock, saturation, and time-delay relationships."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CLEAN_CSV, CHANNELS, TARGET, FIG_DIR, OUT
from src.transforms import adstock, hill


def main():
    df = pd.read_csv(CLEAN_CSV)
    df = df.sort_values("month").reset_index(drop=True)

    required = ["month", TARGET, *CHANNELS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [TARGET, *CHANNELS]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[required[1:]].isna().any().any():
        raise ValueError("Dataset contains missing values in analysis columns.")

    out_dir = OUT / "tables" / "m2"
    fig_dir = FIG_DIR / "m2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Adstock: compare carry-over assumptions for each channel.
    adstock_rows = []
    decay_values = [0.0, 0.3, 0.5, 0.7]

    for channel in CHANNELS:
        for decay in decay_values:
            transformed = adstock(df[channel].to_numpy(), decay)
            adstock_rows.append({
                "channel": channel,
                "decay": decay,
                "original_mean_spend": df[channel].mean(),
                "adstock_mean": transformed.mean(),
                "adstock_max": transformed.max(),
            })

    pd.DataFrame(adstock_rows).to_csv(
        out_dir / "adstock_summary.csv", index=False
    )

    # 2. Saturation: explore Hill curves using each channel's median
    # spend as the half-saturation point. This is illustrative, not fitted.
    saturation_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for channel, ax in zip(CHANNELS, axes.flat):
        spend = df[channel].to_numpy()
        half_sat = float(np.median(spend))

        if half_sat <= 0:
            continue

        grid = np.linspace(0, max(float(spend.max()), half_sat) * 1.2, 100)
        response = hill(grid, half_sat=half_sat, slope=1.5)

        for value, transformed in zip(grid, response):
            saturation_rows.append({
                "channel": channel,
                "spend_lkr": value,
                "half_saturation_lkr": half_sat,
                "hill_slope": 1.5,
                "relative_response": transformed,
            })

        ax.plot(grid, response)
        ax.axvline(half_sat, linestyle="--", alpha=0.7)
        ax.set_title(channel.replace("_spend_lkr", "").replace("_", " "))
        ax.set_xlabel("Monthly spend (LKR)")
        ax.set_ylabel("Relative response")
        ax.grid(alpha=0.25)

    fig.suptitle("M2: Illustrative Hill Saturation Curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "saturation_curves.png", dpi=160)
    plt.close(fig)

    pd.DataFrame(saturation_rows).to_csv(
        out_dir / "saturation_curves.csv", index=False
    )

    # 3. Time-delay: compare current and previous-month spend with revenue.
    # Lag 0 means same month; lag 1 means spend one month earlier, etc.
    lag_rows = []

    for channel in CHANNELS:
        for lag in range(4):
            shifted_spend = df[channel].shift(lag)
            valid = pd.DataFrame({
                "spend": shifted_spend,
                "revenue": df[TARGET],
            }).dropna()

            corr = (
                valid["spend"].corr(valid["revenue"])
                if len(valid) >= 3 else np.nan
            )

            lag_rows.append({
                "channel": channel,
                "lag_months": lag,
                "observations": len(valid),
                "spend_revenue_correlation": corr,
            })

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(out_dir / "time_delay_correlations.csv", index=False)

    pivot = lag_df.pivot(
        index="channel",
        columns="lag_months",
        values="spend_revenue_correlation",
    )
    pivot.columns = [f"lag_{lag}_month" for lag in pivot.columns]
    pivot.to_csv(out_dir / "time_delay_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    for channel in CHANNELS:
        subset = lag_df[lag_df["channel"] == channel]
        ax.plot(
            subset["lag_months"],
            subset["spend_revenue_correlation"],
            marker="o",
            label=channel.replace("_spend_lkr", "").replace("_", " "),
        )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Spend lead time (months before revenue)")
    ax.set_ylabel("Pearson correlation with revenue")
    ax.set_title("M2: Exploratory Spend-to-Revenue Time Delays")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "time_delay_correlations.png", dpi=160)
    plt.close(fig)

    print("M2 analysis completed.")
    print(f"Tables saved to: {out_dir}")
    print(f"Figures saved to: {fig_dir}")
    print("Note: correlations are exploratory, not causal estimates.")


if __name__ == "__main__":
    main()
