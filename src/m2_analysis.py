
"""M2 analysis: adstock, saturation, adjusted lags, and parameter candidates."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import CLEAN_CSV, CHANNELS, TARGET, FIG_DIR, OUT
from src.transforms import adstock, hill, negative_exponential

DECAY_VALUES = [0.0, 0.3, 0.5, 0.7]
HILL_SLOPES = [1.0, 1.5, 2.0]
MAX_LAG = 4
N_BINS = 4


def channel_label(channel):
    return channel.replace("_spend_lkr", "").replace("_", " ").title()


def correlation(x, y):
    paired = pd.DataFrame({
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(paired) < 3:
        return np.nan
    if paired["x"].nunique() < 2 or paired["y"].nunique() < 2:
        return np.nan
    return float(paired["x"].corr(paired["y"]))


def residualize(values, controls):
    """Remove linear effects of the available seasonality/promotion controls."""
    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)

    design = np.column_stack([np.ones(len(values)), controls])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def fit_saturating_curve(spend, response, model):
    """
    Grid-search a bounded saturating curve.

    Fits response = intercept + amplitude * saturation(spend).
    The intercept and amplitude are estimated by least squares for each
    candidate shape. This is exploratory curve fitting, not causal modelling.
    """
    spend = np.asarray(spend, dtype=float)
    response = np.asarray(response, dtype=float)

    valid = np.isfinite(spend) & np.isfinite(response)
    spend = spend[valid]
    response = response[valid]

    if len(spend) < 8 or np.unique(spend).size < 4:
        return {
            "fit_status": "insufficient_data",
            "intercept": np.nan,
            "amplitude": np.nan,
            "half_saturation_lkr": np.nan,
            "slope": np.nan,
            "scale_lkr": np.nan,
            "r_squared": np.nan,
        }

    positive = spend[spend > 0]
    if len(positive) < 4 or np.std(response) == 0:
        return {
            "fit_status": "insufficient_positive_spend_or_variation",
            "intercept": np.nan,
            "amplitude": np.nan,
            "half_saturation_lkr": np.nan,
            "slope": np.nan,
            "scale_lkr": np.nan,
            "r_squared": np.nan,
        }

    # Candidate parameter ranges are intentionally bounded for M3/M4.
    half_grid = np.unique(np.quantile(positive, [0.25, 0.5, 0.75]))
    scale_grid = half_grid.copy()
    slope_grid = np.array(HILL_SLOPES, dtype=float)

    best = None
    total_ss = float(np.sum((response - response.mean()) ** 2))

    if total_ss <= 0:
        total_ss = np.nan

    if model == "hill":
        candidates = [
            (float(half), float(slope))
            for half in half_grid
            for slope in slope_grid
        ]
    elif model == "negative_exponential":
        candidates = [(float(scale), np.nan) for scale in scale_grid]
    else:
        raise ValueError(f"Unknown saturation model: {model}")

    for first, slope in candidates:
        if model == "hill":
            saturation = hill(spend, half_sat=first, slope=slope)
        else:
            saturation = negative_exponential(spend, scale=first)

        design = np.column_stack([np.ones(len(spend)), saturation])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        prediction = design @ coefficients
        sse = float(np.sum((response - prediction) ** 2))

        if best is None or sse < best["sse"]:
            r_squared = (
                1.0 - sse / total_ss if np.isfinite(total_ss) else np.nan
            )
            best = {
                "sse": sse,
                "intercept": float(coefficients[0]),
                "amplitude": float(coefficients[1]),
                "half_saturation_lkr": first if model == "hill" else np.nan,
                "slope": slope if model == "hill" else np.nan,
                "scale_lkr": first if model == "negative_exponential" else np.nan,
                "r_squared": float(r_squared),
            }

    best.pop("sse")
    best["fit_status"] = "ok"
    return best


def main():
    df = pd.read_csv(CLEAN_CSV)

    required = ["month", TARGET, *CHANNELS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["month"] = pd.to_datetime(df["month"], errors="raise")
    df = df.sort_values("month").reset_index(drop=True)

    for column in [TARGET, *CHANNELS]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[required[1:]].isna().any().any():
        raise ValueError("Dataset contains missing values in required analysis columns.")

    if not np.isfinite(df[[TARGET, *CHANNELS]].to_numpy(dtype=float)).all():
        raise ValueError("Analysis columns must contain finite values.")

    if (df[list(CHANNELS)] < 0).any().any():
        raise ValueError("Marketing spend values must be non-negative.")

    # The plan specifically calls for removing seasonality and promotion effects.
    control_columns = [
        column for column in ["seasonal_demand_index", "promotion_discount_pct"]
        if column in df.columns
    ]
    if len(control_columns) < 2:
        raise ValueError(
            "Expected seasonal_demand_index and promotion_discount_pct "
            "for adjusted time-delay analysis."
        )

    controls_df = df[control_columns].apply(pd.to_numeric, errors="raise")
    if controls_df.isna().any().any():
        raise ValueError("Seasonality/promotion controls contain missing values.")
    controls = controls_df.to_numpy(dtype=float)

    out_dir = OUT / "tables" / "m2"
    fig_dir = FIG_DIR / "m2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    revenue = df[TARGET].to_numpy(dtype=float)
    adjusted_revenue = residualize(revenue, controls)

    # 1. Adstock transformation summary
    adstock_rows = []
    adstocked_data = {}

    for channel in CHANNELS:
        spend = df[channel].to_numpy(dtype=float)
        for decay in DECAY_VALUES:
            transformed = adstock(spend, decay)
            adstocked_data[(channel, decay)] = transformed
            adstock_rows.append({
                "channel": channel,
                "decay": decay,
                "original_mean_spend": float(np.mean(spend)),
                "adstock_mean": float(np.mean(transformed)),
                "adstock_max": float(np.max(transformed)),
                "original_total_spend": float(np.sum(spend)),
                "adstock_total": float(np.sum(transformed)),
            })

    pd.DataFrame(adstock_rows).to_csv(
        out_dir / "adstock_summary.csv", index=False
    )

    # 2. Hill saturation sensitivity curves
    saturation_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for channel, ax in zip(CHANNELS, axes.flat):
        spend = df[channel].to_numpy(dtype=float)
        positive = spend[spend > 0]

        if len(positive) == 0:
            ax.set_title(channel_label(channel))
            ax.text(0.5, 0.5, "No positive spend", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        half_sat = float(np.median(positive))
        grid = np.linspace(0, max(float(spend.max()), half_sat) * 1.2, 150)

        for slope in HILL_SLOPES:
            response = hill(grid, half_sat=half_sat, slope=slope)
            ax.plot(grid, response, label=f"Hill slope = {slope}")

            for x_value, y_value in zip(grid, response):
                saturation_rows.append({
                    "channel": channel,
                    "spend_lkr": float(x_value),
                    "half_saturation_lkr": half_sat,
                    "hill_slope": slope,
                    "relative_response": float(y_value),
                })

        ax.axvline(half_sat, linestyle="--", alpha=0.7,
                   label="Half-saturation")
        ax.set_title(channel_label(channel))
        ax.set_xlabel("Monthly spend (LKR)")
        ax.set_ylabel("Relative response")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("M2: Hill Saturation Sensitivity by Channel")
    fig.tight_layout()
    fig.savefig(fig_dir / "saturation_curves.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(saturation_rows).to_csv(
        out_dir / "saturation_curves.csv", index=False
    )

    # 3. Adjusted cross-correlation at lags 0-4.
    # Residualize spend features against the same seasonality/promotion controls,
    # then align spend at t-lag with adjusted revenue at t.
    lag_rows = []
    feature_specs = [("raw", None)] + [
        ("adstock", decay) for decay in DECAY_VALUES if decay > 0
    ]

    for channel in CHANNELS:
        raw_spend = df[channel].to_numpy(dtype=float)

        for feature_type, decay in feature_specs:
            feature = (
                raw_spend if feature_type == "raw"
                else adstocked_data[(channel, decay)]
            )
            adjusted_feature = residualize(feature, controls)

            for lag in range(MAX_LAG + 1):
                x = pd.Series(adjusted_feature).shift(lag).to_numpy()
                y = adjusted_revenue.copy()
                valid = np.isfinite(x) & np.isfinite(y)
                corr = correlation(x[valid], y[valid])

                lag_rows.append({
                    "channel": channel,
                    "feature_type": feature_type,
                    "decay": decay,
                    "lag_months": lag,
                    "observations": int(valid.sum()),
                    "adjusted_spend_revenue_correlation": corr,
                    "adjustment_controls": ", ".join(control_columns),
                })

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(out_dir / "time_delay_correlations.csv", index=False)

    best_rows = []
    for (channel, feature_type, decay), group in lag_df.groupby(
        ["channel", "feature_type", "decay"], dropna=False
    ):
        valid_group = group.dropna(
            subset=["adjusted_spend_revenue_correlation"]
        )
        if valid_group.empty:
            continue

        best = valid_group.loc[
            valid_group["adjusted_spend_revenue_correlation"].abs().idxmax()
        ]
        best_rows.append({
            "channel": channel,
            "feature_type": feature_type,
            "decay": decay,
            "best_absolute_correlation_lag": int(best["lag_months"]),
            "adjusted_correlation_at_selected_lag": float(
                best["adjusted_spend_revenue_correlation"]
            ),
            "observations": int(best["observations"]),
        })

    pd.DataFrame(best_rows).to_csv(
        out_dir / "best_time_delay_by_channel.csv", index=False
    )

    raw_lags = lag_df[lag_df["feature_type"] == "raw"]
    raw_pivot = raw_lags.pivot(
        index="channel",
        columns="lag_months",
        values="adjusted_spend_revenue_correlation",
    )
    raw_pivot.columns = [f"lag_{lag}_month" for lag in raw_pivot.columns]
    raw_pivot.to_csv(out_dir / "time_delay_correlation_matrix.csv")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for channel, ax in zip(CHANNELS, axes.flat):
        subset = lag_df[
            (lag_df["channel"] == channel) &
            (lag_df["feature_type"] == "raw")
        ].sort_values("lag_months")

        ax.plot(
            subset["lag_months"],
            subset["adjusted_spend_revenue_correlation"],
            marker="o",
            label="Adjusted raw spend",
        )
        ax.axhline(0, linewidth=0.8, alpha=0.6)
        ax.set_title(channel_label(channel))
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Adjusted correlation")
        ax.set_xticks(range(MAX_LAG + 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("M2: Seasonality/Promotion-Adjusted Time Delays")
    fig.tight_layout()
    fig.savefig(fig_dir / "time_delay_correlations.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # 4. Diminishing returns: binned spend vs adjusted revenue,
    # plus per-channel Hill and negative-exponential curve fits.
    diminishing_rows = []
    fit_rows = []

    for channel in CHANNELS:
        spend = df[channel].to_numpy(dtype=float)
        adjusted_spend = residualize(spend, controls)

        bin_frame = pd.DataFrame({
            "spend_lkr": spend,
            "adjusted_spend_residual": adjusted_spend,
            "adjusted_revenue": adjusted_revenue,
        })

        try:
            bin_frame["spend_bin"] = pd.qcut(
                bin_frame["spend_lkr"], q=N_BINS, duplicates="drop"
            )
        except ValueError:
            bin_frame["spend_bin"] = np.nan

        grouped = (
            bin_frame.dropna(subset=["spend_bin"])
            .groupby("spend_bin", observed=True)
            .agg(
                observations=("spend_lkr", "size"),
                mean_spend_lkr=("spend_lkr", "mean"),
                mean_adjusted_revenue=("adjusted_revenue", "mean"),
                mean_adjusted_spend=("adjusted_spend_residual", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            diminishing_rows.append({
                "channel": channel,
                "spend_bin": str(row["spend_bin"]),
                "observations": int(row["observations"]),
                "mean_spend_lkr": float(row["mean_spend_lkr"]),
                "mean_adjusted_revenue": float(row["mean_adjusted_revenue"]),
                "mean_adjusted_spend": float(row["mean_adjusted_spend"]),
            })

        # Fit the saturating functions against spend and adjusted revenue.
        # Small monthly samples mean these are candidate curves, not causal
        # channel-effect estimates.
        for model in ["hill", "negative_exponential"]:
            fit = fit_saturating_curve(spend, adjusted_revenue, model)
            fit_rows.append({"channel": channel, "model": model, **fit})

    pd.DataFrame(diminishing_rows).to_csv(
        out_dir / "diminishing_returns_bins.csv", index=False
    )
    pd.DataFrame(fit_rows).to_csv(
        out_dir / "diminishing_returns_fits.csv", index=False
    )

    # 5. Bounded candidate parameter ranges for M3/M4.
    candidate_rows = []
    for channel in CHANNELS:
        spend = df[channel].to_numpy(dtype=float)
        positive = spend[spend > 0]

        if len(positive) == 0:
            half_candidates = []
        else:
            half_candidates = sorted(
                set(float(v) for v in np.quantile(positive, [0.25, 0.5, 0.75]))
            )

        for decay in DECAY_VALUES:
            for half_sat in half_candidates:
                for slope in HILL_SLOPES:
                    candidate_rows.append({
                        "channel": channel,
                        "adstock_decay_candidate": decay,
                        "half_saturation_lkr_candidate": half_sat,
                        "hill_slope_candidate": slope,
                        "decay_min": min(DECAY_VALUES),
                        "decay_max": max(DECAY_VALUES),
                        "half_saturation_min_lkr": (
                            min(half_candidates) if half_candidates else np.nan
                        ),
                        "half_saturation_max_lkr": (
                            max(half_candidates) if half_candidates else np.nan
                        ),
                        "hill_slope_min": min(HILL_SLOPES),
                        "hill_slope_max": max(HILL_SLOPES),
                        "intended_use": "Bounded exploratory candidates for M3/M4",
                    })

    pd.DataFrame(candidate_rows).to_csv(
        out_dir / "parameter_candidates.csv", index=False
    )

    # 6. Analysis metadata and limitations
    metadata = pd.DataFrame([{
        "observations": len(df),
        "first_month": df["month"].min().strftime("%Y-%m"),
        "last_month": df["month"].max().strftime("%Y-%m"),
        "target": TARGET,
        "channels_analysed": len(CHANNELS),
        "adjustment_controls": ", ".join(control_columns),
        "adstock_decay_values": ", ".join(map(str, DECAY_VALUES)),
        "hill_slopes": ", ".join(map(str, HILL_SLOPES)),
        "maximum_lag_months": MAX_LAG,
        "diminishing_returns_bins": N_BINS,
        "interpretation": (
            "Exploratory associations and bounded curve candidates only. "
            "Residualisation and correlations do not establish causality. "
            "Monthly sample size is small; fitted parameters need validation."
        ),
    }])
    metadata.to_csv(out_dir / "analysis_metadata.csv", index=False)

    print("M2 analysis completed successfully.")
    print(f"Tables saved to: {out_dir}")
    print(f"Figures saved to: {fig_dir}")
    print(f"Observations analysed: {len(df)}")
    print(f"Channels analysed: {len(CHANNELS)}")
    print("Includes adjusted lags 0-4, diminishing-return fits, and parameter candidates.")
    print("Note: results are exploratory, not causal estimates.")


if __name__ == "__main__":
    main()
