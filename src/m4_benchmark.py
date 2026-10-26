
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input dataset prepared by M1
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mmm_clean.csv"


def load_dataset():
    """
    Load and validate the cleaned MMM dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    print("\nDataset loaded successfully!")

    print("\nDataset shape:")
    print(df.shape)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    return df


# M4 model variables
CHANNEL_COLUMNS = [
    "tv_spend_lkr",
    "facebook_instagram_spend_lkr",
    "google_search_spend_lkr",
    "youtube_spend_lkr",
    "radio_spend_lkr",
    "outdoor_spend_lkr",
]

CONTROL_COLUMNS = [
    "average_price_index",
    "promotion_discount_pct",
    "competitor_pressure_index_0_100",
    "economic_sentiment_index_0_100",
    "lkr_usd_exchange_rate",
    "inflation_proxy_index",
    "seasonal_demand_index",
    "public_holiday_count",
]


def prepare_model_data(df):
    """
    Create the regression variables and time-based split.
    """

    df = df.copy()

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    required = (
        CHANNEL_COLUMNS
        + CONTROL_COLUMNS
        + ["sales_revenue_lkr"]
    )

    if df[required].isna().any().any():
        raise ValueError("Model variables contain missing values.")

    if (df[CHANNEL_COLUMNS + ["sales_revenue_lkr"]] <= 0).any().any():
        raise ValueError("Revenue and channel spend must be positive.")

    # Apply logarithms to advertising expenditure
    for channel in CHANNEL_COLUMNS:
        df[f"log_{channel}"] = np.log(df[channel])

    # Apply logarithm to the target variable
    df["log_sales_revenue"] = np.log(df["sales_revenue_lkr"])

    # Preserve chronological order
    train_df = df.iloc[:-6].copy()
    test_df = df.iloc[-6:].copy()

    print("\nTraining observations:", len(train_df))
    print("Testing observations:", len(test_df))

    return train_df, test_df


def fit_benchmark_model(train_df):
    """
    Fit a log-log OLS model with HAC standard errors.
    """

    log_channels = [
        f"log_{channel}" for channel in CHANNEL_COLUMNS
    ]

    features = log_channels + CONTROL_COLUMNS

    X_train = train_df[features].astype(float)
    y_train = train_df["log_sales_revenue"].astype(float)

    # Add regression intercept
    X_train = sm.add_constant(
        X_train,
        has_constant="add"
    )

    model = sm.OLS(y_train, X_train)

    # Newey-West HAC standard errors
    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 2}
    )

    print("\n===== M4 BENCHMARK MODEL =====")
    print(results.summary())

    return results


def run_model_diagnostics(results):
    """
    Evaluate regression assumptions and save diagnostic outputs.
    """

    print("\n===== M4 MODEL DIAGNOSTICS =====")

    output_dir = PROJECT_ROOT / "outputs" / "tables" / "m4"
    figure_dir = PROJECT_ROOT / "outputs" / "figures" / "m4"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    residuals = results.resid
    fitted_values = results.fittedvalues

    # 1. Variance Inflation Factor (VIF)
    print("\n1. Variance Inflation Factor (VIF)")

    X = results.model.exog
    feature_names = results.model.exog_names

    vif_rows = []

    for i, feature in enumerate(feature_names):

        if feature == "const":
            continue

        vif = variance_inflation_factor(X, i)

        vif_rows.append({
            "feature": feature,
            "vif": vif,
        })

    vif_df = pd.DataFrame(vif_rows)
    vif_df = vif_df.sort_values(
        by="vif",
        ascending=False
    )

    print(vif_df.to_string(index=False))

    vif_df.to_csv(
        output_dir / "vif_results.csv",
        index=False
    )

    # 2. Durbin-Watson
    dw_stat = durbin_watson(residuals)

    print("\n2. Durbin-Watson:", round(dw_stat, 4))

    # 3. Ljung-Box
    lb_results = acorr_ljungbox(
        residuals,
        lags=[1, 2, 4],
        return_df=True
    )

    print("\n3. Ljung-Box Test")
    print(lb_results)

    lb_results.to_csv(
        output_dir / "ljung_box_results.csv"
    )

    # 4. Breusch-Pagan
    bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = (
        het_breuschpagan(
            residuals,
            X
        )
    )

    print("\n4. Breusch-Pagan Test")
    print("LM statistic:", round(bp_stat, 4))
    print("LM p-value:", round(bp_pvalue, 4))
    print("F p-value:", round(bp_fpvalue, 4))

    # Save diagnostic summary
    diagnostics_df = pd.DataFrame([{
        "durbin_watson": dw_stat,
        "breusch_pagan_lm_stat": bp_stat,
        "breusch_pagan_lm_pvalue": bp_pvalue,
        "breusch_pagan_f_pvalue": bp_fpvalue,
    }])

    diagnostics_df.to_csv(
        output_dir / "diagnostic_summary.csv",
        index=False
    )

    # 5. Residual vs fitted plot
    plt.figure(figsize=(8, 5))

    plt.scatter(fitted_values, residuals)

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Fitted Log Revenue")
    plt.ylabel("Residuals")
    plt.title("M4 - Residuals vs Fitted Values")
    plt.tight_layout()

    plt.savefig(
        figure_dir / "residuals_vs_fitted.png",
        dpi=150
    )

    plt.close()

    # 6. Normal Q-Q plot
    fig = sm.qqplot(residuals, line="45", fit=True)
    fig.suptitle("M4 - Residual Q-Q Plot")
    fig.tight_layout()

    fig.savefig(
        figure_dir / "residual_qq_plot.png",
        dpi=150
    )

    plt.close(fig)

    print("\nDiagnostic outputs saved successfully!")

    return vif_df, diagnostics_df


def fit_distributed_lag_model(df):
    """
    Estimate current and delayed relationships between
    total advertising spend and sales revenue.
    """

    print("\n===== M4 DISTRIBUTED LAG MODEL =====")

    lag_df = df.copy()

    lag_df["month"] = pd.to_datetime(lag_df["month"])
    lag_df = lag_df.sort_values("month").reset_index(drop=True)

    # Current advertising expenditure
    lag_df["log_spend_lag0"] = np.log(
        lag_df["total_ad_spend_lkr"]
    )

    # Advertising expenditure one month earlier
    lag_df["log_spend_lag1"] = (
        lag_df["log_spend_lag0"].shift(1)
    )

    # Advertising expenditure two months earlier
    lag_df["log_spend_lag2"] = (
        lag_df["log_spend_lag0"].shift(2)
    )

    lag_df["log_revenue"] = np.log(
        lag_df["sales_revenue_lkr"]
    )

    # Remove the first two months without lag history
    lag_df = lag_df.dropna(
        subset=["log_spend_lag1", "log_spend_lag2"]
    ).copy()

    features = [
        "log_spend_lag0",
        "log_spend_lag1",
        "log_spend_lag2",
    ] + CONTROL_COLUMNS

    # Preserve the final six months as holdout
    train_df = lag_df.iloc[:-6].copy()
    test_df = lag_df.iloc[-6:].copy()

    X_train = sm.add_constant(
        train_df[features].astype(float),
        has_constant="add"
    )

    y_train = train_df["log_revenue"].astype(float)

    model = sm.OLS(y_train, X_train)

    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 2}
    )

    print("\nTraining observations:", len(train_df))
    print("Testing observations:", len(test_df))
    print(results.summary())

    # Export lag coefficients and confidence intervals
    intervals = results.conf_int()

    lag_results = pd.DataFrame({
        "coefficient": results.params,
        "p_value": results.pvalues,
        "ci_lower": intervals[0],
        "ci_upper": intervals[1],
    })

    lag_results = lag_results.loc[
        ["log_spend_lag0", "log_spend_lag1", "log_spend_lag2"]
    ]

    output_dir = PROJECT_ROOT / "outputs" / "tables" / "m4"
    output_dir.mkdir(parents=True, exist_ok=True)

    lag_results.to_csv(
        output_dir / "distributed_lag_results.csv"
    )

    print("\nAdvertising lag coefficients:")
    print(lag_results)

    print("\nDistributed-lag results saved successfully!")

    return results


def analyse_external_factors(results):
    """
    Analyse estimated external-factor effects using
    the benchmark OLS model and HAC confidence intervals.
    """

    print("\n===== M4 EXTERNAL FACTOR ANALYSIS =====")

    confidence_intervals = results.conf_int()

    # Meaningful changes in each variable
    changes = {
        "average_price_index": 0.01,
        "promotion_discount_pct": 1.0,
        "competitor_pressure_index_0_100": 1.0,
        "economic_sentiment_index_0_100": 1.0,
        "lkr_usd_exchange_rate": 1.0,
        "inflation_proxy_index": 0.01,
        "seasonal_demand_index": 0.1,
        "public_holiday_count": 1.0,
    }

    factor_results = []

    for factor in CONTROL_COLUMNS:

        coefficient = results.params[factor]
        p_value = results.pvalues[factor]

        ci_lower = confidence_intervals.loc[factor, 0]
        ci_upper = confidence_intervals.loc[factor, 1]

        change = changes[factor]

        # Convert changes in log revenue to % changes
        effect_pct = (
            np.exp(coefficient * change) - 1
        ) * 100

        effect_ci_lower = (
            np.exp(ci_lower * change) - 1
        ) * 100

        effect_ci_upper = (
            np.exp(ci_upper * change) - 1
        ) * 100

        factor_results.append({
            "factor": factor,
            "unit_change": change,
            "coefficient": coefficient,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "estimated_revenue_change_pct": effect_pct,
            "effect_ci_lower_pct": effect_ci_lower,
            "effect_ci_upper_pct": effect_ci_upper,
            "significant_at_5pct": p_value < 0.05,
        })

    factor_df = pd.DataFrame(factor_results)

    print("\nExternal factor coefficients:")
    print(
        factor_df[
            ["factor", "coefficient", "p_value",
             "ci_lower", "ci_upper"]
        ].to_string(index=False)
    )

    print("\nEstimated revenue effects:")
    print(
        factor_df[
            ["factor", "unit_change",
             "estimated_revenue_change_pct",
             "effect_ci_lower_pct",
             "effect_ci_upper_pct"]
        ].round(4).to_string(index=False)
    )

    output_dir = PROJECT_ROOT / "outputs" / "tables" / "m4"
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_df.to_csv(
        output_dir / "external_factor_effects.csv",
        index=False
    )

    print("\nExternal factor results saved successfully!")

    return factor_df


def run_ablation_analysis(train_df, test_df, benchmark_results):
    """
    Compare MMM models with and without external controls.
    Use the same training and holdout months.
    """

    print("\n===== M4 ABLATION ANALYSIS =====")

    log_channels = [
        f"log_{channel}" for channel in CHANNEL_COLUMNS
    ]

    # Model B: advertising channels only
    X_train_no_controls = sm.add_constant(
        train_df[log_channels].astype(float),
        has_constant="add"
    )

    y_train = train_df["log_sales_revenue"].astype(float)

    no_controls_model = sm.OLS(
        y_train,
        X_train_no_controls
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 2}
    )

    # Compare advertising coefficients
    rows = []

    for channel in log_channels:

        with_controls = benchmark_results.params[channel]
        without_controls = no_controls_model.params[channel]

        rows.append({
            "channel": channel.replace("log_", ""),
            "with_controls": with_controls,
            "without_controls": without_controls,
            "difference_without_minus_with": (
                without_controls - with_controls
            ),
            "sign_changed": (
                np.sign(with_controls)
                != np.sign(without_controls)
            )
        })

    comparison_df = pd.DataFrame(rows)

    print("\nAdvertising coefficient comparison:")
    print(comparison_df.round(4).to_string(index=False))

    # Holdout predictions using the same six months
    X_test_full = sm.add_constant(
        test_df[log_channels + CONTROL_COLUMNS].astype(float),
        has_constant="add"
    )

    X_test_no_controls = sm.add_constant(
        test_df[log_channels].astype(float),
        has_constant="add"
    )

    # Align columns with fitted regression models
    X_test_full = X_test_full[
        benchmark_results.model.exog_names
    ]

    X_test_no_controls = X_test_no_controls[
        no_controls_model.model.exog_names
    ]

    actual = test_df["sales_revenue_lkr"].to_numpy()

    # Convert predictions back into revenue units.
    # Exponentiated log predictions are used as
    # simple revenue forecasts without bias correction.
    predicted_full = np.exp(
        benchmark_results.predict(X_test_full)
    )

    predicted_no_controls = np.exp(
        no_controls_model.predict(X_test_no_controls)
    )

    def calculate_metrics(predicted):
        predicted = np.asarray(predicted)

        mape = np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100

        rmse = np.sqrt(
            np.mean((actual - predicted) ** 2)
        )

        nrmse = rmse / np.mean(actual)

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        return mape, nrmse, r2

    full_mape, full_nrmse, full_r2 = calculate_metrics(
        predicted_full
    )

    no_mape, no_nrmse, no_r2 = calculate_metrics(
        predicted_no_controls
    )

    metrics_df = pd.DataFrame([
        {
            "model": "with_controls",
            "train_r2_log": benchmark_results.rsquared,
            "holdout_mape_pct": full_mape,
            "holdout_nrmse": full_nrmse,
            "holdout_r2": full_r2,
        },
        {
            "model": "without_controls",
            "train_r2_log": no_controls_model.rsquared,
            "holdout_mape_pct": no_mape,
            "holdout_nrmse": no_nrmse,
            "holdout_r2": no_r2,
        }
    ])

    print("\nModel performance comparison:")
    print(metrics_df.round(4).to_string(index=False))

    output_dir = PROJECT_ROOT / "outputs" / "tables" / "m4"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(
        output_dir / "ablation_coefficients.csv",
        index=False
    )

    metrics_df.to_csv(
        output_dir / "ablation_metrics.csv",
        index=False
    )

    print("\nAblation results saved successfully!")

    return comparison_df, metrics_df


def export_benchmark_results(results):
    """
    Export benchmark regression coefficients,
    HAC confidence intervals, and model statistics.
    """

    print("\n===== M4 BENCHMARK RESULTS EXPORT =====")

    output_dir = PROJECT_ROOT / "outputs" / "tables" / "m4"
    output_dir.mkdir(parents=True, exist_ok=True)

    confidence_intervals = results.conf_int()

    coefficients_df = pd.DataFrame({
        "variable": results.params.index,
        "coefficient": results.params.values,
        "standard_error": results.bse.values,
        "p_value": results.pvalues.values,
        "ci_lower": confidence_intervals.iloc[:, 0].values,
        "ci_upper": confidence_intervals.iloc[:, 1].values,
    })

    coefficients_df.to_csv(
        output_dir / "benchmark_coefficients.csv",
        index=False
    )

    summary_df = pd.DataFrame([{
        "model": "M4_OLS_HAC",
        "training_observations": int(results.nobs),
        "r_squared": results.rsquared,
        "adjusted_r_squared": results.rsquared_adj,
        "aic": results.aic,
        "bic": results.bic,
        "covariance_type": results.cov_type,
    }])

    summary_df.to_csv(
        output_dir / "benchmark_model_summary.csv",
        index=False
    )

    print("\nBenchmark coefficient results saved.")
    print("Benchmark model summary saved.")

    return coefficients_df, summary_df


def main():
    print("M4 - Benchmark MMM and External Factors")

    df = load_dataset()

    train_df, test_df = prepare_model_data(df)

    benchmark_results = fit_benchmark_model(train_df)

    vif_results, diagnostics = run_model_diagnostics(
        benchmark_results
    )

    lag_results = fit_distributed_lag_model(df)

    external_results = analyse_external_factors(
        benchmark_results
    )

    ablation_coefficients, ablation_metrics = (
        run_ablation_analysis(
            train_df,
            test_df,
            benchmark_results
        )
    )

    export_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()