
"""
M4: Comparison between M3 primary MMM and M4 benchmark.

Both models are trained on the same first 30 months
and evaluated on the same final 6 months.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.mmm import MMMModel, metrics

from src.m4_benchmark import (
    CHANNEL_COLUMNS,
    CONTROL_COLUMNS,
    load_dataset,
    prepare_model_data,
    fit_benchmark_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "m4"


def compare_models():

    print("\n===== M3 VS M4 MODEL COMPARISON =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and use identical time-based splits
    df = load_dataset()

    train_df, test_df = prepare_model_data(df)

    actual = test_df["sales_revenue_lkr"].to_numpy(
        dtype=float
    )

    # 2. Fit M3 primary Marketing Mix Model
    print("\nTraining M3 primary MMM...")

    m3_model = MMMModel()
    m3_model.fit(train_df)

    # Continue adstock history from training into testing
    m3_predictions = m3_model.predict(
        test_df,
        continue_history=True
    )

    # 3. Fit M4 Statsmodels benchmark
    print("\nTraining M4 benchmark model...")

    m4_results = fit_benchmark_model(train_df)

    log_channels = [
        f"log_{channel}" for channel in CHANNEL_COLUMNS
    ]

    features = log_channels + CONTROL_COLUMNS

    X_test = sm.add_constant(
        test_df[features].astype(float),
        has_constant="add"
    )

    X_test = X_test[m4_results.model.exog_names]

    # Predictions are initially in log revenue
    m4_log_predictions = m4_results.predict(X_test)

    # Convert predictions into LKR
    m4_predictions = np.exp(m4_log_predictions)

    # 4. Compare model accuracy
    m3_scores = metrics(actual, m3_predictions)
    m4_scores = metrics(actual, m4_predictions)

    comparison_df = pd.DataFrame([
        {"model": "M3_Primary_MMM", **m3_scores},
        {"model": "M4_Statsmodels", **m4_scores},
    ])

    print("\n===== MODEL PERFORMANCE =====")
    print(comparison_df.round(4).to_string(index=False))

    comparison_df.to_csv(
        OUTPUT_DIR / "m3_m4_model_comparison.csv",
        index=False
    )

    # 5. Export actual and predicted revenue
    prediction_df = pd.DataFrame({
        "month": test_df["month"].to_numpy(),
        "actual_revenue_lkr": actual,
        "m3_predicted_revenue_lkr": m3_predictions,
        "m4_predicted_revenue_lkr": m4_predictions,
    })

    prediction_df.to_csv(
        OUTPUT_DIR / "m3_m4_predictions.csv",
        index=False
    )

    # 6. Compare directions of channel relationships
    channel_rows = []

    for i, channel in enumerate(CHANNEL_COLUMNS):

        m3_coefficient = m3_model.coef_[i]
        m4_coefficient = m4_results.params[
            f"log_{channel}"
        ]

        m3_direction = (
            "positive" if m3_coefficient > 1e-10
            else "zero"
        )

        m4_direction = (
            "positive" if m4_coefficient > 0
            else "negative" if m4_coefficient < 0
            else "zero"
        )

        channel_rows.append({
            "channel": channel,
            "m3_direction": m3_direction,
            "m4_direction": m4_direction,
            "m4_elasticity": m4_coefficient,
            "m4_p_value": m4_results.pvalues[
                f"log_{channel}"
            ],
            "direction_disagreement": (
                m3_direction != m4_direction
            ),
        })

    channel_df = pd.DataFrame(channel_rows)

    print("\n===== CHANNEL DIRECTION COMPARISON =====")
    print(channel_df.to_string(index=False))

    channel_df.to_csv(
        OUTPUT_DIR / "m3_m4_channel_comparison.csv",
        index=False
    )

    print("\nComparison results saved successfully!")

    return comparison_df, prediction_df, channel_df


if __name__ == "__main__":
    compare_models()
