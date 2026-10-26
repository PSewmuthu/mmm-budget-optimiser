
"""M3: Train and evaluate the Primary MMM.

Run from project root:
    python -m src.mmm_fit
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CHANNELS,
    CLEAN_CSV,
    MODEL_DIR,
    TARGET,
)
from src.mmm import MMMModel, metrics


def main():

    print("Loading cleaned MMM dataset...")

    # Load M1 cleaned data.
    df = pd.read_csv(
        CLEAN_CSV,
        parse_dates=["month"],
    )

    df = (
        df.sort_values("month")
        .reset_index(drop=True)
    )

    if len(df) != 36:
        raise ValueError(
            f"Expected 36 months, found {len(df)}"
        )

    # Check that months are consecutive.
    month_gaps = (
        df["month"]
        .dt.to_period("M")
        .diff()
        .iloc[1:]
        .apply(lambda difference: difference.n)
    )

    if not (month_gaps == 1).all():
        raise ValueError(
            "Months must be consecutive"
        )

    # Hold out final six months.
    development = df.iloc[:-6].copy()
    holdout = df.iloc[-6:].copy()

    print(
        "Development months:",
        len(development),
    )

    print(
        "Holdout months:",
        len(holdout),
    )

    # Train using only development data.
    print("Training Primary MMM...")

    model = MMMModel()
    model.fit(development)

    # Predict the final six months.
    print("Evaluating final six months...")

    predictions = model.predict(
        holdout,
        continue_history=True,
    )

    holdout_scores = metrics(
        holdout[TARGET].to_numpy(),
        predictions,
    )

    output_dir = Path(MODEL_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save cross-validation results.
    model.cv_results_.to_csv(
        output_dir / "cv_results.csv",
        index=False,
    )

    # Save actual and predicted revenue.
    prediction_table = holdout[
        ["month", TARGET]
    ].copy()

    prediction_table[
        "predicted_revenue_lkr"
    ] = predictions

    prediction_table.to_csv(
        output_dir / "holdout_predictions.csv",
        index=False,
    )

    # Save holdout accuracy metrics.
    with open(
        output_dir / "holdout_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            holdout_scores,
            file,
            indent=2,
        )

    # Record settings from the development-only model.
    development_half_sat = dict(
        zip(
            CHANNELS,
            model.half_sat_.tolist(),
        )
    )

    # Refit coefficients on all 36 months for M5/M6.
    # Do not tune transformation parameters on the holdout.
    model.refit(df)

    # Save trained model.
    joblib.dump(
        model,
        output_dir / "mmm_model.joblib",
    )

    # Save transformation parameters.
    parameters = {
        "shared_decay": float(
            model.theta_[0]
        ),
        "shared_half_sat_multiplier": float(
            model.theta_[1]
        ),
        "shared_hill_slope": float(
            model.theta_[2]
        ),
        "development_half_sat": development_half_sat,
        "deployed_half_sat": dict(
            zip(
                CHANNELS,
                model.half_sat_.tolist(),
            )
        ),
    }

    with open(
        output_dir / "transform_parameters.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            parameters,
            file,
            indent=2,
        )

    # Create response curves for M5/M6.
    curves = []

    for channel in CHANNELS:

        average_spend = float(
            df[channel].mean()
        )

        # Evaluate from zero to 150% of mean spend.
        grid = np.linspace(
            0,
            average_spend * 1.5,
            40,
        )

        incremental = model.response_curve(
            channel,
            grid,
        )

        curve = pd.DataFrame({
            "channel": channel,
            "spend_lkr": grid,
            "incremental_revenue_lkr": incremental,
        })

        curves.append(curve)

    pd.concat(
        curves,
        ignore_index=True,
    ).to_csv(
        output_dir / "response_curves.csv",
        index=False,
    )

    # Print final results.
    print("\nCROSS-VALIDATION RESULTS")

    print(
        model.cv_results_
        .round(3)
        .to_string(index=False)
    )

    print("\nFINAL HOLDOUT RESULTS")

    print(
        json.dumps(
            holdout_scores,
            indent=2,
        )
    )

    print("\nM3 training completed.")

    print(
        f"Results saved in: {output_dir}"
    )


if __name__ == "__main__":
    main()
