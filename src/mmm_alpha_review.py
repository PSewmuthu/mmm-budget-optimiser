
"""M3: Ridge regularisation and baseline comparison.

Run:
    python -m src.mmm_alpha_review

Uses the first 30 months only.
Does not modify the main model or final holdout.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEAN_CSV, MODEL_DIR, TARGET
from src.mmm import MMMModel, metrics


# Ridge regularisation strengths to compare.
ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]

# Same expanding-window folds used by the main MMM.
FOLD_STARTS = (18, 22, 26)
VALIDATION_MONTHS = 4


def main():

    # Load the cleaned M1 dataset.
    data = pd.read_csv(
        CLEAN_CSV,
        parse_dates=["month"],
    )

    data = data.sort_values(
        "month"
    ).reset_index(drop=True)

    if len(data) != 36:
        raise ValueError(
            "Expected exactly 36 months of data"
        )

    # Only the first 30 months are used.
    # The final six-month holdout remains excluded.
    development = data.iloc[:30].copy()

    # STEP 1: Calculate simple baseline predictions.
    baseline_rows = []

    for train_end in FOLD_STARTS:

        train = development.iloc[:train_end]

        validation = development.iloc[
            train_end:train_end + VALIDATION_MONTHS
        ]

        # Predict the historical training average.
        predicted = np.full(
            len(validation),
            float(train[TARGET].mean()),
        )

        score = metrics(
            validation[TARGET].to_numpy(),
            predicted,
        )

        baseline_rows.append({
            "train_months": train_end,
            "validation_months": len(validation),
            **score,
        })

    baseline = pd.DataFrame(baseline_rows)

    baseline_nrmse = float(
        baseline["nrmse_pct"].mean()
    )

    # STEP 2: Compare different Ridge alpha values.
    summaries = []
    fold_results = []

    for alpha in ALPHAS:

        print(f"Evaluating Ridge alpha={alpha}...")

        model = MMMModel(alpha=alpha)

        # Transform search and CV use development only.
        model.fit(development)

        cv = model.cv_results_.copy()

        cv.insert(0, "alpha", alpha)

        fold_results.append(cv)

        summaries.append({
            "alpha": alpha,
            "cv_mape_pct": float(
                cv["mape_pct"].mean()
            ),
            "cv_r2_mean": float(
                cv["r2"].mean()
            ),
            "cv_nrmse_pct": float(
                cv["nrmse_pct"].mean()
            ),
            "baseline_cv_nrmse_pct": baseline_nrmse,
            "nrmse_improvement_vs_baseline_pct_points": (
                baseline_nrmse
                - float(cv["nrmse_pct"].mean())
            ),
            "decay": float(model.theta_[0]),
            "half_sat_multiplier": float(
                model.theta_[1]
            ),
            "hill_slope": float(model.theta_[2]),
        })

    # STEP 3: Rank candidate models by CV NRMSE.
    results = (
        pd.DataFrame(summaries)
        .sort_values(
            "cv_nrmse_pct",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # STEP 4: Save separate review outputs.
    review_dir = Path(MODEL_DIR) / "review"

    review_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline.to_csv(
        review_dir / "baseline_cv.csv",
        index=False,
    )

    results.to_csv(
        review_dir / "alpha_comparison.csv",
        index=False,
    )

    pd.concat(
        fold_results,
        ignore_index=True,
    ).to_csv(
        review_dir / "alpha_fold_results.csv",
        index=False,
    )

    # STEP 5: Display results.
    print("\nTRAINING-MEAN BASELINE")
    print(
        baseline.round(3).to_string(index=False)
    )

    print("\nRIDGE ALPHA COMPARISON")

    columns = [
        "alpha",
        "cv_mape_pct",
        "cv_r2_mean",
        "cv_nrmse_pct",
        "nrmse_improvement_vs_baseline_pct_points",
    ]

    print(
        results[columns]
        .round(3)
        .to_string(index=False)
    )

    best = results.iloc[0]

    print(
        "\nLowest CV NRMSE among tested alphas:",
        best["alpha"],
    )

    if best["cv_nrmse_pct"] < baseline_nrmse:
        print(
            "At least one MMM configuration beats "
            "the mean baseline on CV NRMSE."
        )
    else:
        print(
            "None of the tested MMM configurations "
            "beat the mean baseline on CV NRMSE."
        )

    print("\nReview files saved to:", review_dir)

    print(
        "Note: The same development CV folds were "
        "used for parameter tuning and comparison."
    )

    print(
        "These are model-selection scores, "
        "not independent test estimates."
    )

    print(
        "The final six-month holdout was not used."
    )

    print(
        "The existing trained model was not modified."
    )


if __name__ == "__main__":
    main()
