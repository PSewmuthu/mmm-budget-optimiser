
"""M3: Primary Marketing Mix Model.

Uses M2 adstock and Hill saturation, constrained Ridge
regression, and SciPy parameter optimisation.

Public interface for M5 and M6:
    fit(df)
    predict(df)
    response_curve(channel, grid)
    contributions(df)
"""

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, lsq_linear
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.config import CHANNELS, CONTROLS, SEED, TARGET
from src.transforms import adstock, hill


# Provisional shared transform ranges.
# Confirm final bounds with M2.
PARAMETER_BOUNDS = [
    (0.05, 0.70),  # Adstock decay
    (0.70, 1.70),  # Half-saturation multiplier
    (0.75, 1.50),  # Hill slope
]


def metrics(y_true, y_pred):
    """Return MAPE, R-squared, and NRMSE."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.shape != predicted.shape or len(actual) == 0:
        raise ValueError("Invalid prediction arrays")

    if not np.isfinite(actual).all():
        raise ValueError("Actual revenue contains invalid values")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain invalid values")

    if np.any(actual <= 0):
        raise ValueError("Revenue must be positive")

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    return {
        "mape_pct": float(
            mean_absolute_percentage_error(
                actual, predicted
            ) * 100
        ),
        "r2": float(
            r2_score(actual, predicted)
        ) if len(actual) > 1 else float("nan"),
        "nrmse_pct": float(
            rmse / actual.mean() * 100
        ),
    }


class MMMModel:
    """Constrained Ridge Marketing Mix Model."""

    def __init__(
        self,
        alpha=0.30,
        bounds=PARAMETER_BOUNDS,
        search_iterations=10,
    ):
        self.alpha = float(alpha)
        self.bounds = bounds
        self.search_iterations = int(search_iterations)
        self.fitted_ = False

    @staticmethod
    def _validate(
        df,
        require_target=False,
        require_controls=True,
    ):
        """Check input data before training or prediction."""

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Expected a non-empty DataFrame")

        required = list(CHANNELS)

        if require_controls:
            required += list(CONTROLS)

        if require_target:
            required += [TARGET]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        values = df[required].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Inputs contain missing or invalid numbers"
            )

        if (df[list(CHANNELS)] < 0).any().any():
            raise ValueError(
                "Advertising spend cannot be negative"
            )

        if require_target and (df[TARGET] <= 0).any():
            raise ValueError("Revenue must be positive")

    @staticmethod
    def _adstock_matrix(df, decay, previous=None):
        """Apply M2 geometric adstock to each channel."""

        result = np.empty(
            (len(df), len(CHANNELS))
        )

        last = np.zeros(len(CHANNELS))

        for j, channel in enumerate(CHANNELS):
            spending = df[channel].to_numpy(
                dtype=float
            )

            if previous is None:
                transformed = adstock(
                    spending, decay
                )
            else:
                # Continue adstock from the previous period.
                transformed = adstock(
                    np.r_[previous[j], spending],
                    decay,
                )[1:]

            result[:, j] = transformed
            last[j] = transformed[-1]

        return result, last

    @staticmethod
    def _design(
        df,
        theta,
        half_sat,
        control_means,
        control_stds,
        previous=None,
    ):
        """Create transformed media and control features."""

        decay, _, slope = theta

        carried, last = MMMModel._adstock_matrix(
            df, decay, previous
        )

        # Apply M2 Hill saturation.
        media = np.column_stack([
            hill(
                carried[:, j],
                half_sat[j],
                slope,
            )
            for j in range(len(CHANNELS))
        ])

        # Fill omitted controls using training averages.
        controls_df = df.reindex(
            columns=list(CONTROLS)
        ).copy()

        for col, avg in zip(
            CONTROLS, control_means
        ):
            controls_df[col] = (
                controls_df[col].fillna(avg)
            )

        controls = (
            controls_df.to_numpy(dtype=float)
            - control_means
        ) / control_stds

        X = np.column_stack([
            media, controls
        ])

        return X, last

    @staticmethod
    def _ridge(X, y, alpha):
        """Fit Ridge with nonnegative media coefficients.

        Controls can have positive or negative coefficients.
        The intercept is not regularised.
        """

        x_mean = X.mean(axis=0)
        y_mean = y.mean()

        X_centered = X - x_mean
        y_centered = y - y_mean

        feature_count = X.shape[1]

        A = np.vstack([
            X_centered,
            np.sqrt(alpha) * np.eye(
                feature_count
            ),
        ])

        b = np.r_[
            y_centered,
            np.zeros(feature_count),
        ]

        lower = np.r_[
            np.zeros(len(CHANNELS)),
            np.full(
                len(CONTROLS),
                -np.inf,
            ),
        ]

        upper = np.full(
            feature_count,
            np.inf,
        )

        result = lsq_linear(
            A,
            b,
            bounds=(lower, upper),
            tol=1e-9,
        )

        if not result.success:
            raise RuntimeError(
                "Ridge model fitting failed"
            )

        coefficients = result.x

        intercept = float(
            y_mean - x_mean @ coefficients
        )

        return intercept, coefficients

    def _fit_parameters(self, df, theta):
        """Fit regression for candidate transform settings."""

        carried, last = self._adstock_matrix(
            df, theta[0]
        )

        medians = np.median(
            carried,
            axis=0,
        )

        half_sat = np.maximum(
            medians * theta[1],
            1.0,
        )

        control_means = (
            df[list(CONTROLS)]
            .mean()
            .to_numpy(dtype=float)
        )

        control_stds = (
            df[list(CONTROLS)]
            .std(ddof=0)
            .to_numpy(dtype=float)
        )

        control_stds = np.where(
            control_stds == 0,
            1.0,
            control_stds,
        )

        X, _ = self._design(
            df,
            theta,
            half_sat,
            control_means,
            control_stds,
        )

        # Scaling revenue improves numerical stability.
        revenue_scale = float(
            df[TARGET].mean()
        )

        y = (
            df[TARGET].to_numpy(dtype=float)
            / revenue_scale
        )

        intercept, coefficients = self._ridge(
            X,
            y,
            self.alpha,
        )

        return (
            intercept,
            coefficients,
            revenue_scale,
            half_sat,
            control_means,
            control_stds,
            last,
        )

    def _cv_score(self, df, theta, collect=False):
        """Expanding-window time-series cross-validation."""

        rows = []

        # Three validation folds within first 30 months.
        for train_end in (18, 22, 26):

            if train_end + 4 > len(df):
                continue

            train = df.iloc[:train_end]
            validation = df.iloc[
                train_end:train_end + 4
            ]

            (
                intercept,
                coefficients,
                scale,
                half_sat,
                means,
                stds,
                last,
            ) = self._fit_parameters(
                train, theta
            )

            X_valid, _ = self._design(
                validation,
                theta,
                half_sat,
                means,
                stds,
                previous=last,
            )

            predicted = (
                intercept
                + X_valid @ coefficients
            ) * scale

            score = metrics(
                validation[TARGET].to_numpy(),
                predicted,
            )

            rows.append({
                "train_months": train_end,
                "validation_months": len(validation),
                **score,
            })

        if not rows:
            raise ValueError(
                "At least 30 months are required"
            )

        if collect:
            return pd.DataFrame(rows)

        return float(
            np.mean([
                row["nrmse_pct"]
                for row in rows
            ])
        )

    def fit(self, df):
        """Search for transforms and fit MMM."""

        self._validate(
            df,
            require_target=True,
        )

        if len(df) < 30:
            raise ValueError(
                "At least 30 months required"
            )

        if "month" in df.columns:
            months = pd.to_datetime(df["month"])

            if not months.is_monotonic_increasing:
                raise ValueError(
                    "Sort data chronologically"
                )

        df = df.reset_index(drop=True)

        # SciPy finds transformation settings using CV.
        search = differential_evolution(
            lambda theta: self._cv_score(
                df, theta
            ),
            bounds=self.bounds,
            seed=SEED,
            popsize=4,
            maxiter=self.search_iterations,
            polish=False,
        )

        self.theta_ = np.asarray(
            search.x
        )

        self.cv_results_ = self._cv_score(
            df,
            self.theta_,
            collect=True,
        )

        (
            self.intercept_,
            self.coef_,
            self.revenue_scale_,
            self.half_sat_,
            self.control_means_,
            self.control_stds_,
            self.last_carry_,
        ) = self._fit_parameters(
            df, self.theta_
        )

        self.fitted_ = True

        return self

    def refit(self, df):
        """Refit coefficients using fixed transform settings."""

        if not self.fitted_:
            raise RuntimeError(
                "Fit the model first"
            )

        self._validate(
            df,
            require_target=True,
        )

        if "month" in df.columns:
            months = pd.to_datetime(df["month"])

            if not months.is_monotonic_increasing:
                raise ValueError(
                    "Sort data chronologically"
                )

        (
            self.intercept_,
            self.coef_,
            self.revenue_scale_,
            self.half_sat_,
            self.control_means_,
            self.control_stds_,
            self.last_carry_,
        ) = self._fit_parameters(
            df, self.theta_
        )

        return self

    def _features(
        self,
        df,
        continue_history=False,
    ):
        """Build prediction features."""

        if not self.fitted_:
            raise RuntimeError(
                "Fit the model first"
            )

        self._validate(
            df,
            require_controls=False,
        )

        previous = (
            self.last_carry_
            if continue_history else None
        )

        X, _ = self._design(
            df,
            self.theta_,
            self.half_sat_,
            self.control_means_,
            self.control_stds_,
            previous,
        )

        return X

    def predict(
        self,
        df,
        continue_history=False,
    ):
        """Return predicted revenue in LKR as a NumPy array."""

        X = self._features(
            df,
            continue_history,
        )

        prediction = (
            self.intercept_
            + X @ self.coef_
        ) * self.revenue_scale_

        return np.asarray(
            prediction,
            dtype=float,
        )

    def response_curve(self, channel, grid):
        """Return incremental steady-state channel revenue.

        Output matches the original M3 interface:
        one NumPy value per spending-grid value.
        """

        if not self.fitted_:
            raise RuntimeError(
                "Fit the model first"
            )

        if channel not in CHANNELS:
            raise ValueError(
                f"Unknown channel: {channel}"
            )

        spend = np.atleast_1d(
            np.asarray(
                grid,
                dtype=float,
            )
        )

        if (
            not np.isfinite(spend).all()
            or (spend < 0).any()
        ):
            raise ValueError(
                "Invalid spending amounts"
            )

        j = list(CHANNELS).index(channel)

        # Long-run carryover at constant monthly spend.
        effective_spend = (
            spend / (1 - self.theta_[0])
        )

        transformed = hill(
            effective_spend,
            self.half_sat_[j],
            self.theta_[2],
        )

        incremental_revenue = (
            self.coef_[j]
            * transformed
            * self.revenue_scale_
        )

        return np.asarray(
            incremental_revenue,
            dtype=float,
        )

    def contributions(
        self,
        df,
        continue_history=False,
    ):
        """Return monthly baseline and feature contributions."""

        X = self._features(
            df,
            continue_history,
        )

        result = pd.DataFrame(
            index=df.index
        )

        result["baseline_lkr"] = (
            self.intercept_
            * self.revenue_scale_
        )

        feature_names = (
            list(CHANNELS) + list(CONTROLS)
        )

        for i, name in enumerate(feature_names):
            result[
                f"{name}_contribution_lkr"
            ] = (
                X[:, i]
                * self.coef_[i]
                * self.revenue_scale_
            )

        result["predicted_revenue_lkr"] = (
            self.predict(
                df,
                continue_history,
            )
        )

        return result
