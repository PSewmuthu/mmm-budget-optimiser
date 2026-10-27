
# M3 — Primary Marketing Mix Model

## 1. Objective

Develop a Primary Marketing Mix Model (MMM) to estimate
monthly sales revenue from advertising spending and
external control variables.

The model is designed to support downstream channel
effectiveness analysis and budget optimisation.

## 2. Methodology

### 2.1 Data

The model uses the cleaned synthetic marketing dataset
provided by M1.

The dataset contains 36 consecutive monthly observations.

The target variable is monthly sales revenue in LKR.

### 2.2 Advertising Transformations

M2's geometric adstock function is used to represent
the carryover effects of advertising.

M2's Hill saturation function is used to represent
diminishing returns as advertising expenditure increases.

SciPy differential evolution is used to select bounded
transformation parameters.

Shared transformation parameters are used across the
channels to reduce model complexity.

### 2.3 Regression Model

Constrained Ridge regression is used to predict sales
revenue.

Advertising channel coefficients are restricted to
non-negative values.

External control variables are allowed to have positive
or negative coefficients.

The supplied seasonal demand index is included instead
of twelve monthly dummy variables.

## 3. Model Validation

Random data splitting was avoided because the observations
are ordered over time.

The first 30 months were used for model development.

Expanding-window time-series cross-validation was used
with three folds.

The final six months were reserved for holdout testing.

The evaluation metrics were:

- Mean Absolute Percentage Error (MAPE)
- R-squared (R2)
- Normalized Root Mean Squared Error (NRMSE)

## 4. Results

### 4.1 Cross-Validation Results

| Fold | Training Months | Validation Months | MAPE | R2 | NRMSE |
|---|---:|---:|---:|---:|---:|
| 1 | 18 | 4 | 6.925% | -0.088 | 8.809% |
| 2 | 22 | 4 | 4.651% | 0.432 | 6.786% |
| 3 | 26 | 4 | 1.056% | 0.969 | 1.285% |

### 4.2 Final Six-Month Holdout

- MAPE: 6.177%
- R2: -0.417
- NRMSE: 7.083%

The model achieved an average absolute percentage
prediction error of approximately 6.18%.

However, the negative holdout R2 indicates that its
squared prediction error was worse than that of a
constant prediction based on the holdout mean.

All six holdout observations were overpredicted.

### 4.3 Baseline Comparison

The following models were evaluated on the final
six-month holdout:

| Model | MAPE | R2 |
|---|---:|---:|
| Primary MMM | 6.18% | -0.417 |
| Training mean | 4.31% | -0.337 |
| Last known revenue | 7.41% | -1.682 |

The training-mean baseline performed better than the
Primary MMM on both MAPE and R2.

The MMM performed better than the last-known-revenue
baseline on these two measures.

The results indicate that the model's predictive
performance is limited and should be interpreted
cautiously.

## 5. Model Outputs

The following files are generated in outputs/model/:

- cv_results.csv
- holdout_metrics.json
- holdout_predictions.csv
- transform_parameters.json
- response_curves.csv
- mmm_model.joblib

After holdout evaluation, the model coefficients are
refitted on all 36 months using fixed transformation
parameters for downstream use.

## 6. Integration

The model provides these methods:

- MMMModel.fit(df)
- MMMModel.predict(df)
- MMMModel.response_curve(channel, grid)
- MMMModel.contributions(df)

predict() returns a NumPy array of revenue predictions.

response_curve() returns a NumPy array of estimated
incremental steady-state channel revenue responses.

contributions() returns the model's monthly baseline,
channel contributions, control contributions, and
predicted revenue.

These methods support integration with M5 and M6.

## 7. Limitations

The dataset contains only 36 synthetic monthly
observations, creating a substantial risk of overfitting.

Advertising channels may be correlated, affecting the
stability of estimated channel contributions.

Shared transformation parameters simplify the model
but may not represent differences between channels.

The negative holdout R2 and consistent overprediction
indicate limited forecasting reliability.

The holdout should not be repeatedly used to select
model parameters.

MMM relationships estimated from observational data
should not be interpreted as proven causal effects.

Budget recommendations based on this model should be
treated as exploratory estimates.

Final transformation bounds must be checked against
M2's recommended parameter ranges.
