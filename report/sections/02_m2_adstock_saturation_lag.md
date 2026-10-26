
# M2: Adstock, Saturation and Time-Delay Analysis

## 1. Objective

The objective of M2 is to investigate advertising carry-over effects,
diminishing returns and the time delay between marketing expenditure and
sales revenue. The analysis covers six marketing channels using 36 monthly
observations from January 2023 to December 2025.

The analysis also produces bounded parameter candidates for use in M3 and
M4. These candidates are exploratory starting points and must not be
interpreted as confirmed estimates of advertising effectiveness.

The target variable is `sales_revenue_lkr`. The marketing channels analysed
are:

- `tv_spend_lkr`
- `facebook_instagram_spend_lkr`
- `google_search_spend_lkr`
- `youtube_spend_lkr`
- `radio_spend_lkr`
- `outdoor_spend_lkr`

## 2. Geometric Adstock Analysis

Geometric adstock represents the carry-over of advertising expenditure
between consecutive months. It is defined as:

\[
A_t = X_t + \lambda A_{t-1}
\]

where:

- \(X_t\) is advertising spend in month \(t\).
- \(A_t\) is the transformed advertising value.
- \(\lambda\) is the decay parameter.
- \(A_{t-1}\) is the previous month's adstock value.

The implementation evaluates the following decay candidates:

- 0.0
- 0.3
- 0.5
- 0.7

A decay of zero means that previous advertising expenditure does not carry
over to the next month. Larger decay values represent stronger assumed
carry-over effects.

The decay values are candidate scenarios rather than statistically
confirmed estimates. The best value may differ by channel and should be
validated during subsequent modelling.

The results are saved in:

`outputs/tables/m2/adstock_summary.csv`

The implementation also includes an optional delayed-peak adstock
transformation. This allows an exploratory representation in which the
response to advertising can peak after the initial month. Its use does not
establish that a delayed response exists in the observed data.

## 3. Saturation and Diminishing Returns

Advertising response may exhibit diminishing returns: increasing spend
does not necessarily produce a proportional increase in response.

Two saturation functions are considered.

### 3.1 Hill Saturation

The Hill function is:

\[
S(x) = \frac{x^s}{x^s + K^s}
\]

where:

- \(x\) is advertising spend.
- \(K\) is the half-saturation parameter.
- \(s\) is the Hill slope.
- \(S(x)\) is a relative response between zero and one.

At \(x=K\), the response equals 0.5. The slope controls the shape of the
saturation curve.

The candidate Hill slopes are 1.0, 1.5 and 2.0.

### 3.2 Negative-Exponential Saturation

The negative-exponential function is:

\[
S(x) = 1 - e^{-x/c}
\]

where \(c\) is a positive scale parameter. The response increases with
spend but approaches an upper limit of one.

### 3.3 Fitting Method

For each marketing channel, the analysis fits candidate saturation curves
against the seasonality- and promotion-adjusted revenue response.

The curve-fitting routine evaluates a bounded grid of half-saturation
candidates and Hill slopes. For the negative-exponential model, it
evaluates candidate scale values. For each candidate curve, the intercept
and amplitude are estimated using least squares.

The exported fit results include the selected candidate parameters,
fit status and coefficient of determination (\(R^2\)).

The fitted parameters are exploratory. A successful fit status means that
the calculation completed; it does not mean that the model explains the
data well or that the parameters represent causal advertising effects.

Results:

- `outputs/tables/m2/diminishing_returns_fits.csv`
- `outputs/tables/m2/diminishing_returns_bins.csv`
- `outputs/tables/m2/saturation_curves.csv`

Figure:

`outputs/figures/m2/saturation_curves.png`

## 4. Seasonality and Promotion Adjustment

The time-delay and diminishing-returns analysis accounts for two observed
controls:

- `seasonal_demand_index`
- `promotion_discount_pct`

The analysis uses linear residualisation. An intercept and the two control
variables are used to estimate the component associated with those
controls. The residuals represent the part of each analysed variable not
explained by this linear control model.

This is an exploratory adjustment, not a guarantee that all seasonal,
promotional or other confounding effects have been removed.

## 5. Time-Delay Analysis

The time-delay analysis evaluates monthly advertising expenditure and
adstock-transformed expenditure against adjusted sales revenue.

The evaluated lags are:

- Lag 0: the same month.
- Lag 1: one month earlier.
- Lag 2: two months earlier.
- Lag 3: three months earlier.
- Lag 4: four months earlier.

For each channel, the analysis calculates Pearson correlation coefficients
between adjusted revenue and lagged adjusted advertising features. The
analysis includes raw spend and adstock features at decay values 0.3, 0.5
and 0.7.

The number of available observations decreases as the lag increases
because earlier advertising values are aligned with later revenue values.

The results are saved in:

- `outputs/tables/m2/time_delay_correlations.csv`
- `outputs/tables/m2/best_time_delay_by_channel.csv`
- `outputs/tables/m2/time_delay_correlation_matrix.csv`

Figure:

`outputs/figures/m2/time_delay_correlations.png`

The CSV files should be used when reporting exact correlations or identifying
the strongest observed lag. These associations are descriptive and do not
prove that advertising caused revenue changes.

## 6. Binned Diminishing-Returns Analysis

For each channel, observed spend is grouped into four bins. The analysis
summarises spend and adjusted revenue within the bins to help inspect
whether the observed relationship is consistent with diminishing returns.

The binned analysis is exploratory. With only 36 monthly observations,
each channel has a limited number of observations, and some spend levels
may have relatively few data points.

Results:

`outputs/tables/m2/diminishing_returns_bins.csv`

The fitted curves and binned summaries should be interpreted together.
Neither one alone confirms an optimal advertising budget.

## 7. Bounded Parameter Candidates for M3 and M4

The analysis exports candidate parameter values intended to help M3 and
M4 constrain their search space.

The current candidate settings are:

| Parameter | Candidate range or values |
|---|---|
| Adstock decay | 0.0, 0.3, 0.5, 0.7 |
| Hill slope | 1.0, 1.5, 2.0 |
| Hill half-saturation | Channel-specific positive-spend quantiles |
| Negative-exponential scale | Channel-specific positive-spend quantiles |

The exported `parameter_candidates.csv` file contains the candidate
combinations and associated bounds. The minimum and maximum
half-saturation values depend on the channel's observed positive-spend
distribution.

M3 and M4 should use the bounds for the relevant channel rather than
applying one channel's half-saturation bounds to every channel.

These values are intended as bounded exploratory candidates. They are not
final calibrated parameters and should be validated using appropriate
time-ordered evaluation and model diagnostics.

Results:

`outputs/tables/m2/parameter_candidates.csv`

## 8. Limitations

The analysis has several limitations:

1. The dataset contains only 36 monthly observations.
2. Linear residualisation controls for the selected seasonal and promotion
   variables but does not eliminate every possible confounder.
3. Correlation does not establish causation.
4. Multiple channels and lag comparisons increase the chance of observing
   apparently strong relationships by chance.
5. Saturation curve fits may be weak, even when the fitting procedure
   completes successfully.
6. Candidate decay, half-saturation, slope and scale values require further
   validation before they are used for budget recommendations.
7. The results should not be interpreted as proof of the causal return on
   advertising spend.

Subsequent modelling should evaluate candidate parameters using
time-ordered validation, appropriate diagnostics and sensitivity analysis.

## 9. Output Inventory

The M2 analysis produces the following tables:

- `outputs/tables/m2/adstock_summary.csv`
- `outputs/tables/m2/analysis_metadata.csv`
- `outputs/tables/m2/best_time_delay_by_channel.csv`
- `outputs/tables/m2/diminishing_returns_bins.csv`
- `outputs/tables/m2/diminishing_returns_fits.csv`
- `outputs/tables/m2/parameter_candidates.csv`
- `outputs/tables/m2/saturation_curves.csv`
- `outputs/tables/m2/time_delay_correlations.csv`
- `outputs/tables/m2/time_delay_correlation_matrix.csv`

Figures:

- `outputs/figures/m2/saturation_curves.png`
- `outputs/figures/m2/time_delay_correlations.png`

## 10. Conclusion

M2 implements geometric adstock, an optional delayed-peak transformation,
two saturation functions, adjusted time-delay correlations over lags zero
through four, binned diminishing-returns analysis and bounded parameter
candidate generation.

The outputs provide exploratory evidence and parameter candidates for
subsequent modelling. They do not establish causal advertising effects or
identify a confirmed optimal budget allocation.

M3 and M4 should use the channel-specific candidate bounds as starting
constraints and evaluate their suitability using model validation and
diagnostics.
