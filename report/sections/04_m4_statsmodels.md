# M4 — Benchmark Marketing Mix Modelling and External Factors (Statsmodels)

## 1. Objective

The objective of the M4 workstream was to develop an interpretable statistical benchmark for Marketing Mix Modelling (MMM) using Statsmodels. The benchmark model evaluates the relationship between advertising expenditure and sales revenue while accounting for external economic and business factors.

The implementation also investigates delayed advertising relationships, evaluates regression assumptions, analyses the influence of external factors, and compares the benchmark against the primary Marketing Mix Model developed in M3.

The main objectives were to:

- Develop a log-log Ordinary Least Squares (OLS) benchmark with Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard errors.
- Evaluate regression assumptions using statistical diagnostic tests and residual visualisations.
- Investigate delayed relationships between advertising expenditure and revenue using distributed-lag modelling.
- Estimate external-factor effects with confidence intervals.
- Perform ablation analysis to evaluate the importance of external control variables.
- Compare the benchmark against M3's primary Marketing Mix Model and identify disagreements.

## 2. Dataset Description

The analysis used the cleaned synthetic marketing dataset prepared by M1, containing 36 consecutive monthly observations from January 2023 to December 2025.

The dataset includes advertising expenditure across six marketing channels:

1. Television (TV)
2. Facebook/Instagram
3. Google Search
4. YouTube
5. Radio
6. Outdoor advertising

The target variable was monthly sales revenue (`sales_revenue_lkr`).

Eight external control variables were included:

| External Factor | Dataset Variable |
|---|---|
| Average price | `average_price_index` |
| Promotion discounts | `promotion_discount_pct` |
| Competitor pressure | `competitor_pressure_index_0_100` |
| Economic sentiment | `economic_sentiment_index_0_100` |
| Exchange rate | `lkr_usd_exchange_rate` |
| Inflation | `inflation_proxy_index` |
| Seasonal demand | `seasonal_demand_index` |
| Public holidays | `public_holiday_count` |

The cleaned dataset contained 31 columns with no missing values.

A chronological data split was used to prevent random mixing of past and future observations:

- Training dataset: First 30 months (January 2023–June 2025).
- Holdout dataset: Final six months (July 2025–December 2025).

## 3. Benchmark Model Methodology

### 3.1 Log-Log OLS Regression

A log-log Ordinary Least Squares (OLS) regression model was implemented using Statsmodels.

The model uses the natural logarithm of sales revenue as its dependent variable and the natural logarithms of advertising expenditure across the six channels as marketing predictors.

The external control variables were included in their original units.

The general model specification is:

**ln(Revenue) = β₀ + Σ βᵢ ln(Channel Spendᵢ) + Σ γⱼ External Factorⱼ + ε**

Where:

- β₀ represents the intercept.
- βᵢ represents the advertising-channel elasticity coefficients.
- γⱼ represents the coefficients of external control variables.
- ε represents the regression error term.

Advertising-channel coefficients can be interpreted as conditional elasticities. For example, a channel coefficient of 0.20 indicates that a 1% increase in spending is associated with approximately a 0.20% increase in revenue, holding other model variables constant.

### 3.2 Newey-West HAC Standard Errors

Newey-West HAC standard errors were used with a maximum lag of two months.

This approach provides standard errors that are robust to certain forms of heteroskedasticity and serial correlation.

HAC covariance estimation improves statistical inference under these conditions but does not eliminate multicollinearity, omitted-variable bias, or overfitting.

## 4. Benchmark Model Results

The benchmark model was fitted using 30 training observations.

| Performance Metric | Result |
|---|---:|
| Training R² | 0.8602 |
| Adjusted R² | 0.730 |
| Training observations | 30 |
| Model residual degrees of freedom | 15 |
| AIC | −86.20 |
| BIC | −65.18 |
| Covariance estimator | HAC |
| HAC maximum lags | 2 |

The model achieved a training R² of approximately 0.86, indicating that the included predictors explained approximately 86% of the variation in log-transformed sales revenue within the training sample.

However, the adjusted R² was lower, reflecting the relatively large number of explanatory variables compared with the available observations.

### 4.1 Advertising-Channel Coefficients

| Advertising Channel | Coefficient | p-value | Interpretation |
|---|---:|---:|---|
| TV | 0.0201 | 0.565 | Positive, not statistically significant |
| Facebook/Instagram | −0.0777 | 0.242 | Negative, not statistically significant |
| Google Search | 0.1920 | 0.001 | Positive, statistically significant |
| YouTube | 0.1190 | 0.055 | Positive, borderline significance |
| Radio | −0.2004 | 0.003 | Negative, statistically significant |
| Outdoor | −0.0013 | 0.976 | Near-zero, not statistically significant |

Google Search showed a statistically significant positive relationship with sales revenue. Its coefficient of approximately 0.192 suggests that a 1% increase in Google Search expenditure was associated with approximately a 0.192% increase in revenue, holding the other included factors constant.

Radio advertising produced a statistically significant negative coefficient. However, this result does not establish that radio advertising causes revenue to decline. Correlated spending decisions, omitted factors, model specification, and the small training sample may influence this estimate.

The remaining advertising channels showed limited statistical evidence of individual effects in this benchmark specification.

## 5. Model Diagnostic Analysis

Several diagnostic tests were conducted to evaluate the statistical properties of the fitted regression model.

### 5.1 Variance Inflation Factor (VIF)

VIF was calculated to identify multicollinearity among the explanatory variables.

The most important results were:

| Variable | VIF |
|---|---:|
| Inflation proxy index | 79.37 |
| Average price index | 70.13 |
| Economic sentiment index | 6.35 |
| LKR/USD exchange rate | 4.47 |
| Other advertising-channel predictors | Approximately 2.35–3.00 |

The inflation proxy and average price index showed extremely high VIF values, indicating severe multicollinearity.

High multicollinearity can make estimated coefficients sensitive to small changes in the data, increase uncertainty, and complicate the interpretation of individual factor effects.

Consequently, the individual price and inflation coefficients should be interpreted with particular caution.

### 5.2 Durbin-Watson Test

The Durbin-Watson statistic was:

**Durbin-Watson = 2.0378**

A value close to two suggests that there was no strong indication of first-order residual autocorrelation in the fitted model.

### 5.3 Ljung-Box Test

The Ljung-Box test was performed at lags 1, 2, and 4.

| Lag | Test Statistic | p-value |
|---|---:|---:|
| 1 | 0.2611 | 0.6093 |
| 2 | 1.1134 | 0.5731 |
| 4 | 1.5372 | 0.8200 |

All tested p-values were greater than 0.05.

Therefore, the test did not identify statistically significant residual autocorrelation at the examined lags.

### 5.4 Breusch-Pagan Test

The Breusch-Pagan test was conducted to assess whether the regression residual variance was constant.

| Statistic | Result |
|---|---:|
| LM statistic | 19.6298 |
| LM p-value | 0.1423 |
| F-test p-value | 0.0933 |

The p-values were greater than 0.05, so the test did not provide statistically significant evidence of heteroskedasticity at the 5% significance level.

However, the relatively small training sample limits the statistical power of the diagnostic tests.

### 5.5 Residual Visualisations

Two residual diagnostic plots were generated:

- Residuals versus fitted values
- Normal Q-Q plot

These visualisations support inspection of systematic residual patterns, unusual observations, and departures from distributional assumptions.

The figures are saved under `outputs/figures/m4/`.

## 6. Distributed-Lag Analysis

A distributed-lag regression was implemented to investigate whether advertising expenditure is associated with sales revenue in the current month or subsequent months.

The model used the natural logarithm of total advertising expenditure at three different time points:

- Lag 0: Current-month advertising expenditure.
- Lag 1: Advertising expenditure from the previous month.
- Lag 2: Advertising expenditure from two months earlier.

The first two observations were excluded because the necessary lag history was unavailable. The lag model was trained using 28 observations while maintaining the final six months as holdout data.

### 6.1 Distributed-Lag Results

| Advertising Lag | Coefficient | p-value | 95% Confidence Interval |
|---|---:|---:|---|
| Lag 0 | 0.1099 | 0.107 | [−0.0236, 0.2434] |
| Lag 1 | 0.1137 | 0.135 | [−0.0356, 0.2629] |
| Lag 2 | −0.0988 | 0.032 | [−0.1893, −0.0083] |

The distributed-lag model achieved a training R² of 0.816 and an adjusted R² of 0.689.

The current-month and one-month-lag coefficients were positive but not statistically significant at the 5% level.

The two-month-lag coefficient was negative and statistically significant in this specification.

However, the model does not establish that advertising causes a reduction in sales after two months. The result may reflect correlations between variables, the limited number of observations, or model instability.

The distributed-lag implementation uses total advertising expenditure rather than separate lag terms for each advertising channel. Therefore, it provides aggregate evidence about delayed relationships and does not identify channel-specific carryover effects.

## 7. External-Factor Analysis

The benchmark model estimated the relationships between eight external factors and sales revenue.

HAC-based 95% confidence intervals were extracted for each factor. Estimated revenue percentage changes were calculated using meaningful changes in each variable.

### 7.1 External-Factor Results

| External Factor | Variable Change | Estimated Revenue Change | p-value |
|---|---|---:|---:|
| Average price index | +0.01 index points | −0.98% | 0.230 |
| Promotion discount | +1 percentage point | +0.66% | <0.001 |
| Competitor pressure | +1 index point | −0.01% | 0.861 |
| Economic sentiment | +1 index point | +0.50% | 0.004 |
| Exchange rate | +1 LKR/USD | +0.31% | 0.376 |
| Inflation proxy | +0.01 index points | +0.73% | 0.024 |
| Seasonal demand | +0.10 index points | +3.85% | 0.006 |
| Public holidays | +1 day | +0.14% | 0.799 |

### 7.2 Significant External Factors

The benchmark identified positive and statistically significant associations for promotion discounts, economic sentiment, inflation proxy, and seasonal demand.

**Promotion discounts:** A one-percentage-point increase in the promotion discount was associated with approximately 0.66% higher revenue, with a 95% confidence interval of approximately 0.40% to 0.92%.

**Economic sentiment:** A one-point increase in economic sentiment was associated with approximately 0.50% higher revenue.

**Seasonal demand:** A 0.10-point increase in the seasonal demand index was associated with approximately 3.85% higher revenue.

**Inflation proxy:** The model estimated a positive relationship between the inflation proxy and revenue. However, the severe multicollinearity involving inflation and average price reduces the reliability of the individual coefficient interpretation.

These results represent conditional statistical associations and should not be treated as causal estimates.

## 8. Ablation Analysis

An ablation experiment was performed to examine the importance of external control variables.

Two models were compared:

- Full benchmark model: Six advertising channels and eight external controls.
- Reduced benchmark model: Six advertising channels without external controls.

Both models used the same training and holdout periods.

### 8.1 Model Performance Comparison

| Performance Metric | With External Controls | Without External Controls |
|---|---:|---:|
| Training R² (log revenue) | 0.8602 | 0.4430 |
| Holdout MAPE | 6.64% | 8.23% |
| Holdout NRMSE | 0.0879 | 0.0969 |
| Holdout R² | −1.1830 | −1.6499 |

Including external controls improved training fit and reduced holdout prediction error.

However, both models produced negative holdout R² values, indicating limited performance in explaining the variation in unseen observations.

### 8.2 Advertising-Coefficient Comparison

| Channel | With Controls | Without Controls |
|---|---:|---:|
| TV | 0.0201 | 0.0873 |
| Facebook/Instagram | −0.0777 | 0.0302 |
| Google Search | 0.1920 | 0.1501 |
| YouTube | 0.1190 | 0.0669 |
| Radio | −0.2004 | −0.3406 |
| Outdoor | −0.0013 | 0.1176 |

The results showed that excluding external factors changed the estimated advertising coefficients.

Facebook/Instagram and Outdoor advertising changed coefficient signs when external controls were removed.

Some channel coefficients increased after excluding controls, while others decreased. Therefore, omitted controls did not systematically overstate every channel's estimated relationship with revenue.

The ablation results demonstrate that advertising-channel estimates are sensitive to the inclusion of external factors.

## 9. Comparison with the Primary MMM (M3)

The Statsmodels benchmark developed in M4 was compared against the primary MMM developed in M3.

Both models used the first 30 monthly observations for training and the final six months for holdout evaluation.

### 9.1 Model Approaches

| Characteristic | M3 Primary MMM | M4 Benchmark MMM |
|---|---|---|
| Regression approach | Constrained Ridge regression | Log-log OLS |
| Library | Scikit-learn/SciPy | Statsmodels |
| Advertising transformation | Adstock and Hill saturation | Logarithmic transformation |
| Coefficient constraints | Non-negative marketing effects | Unconstrained coefficients |
| Parameter optimisation | SciPy differential evolution | OLS estimation |
| Main contribution | Marketing response modelling and prediction | Statistical interpretation and validation |

M3 incorporates advertising carryover and diminishing-return effects through adstock and saturation transformations.

M4 provides an interpretable benchmark through elasticity-style coefficients, statistical diagnostics, confidence intervals, and external-factor analysis.

### 9.2 Prediction Performance Comparison

| Metric | M3 Primary MMM | M4 Statsmodels |
|---|---:|---:|
| Holdout MAPE | 6.1774% | 6.6362% |
| Holdout R² | −0.4173 | −1.1830 |
| Holdout NRMSE | 7.0833% | 8.7908% |

M3 achieved better holdout prediction performance across all three evaluation metrics.

Its MAPE was approximately 0.46 percentage points lower than M4.

However, both models had negative holdout R² values. This indicates that neither model performed better than the holdout-mean baseline under the R² metric.

Given that only six holdout observations were available, these results should be interpreted cautiously.

### 9.3 Advertising-Channel Direction Comparison

| Advertising Channel | M3 Direction | M4 Direction | Agreement |
|---|---|---|---|
| TV | Positive | Positive | Yes |
| Facebook/Instagram | Positive | Negative | No |
| Google Search | Positive | Positive | Yes |
| YouTube | Positive | Positive | Yes |
| Radio | Positive | Negative | No |
| Outdoor | Positive | Negative | No |

Both models produced positive advertising relationships for TV, Google Search, and YouTube.

Directional disagreements occurred for Facebook/Instagram, Radio, and Outdoor advertising.

These differences are partly explained by the model specifications. M3 enforces non-negative marketing coefficients, while M4 allows coefficients to take positive or negative values.

Additionally, M3 uses adstock and saturation transformations, whereas M4 uses logarithmic spending variables.

The Facebook/Instagram and Outdoor coefficients were not statistically significant in M4. Consequently, their negative estimated directions should not be regarded as strong evidence of genuinely negative advertising effects.

Radio had a statistically significant negative coefficient in M4, but this observational result does not prove a negative causal advertising effect.

Raw model coefficients were not compared numerically because the two models use different feature transformations and coefficient scales.

### 9.4 Comparison Conclusion

M3 demonstrated stronger holdout prediction performance, while M4 provided additional statistical evidence and interpretability.

The benchmark helped identify potential model instability, influential external factors, and differences in advertising-channel relationships.

The two models therefore provide complementary information for evaluating marketing performance and supporting later channel-effectiveness and budget-optimisation analyses.

## 10. Implementation and Testing

The M4 implementation was developed using Python, Pandas, NumPy, Statsmodels, and Matplotlib.

The main implementation files are:

| File | Purpose |
|---|---|
| `src/m4_benchmark.py` | Benchmark OLS, diagnostics, distributed lags, external-factor analysis, ablation, and exports |
| `src/m4_m3_comparison.py` | M3 versus M4 performance and channel-direction comparison |
| `tests/test_m4_benchmark.py` | Automated tests for M4 functionality and comparison |

The generated results are stored in M4-specific output folders:

- `outputs/tables/m4/` — Benchmark coefficients, model summaries, diagnostic statistics, external-factor effects, distributed-lag results, ablation results, and M3 comparison tables.
- `outputs/figures/m4/` — Residual diagnostic visualisations.

### 10.1 Automated Testing

Seven M4-specific automated tests were implemented, covering dataset loading, chronological splitting, benchmark estimation, external-factor analysis, ablation, distributed-lag estimation, and comparison with M3.

The M4 test suite completed successfully with:

**7 tests passed.**

The complete repository test suite also passed:

**70 tests passed.**

These tests confirm that the tested functions execute correctly and produce expected output structures. They do not independently establish the statistical validity or causal accuracy of the models.

## 11. Limitations

The M4 analysis has several important limitations.

**Small sample size:** The dataset contains only 36 monthly observations. The benchmark uses 30 training observations and 14 explanatory variables, creating a risk of overfitting and unstable estimates.

**Multicollinearity:** High VIF values were observed for the inflation proxy and average price index, making their independent effects difficult to interpret reliably.

**Negative holdout R²:** Both M3 and M4 produced negative holdout R² values, indicating limited predictive performance under this evaluation metric.

**Synthetic dataset:** The analysis uses synthetic data. Results demonstrate modelling techniques rather than verified real-world marketing relationships.

**Observational modelling:** Regression coefficients and distributed-lag relationships represent statistical associations rather than causal advertising impacts.

**Aggregate lag analysis:** The distributed-lag model uses total advertising expenditure rather than channel-specific lag variables.

**Model differences:** M3 and M4 employ different feature transformations and coefficient constraints. Therefore, their coefficients are not directly comparable.

**Prediction transformation:** M4 holdout revenue predictions were obtained by exponentiating predicted log revenue without an additional log-to-level bias correction.

**Limited holdout data:** Model comparisons were based on only six holdout months, which limits confidence in generalisation.

## 12. Conclusion

The M4 workstream successfully implemented an interpretable Statsmodels benchmark for Marketing Mix Modelling, including log-log OLS regression with Newey-West HAC standard errors, distributed-lag modelling, statistical diagnostics, external-factor analysis, and ablation experiments.

The benchmark identified positive relationships for Google Search expenditure and several external factors, while diagnostic tests highlighted severe multicollinearity between price and inflation variables.

The ablation analysis demonstrated that external controls materially influence estimated advertising coefficients and prediction performance.

Comparison against the M3 primary MMM showed that M3 achieved better holdout prediction accuracy, while M4 contributed statistical interpretation, model diagnostics, and sensitivity analysis.

Overall, M4 provides a complementary benchmark for evaluating the primary Marketing Mix Model. Its findings should be interpreted cautiously because of the synthetic dataset, limited sample size, and observational nature of the analysis.