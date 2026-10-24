# M1: Data Preparation and Exploratory Data Analysis

## 1. Scope and status

M1 prepares and explores the monthly synthetic dataset for the
Marketing Mix Modelling pipeline.

Completed:
- Loading the raw data and data dictionary.
- Validating, sorting and cleaning the dataset.
- Creating time features, channel spend shares and log revenue.
- Exporting the cleaned dataset and preparation metadata.
- Screening and reviewing potential outliers.
- Exploring revenue, channel spending and allocation patterns.
- Examining correlations and channel multicollinearity.
- Conducting exploratory stationarity tests.
- Calculating year-on-year revenue growth.
- Running the local automated test suite: 20 tests passed.

Repository integration remains subject to dependency updates,
GitHub Actions verification and pull request review.

## 2. Dataset and analysis periods

Source: `data/raw/Dataset.xlsx`

The workbook contains:
- `MMM_Data`: 36 rows and 25 original columns.
- `Data_Dictionary`: column descriptions and units.

The dataset covers January 2023 to December 2025.
Its context is national FMCG packaged-food sales in Sri Lanka.

Advertising spend and sales revenue are expressed in LKR.

The six advertising channels are:
- TV
- Facebook and Instagram
- Google Search
- YouTube
- Radio
- Outdoor

All observations are synthetic. Findings are illustrative and
must not be presented as evidence of actual market performance.

### Analysis split

Data validation and the cleaned export cover all 36 months.

Exploratory analysis uses the first 30 months:
January 2023 to June 2025.

The final six months, July to December 2025, are reserved for
holdout evaluation and excluded from exploratory statistics,
outlier thresholds, charts and diagnostic tests.

The modelling team should preserve this holdout when selecting
features and tuning models.

## 3. Validation

Validation checks include:
- Required columns are present.
- Exactly 36 observations are available.
- Required raw columns contain no missing values.
- Dates represent unique, consecutive month starts.
- Numeric columns contain numeric, finite values.
- Channel spend and total advertising spend are non-negative.
- Revenue is positive before calculating its logarithm.
- Reported total spend matches the channel sum within LKR 5.
- Metadata columns are constant.
- Every observation is marked SYNTHETIC.

The supplied dataset passed validation.

The maximum absolute difference between reported total
advertising spend and the sum of channel spends was LKR 2,
within the LKR 5 validation tolerance.

Validation results are recorded in:
`data/processed/mmm_metadata.json`

Outlier screening is a separate exploratory check.

## 4. Cleaning decisions

The original Excel workbook remains unchanged.

The cleaning function:
1. Validates the input.
2. Copies the input DataFrame.
3. Converts the month column to datetime.
4. Sorts observations chronologically.
5. Removes constant columns from the cleaned table.
6. Creates the derived features.

No observations were removed, imputed or capped.
Original spend, revenue and control values were preserved.

The following constant columns were removed from the cleaned
table and recorded in metadata:
- `country`
- `brand_category`
- `region_scope`
- `data_type`

The cleaned dataset contains 36 rows and 31 columns.

Dates are exported as YYYY-MM-DD. Downstream code should parse
the month column as datetime when reading the CSV.

## 5. Derived features

| Feature | Definition |
| --- | --- |
| `t` | Chronological index from 0 to 35 |
| `year` | Calendar year |
| `month_num` | Calendar month from 1 to 12 |
| `<channel>_share` | Channel spend divided by the sum of six channel spends |
| `log_revenue` | Natural logarithm of sales revenue |

Share columns append `_share` to each original channel-spend
column name.

Shares are stored as fractions and sum to 1 per month, subject
to floating-point precision.

The denominator uses the sum of channel spends rather than the
reported total, which can differ slightly within tolerance.

The current implementation rejects months with zero total
channel spend because their spend shares would be undefined.

Year-on-year growth is exported separately because its first
12 observations have no previous-year comparison. This preserves
the cleaned dataset's no-missing-values contract.

## 6. Outlier screening and review

The 1.5 × IQR rule was applied separately to revenue, reported
total spend, six channel spends and eight controls using only
the 30 exploratory observations.

For each variable:
- Lower threshold = Q1 − 1.5 × IQR.
- Upper threshold = Q3 + 1.5 × IQR.

One value was flagged:

| Month | Variable | Observed value | Upper threshold |
| --- | --- | ---: | ---: |
| September 2023 | Google Search spend | LKR 5,096,864 | LKR 4,567,065.13 |

No values were flagged in the other 15 variables screened.

The flagged observation was reviewed and retained unchanged.
An IQR flag alone does not establish a data error. No source
evidence supports correction or removal, and its cause is unknown.

The review decision is recorded in `outlier_flags_train.csv`.

This screening does not adjust for seasonality or time trends.
Absence of an IQR flag does not establish absence of anomalies.

## 7. Revenue and spending patterns

Monthly revenue fluctuates without a consistently increasing
pattern across the exploratory period.

Advertising spend is generally lower during much of 2024 than
in 2023 and increases again in early 2025.

Revenue and advertising spend do not move together in every
month. These plots cannot separate advertising effects from
seasonality, promotions, economic conditions or carry-over.

TV and Facebook/Instagram generally receive larger allocations
than the other individual channels. Radio spending is
comparatively low.

TV and Facebook/Instagram together frequently account for
approximately half of monthly channel expenditure.

Spend levels and shares describe budget allocation, not
effectiveness or ROI.

## 8. Calendar-month revenue patterns

April and December have the highest calendar-month mean revenues
in the exploratory sample, both approximately LKR 96 million.

January to June each have three observations.
July to December each have two observations.

The chart displays individual observations alongside the means
to make the small sample sizes visible.

These are descriptive calendar-month patterns, not isolated
seasonal effects. They have not been adjusted for promotions,
trend, advertising spend or other controls.

## 9. Correlation analysis

Pearson correlations were calculated using revenue, the six raw
channel spends and eight controls.

Reported total spend, derived features and leakage columns
were excluded from the correlation matrix.

Same-month channel–revenue correlations were:

| Channel | Pearson correlation |
| --- | ---: |
| TV | 0.359 |
| Facebook / Instagram | 0.307 |
| Google Search | 0.360 |
| YouTube | 0.308 |
| Radio | -0.055 |
| Outdoor | 0.432 |

Outdoor has the largest raw same-month correlation with revenue.
Radio's correlation is close to zero.

These associations do not adjust for controls, carry-over,
saturation or relationships between channels.

They are not causal effects or ROI estimates and must not be
used alone to rank channel effectiveness. Radio's near-zero
correlation does not establish that radio advertising is
ineffective or harmful.

## 10. Channel multicollinearity

Variance Inflation Factors were calculated for the six raw
channel-spend variables using the 30 exploratory observations.

The auxiliary regressions included an intercept. Inputs were
standardised for numerical conditioning.

| Channel | VIF |
| --- | ---: |
| TV | 2.255 |
| YouTube | 2.073 |
| Outdoor | 2.002 |
| Radio | 1.788 |
| Google Search | 1.684 |
| Facebook / Instagram | 1.534 |

These results do not suggest severe linear multicollinearity
within this channel-only predictor set.

No channels were removed based on this diagnostic.

Controls, total spend and derived shares were excluded.
M3 and M4 should reassess multicollinearity using their actual
model inputs, including controls and transformed channel spends.

## 11. Exploratory stationarity tests

ADF and KPSS tests were applied to revenue and the six raw
channel spends using the first 30 months.

Specification:
- Both tests include an intercept without a deterministic trend.
- ADF selects a lag from 0 to 3 using AIC.
- KPSS uses automatic bandwidth selection.
- Decisions use a nominal 5% significance level.
- Multiple-testing adjustments were not applied.

ADF tests the unit-root null hypothesis.
KPSS tests the level-stationarity null hypothesis.

| Variables | Interpretation under the selected specifications |
| --- | --- |
| Revenue, Google Search, Radio, Outdoor | ADF rejects a unit root; KPSS does not reject level stationarity. Results are consistent with level stationarity. |
| TV, Facebook / Instagram, YouTube | Neither test rejects its null. Results are inconclusive. |

All KPSS p-values exceed the upper tabulated bound of 0.10.
They are reported as bounds rather than exact p-values.

Failure to reject a null hypothesis does not prove it is true.
The small sample and sensitivity to lag choices, seasonal
variation and structural changes limit these results.

No automatic differencing or variable removal was performed.
These tests do not validate the final MMM.

## 12. Year-on-year revenue growth

Year-on-year growth is calculated as:

`100 × (current revenue / revenue 12 months earlier − 1)`

The monthly sequence is checked before using a 12-row shift.

The exploratory dataset provides 18 comparisons covering
January 2024 to June 2025.

The first 12 months have no previous-year observations.
Their growth values remain missing and are not replaced with zero.

Findings:
- Nine comparisons show positive growth.
- Nine comparisons show negative growth.
- Largest increase: January 2024, approximately +27.89%.
- Largest decline: August 2024, approximately −20.04%.

These changes describe total revenue movements, not incremental
advertising effects. Comparing the same calendar month does not
remove changes in prices, promotions, economic conditions or spend.

The output is stored separately in:
`outputs/tables/m1/revenue_yoy_growth_train.csv`

The cleaned modelling dataset retains all 36 observations.
Revenue-derived growth must not be used as a contemporaneous
predictor of revenue.

## 13. Deliverables

### Code and cleaned data

| File | Purpose |
| --- | --- |
| `src/data_prep.py` | Loading, validation, cleaning and feature creation |
| `data/processed/mmm_clean.csv` | Cleaned dataset: 36 rows and 31 columns |
| `data/processed/mmm_metadata.json` | Source, constants, validation and preparation notes |
| `tests/test_data_prep.py` | Data-preparation tests |
| `notebooks/01_m1_data_eda.ipynb` | Exploratory analysis and visualisation |
| `report/sections/01_data_eda.md` | M1 methods, findings and limitations |

### Tables

Location: `outputs/tables/m1/`

- `summary_statistics_train.csv`
- `outlier_thresholds_train.csv`
- `outlier_flags_train.csv`
- `seasonality_summary_train.csv`
- `monthly_spend_mix_train.csv`
- `pearson_correlations_train.csv`
- `channel_revenue_correlations_train.csv`
- `channel_vif_train.csv`
- `stationarity_tests_train.csv`
- `revenue_yoy_growth_train.csv`

### Figures

Location: `outputs/figures/m1/`

- `01_revenue_spend_trends_train.png`
- `02_monthly_revenue_pattern_train.png`
- `03_channel_spend_trends_train.png`
- `04_channel_spend_mix_train.png`
- `05_correlation_heatmap_train.png`
- `06_spend_revenue_scatter_train.png`
- `07_channel_vif_train.png`
- `08_revenue_yoy_growth_train.png`

## 14. Automated testing and reproduction

The local test suite completed with 20 passing tests:
- Six existing repository tests.
- Fourteen additional data-preparation test cases.

Tests cover sorting, preservation of inputs and measurements,
derived features, missing data, duplicate or missing months,
negative spend, spend-reconciliation tolerance, zero revenue
and zero total channel spend.

This result describes local execution.
GitHub Actions execution has not yet been verified.

From the repository root, run:

```powershell
python -m src.data_prep
python -m pytest -q
```

Then open `notebooks/01_m1_data_eda.ipynb`, select the project's
Python environment, restart the kernel and run all cells.

Data preparation and EDA use:
- pandas
- NumPy
- openpyxl
- Matplotlib
- Seaborn
- Statsmodels
- IPython and ipykernel for notebook execution

Testing uses pytest. Notebook output stripping uses nbstripout.

At the time of this report, `requirements.txt` lists only pytest
and nbstripout. Runtime and notebook dependencies still need to
be declared in coordination with M8.

## 15. Handover recommendations and limitations

- Use the cleaned 36-month dataset as the shared downstream input.
- Preserve the final six months for holdout evaluation.
- Use chronological validation within the training period.
- Fit data-dependent transformations within each training fold.
- Retain the reviewed Google Search observation; its IQR flag
  alone does not support removal.
- Keep models simple given the small sample.
- Reassess diagnostics using the final model inputs.
- Do not estimate ROI or recommend budget changes from EDA alone.
- Treat all findings as illustrative because the data is synthetic.

Impressions, clicks, conversions and units sold are retained
for reporting but must not be used as revenue predictors.

Log revenue is a target transformation, not a revenue predictor.
Revenue-derived growth must also be excluded as a contemporaneous
revenue predictor.

Do not include total advertising spend alongside all six channel
spends as an additional independent predictor.

Channel shares describe allocation. They should not be added
automatically alongside spend variables.

M2–M6 should estimate delay, saturation, channel contributions,
uncertainty and budget recommendations through the modelling
workflow rather than infer them from raw correlations.

Outstanding integration tasks:
- Update preparation metadata notes that still describe completed
  EDA work as pending.
- Coordinate dependency declarations and pipeline integration
  with M8.
- Verify GitHub Actions.
- Submit the M1 branch for team-leader review.




## 16. Tested local dependency versions

The following versions were installed in the local environment
used for the successful notebook execution and 20 passing tests.

```text
pandas==3.1.0
numpy==2.4.6
openpyxl==3.1.5
matplotlib==3.11.2
seaborn==0.13.2
statsmodels==0.15.0
ipython==9.17.1
ipykernel==7.4.0
pytest==9.1.1
nbstripout==0.9.1
```

This list records direct dependencies and their tested local
versions; it is not a complete environment lock file.

M8 should incorporate the required dependencies into the shared
environment, reconcile them with other workstreams and verify
installation and tests in GitHub Actions.