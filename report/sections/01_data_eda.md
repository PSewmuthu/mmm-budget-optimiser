# M1: Data Preparation and Exploratory Data Analysis

## 1. Scope and current status

This workstream prepares the monthly dataset for the team's Marketing
Mix Modelling pipeline.

Completed:
- Loading the data and data dictionary.
- Raw-data validation.
- Chronological sorting and feature creation.
- Exporting the cleaned dataset and metadata.
- Adding automated data-preparation tests.

Pending:
- Outlier analysis.
- Exploratory charts and interpretation.
- Year-on-year revenue growth output.
- Channel multicollinearity analysis using VIF.
- Stationarity tests and interpretation.

## 2. Dataset

Source: `data/raw/Dataset.xlsx`

The workbook contains:
- `MMM_Data`: 36 observations and 25 original columns.
- `Data_Dictionary`: column descriptions and units.

The observations cover January 2023 to December 2025 at monthly
frequency. The context is Sri Lanka, national coverage, and FMCG
packaged foods.

The six advertising channels are:
- TV
- Facebook and Instagram
- Google Search
- YouTube
- Radio
- Outdoor

Advertising expenditure and sales revenue are expressed in LKR.

All observations are synthetic. Findings are illustrative and must
not be presented as evidence of actual market performance.

## 3. Validation

The validation function checks:
- Required columns are present.
- There are exactly 36 observations.
- Required raw columns contain no missing values.
- Dates represent 36 unique, consecutive month starts.
- Numeric columns have numeric data types and finite values.
- Channel spend and total advertising spend are non-negative.
- Revenue is positive before calculating its natural logarithm.
- Reported total spend matches the channel sum within LKR 5.
- Metadata columns are constant.
- Every observation is marked SYNTHETIC.

The supplied data passed these checks. The maximum absolute
difference between reported total advertising spend and the sum
of channel spends was LKR 2, within the agreed LKR 5 tolerance.

Source for validation results:
`data/processed/mmm_metadata.json`

Outlier assessment is separate from these validation checks and
has not yet been completed.

## 4. Cleaning decisions

The original Excel file remains unchanged.

The cleaning function:
1. Validates the input.
2. Creates a copy of the input DataFrame.
3. Converts the month column to datetime.
4. Sorts observations chronologically.
5. Removes constant columns from the modelling table.
6. Adds the agreed derived features.

No observations were removed or imputed. Original spend, revenue
and control values were preserved, including spend discrepancies
within the agreed tolerance.

The following constants were moved to metadata:
- country
- brand_category
- region_scope
- data_type

## 5. Derived features

| Feature | Definition |
| --- | --- |
| `t` | Chronological index from 0 to 35 |
| `year` | Calendar year |
| `month_num` | Calendar month from 1 to 12 |
| `<channel>_share` | Channel spend divided by the sum of all six channel spends |
| `log_revenue` | Natural logarithm of sales revenue |

Shares are stored as fractions, not percentages. The sum of the
six channel shares is 1 for each month, subject to floating-point
precision.

The channel-spend sum is used as the share denominator rather
than the reported total, which may differ slightly within tolerance.

The current implementation rejects a month with zero total
channel spend because its spend shares would be undefined.

Year-on-year revenue growth remains pending. Its first 12 values
are undefined because earlier-year observations are unavailable.
These values must not be replaced with zero. A separate EDA output
is planned to preserve the current no-missing-values CSV contract.

## 6. Output files

| File | Purpose |
| --- | --- |
| `data/processed/mmm_clean.csv` | Cleaned monthly dataset: 36 rows and 31 columns |
| `data/processed/mmm_metadata.json` | Source information, constants, validation results and preparation notes |
| `src/data_prep.py` | Reusable loading, validation and cleaning functions |
| `tests/test_data_prep.py` | Data-preparation tests |

Dates are exported to CSV as YYYY-MM-DD. Downstream code should
parse the month column as datetime when loading the CSV.

The existing `src/config.py` supplies shared paths, column lists
and the random seed.

## 7. Automated testing

The local test suite completed with 20 passing test cases:
- 6 existing repository tests.
- 14 new data-preparation test cases.

The new tests cover:
- Chronological sorting and preservation of the input DataFrame.
- Preservation of original measurements.
- Time features, spend shares and log-revenue calculations.
- Missing columns and missing values.
- Duplicate months and gaps in the monthly sequence.
- Negative advertising spend.
- Acceptance of spend differences at the LKR 5 boundary.
- Rejection of spend differences beyond that boundary.
- Zero revenue and zero total channel spend.

This result describes local execution. GitHub Actions execution
has not yet been verified.

## 8. Reproduction

Run from the repository root using the project's Python environment:

```powershell
python -m src.data_prep
python -m pytest -q
```

Data preparation requires pandas, NumPy and openpyxl.
Test execution also requires pytest.

The starter requirements file needs these runtime dependencies
to be included through coordination with M8.

## 9. Modelling safeguards and limitations

- Only 36 monthly observations are available.
- The dataset is synthetic.
- Impressions, clicks, conversions and units_sold are retained
  for reporting but must not be used as revenue predictors.
- Log revenue is a target transformation, not a revenue predictor.
- Revenue-derived growth must not be used as a contemporaneous
  revenue predictor.
- Total advertising spend must not be included alongside all six
  channel spends as an additional independent predictor.
- Correlation does not establish causal advertising effects.
- EDA, outlier review, VIF and stationarity findings will be added
  after those analyses are completed.