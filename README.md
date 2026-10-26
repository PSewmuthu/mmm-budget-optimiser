# Marketing Mix Modelling and Budget Optimisation

Measure how advertising spend across six channels drives sales, quantify diminishing returns, time delays and external factors,
and find the budget split that maximises return under a fixed budget. Sri Lanka, FMCG packaged foods, 36 months (synthetic data).

**Tools:** Python, Pandas, NumPy, Scikit-learn, SciPy, Statsmodels, Matplotlib, Seaborn, Power BI.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q        # run test cases
python run_all.py        # runs the pipeline stages merged so far
```

## Project deliverables

- [x] Cleaned dataset and exploratory analysis (M1)
- [ ] Working Marketing Mix Model (M2, M3, M4)
- [ ] Channel effectiveness and ROI analysis (M5)
- [ ] Budget optimisation algorithm (M6)
- [ ] Interactive Power BI dashboard (M7)
- [ ] Final report with data-driven recommendations (M8)

## Team and progress

Status values: Not started / In progress / In review / Merged. Each member updates their own row in their PRs.

| ID  | Workstream                                 | Branch            | Status      | Key output                                                       | Notes                                                              |
| --- | ------------------------------------------ | ----------------- | ----------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| M1  | Data cleaning and EDA                      | `m1/data-eda`     | Completed   | `data/processed/mmm_clean.csv`, `src/data_prep.py`, EDA notebook | Cleaning and metadata exported; 20 local tests pass; EDA completed |
| M2  | Adstock, saturation, time delays           | `m2/transforms`   | Completed   | `src/transforms.py`, lag and saturation findings                 | Data transformations completed and have passed all 59 tests        |
| M3  | Primary MMM (sklearn + SciPy)              | `m3/mmm-core`     | Not started | `src/mmm.py`, model fit, CV results                              |                                                                    |
| M4  | Statsmodels benchmark and external factors | `m4/statsmodels`  | Not started | diagnostics, factor effects                                      |                                                                    |
| M5  | Channel effectiveness and ROI              | `m5/roi`          | Not started | `outputs/powerbi/channel_roi.csv` etc.                           |                                                                    |
| M6  | Budget optimisation                        | `m6/optimiser`    | Not started | `src/optimiser.py`, scenarios                                    |                                                                    |
| M7  | Power BI dashboard                         | `m7/dashboard`    | Not started | `dashboard/*.pbix`                                               |                                                                    |
| M8  | Report and integration                     | `m8/report-infra` | Not started | `report/`, `run_all.py`, CI                                      |                                                                    |
| M9  | Review and merge only                      | `master`          | n/a         |                                                                  |                                                                    |

## Milestones

| Phase                    | Gate                                                   | Target date | Done |
| ------------------------ | ------------------------------------------------------ | ----------- | ---- |
| 1. Foundations           | Clean dataset and `config.py` merged                   |             | ✅   |
| 2. Modelling             | MMM and benchmark merged, signs and fit reviewed       |             | [ ]  |
| 3. Insight and decisions | ROI tables, scenario grid merged; dashboard reads them |             | [ ]  |
| 4. Report and polish     | Report numbers reproduced by `make all`                |             | [ ]  |

## Key results (fill in as they land)

| Item                                       | Result | Source file |
| ------------------------------------------ | ------ | ----------- |
| Model holdout MAPE / R²                    |        |             |
| Best / worst channel ROI                   |        |             |
| Estimated carry-over (adstock) per channel |        |             |
| Recommended budget mix and expected uplift |        |             |

## Repo layout

```
data/raw/            original Excel dataset (never edited)
data/processed/      cleaned data (M1)
src/                 config, data prep, transforms, model, optimiser
notebooks/           EDA and analysis notebooks
outputs/             figures, model files, Power BI tables
dashboard/           Power BI file (M7 only)
report/sections/     one markdown section per member; M8 assembles the report
tests/               pytest suite (runs in CI)
```

## How we work

1. `git checkout master && git pull`, then branch: `git checkout -b m<N>/<short-name>` (e.g. `m3/mmm-core`).
2. Edit only your own files (ownership = the table above). Need a change elsewhere? Open a GitHub issue.
3. Run `python -m pytest -q` before every push. Keep PRs small; rebase first: `git pull --rebase origin master`.
4. Open a PR and update your README row in the same PR.
5. M9 reviews and squash-merges. Nobody pushes to `master` or merges their own PR.
6. Notebooks have outputs stripped automatically (nbstripout); keep logic in `src/`. Only M7 edits the binary `.pbix`.

## Shared contracts

Changing any of these needs an issue.

| Contract                       | Owner  | Definition                                                                                                                                                                      |
| ------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/processed/mmm_clean.csv` | M1     | 36 monthly rows, `month` as date, snake_case columns as raw, spend in LKR, constant columns removed                                                                             |
| `src/config.py`                | M1     | `CHANNELS`, `CONTROLS`, `TARGET`, `LEAKAGE_COLUMNS`, paths, `SEED = 42`, default bounds                                                                                         |
| `src/transforms.py`            | M2     | `adstock(x, decay)`, `hill(x, half_sat, slope)`                                                                                                                                 |
| `src/mmm.py`                   | M3     | `MMMModel`: `fit`, `predict`, `response_curve`, `contributions`                                                                                                                 |
| `outputs/powerbi/*.csv`        | M5, M6 | `monthly_actual_vs_pred`, `contributions`, `channel_roi`, `response_curves`, `lag_profile`, `external_factors`, `optimisation_scenarios`; column names agreed in an issue first |

Rules: `impressions`, `clicks`, `conversions`, `units_sold` are never predictors (leakage). Time-ordered validation only. All randomness uses `config.SEED`.
Definitions: **ROI** = incremental revenue (LKR) / ad spend (LKR); **marginal ROI** = slope of the channel response curve at current spend; with a fixed budget, maximising revenue equals maximising ROI.

## Known limitations

- Only 36 monthly observations: results show direction and ranking, not precise ROI; report intervals.
- Dataset is synthetic; findings are illustrative.
- Impressions, clicks and conversions exist only as totals (not per channel); `total_ad_spend_lkr` equals the channel sum within LKR 2.
- ROI is revenue-based (no margin data); model shows association, not proven causation.
