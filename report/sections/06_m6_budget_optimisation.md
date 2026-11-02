# M6 - Budget Optimisation

## 1. Objective

Find the split of a fixed monthly advertising budget across the six channels that maximises predicted incremental revenue.
With the total budget fixed, maximising revenue and maximising ROI are the same problem, because the denominator (total spend) does not change.

## 2. Method

- **Model used.** M3's fitted Primary MMM, through its `response_curve(channel, grid)` interface only. Revenue is additive across channels, so total ad-driven revenue is the sum of the six channel response curves at constant monthly spend (steady state, including adstock carry-over).
- **Problem.** Maximise total response subject to `sum(spend) = budget` and `low <= spend <= high` for each channel.
- **Solvers.** Multi-start SciPy SLSQP is the primary solver. SciPy differential evolution is an independent cross-check; each candidate is projected onto the feasible set so the budget and bounds hold exactly. A marginal-ROI (equimarginal) check confirms the result: channels strictly inside their bounds must share the same marginal ROI, channels at an upper bound must be at least that, and channels at a lower bound at most that.
- **Baseline for comparison.** The current mix (historical mean monthly spend per channel, LKR 18.58M in total) scaled proportionally to the scenario budget and kept inside the bounds.
- **Bounds.** Each channel may move within a fraction of its historical mean: wide (50% to 150%), medium (70% to 130%, the primary case) or narrow (80% to 120%). Narrow bounds prevent extrapolating far beyond the observed spend range.
- **Scenarios.** Budget changes of -20%, -10%, 0, +10%, +20%, +30% under each bound set. Some combinations are infeasible or leave no freedom (budget equals the sum of the bounds); they are labelled `infeasible` or `forced_by_bounds` in the outputs.
- **Uncertainty.** 200 residual block-bootstrap draws (block length 3, residuals inflated for in-sample fit, draws re-centred on M3's coefficients). Each draw refits the regression coefficients with M3's transform settings fixed. A robust allocation is also optimised on an ensemble of 60 draws.

## 3. Results

Primary case: current budget, medium bounds (70% to 130%).

| Channel            | Current (LKR M/month) | Optimised (LKR M/month) | Change | Position       |
| ------------------ | --------------------: | ----------------------: | -----: | -------------- |
| TV                 |                  4.36 |                    3.50 |   -20% | interior       |
| Facebook/Instagram |                  4.40 |                    3.08 |   -30% | at lower bound |
| Google Search      |                  3.32 |                    4.31 |   +30% | at upper bound |
| YouTube            |                  2.22 |                    2.89 |   +30% | at upper bound |
| Radio              |                  1.67 |                    1.41 |   -16% | interior       |
| Outdoor            |                  2.61 |                    3.40 |   +30% | at upper bound |

- Expected uplift: **LKR 0.95M per month (about LKR 11.4M per year)**, which is 4.7% of the revenue the model attributes to advertising and 1.1% of total monthly revenue.
- Bootstrap range (5th to 95th percentile): LKR 0.31M to 1.65M per month; the uplift is positive in 99% of draws.
- Wide bounds give LKR 1.09M per month and narrow bounds LKR 0.75M per month, so most of the gain comes from the first moderate shifts.
- The direction is the same in every scenario and in the robust run: move money from Facebook/Instagram, TV and Radio towards Google Search, YouTube and Outdoor.
- Solver checks: SLSQP and differential evolution agree on all 17 point-estimate scenarios (relative gap below 0.1%), and the equimarginal condition holds in every scenario.
- A smaller budget (-10%) with medium bounds still gains LKR 1.03M per month over a proportional cut, because the saving is taken from the least productive channels.

Figures: `outputs/figures/m6/allocation_current_vs_optimal.png`, `marginal_roi.png`, `scenario_uplift.png`.
Tables: `outputs/powerbi/optimisation_scenarios.csv` (channel level, for the dashboard) and `outputs/tables/m6/scenario_summary.csv`.

## 4. Limitations and how to read these results

1. **The underlying model has not shown out-of-sample skill.** M3's final six-month holdout gave R² = -0.417 and a MAPE (6.18%) worse than the training-mean baseline (4.31%). These uplifts are therefore model-based scenarios, not validated forecasts. Treat the direction of the shift as the finding and the size of the gain as indicative.
2. **Bootstrap intervals are optimistic.** They reflect coefficient uncertainty only, with adstock and saturation settings fixed, and they come from in-sample residuals. True uncertainty is larger.
3. **Revenue-based ROI.** There is no margin data, so channel ROI below 1 (TV, Facebook/Instagram and Radio at current spend) does not mean a channel loses money.
4. **Steady-state assumption.** The optimiser treats spend as constant every month. It does not plan month-by-month around seasonality.
5. **Correlation, not proven cause.** Recommendations should be implemented as staged, measured shifts (for example one third of the move per quarter) and checked against actual results.

## 5. Reproduce

```bash
python -m src.optimise_run           # about one minute
python -m src.optimise_run --quick   # fast check without the cross-check solver
python -m pytest tests/test_optimiser.py
```
