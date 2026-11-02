"""M6: run budget optimisation scenarios and write outputs.

Run from project root:
    python -m src.optimise_run            # full run
    python -m src.optimise_run --quick    # fewer draws/starts for a fast check

Outputs
    outputs/powerbi/optimisation_scenarios.csv   channel-level rows (for M7)
    outputs/tables/m6/scenario_summary.csv       one row per scenario
    outputs/figures/m6/*.png
"""

from src.optimiser import (
    channel_bounds,
    channel_responses,
    is_feasible,
    marginal_roi,
    optimise_full,
    project
)
from src.m6_uncertainty import (
    bootstrap_models,
    summarise_uplift,
    uplift_distribution
)
from src.config import (
    CHANNELS,
    CLEAN_CSV,
    FIG_DIR,
    MODEL_DIR,
    OUT,
    POWERBI_DIR,
    SEED,
    TARGET
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")


BUDGET_CHANGES_PCT = [-20, -10, 0, 10, 20, 30]

# label -> (low fraction, high fraction) of each channel's historical mean
BOUND_SETS = {
    "wide_50_150": (0.5, 1.5),
    "medium_70_130": (0.7, 1.3),
    "narrow_80_120": (0.8, 1.2),
}
PRIMARY_BOUNDS = "medium_70_130"

NICE = {
    "tv_spend_lkr": "TV",
    "facebook_instagram_spend_lkr": "Facebook/Instagram",
    "google_search_spend_lkr": "Google Search",
    "youtube_spend_lkr": "YouTube",
    "radio_spend_lkr": "Radio",
    "outdoor_spend_lkr": "Outdoor",
}


def _scenario_id(method, bounds_label, pct):
    sign = "p" if pct >= 0 else "m"
    return f"{method}_{bounds_label}_{sign}{abs(pct)}"


def _scaled_current(current, budget, low, high):
    """Current mix scaled to the budget, then forced inside the bounds."""
    return project(current * budget / current.sum(), budget, low, high)


def run_scenario(model, boot_models, current, base_budget, bounds_label, pct,
                 method, n_starts, run_de, mean_revenue):
    lo_f, hi_f = BOUND_SETS[bounds_label]
    low, high = channel_bounds(current, lo_f, hi_f)
    budget = base_budget * (1 + pct / 100)
    sid = _scenario_id(method, bounds_label, pct)

    summary = {
        "scenario_id": sid,
        "method": method,
        "bounds_label": bounds_label,
        "bound_low_frac": lo_f,
        "bound_high_frac": hi_f,
        "budget_change_pct": pct,
        "monthly_budget_lkr": budget,
    }
    if not is_feasible(budget, low, high):
        summary["status"] = "infeasible"
        return summary, []

    baseline = _scaled_current(current, budget, low, high)
    forced = (abs(budget - low.sum()) < 1e-6 * budget
              or abs(budget - high.sum()) < 1e-6 * budget)
    solve_model = boot_models if method == "bootstrap_robust" else model
    res = optimise_full(solve_model, budget, low, high, current=baseline,
                        n_starts=n_starts, seed=SEED, run_de=run_de)
    optimal = res["allocation"]

    # Point-estimate responses (used for reporting) and bootstrap uplift.
    base_resp = channel_responses(model, baseline)
    opt_resp = channel_responses(model, optimal)
    uplift = float(opt_resp.sum() - base_resp.sum())
    draws = uplift_distribution(boot_models, optimal, baseline)

    summary.update({
        "status": "forced_by_bounds" if forced else "ok",
        "baseline_incremental_revenue_lkr": float(base_resp.sum()),
        "optimal_incremental_revenue_lkr": float(opt_resp.sum()),
        "uplift_lkr_per_month": uplift,
        "uplift_lkr_per_year": uplift * 12,
        "uplift_pct_of_ad_revenue": float(uplift / base_resp.sum() * 100),
        "uplift_pct_of_total_revenue": float(uplift / mean_revenue * 100),
        **(summarise_uplift(draws) if not forced else {
            "uplift_p05_lkr": np.nan, "uplift_p50_lkr": np.nan,
            "uplift_p95_lkr": np.nan, "prob_uplift_positive": np.nan}),
        "solver_used": res["solver"],
        "de_agrees_with_slsqp": res["solvers_agree"],
        "rel_gap_de": res["rel_gap_de"],
        "kkt_ok": res["kkt_ok"],
        "lambda_marginal_roi": res["lambda_mroi"],
        "max_channel_shift_pct": float(
            np.max(np.abs(optimal / baseline - 1)) * 100),
    })

    mroi_opt = res["marginal_roi"]
    mroi_base = marginal_roi(model, baseline)
    status = res["bound_status"]
    rows = []
    for j, ch in enumerate(CHANNELS):
        rows.append({
            **{k: summary[k] for k in (
                "scenario_id", "method", "bounds_label", "bound_low_frac",
                "bound_high_frac", "budget_change_pct", "monthly_budget_lkr")},
            "channel": NICE[ch],
            "baseline_spend_lkr": baseline[j],
            "optimal_spend_lkr": optimal[j],
            "change_lkr": optimal[j] - baseline[j],
            "change_pct": (optimal[j] / baseline[j] - 1) * 100,
            "optimal_share_pct": optimal[j] / budget * 100,
            "baseline_share_pct": baseline[j] / budget * 100,
            "baseline_roi": base_resp[j] / baseline[j],
            "optimal_roi": opt_resp[j] / optimal[j],
            "baseline_marginal_roi": mroi_base[j],
            "optimal_marginal_roi": mroi_opt[j],
            "bound_status": status[j],
            "uplift_lkr_per_month": uplift,
            "uplift_p05_lkr": summary["uplift_p05_lkr"],
            "uplift_p95_lkr": summary["uplift_p95_lkr"],
            "prob_uplift_positive": summary["prob_uplift_positive"],
        })
    return summary, rows


# ------------------------------- figures ----------------------------------
def _fig_allocation(rows, path):
    d = pd.DataFrame(rows)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, d["baseline_spend_lkr"] / 1e6, 0.4,
           label="Current mix", color="#9aa0a6")
    ax.bar(x + 0.2, d["optimal_spend_lkr"] / 1e6, 0.4,
           label="Optimised mix", color="#1D9E75")
    for i, pct in enumerate(d["change_pct"]):
        ax.text(i + 0.2, d["optimal_spend_lkr"].iloc[i] / 1e6 + 0.05,
                f"{pct:+.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["channel"], rotation=20, ha="right")
    ax.set_ylabel("Monthly spend (LKR million)")
    ax.set_title("Current vs optimised allocation (same budget, "
                 f"{PRIMARY_BOUNDS} bounds)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_marginal(rows, path):
    d = pd.DataFrame(rows)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, d["baseline_marginal_roi"], 0.4,
           label="At current mix", color="#9aa0a6")
    ax.bar(x + 0.2, d["optimal_marginal_roi"], 0.4,
           label="At optimised mix", color="#534AB7")
    ax.set_xticks(x)
    ax.set_xticklabels(d["channel"], rotation=20, ha="right")
    ax.set_ylabel("Extra revenue per extra LKR")
    ax.set_title("Marginal ROI by channel: closer together after optimising")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_scenarios(summary, path):
    d = pd.DataFrame(summary)
    d = d[(d["status"] == "ok") & (
        d["method"] == "point_estimate")]  # skip forced
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"wide_50_150": "#D85A30", "medium_70_130": "#1D9E75",
               "narrow_80_120": "#534AB7"}
    for label, g in d.groupby("bounds_label"):
        g = g.sort_values("budget_change_pct")
        ax.plot(g["budget_change_pct"], g["uplift_lkr_per_month"] / 1e6,
                marker="o", label=label, color=colours[label])
        if label == PRIMARY_BOUNDS:
            ax.fill_between(g["budget_change_pct"],
                            g["uplift_p05_lkr"] / 1e6,
                            g["uplift_p95_lkr"] / 1e6,
                            color=colours[label], alpha=0.15,
                            label="5-95% bootstrap range (medium)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Budget change vs current (%)")
    ax.set_ylabel("Uplift vs proportional mix (LKR million / month)")
    ax.set_title("Uplift from re-allocating, by budget level and bounds")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------- main -------------------------------------
def main(draws=200, block=3, n_starts=12, run_de=True, robust_draws=60):
    df = pd.read_csv(CLEAN_CSV, parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)
    model = joblib.load(MODEL_DIR / "mmm_model.joblib")

    current = df[CHANNELS].mean().to_numpy(dtype=float)
    base_budget = float(current.sum())
    mean_revenue = float(df[TARGET].mean())
    print(f"Current mean monthly budget: LKR {base_budget:,.0f}")

    print(f"Bootstrapping {draws} model draws...")
    boots = bootstrap_models(model, df, n_draws=draws, block=block, seed=SEED)
    robust_models = boots[:robust_draws]

    summaries, channel_rows = [], []

    print("Point-estimate scenarios...")
    for label in BOUND_SETS:
        for pct in BUDGET_CHANGES_PCT:
            s, r = run_scenario(model, boots, current, base_budget, label,
                                pct, "point_estimate", n_starts, run_de,
                                mean_revenue)
            summaries.append(s)
            channel_rows.extend(r)

    print(f"Robust scenarios (ensemble of {len(robust_models)} draws)...")
    for label in BOUND_SETS:
        s, r = run_scenario(model, robust_models, current, base_budget,
                            label, 0, "bootstrap_robust",
                            max(n_starts // 2, 4), False, mean_revenue)
        # Report uplift uncertainty on the full set of draws.
        summaries.append(s)
        channel_rows.extend(r)

    summary_df = pd.DataFrame(summaries)
    rows_df = pd.DataFrame(channel_rows)

    tables = OUT / "tables" / "m6"
    figs = FIG_DIR / "m6"
    for p in (tables, figs, POWERBI_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(tables / "scenario_summary.csv", index=False)
    rows_df.to_csv(POWERBI_DIR / "optimisation_scenarios.csv", index=False)

    primary = rows_df[(rows_df["scenario_id"]
                       == _scenario_id("point_estimate", PRIMARY_BOUNDS, 0))]
    primary_rows = primary.to_dict("records")
    if primary_rows:
        _fig_allocation(primary_rows, figs /
                        "allocation_current_vs_optimal.png")
        _fig_marginal(primary_rows, figs / "marginal_roi.png")
    _fig_scenarios(summaries, figs / "scenario_uplift.png")

    meta = {
        "base_monthly_budget_lkr": base_budget,
        "bootstrap_draws": draws,
        "bootstrap_block_length": block,
        "robust_ensemble_size": len(robust_models),
        "primary_bounds": PRIMARY_BOUNDS,
        "seed": SEED,
        "note": ("Uplift is measured against the current mix scaled "
                 "proportionally to the scenario budget. Intervals cover "
                 "coefficient uncertainty only."),
    }
    (tables / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    ok = summary_df[summary_df["status"].isin(["ok", "forced_by_bounds"])]
    print(f"Scenarios solved: {len(ok)} / {len(summary_df)} "
          f"({int((summary_df['status'] == 'forced_by_bounds').sum())} "
          "forced by bounds, no freedom to re-allocate)")
    point = ok[ok["method"] == "point_estimate"]
    print("Solver agreement (DE vs SLSQP):",
          int(point["de_agrees_with_slsqp"].sum()), "/", len(point))
    print("KKT satisfied:", int(ok["kkt_ok"].sum()), "/", len(ok))
    print(f"Wrote {POWERBI_DIR / 'optimisation_scenarios.csv'}")
    return summary_df, rows_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--block", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="fewer draws and starts, no DE cross-check")
    args = parser.parse_args()
    if args.quick:
        main(draws=30, block=args.block, n_starts=4, run_de=False,
             robust_draws=15)
    else:
        main(draws=args.draws, block=args.block)
