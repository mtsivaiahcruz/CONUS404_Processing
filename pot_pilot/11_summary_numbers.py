"""Print the headline numbers that the project notes quote for Steps 3, 4 and 5.

Everything is worked out from the result files, so the notes can be checked against it.

Reads:
    output/pilot_pot_results.csv
    output/work/cadence_vs_mle_long.csv, output/work/gev_vs_pot_full.csv
Writes:
    output/summary_numbers.txt (the same text it prints)
"""
import io
import warnings

import pandas as pd

from paths import OUT, WORK, require

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250, "display.max_columns", 60)
require(OUT / "pilot_pot_results.csv", OUT / "pilot_gev_vs_pot.csv",
        WORK / "cadence_vs_mle_long.csv", WORK / "gev_vs_pot_full.csv")
text = io.StringIO()


def say(*parts):
    """Print a line and keep a copy for summary_numbers.txt."""
    line = " ".join(str(part) for part in parts)
    print(line)
    text.write(line + "\n")


def median_and_range(values):
    return f"{values.median():+.0f} ({values.min():+.0f} to {values.max():+.0f})"


# Step 3. CADENCE's L-moments fit against the documented maximum-likelihood fit.
cadence = pd.read_csv(WORK / "cadence_vs_mle_long.csv")
say("STEP 3. CADENCE (L-moments) vs documented GEV (MLE), native m/s")
say("largest |difference| across the 13 cells, % of the MLE estimate, by return period:")
largest_diff = cadence.groupby("return_period").diff_pct.apply(lambda x: x.abs().max())
say(largest_diff.round(1).to_string())

checks = cadence[cadence.return_period.isin([20, 50, 100])].copy()
checks["delta_w"] = checks.mle_delta_upper_native_ms - checks.mle_delta_lower_native_ms
checks["boot_w"] = checks.mle_boot_upper_native_ms - checks.mle_boot_lower_native_ms
# how big the gap between the two fits is, next to the uncertainty of the MLE fit
checks["gap_vs_halfwidth"] = checks.diff_cadence_minus_mle_ms.abs() / (checks.delta_w / 2)
checks["inside"] = ((checks.cadence_native_ms >= checks.mle_delta_lower_native_ms)
                    & (checks.cadence_native_ms <= checks.mle_delta_upper_native_ms))
checks["w_ratio"] = checks.delta_w / checks.boot_w
upper_gap = checks.mle_boot_upper_native_ms - checks.mle_delta_upper_native_ms
checks["upper_gap_pct"] = 100 * upper_gap / checks.mle_native_ms
say("\ndecision checks at 20, 50 and 100 years:")
say(f"  fit gap as a share of the MLE interval's half-width: "
    f"largest {checks.gap_vs_halfwidth.max():.2f}")
say(f"  L-moments estimate inside the MLE 95% delta interval: "
    f"{int(checks.inside.sum())} of {len(checks)}")
say("  GEV delta width / bootstrap width (min, median, max) by return period:")
say(checks.groupby("return_period").w_ratio.agg(["min", "median", "max"]).round(2).to_string())
say("  bootstrap upper bound minus delta upper bound, % of estimate (median, max) "
    "by return period:")
say(checks.groupby("return_period").upper_gap_pct.agg(["median", "max"]).round(1).to_string())
rp100 = cadence[cadence.return_period == 100]
lmom_width = rp100.cadence_boot_upper_native_ms - rp100.cadence_boot_lower_native_ms
mle_width = rp100.mle_delta_upper_native_ms - rp100.mle_delta_lower_native_ms
say(f"  100-year interval widths, m/s: L-moments bootstrap {lmom_width.min():.1f}"
    f" to {lmom_width.max():.1f};"
    f" MLE delta {mle_width.min():.1f}"
    f" to {mle_width.max():.1f}")

# Step 4. The POT tests.
pot_results = pd.read_csv(OUT / "pilot_pot_results.csv")
# the locations from west to east
order = pot_results.drop_duplicates("candidate").sort_values("longitude").candidate.tolist()
say("\n\nSTEP 4. POT tests")
n_flagged = int(pot_results["flags"].fillna("").ne("").sum())
say(f"fits: {len(pot_results)}; flagged: {n_flagged}; "
    f"shape range {pot_results.gpd_shape_xi.min():.2f} to {pot_results.gpd_shape_xi.max():.2f}; "
    f"days above threshold per event {pot_results.mean_days_per_event.min():.2f} "
    f"to {pot_results.mean_days_per_event.max():.2f}; "
    f"lowest KS p-value {pot_results.ks_pvalue.min():.3f}")
say("settings and event counts (ranges across cells and separation rules):")
by_threshold = pot_results.groupby("threshold_pct").agg(
    threshold_min=("threshold_ms", "min"), threshold_max=("threshold_ms", "max"),
    days_above=("n_exceedance_days", "max"), events_min=("n_events", "min"),
    events_max=("n_events", "max"), per_year_min=("events_per_year", "min"),
    per_year_max=("events_per_year", "max"),
)
say(by_threshold.round(1).to_string())

sep2 = pot_results[pot_results.separation_days == 2]
say("\n100-year estimate by threshold, native m/s, 2-day separation:")
rp100_by_threshold = sep2.pivot_table(index="candidate", columns="threshold_pct",
                                      values="rp100_ms")
say(rp100_by_threshold.loc[order].round(1).to_string())

# effect of the separation rule: 3 days against 1 day
by_separation = pot_results.pivot_table(index=["candidate", "threshold_pct"],
                                        columns="separation_days", values=["n_events", "rp100_ms"])
events_change = 100 * (by_separation[("n_events", 3)] / by_separation[("n_events", 1)] - 1)
level_change = (100 * (by_separation[("rp100_ms", 3)] / by_separation[("rp100_ms", 1)] - 1)).abs()
say("\nseparation rule, 3 days vs 1 day: median % change in event count by threshold:",
    events_change.groupby("threshold_pct").median().round(1).to_dict())
say(f"largest change in the 100-year estimate from the separation rule: {level_change.max():.1f}%")

all_settings = pot_results.pivot_table(index="candidate",
                                       columns=["threshold_pct", "separation_days"],
                                       values="rp100_ms").loc[order]
spread = pd.DataFrame({
    "min": all_settings.min(axis=1),
    "max": all_settings.max(axis=1),
    "spread_pct": (100 * (all_settings.max(axis=1) - all_settings.min(axis=1))
                   / all_settings.median(axis=1)),
})
say("\n100-year estimate across all 12 settings (min, max, spread as % of the median):")
say(spread.round(1).to_string())
say("\nmedian standard error of the 100-year estimate by threshold, m/s:",
    pot_results.groupby("threshold_pct").rp100_se_ms.median().round(2).to_dict())
say("fitted shape by threshold, 2-day separation:")
shape_by_threshold = sep2.pivot_table(index="candidate", columns="threshold_pct",
                                      values="gpd_shape_xi")
say(shape_by_threshold.loc[order].round(2).to_string())

# Step 5. GEV against POT.
full = pd.read_csv(WORK / "gev_vs_pot_full.csv")
full["flags"] = full["flags"].fillna("")
full_sep2 = full[full.separation_days == 2]
say("\n\nSTEP 5. GEV vs POT (negative = POT interval narrower)")
method_names = (("delta", "delta method"), ("boot", "parametric bootstrap"),
                ("years", "resampled-years bootstrap"))
for method, name in method_names:
    say(f"\nmedian % change in width across the 13 cells, 2-day separation, {name}:")
    median_change = full_sep2.pivot_table(index="threshold_pct", columns="return_period",
                                          values=f"width_change_{method}_pct", aggfunc="median")
    say(median_change.round(0).to_string())
say("\ncells (of 13) where POT is narrower, 2-day separation, delta method:")
n_narrower = full_sep2.groupby(["threshold_pct", "return_period"]).pot_narrower_delta.sum()
say(n_narrower.unstack().to_string())

say("\n100-year: GEV estimate and delta interval; % change in width by threshold "
    "(delta, 2-day separation):")
# the GEV values are the same on every POT row of a cell, so the 98th percentile rows will do
gev100 = full_sep2[(full_sep2.return_period == 100) & (full_sep2.threshold_pct == 98)]
gev100 = gev100.set_index("candidate").loc[order]
gev_and_change = pd.DataFrame({
    "gev_est": gev100.gev_est_ms.round(1),
    "gev_lower": gev100.gev_delta_lower_ms.round(1),
    "gev_upper": gev100.gev_delta_upper_ms.round(1),
})
change_by_threshold = full_sep2[full_sep2.return_period == 100].pivot_table(
    index="candidate", columns="threshold_pct", values="width_change_delta_pct")
gev_and_change = gev_and_change.join(change_by_threshold.round(0))
say(gev_and_change.to_string())

agreement = full.assign(
    all_narrower=full.pot_narrower_delta & full.pot_narrower_boot & full.pot_narrower_years,
    all_wider=~(full.pot_narrower_delta | full.pot_narrower_boot | full.pot_narrower_years),
)
say("\nper cell, of its 36 rows (4 thresholds x 3 separations x 3 return periods): "
    "rows where all three methods say narrower / wider:")
say(agreement.groupby("candidate")[["all_narrower", "all_wider"]].sum().loc[order].to_string())
say("\nPOT estimate minus GEV estimate, %, median (min to max) across cells and separations:")
estimate_diff = full.pivot_table(index="threshold_pct", columns="return_period",
                                 values="est_diff_pct", aggfunc=median_and_range)
say(estimate_diff.to_string())
say(f"\nPOT estimate inside the GEV delta interval: "
    f"{int(full.pot_est_inside_gev_delta_interval.sum())} of {len(full)} rows")
outside = full[~full.gev_est_inside_pot_delta_interval]
say(f"GEV estimate outside the POT delta interval: {len(outside)} rows")
say(outside.groupby(["candidate", "return_period", "threshold_pct"]).size().to_string())

say(f"\nflagged rows: {int(full['flags'].ne('').sum())} of {len(full)}")
boot_disagrees = full.pot_narrower_delta != full.pot_narrower_boot
say("  delta and bootstrap disagree:", int(boot_disagrees.sum()),
    full[boot_disagrees].groupby("candidate").size().to_dict())
years_disagrees = full.pot_narrower_delta != full.pot_narrower_years
say("  delta and resampled-years disagree:", int(years_disagrees.sum()),
    full[years_disagrees].groupby("candidate").size().to_dict())
say("\nPOT delta width / POT bootstrap width (min, median, max) by return period:")
width_ratio = full.pot_delta_width_ms / full.pot_boot_width_ms
say(width_ratio.groupby(full.return_period).agg(["min", "median", "max"]).round(2).to_string())

(OUT / "summary_numbers.txt").write_text(text.getvalue(), encoding="utf-8")
print(f"\nwrote {OUT / 'summary_numbers.txt'}")
