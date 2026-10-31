"""Results table: steps 15 to 22 combined, by cell and version, and one row per version.

This step refits nothing. It reads the outputs of steps 15 to 22 and puts them side by side,
so each version can be judged on everything at once. The six versions are the GEV and POT,
each with the flat, main and wider priors on the shape (98th percentile threshold and 2-day
separation for POT).

By cell and version (13 cells x 6 versions):
    estimates    20-, 50- and 100-year estimates and 95% credible intervals (steps 15 and 16),
                 and the 100-year width against the flat GEV at the same cell
    held out     threshold-weighted CRPS, quantile score and 90% interval score against the
                 flat GEV, with bootstrap ranges, and the 90% predictive coverage (steps 18, 18a)
    stability    median and largest change in the 100-year estimate with one year left out,
                 and the year behind the largest (step 19)
    threshold    spread of the 100-year estimate over the 95th to 99th percentiles, and whether
                 the 98th's estimate is inside every other threshold's interval (step 20, POT
                 with the wider prior only)
    separation   largest change in the 100-year estimate from the 2-day rule over 1- to 7-day
                 rules (step 21, POT only)
    trend        Kendall's tau and p-value of the annual maxima (step 21a)
    mixed storms coverage and bias of the 100-year level at each hurricane tail (step 22, the
                 four coastal cells only)

By version (6 rows): the simulation study at the three true models (steps 17 and 18a), the
mixed-storm simulation across the four coastal cells (step 22), the pooled held-out scores
(steps 18 and 18a), stability at a typical cell (step 19), the median 100-year width, and the
largest separation-rule change at any cell (step 21).

Reads (all in output/):
    bayes_fits.csv, prior_sensitivity.csv, holdout_scores.csv, holdout_tail_scores.csv,
    stability_summary.csv, threshold_check_summary.csv, separation_check.csv, trend_check.csv,
    simulation_coverage.csv, simulation_interval_scores.csv, mixed_storm_simulation.csv,
    pilot_locations_selected.csv
Writes:
    output/checkpoint2_results_table.csv     one row per cell and version
    output/checkpoint2_version_summary.csv   one row per version
"""
import numpy as np
import pandas as pd

from paths import OUT, require

VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("GEV", "wider"),
            ("POT", "flat"), ("POT", "main"), ("POT", "wider")]
LABELS = {("GEV", "flat"): "GEV, flat", ("GEV", "main"): "GEV, main prior",
          ("GEV", "wider"): "GEV, wider prior", ("POT", "flat"): "POT, flat",
          ("POT", "main"): "POT, main prior", ("POT", "wider"): "POT, wider prior"}
RETURN_PERIODS = [20, 50, 100]
TRUE_MODELS = ["light", "zero", "heavy"]
TAILS = ["bounded", "central", "heavy"]
INPUTS = ["bayes_fits.csv", "prior_sensitivity.csv", "holdout_scores.csv",
          "holdout_tail_scores.csv", "stability_summary.csv", "threshold_check_summary.csv",
          "separation_check.csv", "trend_check.csv", "simulation_coverage.csv",
          "simulation_interval_scores.csv", "mixed_storm_simulation.csv",
          "pilot_locations_selected.csv"]
KEYS = ["cell_id", "model", "prior"]


def read(name):
    return pd.read_csv(OUT / name)


def estimates():
    """The 20-, 50- and 100-year estimates and intervals, one row per cell and version."""
    step15 = read("bayes_fits.csv")
    step16 = read("prior_sensitivity.csv")
    fits = pd.concat([step15[["cell_id", "model", "prior", "return_period", "estimate_ms",
                              "lower_95_ms", "upper_95_ms"]],
                      step16[step16.prior == "wider"][["cell_id", "model", "prior",
                                                       "return_period", "estimate_ms",
                                                       "lower_95_ms", "upper_95_ms"]]])
    wide = fits.pivot_table(index=KEYS, columns="return_period",
                            values=["estimate_ms", "lower_95_ms", "upper_95_ms"])
    wide.columns = [f"rp{period}_{name.replace('_ms', '')}_ms" for name, period in wide.columns]
    wide = wide.reset_index()
    wide = wide[KEYS + [f"rp{p}_{q}_ms" for p in RETURN_PERIODS
                        for q in ("estimate", "lower_95", "upper_95")]]
    wide["rp100_width_ms"] = wide.rp100_upper_95_ms - wide.rp100_lower_95_ms
    flat_gev = wide[(wide.model == "GEV") & (wide.prior == "flat")].set_index(
        "cell_id").rp100_width_ms
    wide["rp100_width_vs_gev_flat_pct"] = 100 * (
        wide.rp100_width_ms / wide.cell_id.map(flat_gev) - 1)
    return wide


def held_out():
    """Steps 18 and 18a at each cell, as the difference from the flat GEV in percent."""
    scores = read("holdout_scores.csv")
    tail = read("holdout_tail_scores.csv")
    reference = scores[(scores.model == "GEV") & (scores.prior == "flat")].set_index(
        "location").mean_twcrps
    out = scores[["location", "cell_id", "model", "prior", "coverage_90_pct",
                  "twcrps_diff_vs_gev_flat_pct"]].copy()
    # the bootstrap range is stored in score units, so put it in percent of the flat GEV
    for end in ("low", "high"):
        out[f"twcrps_diff_vs_gev_flat_{end}_pct"] = (
            100 * scores[f"twcrps_diff_vs_gev_flat_{end}"] / scores.location.map(reference))
    tail = tail[["location", "model", "prior"] + [c for c in tail.columns if "_diff_" in c]]
    return out.merge(tail, on=["location", "model", "prior"])


def by_cell():
    cells = read("pilot_locations_selected.csv")[["cell_id", "candidate"]].rename(
        columns={"candidate": "location"})
    table = cells.merge(estimates(), on="cell_id")

    scores = held_out()
    table = table.merge(scores[scores.location != "All 13 cells"].drop(columns="location"),
                        on=KEYS, how="left")

    stability = read("stability_summary.csv")
    stability = stability[stability.cell_id.notna()].astype({"cell_id": int})
    table = table.merge(stability[KEYS + ["median_abs_change_pct", "largest_abs_change_pct",
                                          "largest_change_year"]].rename(columns={
        "median_abs_change_pct": "leave_one_out_median_change_pct",
        "largest_abs_change_pct": "leave_one_out_largest_change_pct",
        "largest_change_year": "leave_one_out_largest_year"}), on=KEYS, how="left")

    threshold = read("threshold_check_summary.csv")[[
        "cell_id", "spread_95_to_99_pct", "inside_every_other_interval"]].rename(columns={
            "spread_95_to_99_pct": "threshold_spread_95_to_99_pct",
            "inside_every_other_interval": "threshold_98th_inside_every_other"})
    threshold["model"], threshold["prior"] = "POT", "wider"
    table = table.merge(threshold, on=KEYS, how="left")

    separation = read("separation_check.csv")
    separation = separation[separation.percentile == 98]
    separation = (separation.assign(change=separation.rp100_change_from_2day_pct.abs())
                  .groupby(["cell_id", "prior"]).change.max()
                  .rename("separation_largest_change_pct").reset_index())
    separation["model"] = "POT"
    table = table.merge(separation, on=KEYS, how="left")

    trend = read("trend_check.csv")[["cell_id", "annual_max_tau", "annual_max_p"]]
    table = table.merge(trend, on="cell_id", how="left")

    mixed = read("mixed_storm_simulation.csv")
    mixed = mixed[mixed.return_period == 100]
    wide = mixed.pivot_table(index=["candidate", "model", "prior"], columns="hurricane_tail",
                             values=["coverage_pct", "bias_pct"])
    wide.columns = [f"mixed_storm_{name.replace('_pct', '')}_{tail}_pct"
                    for name, tail in wide.columns]
    wide = wide.reset_index().rename(columns={"candidate": "location"})
    wide = wide[["location", "model", "prior"]
                + [f"mixed_storm_{q}_{t}_pct" for q in ("coverage", "bias") for t in TAILS]]
    table = table.merge(wide, on=["location", "model", "prior"], how="left")

    table.insert(2, "version", [LABELS[(m, p)] for m, p in zip(table.model, table.prior)])
    order = {version: i for i, version in enumerate(VERSIONS)}
    cell_order = {c: i for i, c in enumerate(cells.cell_id)}    # west to east
    table["_order"] = [cell_order[c] * 10 + order[(m, p)]
                       for c, m, p in zip(table.cell_id, table.model, table.prior)]
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def by_version(cell_table):
    coverage = read("simulation_coverage.csv")
    coverage = coverage[coverage.return_period == 100]
    interval = read("simulation_interval_scores.csv")
    interval = interval[interval.return_period == 100]
    mixed = read("mixed_storm_simulation.csv")
    mixed = mixed[mixed.return_period == 100]
    scores = held_out()
    pooled = scores[scores.location == "All 13 cells"]
    stability = read("stability_summary.csv")
    stability = stability[stability.cell_id.isna()]

    rows = []
    for model, prior in VERSIONS:
        def pick(frame):
            return frame[(frame.model == model) & (frame.prior == prior)]
        row = {"version": LABELS[(model, prior)], "model": model, "prior": prior}
        sim = pick(coverage).set_index("true_model")
        score = pick(interval).set_index("true_model")
        for truth in TRUE_MODELS:
            row[f"simulation_coverage_{truth}_pct"] = sim.loc[truth, "coverage_pct"]
        row["simulation_bias_heavy_pct"] = sim.loc["heavy", "bias_pct"]
        row["simulation_rmse_heavy_pct"] = sim.loc["heavy", "rmse_pct"]
        row["simulation_median_width_heavy_ms"] = sim.loc["heavy", "median_width_ms"]
        for truth in TRUE_MODELS:
            row[f"simulation_interval_score_{truth}_ms"] = score.loc[truth, "interval_score_ms"]
        mine = pick(mixed)
        for tail in TAILS:
            part = mine[mine.hurricane_tail == tail]
            row[f"mixed_storm_coverage_{tail}_min_pct"] = part.coverage_pct.min()
            row[f"mixed_storm_coverage_{tail}_max_pct"] = part.coverage_pct.max()
        central = mine[mine.hurricane_tail == "central"]
        row["mixed_storm_bias_central_min_pct"] = central.bias_pct.min()
        row["mixed_storm_bias_central_max_pct"] = central.bias_pct.max()
        row["mixed_storm_width_central_mean_ms"] = central.median_width_ms.mean()
        row["mixed_storm_interval_score_central_mean_ms"] = central.interval_score_ms.mean()
        held = pick(pooled).iloc[0]
        for column in ["coverage_90_pct", "twcrps_diff_vs_gev_flat_pct",
                       "twcrps_diff_vs_gev_flat_low_pct", "twcrps_diff_vs_gev_flat_high_pct",
                       "quantile_95_diff_vs_gev_flat_pct", "interval_90_diff_vs_gev_flat_pct"]:
            row[f"holdout_{column}"] = held[column]
        stable = pick(stability).iloc[0]
        row["leave_one_out_largest_change_typical_cell_pct"] = (
            stable.median_largest_abs_change_pct)
        cells = pick(cell_table)
        row["median_rp100_width_ms"] = cells.rp100_width_ms.median()
        row["median_rp100_width_vs_gev_flat_pct"] = cells.rp100_width_vs_gev_flat_pct.median()
        row["separation_largest_change_pct"] = cells.separation_largest_change_pct.max()
        rows.append(row)
    return pd.DataFrame(rows)


def check(cell_table, version_table):
    """The table must hold every cell and version once, and agree with the step outputs."""
    assert len(cell_table) == 13 * len(VERSIONS)
    assert not cell_table.duplicated(KEYS).any()
    assert cell_table.rp100_estimate_ms.notna().all()
    assert (cell_table.rp100_lower_95_ms < cell_table.rp100_estimate_ms).all()
    assert (cell_table.rp100_estimate_ms < cell_table.rp100_upper_95_ms).all()
    flat_gev = cell_table[(cell_table.model == "GEV") & (cell_table.prior == "flat")]
    assert np.allclose(flat_gev.rp100_width_vs_gev_flat_pct, 0)
    assert flat_gev.twcrps_diff_vs_gev_flat_pct.isna().all()
    coastal = cell_table.mixed_storm_coverage_central_pct.notna()
    assert set(cell_table[coastal].location) == {"Houston, TX", "New Orleans, LA",
                                                 "Miami, FL", "Boston, MA"}
    assert cell_table.threshold_spread_95_to_99_pct.notna().sum() == 13
    assert cell_table[cell_table.model == "POT"].separation_largest_change_pct.notna().all()
    assert len(version_table) == len(VERSIONS)


if __name__ == "__main__":
    require(*[OUT / name for name in INPUTS])
    cell_table = by_cell()
    version_table = by_version(cell_table)
    check(cell_table, version_table)

    numbers = cell_table.select_dtypes("number").columns.difference(
        ["cell_id", "leave_one_out_largest_year"])
    cell_table[numbers] = cell_table[numbers].round(4)
    years = cell_table.leave_one_out_largest_year
    cell_table["leave_one_out_largest_year"] = years.astype("Int64")
    cell_table.to_csv(OUT / "checkpoint2_results_table.csv", index=False)
    numbers = version_table.select_dtypes("number").columns
    version_table[numbers] = version_table[numbers].round(4)
    version_table.to_csv(OUT / "checkpoint2_version_summary.csv", index=False)

    show = version_table.set_index("version")
    pd.set_option("display.width", 160)
    print("Simulation study, 100-year level (steps 17 and 18a):")
    print(show[[f"simulation_coverage_{t}_pct" for t in TRUE_MODELS]
               + ["simulation_rmse_heavy_pct", "simulation_interval_score_heavy_ms"]]
          .round(1).to_string())
    print("\nMixed-storm simulation, coverage of the 100-year level across the four coastal "
          "cells (step 22):")
    print(show[[f"mixed_storm_coverage_{t}_{e}_pct" for t in TAILS for e in ("min", "max")]]
          .round(1).to_string())
    print("\nHeld-out years, all 13 cells, % against the flat GEV (steps 18 and 18a), and "
          "stability (step 19):")
    print(show[["holdout_twcrps_diff_vs_gev_flat_pct", "holdout_quantile_95_diff_vs_gev_flat_pct",
                "holdout_interval_90_diff_vs_gev_flat_pct",
                "leave_one_out_largest_change_typical_cell_pct", "median_rp100_width_ms"]]
          .round(2).to_string())
    print(f"\nwrote {OUT / 'checkpoint2_results_table.csv'} ({len(cell_table)} rows), "
          f"{OUT / 'checkpoint2_version_summary.csv'} ({len(version_table)} rows)")
