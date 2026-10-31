"""Plots for checkpoint 2.

Figure 6  Return-level curves at the 13 pilot locations: the GEV and POT, each with the flat
          prior and the wider prior, with 95% credible intervals for the flat GEV (what CADENCE
          does now) and POT with the wider prior (the recommended version away from the
          hurricane cells), and the observed annual maxima.
Figure 7  Simulation study (step 17): how often each version's 95% interval contains the true
          100-year level, at the three true models.
Figure 8  Mixed-storm simulation (step 22): the same at the four coastal cells, for each of the
          three hurricane tails.
Figure 9  Held-out years (steps 18 and 18a): each version's score against the flat GEV, with the
          bootstrap 95% range, for the threshold-weighted CRPS, the quantile score and the 90%
          interval score.

Where the numbers come from. Figure 6 refits the four versions at each cell with the same data,
starting points and seeds as steps 15 and 16, so the draws are the same, and then checks that
the curves equal output/bayes_fits.csv and output/prior_sensitivity.csv at 20, 50 and 100
years. Figures 7 to 9 are drawn straight from the step outputs.

Wind speeds are native model values (SPDUV10MAX, m/s). Colors, fonts and layout follow
13_make_plots.py: the GEV is teal and POT is blue in every figure.

Usage:
    python 24_make_plots.py [--dpi 200]
Reads:
    data/pilot_daily_spduv10max.csv, output/pilot_locations_selected.csv,
    output/pilot_gev_parameters.csv, output/pilot_pot_results.csv, output/bayes_fits.csv,
    output/prior_sensitivity.csv, output/simulation_coverage.csv,
    output/mixed_storm_simulation.csv, output/holdout_scores.csv,
    output/holdout_tail_scores.csv
Writes, all in output/plots/:
    fig6_return_levels_checkpoint2.png, fig6_return_levels_checkpoint2_data.csv
    fig7_simulation_coverage.png
    fig8_mixed_storm_coverage.png
    fig9_holdout_scores.png
"""
import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import bayes
import pot
from paths import DAILY_CSV, FIRST_WY, LAST_WY, N_YEARS, OUT, require


def load_step(name, file_name):
    """Load another step's code by file name, since the names start with a digit."""
    path = Path(__file__).resolve().parent / file_name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


style = load_step("make_plots", "13_make_plots.py")    # the checkpoint 1 figures
GEV_COLOR, POT_COLOR = style.GEV_COLOR, style.POT_COLOR
TEXT_COLOR, MUTED_TEXT, AXIS_COLOR = style.TEXT_COLOR, style.MUTED_TEXT, style.AXIS_COLOR
MINUS, DASH = style.MINUS, style.DASH

PERCENTILE = 98
SEPARATION = 2
TABLE_RETURN_PERIODS = [20, 50, 100]
CURVE_MIN_YEARS = 1.01
CURVE_MAX_YEARS = 200.0
VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("GEV", "wider"),
            ("POT", "flat"), ("POT", "main"), ("POT", "wider")]
LABELS = {("GEV", "flat"): "GEV, flat", ("GEV", "main"): "GEV, main prior",
          ("GEV", "wider"): "GEV, wider prior", ("POT", "flat"): "POT, flat",
          ("POT", "main"): "POT, main prior", ("POT", "wider"): "POT, wider prior"}
COLOR = {"GEV": GEV_COLOR, "POT": POT_COLOR}
MARKER = {"flat": "o", "main": "s", "wider": "D"}    # the prior, as a second cue to the label
TRUE_MODELS = {"light": "Bounded tail (Atlanta)", "zero": "Near zero (Chicago)",
               "heavy": "Heavy tail (New Orleans)"}
COASTAL = ["Houston, TX", "New Orleans, LA", "Miami, FL", "Boston, MA"]
TAILS = {"bounded": "bounded", "central": "central", "heavy": "heavy"}


def seed(k, model, prior):
    """The seed steps 15 and 16 used for this cell and version."""
    if prior in ("flat", "main"):    # step 15
        return 1000 * k + [("GEV", "flat"), ("GEV", "main"), ("POT", "flat"),
                           ("POT", "main")].index((model, prior))
    return 1000 * k + 10 + (model == "POT")    # step 16, the wider prior is its first


def curve(fit, periods, threshold=None):
    """The median and 95% interval of the return level at each period."""
    low, mid, high = np.percentile(fit["levels"], [2.5, 50, 97.5], axis=0)
    if threshold is not None:
        # POT only describes values above its threshold, so the curve is blank below it
        for band in (low, mid, high):
            band[mid < threshold] = np.nan
    return mid, low, high


def figure_return_levels(cells, values, annual_max, dpi):
    """Figure 6."""
    periods = np.unique(np.r_[np.geomspace(CURVE_MIN_YEARS, CURVE_MAX_YEARS, 120),
                              TABLE_RETURN_PERIODS])
    table_index = [int(np.flatnonzero(periods == N)[0]) for N in TABLE_RETURN_PERIODS]
    step15 = pd.read_csv(OUT / "bayes_fits.csv")
    step16 = pd.read_csv(OUT / "prior_sensitivity.csv")
    tables = pd.concat([step15, step16[step16.prior == "wider"]])
    mle_gev = pd.read_csv(OUT / "pilot_gev_parameters.csv").set_index("cell_id")
    mle_pot = pd.read_csv(OUT / "pilot_pot_results.csv")
    mle_pot = mle_pot[(mle_pot.threshold_pct == PERCENTILE)
                      & (mle_pot.separation_days == SEPARATION)].set_index("cell_id")
    shown = [("GEV", "flat"), ("GEV", "wider"), ("POT", "flat"), ("POT", "wider")]
    banded = {("GEV", "flat"), ("POT", "wider")}

    fig, axes, bottom_panels = style.grid_axes(len(cells), (13.5, 8.8), top=0.87, bottom=0.12)
    curve_tables = []
    start = time.time()
    for k, cell in enumerate(cells.itertuples()):
        series = values[:, k]
        threshold = float(np.percentile(series, PERCENTILE))
        _, peaks, _ = pot.decluster(series, threshold, SEPARATION)
        excesses = peaks - threshold
        ax = axes.flat[k]
        lines = {}
        for model, prior in shown:
            if model == "GEV":
                mle = mle_gev.loc[cell.cell_id]
                fit = bayes.fit_version("GEV", prior, annual_max[:, k],
                                        [mle.location, np.log(mle.scale), mle.xi],
                                        seed(k, model, prior), periods, N_YEARS)
                mid, low, high = curve(fit, periods)
            else:
                mle = mle_pot.loc[cell.cell_id]
                fit = bayes.fit_version("POT", prior, excesses,
                                        [np.log(mle.gpd_scale), mle.gpd_shape_xi],
                                        seed(k, model, prior), periods, N_YEARS, threshold)
                mid, low, high = curve(fit, periods, threshold)
            # the curves must equal the step 15 and 16 tables at 20, 50 and 100 years
            row = tables[(tables.cell_id == cell.cell_id) & (tables.model == model)
                         & (tables.prior == prior)].set_index("return_period")
            for name, got, column in [("estimate", mid, "estimate_ms"),
                                      ("lower bound", low, "lower_95_ms"),
                                      ("upper bound", high, "upper_95_ms")]:
                style.must_match(f"{cell.candidate} {LABELS[(model, prior)]} {name}",
                                 got[table_index], row.loc[TABLE_RETURN_PERIODS, column])
            lines[(model, prior)] = (mid, low, high)
            curve_tables.append(pd.DataFrame({
                "cell_id": cell.cell_id, "location": cell.candidate,
                "version": LABELS[(model, prior)], "return_period_years": periods,
                "estimate_ms": mid, "lower_95_ms": low, "upper_95_ms": high}))

        for model, prior in shown:
            mid, low, high = lines[(model, prior)]
            color, solid = COLOR[model], prior == "wider"
            z = 3 if model == "POT" else 2
            if (model, prior) in banded:
                ax.fill_between(periods, low, high, color=color, alpha=0.13, linewidth=0,
                                zorder=z)
                for edge in (low, high):
                    ax.plot(periods, edge, color=color, linewidth=0.7, alpha=0.55, zorder=z)
            ax.plot(periods, mid, color=color, linewidth=2 if solid else 1.6,
                    linestyle="-" if solid else (0, (4, 2)), solid_capstyle="round",
                    zorder=z + 2)
        obs_period, obs_value = style.plotting_positions(annual_max[:, k])
        ax.scatter(obs_period, obs_value, s=15, color=TEXT_COLOR, alpha=0.7,
                   edgecolors="white", linewidths=0.5, zorder=6)

        ax.set_xscale("log")
        ax.set_xlim(CURVE_MIN_YEARS, CURVE_MAX_YEARS)
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
        ax.minorticks_off()
        bands = [lines[v][1] for v in banded] + [lines[v][2] for v in banded]
        y_low = min(min(np.nanmin(b) for b in bands), obs_value.min())
        y_high = max(max(np.nanmax(b) for b in bands), obs_value.max())
        pad = 0.04 * (y_high - y_low)
        ax.set_ylim(y_low - pad, y_high + 5.5 * pad)    # room at the top for the label
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_title(cell.candidate)
        gev_width = lines[("GEV", "flat")][2][table_index[2]] - lines[("GEV", "flat")][1][
            table_index[2]]
        pot_width = lines[("POT", "wider")][2][table_index[2]] - lines[("POT", "wider")][1][
            table_index[2]]
        change = 100 * (pot_width / gev_width - 1)
        word = "narrower" if change < 0 else "wider"
        ax.text(0.03, 0.965, f"POT, wider prior, 100-year\ninterval: {abs(change):.0f}% {word}",
                transform=ax.transAxes, fontsize=9, color=MUTED_TEXT, ha="left", va="top",
                linespacing=1.3)
        if k % 5 == 0:
            ax.set_ylabel("Wind speed (m/s)")
        if k in bottom_panels:
            ax.set_xlabel("Return period (years)")
            ax.tick_params(labelbottom=True)
        print(f"  {cell.candidate:<18} curves match steps 15 and 16  "
              f"({time.time() - start:.0f} s)", flush=True)

    handles = [
        Line2D([], [], color=GEV_COLOR, linewidth=1.6, linestyle=(0, (4, 2)),
               label="GEV, flat prior (CADENCE's method)"),
        Line2D([], [], color=GEV_COLOR, linewidth=2, label="GEV, wider prior"),
        Line2D([], [], color=POT_COLOR, linewidth=1.6, linestyle=(0, (4, 2)),
               label=f"POT, flat prior ({PERCENTILE}th percentile, {SEPARATION}-day rule)"),
        Line2D([], [], color=POT_COLOR, linewidth=2, label="POT, wider prior"),
        Patch(facecolor=MUTED_TEXT, alpha=0.2, linewidth=0,
              label="95% credible interval, in the color of its line"),
        Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=TEXT_COLOR,
               alpha=0.7, markeredgecolor="white", markeredgewidth=0.5,
               label="Observed annual maximum"),
    ]
    style.legend_box(fig, axes, handles, len(cells),
                     extra="Bands are shown for the flat GEV and POT with the wider prior.\n"
                           "Panel labels: 100-year interval width, POT with the\n"
                           "wider prior against the flat GEV.")
    style.heading(fig, "Return-level curves at the 13 pilot locations: checkpoint 2 versions",
                  f"Daily maximum 10 m wind speed (SPDUV10MAX), water years {FIRST_WY}{DASH}"
                  f"{LAST_WY}. Lines are posterior medians, and shaded bands are 95% "
                  "credible intervals.")
    style.footnote(fig, "Wind speed is the native model value, not a 3-second gust. Vertical "
                   "scales differ between panels.\nThe wider prior on the shape is normal, mean "
                   f"{MINUS}0.10, sd 0.20. Dots are placed at Gringorten plotting positions. "
                   "Locations run west to east.")
    name = "fig6_return_levels_checkpoint2"
    style.PLOTS.mkdir(parents=True, exist_ok=True)
    pd.concat(curve_tables, ignore_index=True).round(4).to_csv(
        style.PLOTS / f"{name}_data.csv", index=False)
    return style.save(fig, f"{name}.png", dpi)


def coverage_panel(ax, table, show_labels):
    """One panel of coverage dots: a row per version, the 95% target as a line."""
    for i, (model, prior) in enumerate(VERSIONS):
        row = table[(table.model == model) & (table.prior == prior)].iloc[0]
        half = 1.96 * row.coverage_se_pct
        ax.hlines(i, row.coverage_pct - half, row.coverage_pct + half, color=COLOR[model],
                  linewidth=2, zorder=3)
        ax.scatter(row.coverage_pct, i, s=48, marker=MARKER[prior], color=COLOR[model],
                   edgecolors="white", linewidths=1.2, zorder=4)
    ax.axvline(95, color=TEXT_COLOR, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
    ax.set_ylim(len(VERSIONS) - 0.4, -0.6)
    ax.set_yticks(range(len(VERSIONS)))
    ax.set_yticklabels([LABELS[v] for v in VERSIONS])
    ax.grid(axis="y", visible=False)
    # the axes share their y labels, so only the first column shows them
    ax.tick_params(axis="y", length=0, labelleft=show_labels)


def figure_simulation_coverage(dpi):
    """Figure 7."""
    table = pd.read_csv(OUT / "simulation_coverage.csv")
    table = table[table.return_period == 100]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.77, bottom=0.25, wspace=0.12)
    low = min(70.0, (table.coverage_pct - 1.96 * table.coverage_se_pct).min() - 2)
    for j, (truth, title) in enumerate(TRUE_MODELS.items()):
        ax = axes[j]
        coverage_panel(ax, table[table.true_model == truth], show_labels=j == 0)
        ax.set_xlim(low, 100)
        ax.set_title(title)
        ax.set_xlabel("Coverage of the true 100-year level (%)")
    style.heading(fig, "Simulation study: how often each 95% interval contains the true "
                  "100-year level",
                  "500 simulated 45-year records from each of three true models built from the "
                  "POT fits at three pilot cells (step 17).", y=0.97, gap=0.07)
    style.footnote(fig, "Dots are the share of the 500 intervals that contain the true level, "
                   "and bars are their 95% sampling range. The dashed line is 95%,\nwhere a "
                   "calibrated interval should sit. Teal is the GEV and blue is POT, and the "
                   "marker shape is the prior (circle flat, square main, diamond wider).", y=0.015)
    return style.save(fig, "fig7_simulation_coverage.png", dpi)


def figure_mixed_storm_coverage(dpi):
    """Figure 8."""
    table = pd.read_csv(OUT / "mixed_storm_simulation.csv")
    table = table[table.return_period == 100]
    xi = table.drop_duplicates("hurricane_tail").set_index("hurricane_tail").hurricane_xi
    fig, axes = plt.subplots(len(COASTAL), 3, figsize=(13.5, 11.2), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.11, wspace=0.12,
                        hspace=0.32)
    for r, candidate in enumerate(COASTAL):
        for j, tail in enumerate(TAILS):
            ax = axes[r, j]
            part = table[(table.candidate == candidate) & (table.hurricane_tail == tail)]
            coverage_panel(ax, part, show_labels=j == 0)
            ax.set_xlim(0, 100)
            sign = "+" if xi[tail] > 0 else MINUS if xi[tail] < 0 else ""
            ax.set_title(f"{candidate}, {tail} hurricane tail ({sign}{abs(xi[tail]):.2f})")
            if r == len(COASTAL) - 1:
                ax.set_xlabel("Coverage of the true 100-year level (%)")
                ax.tick_params(labelbottom=True)
    style.heading(fig, "Mixed-storm simulation: coverage at the four coastal cells",
                  "500 simulated 45-year records per cell and hurricane tail, with hurricanes and "
                  "other storms simulated separately (step 22). The hurricane\ntail's shape is "
                  "the pooled estimate (central) and the two ends of its 95% interval.",
                  y=0.975, gap=0.03)
    style.footnote(fig, "Dots are the share of the 500 intervals that contain the true level, "
                   "and bars are their 95% sampling range. The dashed line is 95%.\nTeal is the "
                   "GEV and blue is POT, and the marker shape is the prior (circle flat, square "
                   "main, diamond wider).", y=0.012)
    return style.save(fig, "fig8_mixed_storm_coverage.png", dpi)


def figure_holdout_scores(dpi):
    """Figure 9."""
    scores = pd.read_csv(OUT / "holdout_scores.csv")
    scores = scores[scores.location == "All 13 cells"]
    reference = scores[(scores.model == "GEV") & (scores.prior == "flat")].mean_twcrps.iloc[0]
    tail = pd.read_csv(OUT / "holdout_tail_scores.csv")
    tail = tail[tail.location == "All 13 cells"]
    panels = [
        ("Threshold-weighted CRPS", scores, "twcrps_diff_vs_gev_flat_pct",
         lambda row, end: 100 * row[f"twcrps_diff_vs_gev_flat_{end}"] / reference),
        ("Quantile score, 95th percentile", tail, "quantile_95_diff_vs_gev_flat_pct",
         lambda row, end: row[f"quantile_95_diff_vs_gev_flat_{end}_pct"]),
        ("90% interval score", tail, "interval_90_diff_vs_gev_flat_pct",
         lambda row, end: row[f"interval_90_diff_vs_gev_flat_{end}_pct"]),
    ]
    compared = VERSIONS[1:]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.77, bottom=0.26, wspace=0.12)
    crosses_zero = True
    for j, (title, table, column, end) in enumerate(panels):
        ax = axes[j]
        for i, (model, prior) in enumerate(compared):
            row = table[(table.model == model) & (table.prior == prior)].iloc[0]
            low, high = end(row, "low"), end(row, "high")
            crosses_zero = crosses_zero and low < 0 < high
            ax.hlines(i, low, high, color=COLOR[model], linewidth=2, zorder=3)
            ax.scatter(row[column], i, s=48, marker=MARKER[prior], color=COLOR[model],
                       edgecolors="white", linewidths=1.2, zorder=4)
        ax.axvline(0, color=TEXT_COLOR, linewidth=1, zorder=2)
        limit = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(compared) - 0.4, -0.6)
        ax.set_yticks(range(len(compared)))
        ax.set_yticklabels([LABELS[v] for v in compared])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelleft=j == 0)
        ax.set_title(title)
        ax.set_xlabel("Difference from the flat GEV (%)")
    style.heading(fig, "Held-out years: each version's score against the flat GEV",
                  "All 13 cells and 585 held-out annual maxima (steps 18 and 18a). Below 0 "
                  "scores better than the flat GEV.", y=0.97, gap=0.07)
    style.footnote(fig, "Dots are the pooled difference, and bars are the 95% range from 2,000 "
                   "bootstrap resamples of the held-out years."
                   + (" Every range crosses 0." if crosses_zero else "") + "\nTeal is the "
                   "GEV and blue is POT, and the marker shape is the prior (circle flat, square "
                   "main, diamond wider).", y=0.015)
    return style.save(fig, "fig9_holdout_scores.png", dpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    require(DAILY_CSV, *[OUT / name for name in [
        "pilot_locations_selected.csv", "pilot_gev_parameters.csv", "pilot_pot_results.csv",
        "bayes_fits.csv", "prior_sensitivity.csv", "simulation_coverage.csv",
        "mixed_storm_simulation.csv", "holdout_scores.csv", "holdout_tail_scores.csv"]])
    style.apply_style()
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")    # west to east
    values, annual_max = bayes.pilot_series(cells)
    figure_return_levels(cells, values, annual_max, args.dpi)
    figure_simulation_coverage(args.dpi)
    figure_mixed_storm_coverage(args.dpi)
    figure_holdout_scores(args.dpi)
    print("Figure 6 curves equal steps 15 and 16 at 20, 50 and 100 years (within 0.01 m/s) at "
          "every location.")
