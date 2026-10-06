"""Step 7. Make the checkpoint plots.

Figure 1  Return-level curves, GEV against POT, one panel per pilot location, with 95%
          delta-method intervals and the observed annual maxima.
Figure 2  POT sensitivity: the N-year estimate and its interval at each threshold and
          separation rule, against the GEV estimate and interval.
Figure 3  Change in the width of the 95% interval, POT against GEV, by location, return period
          and threshold.
Figure 4  The pilot locations, colored by daily completeness.

Units. All wind speeds are native model values (SPDUV10MAX, m/s). With --gust3s, Figures 1 and
2 are also written as separate files in 3-second gust (native x 1.1176, the pipeline's Durst
factor), with "3s_gust" in the file name and the unit stated on the figure. Figure 3 does not
depend on the unit, because a percent change in width is the same in either unit. Figure 4 is
a map.

Where the numbers come from. Figure 1 refits the GEV and the POT model from the daily values,
so that the curves are smooth, and then checks that the curves equal
output/pilot_results_table.csv at 20, 50 and 100 years. Figures 2 and 3 are drawn straight
from output/pilot_results_table.csv. Figure 4 uses output/pilot_locations_selected.csv.

Colors and fonts follow CONUS404_Processing/CADENCE_Design_System.md.

Usage:
    python 13_make_plots.py [--threshold 98] [--separation 2] [--return-period 100] [--gust3s]
Reads:
    data/pilot_daily_spduv10max.csv, data/manifest.json (for the map projection)
    output/pilot_locations_selected.csv, output/pilot_results_table.csv
    natural_earth/ne_50m_admin_1_states_provinces_lakes.geojson
Writes, all in output/plots/:
    fig1_return_levels_pot<threshold>_sep<separation>.png
    fig1_return_levels_pot<threshold>_sep<separation>_data.csv    (the plotted curves)
    fig2_pot_sensitivity_rp<return period>.png
    fig3_width_change_sep<separation>.png
    fig4_pilot_locations.png
"""
import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")    # draw straight to files; no window is opened

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import genextreme

import pot
from paths import (DAILY_CSV, FIRST_WY, GUST_FACTOR, LAST_WY, MANIFEST, N_YEARS, NATURAL_EARTH,
                   OUT, require)

warnings.filterwarnings("ignore")
STATES = NATURAL_EARTH / "ne_50m_admin_1_states_provinces_lakes.geojson"
PLOTS = OUT / "plots"
Z95 = 1.96    # a 95% interval is the estimate plus or minus 1.96 standard errors
TABLE_RETURN_PERIODS = [20, 50, 100]    # the return periods in the results table
PERCENTILES = [95, 97, 98, 99]
SEPARATIONS = [1, 2, 3]
# Return-period range of Figure 1. Starting at 1.01 years shows all 45 annual maxima.
CURVE_MIN_YEARS = 1.01
CURVE_MAX_YEARS = 200.0

# Colors from the CADENCE design system.
TEAL = "#2F8F7F"
GREY = "#333333"
TAN = "#D9A341"
BLUE = "#205196"
LIGHT_BLUE = "#6FA8DC"
ICE = "#E9ECEF"
GEV_COLOR = TEAL    # GEV and POT keep the same two colors in every figure
POT_COLOR = BLUE
TEXT_COLOR = GREY
MUTED_TEXT = "#5C5C5C"
AXIS_COLOR = "#CCCCCC"
# The separation rule has an order (1 < 2 < 3 days), so it gets one hue from light to dark,
# plus a different marker shape for each.
SEP_COLOR = {1: LIGHT_BLUE, 2: "#4279B9", 3: BLUE}
SEP_MARKER = {1: "o", 2: "s", 3: "D"}
# For a change that can go either way: blue = POT narrower, neutral at zero, tan = POT wider.
DIVERGING = LinearSegmentedColormap.from_list("cadence_diverging", [BLUE, ICE, TAN])
# For completeness, one hue from light to dark. Only used if completeness differs between
# locations.
SEQUENTIAL = LinearSegmentedColormap.from_list("cadence_sequential", ["#D3E8E4", TEAL])
MINUS = "\u2212"    # a proper minus sign
DASH = "\u2013"     # en dash, for ranges of years
TIMES = "\u00d7"    # multiplication sign


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
        "axes.edgecolor": AXIS_COLOR, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": ICE, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 5,
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.frameon": False, "legend.fontsize": 10.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def heading(fig, title, subtitle, x=0.05, y=0.975, gap=0.036):
    """Title and subtitle at the top left of the figure."""
    fig.text(x, y, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(x, y - gap, subtitle, fontsize=10.5, color=MUTED_TEXT, ha="left", va="top",
             linespacing=1.35)


def footnote(fig, text, x=0.05, y=0.012):
    fig.text(x, y, text, fontsize=9, color=MUTED_TEXT, ha="left", va="bottom", linespacing=1.35)


def save(fig, name, dpi):
    """Save the figure, and warn if any text or legend runs off its edge."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    # Fonts differ between computers, so text that fits on one may not fit on another.
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    for item in list(fig.texts) + list(fig.legends):
        box = item.get_window_extent().transformed(to_figure)
        if box.x0 < 0 or box.x1 > 1 or box.y0 < 0 or box.y1 > 1:
            what = item.get_text()[:60] if hasattr(item, "get_text") else "legend"
            print(f"WARNING: text runs off the edge of {name}: {what!r}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# Model curves for Figure 1.

def gev_grad(R, shape, loc, scale):
    """Derivatives of the GEV return level with respect to (shape, loc, scale).

    The return level is loc + scale * (1 - y**shape) / shape, as in 04_gev_baseline.py.
    """
    log_y = np.log(-np.log(1.0 - 1.0 / R))
    if abs(shape) < 1e-6:
        return np.array([-scale * log_y * log_y / 2.0, 1.0, -log_y])
    y_shape = np.exp(shape * log_y)
    return np.array([
        scale * (-(1.0 - y_shape) / shape**2 - y_shape * log_y / shape),
        1.0,
        (1.0 - y_shape) / shape,
    ])


def gev_curve(annual_max, periods):
    """Fit the GEV to the annual maxima by maximum likelihood.

    Returns the estimate and the lower and upper 95% delta-method bounds at each return period.
    """
    shape, loc, scale = genextreme.fit(annual_max)
    params = np.array([shape, loc, scale])
    cov = np.linalg.inv(pot.hessian(lambda p: genextreme.nnlf(p, annual_max), params))
    estimate = genextreme.isf(1.0 / periods, shape, loc, scale)
    grads = (gev_grad(period, shape, loc, scale) for period in periods)
    se = np.array([np.sqrt(grad @ cov @ grad) for grad in grads])
    return estimate, estimate - Z95 * se, estimate + Z95 * se


def pot_curve(values, percentile, separation, periods):
    """Fit the POT model (see pot.py) to one cell's daily values.

    Returns the estimate and the lower and upper 95% delta-method bounds at each return period.
    """
    threshold = float(np.percentile(values, percentile))
    _, peaks, _ = pot.decluster(values, threshold, separation)
    excesses = (peaks - threshold).astype(np.float64)
    fit = pot.fit_gpd(excesses)
    sigma, xi = fit["sigma"], fit["xi"]
    cov = np.linalg.inv(pot.hessian(lambda p: pot.gpd_nll(p, excesses), np.array([sigma, xi])))
    rate = excesses.size / N_YEARS
    estimate = np.empty(periods.size)
    se = np.empty(periods.size)
    for k, period in enumerate(periods):
        estimate[k] = pot.return_level(period, threshold, rate, sigma, xi)
        grad = pot.return_level_grad(period, rate, sigma, xi)
        se[k] = np.sqrt(grad[0] ** 2 * rate / N_YEARS + grad[1:] @ cov @ grad[1:])
    # The POT model only describes values above its threshold, so the curve is left blank
    # below it.
    below = estimate < threshold
    estimate[below] = np.nan
    se[below] = np.nan
    return estimate, estimate - Z95 * se, estimate + Z95 * se


def plotting_positions(annual_max):
    """Return period to plot each observed annual maximum at (Gringorten plotting positions)."""
    ordered = np.sort(annual_max)
    probability = (np.arange(1, ordered.size + 1) - 0.44) / (ordered.size + 0.12)
    return 1.0 / (1.0 - probability), ordered


def luminance(rgb):
    """How light a color looks, from 0 (black) to 1 (white). WCAG relative luminance."""
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


TEXT_RGB = tuple(int(GREY[i:i + 2], 16) / 255 for i in (1, 3, 5))


def signed(value, unit=""):
    """A whole number with its sign written out. A value that rounds to zero is written 0."""
    n = int(round(abs(value)))
    if n == 0:
        return f"0{unit}"
    return ("+" if value > 0 else MINUS) + f"{n}{unit}"


def must_match(what, got, want, tol=0.01):
    """Stop if the plotted values differ from the results table."""
    if not np.allclose(got, want, atol=tol):
        raise SystemExit(
            f"{what}: the plotted values {np.round(got, 3)} differ from the results table "
            f"{np.round(want, 3)} by more than {tol} m/s. Re-run the earlier steps."
        )


def grid_axes(n_panels, figsize, top, bottom):
    """A 3 x 5 grid of small panels. The first n_panels are used; the rest hold the legend.

    Also returns the lowest used panel of each column, which is where the x labels go.
    """
    fig, axes = plt.subplots(3, 5, figsize=figsize, sharex=True)
    fig.subplots_adjust(left=0.05, right=0.985, top=top, bottom=bottom, wspace=0.27,
                        hspace=0.36)
    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    last_in_column = {}
    for k in range(n_panels):
        last_in_column[k % 5] = k
    return fig, axes, set(last_in_column.values())


def legend_box(fig, axes, handles, n_used, extra=None):
    """Put the legend in the first unused panel, with an optional note under it."""
    position = axes.flat[n_used].get_position()
    legend = fig.legend(handles=handles, loc="upper left",
                        bbox_to_anchor=(position.x0 - 0.005, position.y1 + 0.01),
                        handlelength=2.2, labelspacing=0.75, borderaxespad=0.0)
    if extra:
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(position.x0 - 0.002, box.y0 - 0.02, extra, fontsize=9.5, color=MUTED_TEXT,
                 ha="left", va="top", linespacing=1.4)


def figure_return_levels(cells, table, daily, water_year, percentile, separation, factor,
                         quantity, suffix, dpi):
    """Figure 1. `factor` is 1 for native wind speed or GUST_FACTOR for 3-second gust."""
    periods = np.unique(np.r_[np.geomspace(CURVE_MIN_YEARS, CURVE_MAX_YEARS, 120),
                              TABLE_RETURN_PERIODS])
    # where 20, 50 and 100 years sit in `periods`
    table_index = [int(np.flatnonzero(periods == N)[0]) for N in TABLE_RETURN_PERIODS]
    fig, axes, bottom_panels = grid_axes(len(cells), (13.5, 8.4), top=0.865, bottom=0.125)
    curve_tables = []
    for k, cell in enumerate(cells.itertuples()):
        values = daily[str(cell.cell_id)].to_numpy(np.float32).astype(np.float64)
        annual_max = np.array([values[water_year == year].max()
                               for year in range(FIRST_WY, LAST_WY + 1)])
        gev_est, gev_lower, gev_upper = gev_curve(annual_max, periods)
        pot_est, pot_lower, pot_upper = pot_curve(values, percentile, separation, periods)

        # The curves must agree with the results table at 20, 50 and 100 years.
        cell_table = table[table.cell_id == cell.cell_id]
        gev_row = cell_table[cell_table.method == "GEV"].iloc[0]
        pot_row = cell_table[(cell_table.method == "POT")
                             & (cell_table.threshold_pct == percentile)
                             & (cell_table.separation_days == separation)].iloc[0]
        to_check = (((gev_est, gev_lower, gev_upper), gev_row, "GEV"),
                    ((pot_est, pot_lower, pot_upper), pot_row, "POT"))
        for (est, lower, upper), row, name in to_check:
            must_match(f"{cell.candidate} {name} estimate", est[table_index],
                       [row[f"rp{N}_est_ms"] for N in TABLE_RETURN_PERIODS])
            must_match(f"{cell.candidate} {name} lower bound", lower[table_index],
                       [row[f"rp{N}_delta_lower_ms"] for N in TABLE_RETURN_PERIODS])
            must_match(f"{cell.candidate} {name} upper bound", upper[table_index],
                       [row[f"rp{N}_delta_upper_ms"] for N in TABLE_RETURN_PERIODS])

        # Draw the band, its two edges and the estimate line, for the GEV and then the POT.
        ax = axes.flat[k]
        to_draw = ((gev_est, gev_lower, gev_upper, GEV_COLOR, 2),
                   (pot_est, pot_lower, pot_upper, POT_COLOR, 3))
        for est, lower, upper, color, zorder in to_draw:
            ax.fill_between(periods, lower * factor, upper * factor, color=color, alpha=0.13,
                            linewidth=0, zorder=zorder)
            for edge in (lower, upper):
                ax.plot(periods, edge * factor, color=color, linewidth=0.7, alpha=0.55,
                        zorder=zorder)
            ax.plot(periods, est * factor, color=color, linewidth=2, solid_capstyle="round",
                    zorder=zorder + 2)
        obs_period, obs_value = plotting_positions(annual_max)
        ax.scatter(obs_period, obs_value * factor, s=15, color=TEXT_COLOR, alpha=0.7,
                   edgecolors="white", linewidths=0.5, zorder=6)

        ax.set_xscale("log")
        ax.set_xlim(CURVE_MIN_YEARS, CURVE_MAX_YEARS)
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
        ax.minorticks_off()
        y_low = min(gev_lower.min(), np.nanmin(pot_lower), obs_value.min()) * factor
        y_high = max(gev_upper.max(), np.nanmax(pot_upper), obs_value.max()) * factor
        pad = 0.04 * (y_high - y_low)
        ax.set_ylim(y_low - pad, y_high + 5.5 * pad)    # extra room at the top for the label
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        ax.set_title(cell.candidate)
        change = pot_row["rp100_width_change_delta_pct"]
        word = "narrower" if change < 0 else "wider"
        ax.text(0.03, 0.965, f"POT 100-year interval:\n{abs(change):.0f}% {word}",
                transform=ax.transAxes, fontsize=9, color=MUTED_TEXT, ha="left", va="top",
                linespacing=1.3)
        if k % 5 == 0:
            ax.set_ylabel(f"{quantity} (m/s)")
        if k in bottom_panels:
            ax.set_xlabel("Return period (years)")
            ax.tick_params(labelbottom=True)
        curve_tables.append(pd.DataFrame({
            "cell_id": cell.cell_id, "location": cell.candidate,
            "return_period_years": periods,
            "gev_est": gev_est * factor,
            "gev_lower": gev_lower * factor, "gev_upper": gev_upper * factor,
            "pot_est": pot_est * factor,
            "pot_lower": pot_lower * factor, "pot_upper": pot_upper * factor,
        }))

    handles = [
        Line2D([], [], color=GEV_COLOR, linewidth=2, label="GEV estimate (45 annual maxima)"),
        Line2D([], [], color=POT_COLOR, linewidth=2,
               label=f"POT estimate ({percentile}th percentile, {separation}-day separation)"),
        Patch(facecolor=MUTED_TEXT, alpha=0.2, linewidth=0,
              label="95% delta-method interval, in the color of its line"),
        Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=TEXT_COLOR,
               alpha=0.7, markeredgecolor="white", markeredgewidth=0.5,
               label="Observed annual maximum"),
    ]
    legend_box(fig, axes, handles, len(cells),
               extra="The label in each panel is the change in interval width at 100 years,\n"
                     "POT against GEV, from the results table.")
    if factor == 1:
        what = "Daily maximum 10 m wind speed (SPDUV10MAX)"
    else:
        what = f"SPDUV10MAX converted to 3-second gust (native {TIMES} {GUST_FACTOR})"
    heading(fig, "Return-level curves at the 13 pilot locations: GEV vs POT",
            f"{what}, water years {FIRST_WY}{DASH}{LAST_WY}. "
            "Lines are estimates; shaded bands are 95% delta-method intervals.")
    if factor == 1:
        units_note = "Wind speed is the native model value, not a 3-second gust. "
    else:
        units_note = (f"Values are 3-second gusts: the native model value multiplied by "
                      f"{GUST_FACTOR}. ")
    footnote(fig, units_note + "Vertical scales differ between panels.\n"
             f"POT threshold: the {percentile}th percentile of each location's own daily values. "
             "Dots are placed at Gringorten plotting positions. Locations run west to east.")

    name = f"fig1_return_levels_pot{percentile}_sep{separation}{suffix}"
    curves = pd.concat(curve_tables, ignore_index=True)
    curves.insert(3, "quantity", f"{quantity} (m/s)")
    PLOTS.mkdir(parents=True, exist_ok=True)
    curves.round(4).to_csv(PLOTS / f"{name}_data.csv", index=False)
    return save(fig, f"{name}.png", dpi)


def figure_sensitivity(cells, table, N, factor, quantity, suffix, dpi):
    """Figure 2. The N-year estimate at each of the 12 POT settings, against the GEV."""
    fig, axes, bottom_panels = grid_axes(len(cells), (13.5, 8.4), top=0.865, bottom=0.125)
    # the three separation rules are drawn side by side within each threshold
    offsets = {1: -0.22, 2: 0.0, 3: 0.22}
    for k, cell in enumerate(cells.itertuples()):
        ax = axes.flat[k]
        cell_table = table[table.cell_id == cell.cell_id]
        gev = cell_table[cell_table.method == "GEV"].iloc[0]
        ax.axhspan(gev[f"rp{N}_delta_lower_ms"] * factor, gev[f"rp{N}_delta_upper_ms"] * factor,
                   color=GEV_COLOR, alpha=0.13, linewidth=0, zorder=1)
        ax.axhline(gev[f"rp{N}_est_ms"] * factor, color=GEV_COLOR, linewidth=2, zorder=2)
        pot_table = cell_table[cell_table.method == "POT"]
        for separation in SEPARATIONS:
            by_threshold = pot_table[pot_table.separation_days == separation]
            by_threshold = by_threshold.set_index("threshold_pct").loc[PERCENTILES]
            x = np.arange(len(PERCENTILES)) + offsets[separation]
            est = by_threshold[f"rp{N}_est_ms"].to_numpy() * factor
            lower = by_threshold[f"rp{N}_delta_lower_ms"].to_numpy() * factor
            upper = by_threshold[f"rp{N}_delta_upper_ms"].to_numpy() * factor
            ax.vlines(x, lower, upper, color=SEP_COLOR[separation], linewidth=1.6, zorder=3)
            ax.scatter(x, est, s=34, marker=SEP_MARKER[separation], color=SEP_COLOR[separation],
                       edgecolors="white", linewidths=0.9, zorder=4)
        y_low = min(pot_table[f"rp{N}_delta_lower_ms"].min(),
                    gev[f"rp{N}_delta_lower_ms"]) * factor
        y_high = max(pot_table[f"rp{N}_delta_upper_ms"].max(),
                     gev[f"rp{N}_delta_upper_ms"]) * factor
        pad = 0.07 * (y_high - y_low)
        ax.set_ylim(y_low - pad, y_high + pad)
        ax.set_xlim(-0.55, len(PERCENTILES) - 0.45)
        ax.set_xticks(range(len(PERCENTILES)))
        ax.set_xticklabels([f"{percentile}th" for percentile in PERCENTILES])
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
        ax.set_title(cell.candidate)
        if k % 5 == 0:
            ax.set_ylabel(f"{N}-year {quantity.lower()} (m/s)")
        if k in bottom_panels:
            ax.set_xlabel("POT threshold (percentile)")
            ax.tick_params(labelbottom=True)

    handles = [
        Line2D([], [], color=GEV_COLOR, linewidth=2, label="GEV estimate"),
        Patch(facecolor=GEV_COLOR, alpha=0.2, linewidth=0, label="GEV 95% delta-method interval"),
    ]
    for separation in SEPARATIONS:
        handles.append(Line2D(
            [], [], color=SEP_COLOR[separation], linewidth=1.6, marker=SEP_MARKER[separation],
            markersize=6.5, markeredgecolor="white", markeredgewidth=0.9,
            label=f"POT, {separation}-day separation: estimate and 95% interval"))
    legend_box(fig, axes, handles, len(cells),
               extra="Within each threshold the three separation rules sit left to right:\n"
                     "1, 2 and 3 days.")
    if factor == 1:
        what = "SPDUV10MAX"
    else:
        what = f"3-second gust (native {TIMES} {GUST_FACTOR})"
    heading(fig, f"POT sensitivity to threshold and separation rule: {N}-year {quantity.lower()}",
            f"POT estimate and 95% delta-method interval at each of the 12 settings, "
            f"against the GEV. {what}, water years {FIRST_WY}{DASH}{LAST_WY}.")
    if factor == 1:
        units_note = "Wind speed is the native model value, not a 3-second gust. "
    else:
        units_note = (f"Values are 3-second gusts: the native model value multiplied by "
                      f"{GUST_FACTOR}. ")
    footnote(fig, units_note + "Vertical scales differ between panels.\n"
             "The threshold is a percentile of each location's own daily values. "
             "Locations run west to east.")
    return save(fig, f"fig2_pot_sensitivity_rp{N}{suffix}.png", dpi)


def figure_width_change(cells, table, separation, dpi):
    """Figure 3. A grid of numbers: percent change in interval width, POT against GEV."""
    pot_table = table[(table.method == "POT") & (table.separation_days == separation)]
    names = list(cells.candidate)

    def by_location_and_threshold(column):
        wide = pot_table.pivot(index="location", columns="threshold_pct", values=column)
        return wide.loc[names, PERCENTILES].to_numpy()

    change = {}
    unsure = {}
    for N in TABLE_RETURN_PERIODS:
        delta = by_location_and_threshold(f"rp{N}_width_change_delta_pct")
        boot = by_location_and_threshold(f"rp{N}_width_change_boot_pct")
        years = by_location_and_threshold(f"rp{N}_width_change_years_pct")
        change[N] = delta
        # A cell is marked unsure if either check gives the opposite sign to the delta method,
        # that is, narrower where the delta method says wider or the other way round.
        unsure[N] = (np.sign(boot) != np.sign(delta)) | (np.sign(years) != np.sign(delta))
    # the color scale runs to the next multiple of 10 above the largest change, both ways
    limit = 10 * np.ceil(max(np.abs(block).max() for block in change.values()) / 10)
    norm = Normalize(-limit, limit)

    fig = plt.figure(figsize=(12.5, 7.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[len(names), 1], left=0.135, right=0.985,
                            top=0.795, bottom=0.185, wspace=0.045, hspace=0.07)

    def draw_block(ax, values, marks):
        """Colored cells with the number written in each. `marks` adds an asterisk."""
        ax.pcolormesh(np.arange(values.shape[1] + 1), np.arange(values.shape[0] + 1), values,
                      cmap=DIVERGING, norm=norm, edgecolors="white", linewidth=2)
        ax.invert_yaxis()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                # use white text on dark cells and dark text on light cells, whichever has
                # the better contrast
                fill = luminance(DIVERGING(norm(value))[:3])
                contrast_with_white = (1.05) / (fill + 0.05)
                contrast_with_dark = (fill + 0.05) / (luminance(TEXT_RGB) + 0.05)
                white_better = contrast_with_white > contrast_with_dark
                label = signed(value) + ("*" if marks is not None and marks[i, j] else "")
                ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=9.5,
                        color="white" if white_better else TEXT_COLOR)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    for k, N in enumerate(TABLE_RETURN_PERIODS):
        ax = fig.add_subplot(grid[0, k])
        draw_block(ax, change[N], unsure[N])
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(len(PERCENTILES)) + 0.5)
        ax.set_xticklabels([f"{percentile}th" for percentile in PERCENTILES])
        ax.set_yticks(np.arange(len(names)) + 0.5)
        ax.set_yticklabels(names if k == 0 else [])
        ax.set_title(f"{N}-year return period", loc="center", pad=26, fontsize=11.5)
        if k == 0:
            ax.annotate("POT threshold:", xy=(0, 1), xycoords="axes fraction", xytext=(-8, 6),
                        textcoords="offset points", ha="right", va="bottom", fontsize=9.5,
                        color=MUTED_TEXT)
        # the row of medians under each block
        median_ax = fig.add_subplot(grid[1, k])
        draw_block(median_ax, np.median(change[N], axis=0, keepdims=True), None)
        median_ax.set_xticks([])
        median_ax.set_yticks([0.5])
        median_ax.set_yticklabels(["Median of the 13"] if k == 0 else [], fontweight="bold")

    colorbar_ax = fig.add_axes([0.36, 0.118, 0.40, 0.02])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=colorbar_ax,
                            orientation="horizontal")
    colorbar.outline.set_visible(False)
    ticks = np.arange(-limit, limit + 1, limit / 4)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([signed(tick, "%") for tick in ticks])
    colorbar_ax.tick_params(length=0, labelsize=9.5)
    fig.text(0.352, 0.128, "POT interval narrower", ha="right", va="center", fontsize=10.5)
    fig.text(0.768, 0.128, "POT interval wider", ha="left", va="center", fontsize=10.5)

    heading(fig, "Change in 95% interval width: POT vs GEV",
            f"Percent change in the width of the 95% delta-method interval, POT against GEV, "
            f"{separation}-day separation. "
            f"Negative means the POT interval is narrower.\n"
            f"The threshold is a percentile of each location's own daily values. "
            f"SPDUV10MAX, water years {FIRST_WY}{DASH}{LAST_WY}. Locations run west to east.",
            x=0.02, gap=0.045)
    footnote(fig, "* The parametric bootstrap or the resampled-years check gives the opposite "
             "sign at this setting, so narrower-or-wider is not settled there.\n"
             "Values are from pilot_results_table.csv. "
             "A percent change in width is the same in native m/s and in 3-second gust.", x=0.02)
    return save(fig, f"fig3_width_change_sep{separation}.png", dpi)


def lambert(manifest_path):
    """The Lambert conformal projection of the CONUS404 grid, as in step 01.

    Returns a function that turns longitude and latitude into projected x and y in km.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        attrs = json.load(fh)["dataset_attrs"]
    truelat1 = np.radians(float(attrs["TRUELAT1"]))
    truelat2 = np.radians(float(attrs["TRUELAT2"]))
    stand_lon = float(attrs["STAND_LON"])
    cone = (np.log(np.cos(truelat1) / np.cos(truelat2))
            / np.log(np.tan(np.pi / 4 + truelat2 / 2) / np.tan(np.pi / 4 + truelat1 / 2)))
    F = np.cos(truelat1) * np.tan(np.pi / 4 + truelat1 / 2) ** cone / cone
    radius_km = 6370.0    # the sphere WRF uses

    def project(lon, lat):
        rho = radius_km * F / np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2) ** cone
        theta = cone * np.radians(np.asarray(lon, float) - stand_lon)
        return rho * np.sin(theta), -rho * np.cos(theta)

    return project


def lower48_rings(project):
    """Projected outlines of the lower 48 states and DC, from the Natural Earth file."""
    with open(STATES, encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    rings = []
    for feature in features:
        properties = feature["properties"]
        if properties.get("adm0_a3") != "USA" or properties.get("postal") in ("AK", "HI"):
            continue
        geometry = feature["geometry"]
        if geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            polygons = [geometry["coordinates"]]
        for polygon in polygons:
            lon_lat = np.asarray(polygon[0], float)
            rings.append(np.column_stack(project(lon_lat[:, 0], lon_lat[:, 1])))
    us_states = {feature["properties"].get("postal") for feature in features
                 if feature["properties"].get("adm0_a3") == "USA"}
    if len(us_states) < 51:
        raise SystemExit(f"{STATES.name} does not hold the 50 states and DC")
    return rings


# Where each city's label sits relative to its dot: x offset, y offset (in points) and which
# side of the label is anchored. Cities not listed get their label to the right of the dot.
LABEL_OFFSET = {
    "Houston, TX": (-9, -3, "right"),
    "Miami, FL": (-9, 0, "right"),
    "Boston, MA": (-9, 5, "right"),
    "Seattle, WA": (9, -2, "left"),
    "Rapid City, SD": (9, 3, "left"),
}


def figure_map(cells, dpi):
    """Figure 4. The pilot locations on a map of the lower 48 states."""
    project = lambert(MANIFEST)
    rings = lower48_rings(project)
    fig = plt.figure(figsize=(11, 7.4))
    ax = fig.add_axes([0.02, 0.105, 0.96, 0.755])
    ax.add_collection(PolyCollection(rings, facecolors=ICE, edgecolors="white", linewidths=0.8,
                                     zorder=1))
    x, y = project(cells.longitude.to_numpy(), cells.latitude.to_numpy())
    completeness = cells.completeness_pct.to_numpy(float)
    # If every location has the same completeness, one color says so more plainly than a scale.
    uniform = np.ptp(completeness) == 0
    if uniform:
        ax.scatter(x, y, s=95, color=TEAL, edgecolors="white", linewidths=2, zorder=3)
    else:
        norm = Normalize(min(95.0, np.floor(completeness.min())), 100.0)
        dots = ax.scatter(x, y, s=95, c=completeness, cmap=SEQUENTIAL, norm=norm,
                          edgecolors="white", linewidths=2, zorder=3)
        colorbar_ax = fig.add_axes([0.07, 0.17, 0.22, 0.02])
        colorbar = fig.colorbar(dots, cax=colorbar_ax, orientation="horizontal")
        colorbar.outline.set_visible(False)
        colorbar.set_label("Daily values present (%)")
    for cell, dot_x, dot_y in zip(cells.itertuples(), x, y):
        dx, dy, anchor = LABEL_OFFSET.get(cell.candidate, (9, 0, "left"))
        if uniform:
            label = cell.candidate
        else:
            label = f"{cell.candidate}  {cell.completeness_pct:.1f}%"
        ax.annotate(label, (dot_x, dot_y), xytext=(dx, dy), textcoords="offset points",
                    ha=anchor, va="center", fontsize=10.5, zorder=4)
    outline = np.vstack(rings)
    margin_x = 0.03 * np.ptp(outline[:, 0])
    margin_y = 0.03 * np.ptp(outline[:, 1])
    ax.set_xlim(outline[:, 0].min() - margin_x, outline[:, 0].max() + margin_x)
    ax.set_ylim(outline[:, 1].min() - margin_y, outline[:, 1].max() + margin_y)
    ax.set_aspect("equal")
    ax.axis("off")

    days = f"{int(cells.valid_days.min()):,} of {int(cells.expected_days.max()):,}"
    if uniform:
        handle = Line2D(
            [], [], linestyle="none", marker="o", markersize=10, markerfacecolor=TEAL,
            markeredgecolor="white", markeredgewidth=2,
            label=(f"{completeness[0]:.0f}% of daily values present "
                   f"({len(cells)} of {len(cells)} locations)"))
        fig.legend(handles=[handle], loc="lower left", bbox_to_anchor=(0.045, 0.135),
                   handletextpad=0.4)
        subtitle = (f"Every pilot cell has {days} daily values and "
                    f"{int(cells.valid_years.min())} of {N_YEARS} "
                    f"water years ({FIRST_WY}{DASH}{LAST_WY}), "
                    f"so all {len(cells)} share one color.")
    else:
        subtitle = (f"Share of the {int(cells.expected_days.max()):,} daily values that are "
                    f"present, water years {FIRST_WY}{DASH}{LAST_WY}.")
    heading(fig, f"The {len(cells)} pilot locations, colored by daily completeness", subtitle,
            x=0.045, gap=0.045)
    if uniform:
        same_everywhere = ("Completeness is the same at every cell on the grid, so the locations "
                           "were chosen for geographic variety.\n")
    else:
        same_everywhere = ""
    footnote(fig, "Each dot is the CONUS404 4 km grid cell nearest the city.\n" + same_everywhere
             + "Map projection: Lambert conformal, the CONUS404 grid's own projection. "
             "State outlines: Natural Earth.",
             x=0.045)
    return save(fig, "fig4_pilot_locations.png", dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=int, default=98, choices=PERCENTILES,
                        help="POT threshold percentile shown in Figure 1 (default 98)")
    parser.add_argument("--separation", type=int, default=2, choices=SEPARATIONS,
                        help="POT separation rule in Figures 1 and 3 (default 2)")
    parser.add_argument("--return-period", type=int, default=100, choices=TABLE_RETURN_PERIODS,
                        help="return period of Figure 2 (default 100)")
    parser.add_argument("--gust3s", action="store_true",
                        help="also write Figures 1 and 2 in 3-second gust")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    require(DAILY_CSV, MANIFEST, STATES, OUT / "pilot_locations_selected.csv",
            OUT / "pilot_results_table.csv")
    apply_style()
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")    # west to east
    table = pd.read_csv(OUT / "pilot_results_table.csv")
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    # a water year is named after the year it ends in (October to September)
    water_year = np.where(daily.date.dt.month >= 10, daily.date.dt.year + 1, daily.date.dt.year)
    water_year = water_year.astype(int)
    same_cells = list(table.drop_duplicates("cell_id").cell_id) == list(cells.cell_id)
    assert same_cells, "results table and pilot table disagree"
    assert set(np.unique(water_year)) == set(range(FIRST_WY, LAST_WY + 1))

    figure_return_levels(cells, table, daily, water_year, args.threshold, args.separation,
                         1.0, "Wind speed", "", args.dpi)
    figure_sensitivity(cells, table, args.return_period, 1.0, "Wind speed", "", args.dpi)
    figure_width_change(cells, table, args.separation, args.dpi)
    figure_map(cells, args.dpi)
    if args.gust3s:
        figure_return_levels(cells, table, daily, water_year, args.threshold, args.separation,
                             GUST_FACTOR, "3-second gust", "_3s_gust", args.dpi)
        figure_sensitivity(cells, table, args.return_period, GUST_FACTOR, "3-second gust",
                           "_3s_gust", args.dpi)
    print("Figure 1 curves equal the results table at 20, 50 and 100 years "
          "(within 0.01 m/s) at every location.")


if __name__ == "__main__":
    main()
