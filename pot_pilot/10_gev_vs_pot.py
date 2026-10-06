"""Step 5b. Compare GEV and POT at 20, 50 and 100 years.

For every cell, POT setting and return period this writes the two estimates, the endpoints of
their 95% intervals, the interval widths and the percent change in width.

The GEV side is the documented maximum-likelihood fit to the 45 water-year maxima (step 04),
in native m/s. The POT side is the 12 settings per cell from step 08.

Three kinds of interval are compared:
    "delta"   the delta method. This is the headline result.
    "boot"    the parametric bootstrap (steps 05 and 09, part A). This is the check.
    "years"   the resampled-years bootstrap (step 09, part B), where the POT threshold is
              recomputed each time. This is a robustness check.

    percent change in width = 100 * (POT width - GEV width) / GEV width

The two widths always come from the same kind of interval. A negative number means the POT
interval is narrower.

Reads:
    data/pilot_daily_spduv10max.csv
    output/pilot_locations_selected.csv, output/pilot_pot_results.csv,
    output/pilot_gev_baseline.csv, output/work/cadence_vs_mle_long.csv,
    output/work/boot_param.npz, output/work/boot_years.npz
Writes:
    output/pilot_gev_vs_pot.csv         468 rows = 13 cells x 12 settings x 3 return periods
    output/work/gev_vs_pot_full.csv     the same table before rounding
"""
import warnings

import numpy as np
import pandas as pd

import pot
from paths import DAILY_CSV, N_YEARS, OUT, WORK, require

warnings.filterwarnings("ignore")
BOOT_RETURN_PERIODS = [10, 20, 25, 50, 100, 200, 500]    # the return periods in the .npz files
RETURN_PERIODS = [20, 50, 100]
PERCENTILES = [95, 97, 98, 99]
SEPARATIONS = [1, 2, 3]
METHODS = ("delta", "boot", "years")
require(DAILY_CSV, OUT / "pilot_locations_selected.csv", OUT / "pilot_pot_results.csv",
        OUT / "pilot_gev_baseline.csv", WORK / "cadence_vs_mle_long.csv",
        WORK / "boot_param.npz", WORK / "boot_years.npz")

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
pot_results = pd.read_csv(OUT / "pilot_pot_results.csv")
daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
# GEV numbers: the MLE estimate with its delta and bootstrap bounds, and the L-moments estimate
gev = pd.read_csv(WORK / "cadence_vs_mle_long.csv")
gev_published = pd.read_csv(OUT / "pilot_gev_baseline.csv")
boot_param = np.load(WORK / "boot_param.npz")["levels"]    # (156 fits, 1000, 7 return periods)
boot_years = np.load(WORK / "boot_years.npz")
assert list(boot_years["cell_id"]) == list(cells.cell_id)
assert boot_param.shape[0] == len(pot_results)


def one_row(table, cell_id, return_period):
    """The row of a table for one cell and one return period."""
    return table[(table.cell_id == cell_id) & (table.return_period == return_period)].iloc[0]


rows = []
for k, saved in enumerate(pot_results.itertuples()):
    # where this fit sits in the bootstrap arrays
    i_cell = list(cells.cell_id).index(saved.cell_id)
    i_pct = PERCENTILES.index(saved.threshold_pct)
    i_sep = SEPARATIONS.index(saved.separation_days)

    # redo the POT fit at full precision (the saved results are rounded)
    series = daily[str(saved.cell_id)].to_numpy(np.float64)
    fit = pot.pot_fit(series, saved.threshold_pct, saved.separation_days, N_YEARS, RETURN_PERIODS)

    for N in RETURN_PERIODS:
        i_rp = BOOT_RETURN_PERIODS.index(N)
        gev_row = one_row(gev, saved.cell_id, N)
        published = one_row(gev_published, saved.cell_id, N)
        assert abs(gev_row.mle_native_ms - published.native_ms) < 1e-3
        assert abs(gev_row.mle_delta_lower_native_ms - published.delta_lower_native_ms) < 1e-3

        # bootstrap intervals: the 2.5th and 97.5th percentiles of the simulated return levels
        pot_boot_lower, pot_boot_upper = np.percentile(boot_param[k, :, i_rp], [2.5, 97.5])
        gev_years_lower, gev_years_upper = np.percentile(
            boot_years["gev"][i_cell, :, i_rp], [2.5, 97.5])
        pot_years_lower, pot_years_upper = np.percentile(
            boot_years["pot"][i_cell, :, i_pct, i_sep, i_rp], [2.5, 97.5])

        gev_estimate = gev_row.mle_native_ms
        pot_estimate = fit[f"rp{N}_ms"]
        row = {
            "cell_id": saved.cell_id, "candidate": saved.candidate,
            "latitude": saved.latitude, "longitude": saved.longitude,
            "return_period": N,
            "threshold_pct": saved.threshold_pct, "separation_days": saved.separation_days,
            "threshold_ms": fit["threshold_ms"], "n_events": fit["n_events"],
            "events_per_year": fit["events_per_year"],
            "gpd_shape_xi": fit["gpd_shape_xi"],
            "gev_est_ms": gev_estimate, "pot_est_ms": pot_estimate,
            "est_diff_pct": 100 * (pot_estimate - gev_estimate) / gev_estimate,
            "cadence_lmom_est_ms": gev_row.cadence_native_ms,
            "gev_delta_lower_ms": gev_row.mle_delta_lower_native_ms,
            "gev_delta_upper_ms": gev_row.mle_delta_upper_native_ms,
            "pot_delta_lower_ms": fit[f"rp{N}_lower_ms"],
            "pot_delta_upper_ms": fit[f"rp{N}_upper_ms"],
            "gev_boot_lower_ms": gev_row.mle_boot_lower_native_ms,
            "gev_boot_upper_ms": gev_row.mle_boot_upper_native_ms,
            "pot_boot_lower_ms": pot_boot_lower, "pot_boot_upper_ms": pot_boot_upper,
            "gev_years_lower_ms": gev_years_lower, "gev_years_upper_ms": gev_years_upper,
            "pot_years_lower_ms": pot_years_lower, "pot_years_upper_ms": pot_years_upper,
        }

        # widths, and the percent change in width, for each kind of interval
        for method in METHODS:
            for model in ("gev", "pot"):
                upper = row[f"{model}_{method}_upper_ms"]
                lower = row[f"{model}_{method}_lower_ms"]
                row[f"{model}_{method}_width_ms"] = upper - lower
            gev_width = row[f"gev_{method}_width_ms"]
            pot_width = row[f"pot_{method}_width_ms"]
            row[f"width_change_{method}_pct"] = 100 * (pot_width - gev_width) / gev_width

        row["pot_est_inside_gev_delta_interval"] = bool(
            row["gev_delta_lower_ms"] <= row["pot_est_ms"] <= row["gev_delta_upper_ms"])
        row["gev_est_inside_pot_delta_interval"] = bool(
            row["pot_delta_lower_ms"] <= row["gev_est_ms"] <= row["pot_delta_upper_ms"])

        # -1 where POT is narrower, +1 where it is wider
        signs = {method: np.sign(row[f"width_change_{method}_pct"]) for method in METHODS}
        row["pot_narrower_delta"] = bool(signs["delta"] < 0)
        row["pot_narrower_boot"] = bool(signs["boot"] < 0)
        row["pot_narrower_years"] = bool(signs["years"] < 0)

        flags = []
        if signs["delta"] != signs["boot"]:
            flags.append("delta and bootstrap disagree on narrower/wider")
        if signs["delta"] != signs["years"]:
            flags.append("delta and resampled-years disagree on narrower/wider")
        if not row["gev_est_inside_pot_delta_interval"]:
            flags.append("GEV estimate outside POT delta interval")
        row["flags"] = "; ".join(flags)
        rows.append(row)

table = pd.DataFrame(rows)
assert len(table) == 13 * 12 * 3
assert table.drop(columns="flags").notna().all().all()
for method in METHODS:
    assert (table[f"gev_{method}_width_ms"] > 0).all()
    assert (table[f"pot_{method}_width_ms"] > 0).all()
    # every interval must contain its own estimate
    for model in ("gev", "pot"):
        estimate = table[f"{model}_est_ms"]
        assert (table[f"{model}_{method}_lower_ms"] < estimate).all(), (model, method)
        assert (estimate < table[f"{model}_{method}_upper_ms"]).all(), (model, method)

table.to_csv(WORK / "gev_vs_pot_full.csv", index=False)    # not rounded; steps 11 and 12 use it

# Round to 3 decimals. Identifiers, counts and coordinates are left alone.
rounded = table.copy()
not_rounded = ["cell_id", "return_period", "threshold_pct", "separation_days", "n_events",
               "latitude", "longitude"]
to_round = rounded.select_dtypes("number").columns.difference(not_rounded)
rounded[to_round] = rounded[to_round].round(3)
rounded.to_csv(OUT / "pilot_gev_vs_pot.csv", index=False)
print(f"wrote {len(rounded)} rows x {rounded.shape[1]} columns to {OUT / 'pilot_gev_vs_pot.csv'}")
print("flagged rows:", int((rounded["flags"] != "").sum()))
