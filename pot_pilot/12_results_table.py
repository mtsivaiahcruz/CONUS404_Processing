"""Step 6. Build the results table: data quality and model results by location and method.

One row per location and method setting:
    GEV    1 row per location     documented maximum-likelihood fit to the 45 water-year maxima
    POT    12 rows per location   4 thresholds (95th to 99th percentile) x 3 separation rules
                                  (1 to 3 days)
That is 13 rows for each of the 13 locations, 169 rows in all.

For each of the 20-, 50- and 100-year return periods a row has the estimate and its 95%
interval (lower, upper, width), by the delta method and by the parametric bootstrap. POT rows
also have the percent change in width against the same location's GEV row (for the delta
method, the bootstrap and the resampled-years check):

    width change = 100 * (POT width - GEV width) / GEV width      (negative = POT narrower)

The two widths always come from the same kind of interval. All wind speeds are native model
values in m/s.

Every number comes from the unrounded intermediate files and is rounded once, here.

Reads:
    output/pilot_locations_selected.csv
    output/work/gev_vs_pot_full.csv, output/work/gev_baseline_params.csv
    output/pilot_gev_baseline.csv, output/pilot_pot_results.csv (for the checks)
Writes:
    output/pilot_results_table.csv
"""
import numpy as np
import pandas as pd

from paths import FIRST_WY, LAST_WY, N_YEARS, OUT, WORK, require

RETURN_PERIODS = [20, 50, 100]
require(OUT / "pilot_locations_selected.csv", WORK / "gev_vs_pot_full.csv",
        WORK / "gev_baseline_params.csv")

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")    # west to east
# one row per cell, POT setting and return period
full = pd.read_csv(WORK / "gev_vs_pot_full.csv")
full["flags"] = full["flags"].fillna("")
gev_params = pd.read_csv(WORK / "gev_baseline_params.csv").set_index("cell_id")


def quality_columns(cell):
    """The location and data-quality columns. They are the same on every row of a location."""
    return {
        "cell_id": cell.cell_id, "location": cell.candidate, "region": cell.region,
        "lat_idx": cell.lat_idx, "lon_idx": cell.lon_idx,
        "latitude": cell.latitude, "longitude": cell.longitude,
        "valid_years": cell.valid_years,
        "first_water_year": FIRST_WY, "last_water_year": LAST_WY,
        "valid_days": cell.valid_days, "expected_days": cell.expected_days,
        "missing_days": cell.expected_days - cell.valid_days,
        "completeness_pct": cell.completeness_pct,
    }


rows = []
for cell in cells.itertuples():
    cell_rows = full[full.cell_id == cell.cell_id]
    assert len(cell_rows) == 12 * len(RETURN_PERIODS)

    # The GEV row, which is the baseline.
    # In the input file the GEV values repeat on every POT row of the cell, so check they are
    # all the same and take the first.
    row = quality_columns(cell)
    row.update(method="GEV", fit="maximum likelihood", threshold_pct=np.nan,
               threshold_ms=np.nan, separation_days=np.nan, n_events=np.nan,
               events_per_year=np.nan, values_fitted=N_YEARS,
               shape_xi=gev_params.loc[cell.cell_id, "xi"])
    gev_columns = ("gev_est_ms", "gev_delta_lower_ms", "gev_delta_upper_ms", "gev_boot_lower_ms",
                   "gev_boot_upper_ms", "cadence_lmom_est_ms")
    for N in RETURN_PERIODS:
        at_period = cell_rows[cell_rows.return_period == N]
        for col in gev_columns:
            assert at_period[col].nunique() == 1, (cell.candidate, N, col)
        gev_row = at_period.iloc[0]
        row.update({
            f"rp{N}_est_ms": gev_row.gev_est_ms, f"rp{N}_est_diff_pct": np.nan,
            f"rp{N}_delta_lower_ms": gev_row.gev_delta_lower_ms,
            f"rp{N}_delta_upper_ms": gev_row.gev_delta_upper_ms,
            f"rp{N}_delta_width_ms": gev_row.gev_delta_width_ms,
            f"rp{N}_width_change_delta_pct": np.nan,
            f"rp{N}_boot_lower_ms": gev_row.gev_boot_lower_ms,
            f"rp{N}_boot_upper_ms": gev_row.gev_boot_upper_ms,
            f"rp{N}_boot_width_ms": gev_row.gev_boot_width_ms,
            f"rp{N}_width_change_boot_pct": np.nan,
            f"rp{N}_width_change_years_pct": np.nan,
            f"cadence_lmom_rp{N}_ms": gev_row.cadence_lmom_est_ms,
        })
    row["flags"] = ""
    rows.append(row)
    # CADENCE's L-moments estimates are repeated on the POT rows too
    lmom_columns = {}
    for N in RETURN_PERIODS:
        lmom_columns[f"cadence_lmom_rp{N}_ms"] = row[f"cadence_lmom_rp{N}_ms"]

    # The 12 POT rows.
    settings = cell_rows.groupby(["threshold_pct", "separation_days"], sort=True)
    for (percentile, separation), setting_rows in settings:
        setting_rows = setting_rows.set_index("return_period")
        first = setting_rows.iloc[0]
        row = quality_columns(cell)
        row.update(method="POT", fit="maximum likelihood", threshold_pct=percentile,
                   threshold_ms=first.threshold_ms, separation_days=separation,
                   n_events=first.n_events, events_per_year=first.events_per_year,
                   values_fitted=first.n_events, shape_xi=first.gpd_shape_xi)
        flags = []
        for N in RETURN_PERIODS:
            pot_row = setting_rows.loc[N]
            row.update({
                f"rp{N}_est_ms": pot_row.pot_est_ms, f"rp{N}_est_diff_pct": pot_row.est_diff_pct,
                f"rp{N}_delta_lower_ms": pot_row.pot_delta_lower_ms,
                f"rp{N}_delta_upper_ms": pot_row.pot_delta_upper_ms,
                f"rp{N}_delta_width_ms": pot_row.pot_delta_width_ms,
                f"rp{N}_width_change_delta_pct": pot_row.width_change_delta_pct,
                f"rp{N}_boot_lower_ms": pot_row.pot_boot_lower_ms,
                f"rp{N}_boot_upper_ms": pot_row.pot_boot_upper_ms,
                f"rp{N}_boot_width_ms": pot_row.pot_boot_width_ms,
                f"rp{N}_width_change_boot_pct": pot_row.width_change_boot_pct,
                f"rp{N}_width_change_years_pct": pot_row.width_change_years_pct,
            })
            if pot_row["flags"]:
                flags.append(f"{N}-year: {pot_row['flags']}")
        row.update(lmom_columns)
        row["flags"] = " | ".join(flags)
        rows.append(row)

table = pd.DataFrame(rows)

# Checks, done on the unrounded values.
assert len(table) == 13 * 13
assert (table.method == "GEV").sum() == 13 and (table.method == "POT").sum() == 156
assert not table.duplicated(["cell_id", "method", "threshold_pct", "separation_days"]).any()
assert (table.valid_years == N_YEARS).all()
assert (table.missing_days == 0).all()
assert (table.completeness_pct == 100).all()
pot_rows = table[table.method == "POT"]
gev_rows = table[table.method == "GEV"].set_index("cell_id")
for N in RETURN_PERIODS:
    estimate = table[f"rp{N}_est_ms"]
    for method in ("delta", "boot"):
        lower = table[f"rp{N}_{method}_lower_ms"]
        upper = table[f"rp{N}_{method}_upper_ms"]
        width = table[f"rp{N}_{method}_width_ms"]
        assert np.allclose(upper - lower, width) and (width > 0).all()
        assert (lower < estimate).all() and (estimate < upper).all()
        # the width change must be reproducible from this table's own GEV row
        gev_width = pot_rows.cell_id.map(gev_rows[f"rp{N}_{method}_width_ms"])
        pot_width = pot_rows[f"rp{N}_{method}_width_ms"]
        assert np.allclose(100 * (pot_width - gev_width) / gev_width,
                           pot_rows[f"rp{N}_width_change_{method}_pct"])
    gev_estimate = pot_rows.cell_id.map(gev_rows[f"rp{N}_est_ms"])
    assert np.allclose(100 * (pot_rows[f"rp{N}_est_ms"] - gev_estimate) / gev_estimate,
                       pot_rows[f"rp{N}_est_diff_pct"])
    # the comparison columns are empty on the GEV rows
    empty_on_gev = [f"rp{N}_width_change_delta_pct", f"rp{N}_width_change_boot_pct",
                    f"rp{N}_width_change_years_pct", f"rp{N}_est_diff_pct"]
    assert gev_rows[empty_on_gev].isna().all().all()
assert pot_rows.drop(columns="flags").notna().all().all()

# Check against the tables from the earlier steps, which are rounded: the GEV rows against
# step 06 and the POT rows against step 08.
gev_baseline = pd.read_csv(OUT / "pilot_gev_baseline.csv")
gev_baseline = gev_baseline.set_index(["cell_id", "return_period"])
pot_results = pd.read_csv(OUT / "pilot_pot_results.csv")
pot_results = pot_results.set_index(["cell_id", "threshold_pct", "separation_days"])
for N in RETURN_PERIODS:
    baseline_estimate = [gev_baseline.loc[(cell_id, N), "native_ms"]
                         for cell_id in gev_rows.index]
    baseline_lower = [gev_baseline.loc[(cell_id, N), "delta_lower_native_ms"]
                      for cell_id in gev_rows.index]
    assert np.allclose(gev_rows[f"rp{N}_est_ms"], baseline_estimate, atol=6e-4)
    assert np.allclose(gev_rows[f"rp{N}_delta_lower_ms"], baseline_lower, atol=6e-4)
    key = list(zip(pot_rows.cell_id, pot_rows.threshold_pct.astype(int),
                   pot_rows.separation_days.astype(int)))
    assert np.allclose(pot_rows[f"rp{N}_est_ms"],
                       pot_results.loc[key, f"rp{N}_ms"].to_numpy(), atol=6e-5)
    assert np.allclose(pot_rows[f"rp{N}_delta_upper_ms"],
                       pot_results.loc[key, f"rp{N}_upper_ms"].to_numpy(), atol=6e-5)
assert (pot_rows.n_events.to_numpy() == pot_results.loc[key, "n_events"].to_numpy()).all()

# Put the columns in order, round once, and write the file.
first_columns = [
    "cell_id", "location", "region", "lat_idx", "lon_idx", "latitude", "longitude",
    "completeness_pct", "valid_years", "first_water_year", "last_water_year",
    "valid_days", "expected_days", "missing_days",
    "method", "fit", "threshold_pct", "threshold_ms", "separation_days", "n_events",
    "events_per_year", "values_fitted", "shape_xi",
]
period_columns = []
for N in RETURN_PERIODS:
    period_columns += [
        f"rp{N}_est_ms", f"rp{N}_delta_lower_ms", f"rp{N}_delta_upper_ms",
        f"rp{N}_delta_width_ms", f"rp{N}_width_change_delta_pct",
        f"rp{N}_boot_lower_ms", f"rp{N}_boot_upper_ms", f"rp{N}_boot_width_ms",
        f"rp{N}_width_change_boot_pct", f"rp{N}_width_change_years_pct", f"rp{N}_est_diff_pct",
    ]
last_columns = [f"cadence_lmom_rp{N}_ms" for N in RETURN_PERIODS] + ["flags"]
table = table[first_columns + period_columns + last_columns]
assert set(table.columns) == set(first_columns + period_columns + last_columns)
# these four are whole numbers that are empty on the GEV rows
for col in ["threshold_pct", "separation_days", "n_events", "values_fitted"]:
    table[col] = table[col].astype("Int64")
to_round = (["threshold_ms", "events_per_year", "shape_xi"] + period_columns
            + last_columns[:-1])
table[to_round] = table[to_round].round(3)
table.to_csv(OUT / "pilot_results_table.csv", index=False)

print(f"wrote {len(table)} rows x {table.shape[1]} columns to {OUT / 'pilot_results_table.csv'}")
n_gev = int((table.method == "GEV").sum())
n_pot = int((table.method == "POT").sum())
n_flagged = int((table["flags"] != "").sum())
print(f"  {n_gev} GEV rows, {n_pot} POT rows, "
      f"{n_flagged} POT rows with a flag at one or more return periods")
example = table[table.location == "New Orleans, LA"]
example = example[["method", "threshold_pct", "separation_days", "n_events", "rp100_est_ms",
                   "rp100_delta_lower_ms", "rp100_delta_upper_ms",
                   "rp100_width_change_delta_pct"]]
print(example.to_string(index=False))
