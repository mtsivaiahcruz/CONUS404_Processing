"""Step 4. Peaks-over-threshold (POT) tests at the pilot cells.

Each cell is fitted at 12 settings: 4 thresholds (the 95th, 97th, 98th and 99th percentile of
that cell's own daily values) x 3 event-separation rules (1, 2 and 3 days). The method itself
is in pot.py.

Reads:
    data/pilot_daily_spduv10max.csv (written by conus404_pilot_daily_extract.py)
    output/pilot_locations_selected.csv
    data/spduv10max_water_1980-2024.nc (only to cross-check the daily file)
Writes:
    output/pilot_pot_results.csv    one row per cell and setting
    output/pilot_pot_events.csv     every event that was kept: cell, setting, date of the peak,
                                    peak value
"""
import warnings

import numpy as np
import pandas as pd

import pot
from h5io import open_h5
from paths import DAILY_CSV, FIRST_WY, LAST_WY, N_YEARS, OUT, WATER_NC, WATER_YEARS, require

warnings.filterwarnings("ignore")
PERCENTILES = [95, 97, 98, 99]
SEPARATIONS = [1, 2, 3]
RETURN_PERIODS = [10, 20, 25, 50, 100, 200, 500]
require(DAILY_CSV, OUT / "pilot_locations_selected.csv", WATER_NC)

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])

# Checks on the daily file: every date present, one column per pilot cell, no bad values.
expected = pd.date_range(f"{FIRST_WY - 1}-10-01", f"{LAST_WY}-09-30", freq="D")
dates_ok = len(daily) == len(expected) and (pd.DatetimeIndex(daily.date) == expected).all()
assert dates_ok, "daily dates are not the full 1979-10-01..2024-09-30 run"
columns = [str(cell_id) for cell_id in cells.cell_id]
assert set(columns) == set(daily.columns) - {"date"}, "cell columns do not match the pilot table"
# One row per day, one column per cell. Read as float32, the precision the store holds, so the
# maxima can be compared exactly with the full-grid extraction.
values = daily[columns].to_numpy(np.float32)
n_missing = int(np.isnan(values).sum())
assert n_missing == 0, f"{n_missing} missing daily values"
assert (values >= 0).all(), "negative daily values"

# The water-year maxima of the daily file must equal the annual maxima from the full-grid
# extraction. A water year is named after the year it ends in (October to September).
water_year = np.where(daily.date.dt.month >= 10, daily.date.dt.year + 1, daily.date.dt.year)
water_year = water_year.astype(int)
water_file = open_h5(WATER_NC)
assert list(water_file["year"][:]) == WATER_YEARS
lat_idx, lon_idx = cells.lat_idx.to_numpy(), cells.lon_idx.to_numpy()
extraction_max = water_file["annual_max"][:][:, lat_idx, lon_idx].astype(np.float32)
daily_max = []
for year in WATER_YEARS:
    daily_max.append(values[water_year == year].max(axis=0))
daily_max = np.array(daily_max, np.float32)
max_diff = float(np.abs(daily_max - extraction_max).max())
print(f"daily file: {len(daily)} days x {len(columns)} cells, 0 missing; "
      f"water-year maxima vs the full-grid extraction: largest difference {max_diff:.1e} m/s")
assert max_diff == 0.0, "daily series do not reproduce the annual maxima"

# Fit the POT model at every cell and setting.
rows = []
event_tables = []
for k, cell in enumerate(cells.itertuples()):
    series = values[:, k].astype(np.float64)
    for percentile in PERCENTILES:
        for separation in SEPARATIONS:
            fit = pot.pot_fit(series, percentile, separation, N_YEARS, RETURN_PERIODS)
            rows.append({
                "cell_id": cell.cell_id, "candidate": cell.candidate, "region": cell.region,
                "lat_idx": cell.lat_idx, "lon_idx": cell.lon_idx,
                "latitude": cell.latitude, "longitude": cell.longitude,
                "n_days": len(series), "n_years": N_YEARS, **fit,
            })
            # keep the list of events too: the day and value of each event's peak
            threshold = fit["threshold_ms"]
            peak_days, peaks, _ = pot.decluster(series, threshold, separation)
            event_tables.append(pd.DataFrame({
                "cell_id": cell.cell_id, "candidate": cell.candidate,
                "threshold_pct": percentile, "separation_days": separation,
                "peak_date": daily.date.to_numpy()[peak_days],
                "peak_ms": peaks, "excess_ms": peaks - threshold,
            }))
results = pd.DataFrame(rows)
events = pd.concat(event_tables, ignore_index=True)

# Checks that the results are consistent with each other.
setting = ["cell_id", "threshold_pct", "separation_days"]
assert len(results) == len(cells) * len(PERCENTILES) * len(SEPARATIONS)
events_per_fit = events.groupby(setting, sort=False).size()
assert (results.n_events == events_per_fit.to_numpy()).all()
by_setting = results.set_index(setting)
for cell_id in cells.cell_id:
    for percentile in PERCENTILES:
        n1, n2, n3 = (by_setting.loc[(cell_id, percentile, separation), "n_events"]
                      for separation in SEPARATIONS)
        assert n1 >= n2 >= n3, "a longer separation rule gave more events"
    days_above = [by_setting.loc[(cell_id, percentile, 1), "n_exceedance_days"]
                  for percentile in PERCENTILES]
    never_increases = days_above[0] >= days_above[1] >= days_above[2] >= days_above[3]
    assert never_increases, "a higher threshold gave more days above it"
level_columns = [f"rp{N}_ms" for N in RETURN_PERIODS]
levels = results[level_columns].to_numpy()
assert (np.diff(levels, axis=1) > 0).all(), "return levels not increasing"
# The largest event at every setting must be the cell's highest daily value on record.
record_values = [values[:, columns.index(str(cell_id))].max() for cell_id in results.cell_id]
assert np.allclose(results.max_peak_ms, record_values)
assert results.hessian_ok.all()

# Round the fitted values to 4 decimals. Identifiers, counts and coordinates are left alone.
not_rounded = ["cell_id", "lat_idx", "lon_idx", "n_days", "n_years", "threshold_pct",
               "separation_days", "n_exceedance_days", "n_events", "latitude", "longitude"]
to_round = results.select_dtypes("number").columns.difference(not_rounded)
results[to_round] = results[to_round].round(4)
results.to_csv(OUT / "pilot_pot_results.csv", index=False)
events["peak_ms"] = events.peak_ms.round(4)
events["excess_ms"] = events.excess_ms.round(4)
events.to_csv(OUT / "pilot_pot_events.csv", index=False, date_format="%Y-%m-%d")

print(f"wrote {len(results)} rows to pilot_pot_results.csv "
      f"and {len(events)} events to pilot_pot_events.csv")
print("flags:", results["flags"].replace("", np.nan).dropna().value_counts().to_dict() or "none")
summary = results.groupby(["threshold_pct", "separation_days"]).agg(
    events_min=("n_events", "min"), events_med=("n_events", "median"),
    events_max=("n_events", "max"), per_year_med=("events_per_year", "median"),
    xi_min=("gpd_shape_xi", "min"), xi_med=("gpd_shape_xi", "median"),
    xi_max=("gpd_shape_xi", "max"), ks_min=("ks_pvalue", "min"),
)
print(summary.round(3).to_string())
