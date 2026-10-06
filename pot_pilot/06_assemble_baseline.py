"""Step 3c. Build the two GEV baseline result tables.

Reads:
    output/work/gev_baseline_bootstrap.csv (step 05)
    output/work/gev_baseline_params.csv (step 04)
Writes:
    output/pilot_gev_baseline.csv      one row per cell and return period (10, 20, 25, 50, 100,
                                       200, 500 years): the native and 3-second estimates, with
                                       the delta-method, pipeline and bootstrap 95% intervals
    output/pilot_gev_parameters.csv    the fitted location, scale and shape, with standard errors
"""
import pandas as pd

from paths import FIRST_WY, LAST_WY, N_YEARS, OUT, WORK, require

require(WORK / "gev_baseline_bootstrap.csv", WORK / "gev_baseline_params.csv")
results = pd.read_csv(WORK / "gev_baseline_bootstrap.csv")
fitted = pd.read_csv(WORK / "gev_baseline_params.csv")

columns = [
    "cell_id", "lat_idx", "lon_idx", "latitude", "longitude", "candidate", "return_period",
    "native_ms", "delta_se_native_ms", "delta_lower_native_ms", "delta_upper_native_ms",
    "gust3s_ms", "delta_lower_3s_ms", "delta_upper_3s_ms",
    "pipeline_lower_3s_ms", "pipeline_upper_3s_ms", "boot_lower_3s_ms", "boot_upper_3s_ms",
]
baseline = results[columns].copy()
baseline.insert(7, "n_water_years", N_YEARS)
baseline.insert(8, "first_water_year", FIRST_WY)
baseline.insert(9, "last_water_year", LAST_WY)
# CADENCE's file has no 20-year level; it was added here for the GEV-vs-POT comparison.
baseline["in_documented_set"] = baseline.return_period != 20
baseline = baseline.sort_values(["longitude", "return_period"])

# Round the wind speeds to 3 decimals. Identifiers, counts and coordinates are left alone.
not_rounded = ["cell_id", "lat_idx", "lon_idx", "return_period", "n_water_years",
               "first_water_year", "last_water_year", "latitude", "longitude"]
to_round = baseline.select_dtypes("number").columns.difference(not_rounded)
baseline[to_round] = baseline[to_round].round(3)

assert len(baseline) == 13 * 7 and baseline.native_ms.notna().all()
# a longer return period must always give a higher wind speed
increasing = baseline.groupby("cell_id").native_ms.apply(lambda s: s.is_monotonic_increasing)
assert increasing.all()
baseline.to_csv(OUT / "pilot_gev_baseline.csv", index=False)

parameters = fitted.copy()
to_round = parameters.select_dtypes("number").columns.difference(["cell_id", "n_years"])
parameters[to_round] = parameters[to_round].round(5)
parameters.to_csv(OUT / "pilot_gev_parameters.csv", index=False)

rp100 = baseline[baseline.return_period == 100]
pipeline_half_width = (rp100.pipeline_upper_3s_ms - rp100.pipeline_lower_3s_ms) / 2
delta_half_width = (rp100.delta_upper_3s_ms - rp100.delta_lower_3s_ms) / 2
print("100-year 3-second half-widths, m/s   pipeline:",
      pipeline_half_width.agg(["min", "max"]).round(0).tolist(),
      "  delta:", delta_half_width.agg(["min", "max"]).round(1).tolist())
print("pipeline intervals with a negative lower bound:",
      int((baseline.pipeline_lower_3s_ms < 0).sum()), "of", len(baseline))
print(f"wrote {OUT / 'pilot_gev_baseline.csv'} and pilot_gev_parameters.csv")
