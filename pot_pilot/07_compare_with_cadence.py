"""Step 3d. Compare the documented GEV baseline with CADENCE's current estimates.

CADENCE's current files (gev_parameters.h5 and gev_return_periods.nc) were made by
30a_fit_gev_parameters.py with lmoments3 installed. That is an L-moments fit, on all 45 water
years.

What this script does:
1. Reads CADENCE's parameters, return levels and intervals at the 13 pilot cells.
2. Re-runs the logic of 30a (L-moments through lmoments3, bounded maximum likelihood as the
   fallback) on the same annual maxima and confirms it reproduces CADENCE's parameters.
3. Recomputes CADENCE's return levels and intervals from its parameters with the formulas
   script 40 had when those files were made, and prints how closely they reproduce its
   return-period file.
4. Compares CADENCE (L-moments) with the documented maximum-likelihood baseline for 10 to 500
   years, plus 20 years.
5. Bootstraps the L-moments fit (1000 simulated records) to get intervals for it that can be
   used. Most of CADENCE's reported intervals have a negative lower bound.
6. Across the whole grid, counts how many of CADENCE's reported intervals have a negative
   lower bound, and finds the cells that go through the Gumbel branch of gev_return_level in
   script 40 (shape exactly 0), where the sign was wrong.

Reads:
    gev_parameters.h5, gev_return_periods.nc, year_raw_data
    output/pilot_locations_selected.csv
    output/work/gev_baseline_long.csv, output/work/gev_baseline_bootstrap.csv
Writes:
    output/pilot_cadence_vs_baseline.csv, output/cadence_gumbel_branch_cells.csv
    output/work/cadence_vs_mle_long.csv, output/work/cadence_vs_repro_params.csv
"""
import warnings

import lmoments3 as lm
import numpy as np
import pandas as pd
from lmoments3 import distr
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r

from h5io import open_h5
from paths import (CADENCE_H5, CADENCE_NC, GUST_FACTOR, N_YEARS, OUT, WATER_YEARS, WORK, require,
                   year_raw)

warnings.filterwarnings("ignore")
CADENCE_RETURN_PERIODS = [10, 25, 50, 100, 200, 500]    # the return periods in CADENCE's file
RETURN_PERIODS = [10, 20, 25, 50, 100, 200, 500]
N_BOOT = 1000
SEED = 20260928
require(CADENCE_H5, CADENCE_NC, OUT / "pilot_locations_selected.csv",
        WORK / "gev_baseline_long.csv", WORK / "gev_baseline_bootstrap.csv",
        *(year_raw(year) for year in WATER_YEARS))

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
lat_idx, lon_idx = cells.lat_idx.to_numpy(), cells.lon_idx.to_numpy()

# Annual maxima: one row per water year, one column per pilot cell.
annual_max = []
for year in WATER_YEARS:
    annual_max.append(open_h5(year_raw(year))["SPDUV10MAX"][:][lat_idx, lon_idx])
annual_max = np.array(annual_max, np.float64)

# CADENCE's fitted parameters, for every cell on the grid.
params_file = open_h5(CADENCE_H5)
loc_grid = params_file["location"][:]
scale_grid = params_file["scale"][:]
shape_grid = params_file["shape"][:]
converged_grid = params_file["converged"][:]
lat_grid, lon_grid = params_file["lat_2d"][:], params_file["lon_2d"][:]
assert np.allclose(lat_grid[lat_idx, lon_idx], cells.latitude, atol=1e-4)

# CADENCE's return levels and intervals: arrays of (return period, row, column).
levels_file = open_h5(CADENCE_NC)
assert list(levels_file["return_period"][:]) == CADENCE_RETURN_PERIODS
native_grid = levels_file["wind_speed_native"][:]
lower_grid = levels_file["wind_speed_native_lower_ci"][:]
upper_grid = levels_file["wind_speed_native_upper_ci"][:]
gust_grid = levels_file["wind_speed_3sec_gust"][:]


def pipeline_return_level(R, loc, scale, shape):
    """40_calculate_return_periods.gev_return_level as it was before it was fixed, same logic.

    The shape = 0 branch keeps the sign error on purpose. It is examined further down.
    """
    if scale <= 0 or np.isnan(loc) or np.isnan(scale) or np.isnan(shape):
        return np.nan
    p = 1.0 - 1.0 / R
    if np.abs(shape) < 1e-6:
        return loc + scale * np.log(-np.log(p))
    return loc + (scale / shape) * (1.0 - (-np.log(p)) ** shape)


def pipeline_interval(R, loc, scale, shape, n):
    """40_calculate_return_periods.gev_return_level_ci as it was before it was fixed, same logic.

    Copied as it is, to reproduce the intervals CADENCE reports. It is not the standard delta
    method.
    """
    level = pipeline_return_level(R, loc, scale, shape)
    p = 1.0 - 1.0 / R
    y = -np.log(p)
    lly = np.log(y)
    if abs(shape) > 1e-5:
        d_scale = (1.0 - y**shape) / shape
        d_shape = -(1.0 - y**shape) / (shape**2) + (y**shape) * lly / shape
    else:
        d_scale = -lly
        d_shape = -0.5 * (lly**2)
    se = (scale / np.sqrt(n)) * np.sqrt(d_scale**2 + d_shape**2) * np.log(R + 1)
    return level - 1.96 * se, level + 1.96 * se


def bounded_mle(values):
    """30a_fit_gev_parameters._fit_bounded_mle, same logic."""
    result = minimize(genextreme.nnlf, x0=np.array([0.01, values.mean(), values.std()]),
                      args=(values,), bounds=[(-0.5, 0.5), (None, None), (0.001, None)],
                      method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


def fit_like_30a(values):
    """The fit 30a does when lmoments3 is installed: L-moments, bounded MLE as the fallback.

    Returns shape, location, scale and the name of the method that was used.
    """
    try:
        fit = distr.gev.lmom_fit(values)
        return float(fit["c"]), float(fit["loc"]), float(fit["scale"]), "lmom"
    except Exception:
        return (*bounded_mle(values), "bounded_mle")


def one_row(table, cell_id, return_period):
    """The row of a table for one cell and one return period."""
    return table[(table.cell_id == cell_id) & (table.return_period == return_period)].iloc[0]


baseline = pd.read_csv(WORK / "gev_baseline_long.csv")    # the documented MLE baseline
baseline_boot = pd.read_csv(WORK / "gev_baseline_bootstrap.csv")
rng = np.random.default_rng(SEED)
rows = []
param_rows = []
for k, cell in enumerate(cells.itertuples()):
    row, col = cell.lat_idx, cell.lon_idx
    values = annual_max[:, k]

    # Three sets of parameters: CADENCE's file, my re-run of 30a, and the MLE fit.
    cadence_loc = float(loc_grid[row, col])
    cadence_scale = float(scale_grid[row, col])
    cadence_shape = float(shape_grid[row, col])
    rerun_shape, rerun_loc, rerun_scale, rerun_method = fit_like_30a(values)
    mle_shape, mle_loc, mle_scale = genextreme.fit(values)
    param_rows.append({
        "cell_id": cell.cell_id, "candidate": cell.candidate,
        "converged_flag": int(converged_grid[row, col]),
        "cadence_location": cadence_loc, "cadence_scale": cadence_scale,
        "cadence_shape_c": cadence_shape,
        "repro_location": rerun_loc, "repro_scale": rerun_scale, "repro_shape_c": rerun_shape,
        "repro_method": rerun_method,
        "max_param_abs_diff": max(abs(cadence_loc - rerun_loc), abs(cadence_scale - rerun_scale),
                                  abs(cadence_shape - rerun_shape)),
        "mle_location": mle_loc, "mle_scale": mle_scale, "mle_shape_c": mle_shape,
    })

    # Bootstrap the L-moments fit, the method CADENCE used: simulate records from CADENCE's
    # parameters and refit each one the way 30a does.
    boot_levels = np.empty((N_BOOT, len(RETURN_PERIODS)))
    for b in range(N_BOOT):
        sample = genextreme.rvs(cadence_shape, cadence_loc, cadence_scale, size=len(values),
                                random_state=rng)
        shape_b, loc_b, scale_b, _ = fit_like_30a(sample)
        boot_levels[b] = genextreme.isf(1 / np.array(RETURN_PERIODS), shape_b, loc_b, scale_b)

    for j, R in enumerate(RETURN_PERIODS):
        # recomputed from CADENCE's parameters with script 40's formulas
        estimate = pipeline_return_level(R, cadence_loc, cadence_scale, cadence_shape)
        lower, upper = pipeline_interval(R, cadence_loc, cadence_scale, cadence_shape, N_YEARS)

        # the values stored in CADENCE's return-period file
        if R in CADENCE_RETURN_PERIODS:
            q = CADENCE_RETURN_PERIODS.index(R)
            file_estimate = float(native_grid[q, row, col])
            file_lower = float(lower_grid[q, row, col])
            file_upper = float(upper_grid[q, row, col])
            file_gust = float(gust_grid[q, row, col])
        else:
            # the 20-year level is not in CADENCE's file
            file_estimate = file_lower = file_upper = file_gust = np.nan

        mle = one_row(baseline, cell.cell_id, R)
        mle_boot = one_row(baseline_boot, cell.cell_id, R)
        boot_lower, boot_upper = np.percentile(boot_levels[:, j], [2.5, 97.5])
        rows.append({
            "cell_id": cell.cell_id, "lat_idx": row, "lon_idx": col,
            "latitude": cell.latitude, "longitude": cell.longitude,
            "candidate": cell.candidate, "return_period": R,
            "in_documented_set": R in CADENCE_RETURN_PERIODS,
            "cadence_native_ms": estimate, "cadence_nc_native_ms": file_estimate,
            "cadence_nc_3s_ms": file_gust,
            "cadence_ci_lower_native_ms": lower, "cadence_ci_upper_native_ms": upper,
            "cadence_nc_ci_lower_native_ms": file_lower,
            "cadence_nc_ci_upper_native_ms": file_upper,
            "cadence_boot_lower_native_ms": boot_lower,
            "cadence_boot_upper_native_ms": boot_upper,
            "mle_native_ms": mle.native_ms,
            "mle_delta_lower_native_ms": mle.delta_lower_native_ms,
            "mle_delta_upper_native_ms": mle.delta_upper_native_ms,
            # the baseline bootstrap was saved as 3-second gust; convert back to native
            "mle_boot_lower_native_ms": mle_boot.boot_lower_3s_ms / GUST_FACTOR,
            "mle_boot_upper_native_ms": mle_boot.boot_upper_3s_ms / GUST_FACTOR,
            "diff_cadence_minus_mle_ms": estimate - mle.native_ms,
            "diff_pct": 100 * (estimate - mle.native_ms) / mle.native_ms,
        })
    print(f"  {cell.candidate}: done", flush=True)

fitted = pd.DataFrame(param_rows)
comparison = pd.DataFrame(rows)
fitted.to_csv(WORK / "cadence_vs_repro_params.csv", index=False)
comparison.to_csv(WORK / "cadence_vs_mle_long.csv", index=False)

# Checks on CADENCE's files.
documented = comparison[comparison.in_documented_set]
print("\n30a path used at the 13 cells:", fitted.repro_method.value_counts().to_dict())
print(f"largest parameter difference, CADENCE vs my re-run of 30a: "
      f"{fitted.max_param_abs_diff.max():.2e}")
level_gap = np.nanmax(np.abs(documented.cadence_nc_native_ms - documented.cadence_native_ms))
print(f"largest difference, CADENCE's return levels vs recomputed from its parameters: "
      f"{level_gap:.2e} m/s")
interval_gap = np.nanmax(np.abs(np.r_[
    documented.cadence_nc_ci_lower_native_ms - documented.cadence_ci_lower_native_ms,
    documented.cadence_nc_ci_upper_native_ms - documented.cadence_ci_upper_native_ms,
]))
print(f"largest difference, CADENCE's intervals vs recomputed: {interval_gap:.2e} m/s")
gust_gap = np.nanmax(np.abs(
    documented.cadence_nc_3s_ms - documented.cadence_nc_native_ms * GUST_FACTOR
))
print(f"largest difference, CADENCE's 3-second values vs native x {GUST_FACTOR}: "
      f"{gust_gap:.2e} m/s")
assert fitted.max_param_abs_diff.max() < 1e-5 and (fitted.repro_method == "lmom").all()

print("\nlargest |difference| between CADENCE (L-moments) and the MLE baseline, "
      "% of the MLE estimate:")
largest_diff = comparison.groupby("return_period").diff_pct.apply(lambda x: x.abs().max())
print(largest_diff.round(1).to_dict())

# Build pilot_cadence_vs_baseline.csv.
# The long table is read back from the file so the numbers are exactly the ones saved.
saved = pd.read_csv(WORK / "cadence_vs_mle_long.csv")
columns = [
    "cell_id", "lat_idx", "lon_idx", "latitude", "longitude", "candidate", "return_period",
    "in_documented_set",
    "cadence_native_ms", "cadence_ci_lower_native_ms", "cadence_ci_upper_native_ms",
    "cadence_boot_lower_native_ms", "cadence_boot_upper_native_ms",
    "mle_native_ms", "mle_delta_lower_native_ms", "mle_delta_upper_native_ms",
    "mle_boot_lower_native_ms", "mle_boot_upper_native_ms",
    "diff_cadence_minus_mle_ms", "diff_pct",
]
table = saved[columns].copy()
table.insert(8, "cadence_method", "30a L-moments (lmoments3), gev_parameters.h5 fit 2026-07-12")
table["cadence_ci_source"] = np.where(
    table.in_documented_set,
    "gev_return_periods.nc (ci_method: pipeline gev_return_level_ci)",
    "recomputed from h5 with pipeline gev_return_level_ci (RP20 not in file)",
)
table = table.sort_values(["longitude", "return_period"])
not_rounded = ["cell_id", "lat_idx", "lon_idx", "return_period", "latitude", "longitude"]
to_round = table.select_dtypes("number").columns.difference(not_rounded)
table[to_round] = table[to_round].round(3)
table.to_csv(OUT / "pilot_cadence_vs_baseline.csv", index=False)
print("CADENCE intervals with a negative lower bound at the pilot cells:",
      int((table.cadence_ci_lower_native_ms < 0).sum()), "of", len(table))

# Whole grid: CADENCE's reported intervals.
print(f"\nwhole grid, {shape_grid.size} cells:")
for q, R in enumerate(CADENCE_RETURN_PERIODS):
    lower = lower_grid[q]
    print(f"  {R:>3}-year: {100 * (lower < 0).mean():.1f}% of cells have a negative lower bound; "
          f"median half-width {np.nanmedian((upper_grid[q] - lower) / 2):.1f} m/s (native)")

# Whole grid: the Gumbel branch of script 40's gev_return_level, which had a sign error.
# A cell goes through that branch when its shape is (almost) exactly 0.
is_gumbel = np.abs(shape_grid) < 1e-6
gumbel_rows, gumbel_cols = np.where(is_gumbel)
print(f"\ncells with |shape| < 1e-6: {int(is_gumbel.sum())}; "
      f"of those with shape exactly 0: {int((shape_grid == 0).sum())}")
periods = np.array(CADENCE_RETURN_PERIODS)
records = []
for row, col in zip(gumbel_rows, gumbel_cols):
    # the correct Gumbel return levels for this cell's parameters
    correct = genextreme.isf(1 / periods, 0.0, loc_grid[row, col], scale_grid[row, col])
    # index 0 is the 10-year return period and index 3 is the 100-year one
    records.append({
        "lat_idx": row, "lon_idx": col, "lat": lat_grid[row, col], "lon": lon_grid[row, col],
        "location": loc_grid[row, col], "scale": scale_grid[row, col],
        "shape": shape_grid[row, col],
        "cadence_rp10": native_grid[0, row, col], "correct_rp10": correct[0],
        "cadence_rp100": native_grid[3, row, col], "correct_rp100": correct[3],
        "cadence_rp100_3s": gust_grid[3, row, col], "correct_rp100_3s": correct[3] * GUST_FACTOR,
    })
gumbel = pd.DataFrame(records)
gumbel["rp100_error_ms"] = gumbel.cadence_rp100 - gumbel.correct_rp100

# For comparison: the median 100-year value of each cell's eight neighbours.
neighbor_medians = []
for row, col in zip(gumbel_rows, gumbel_cols):
    window = native_grid[3, max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].copy()
    # blank out the cell itself: index 1 of the 3 x 3 window, or 0 on the edge of the grid
    window[min(row, 1), min(col, 1)] = np.nan
    neighbor_medians.append(np.nanmedian(window))
gumbel["neighbor_median_rp100"] = neighbor_medians
gumbel.round(4).to_csv(WORK / "cadence_gumbel_branch_cells.csv", index=False)
# the copy in output/ is written from the saved file, so the two are identical
pd.read_csv(WORK / "cadence_gumbel_branch_cells.csv").to_csv(
    OUT / "cadence_gumbel_branch_cells.csv", index=False)

print(f"  100-year native value at those cells: "
      f"CADENCE mean {gumbel.cadence_rp100.mean():.1f} "
      f"(median {gumbel.cadence_rp100.median():.1f}) m/s; "
      f"correct mean {gumbel.correct_rp100.mean():.1f} "
      f"(median {gumbel.correct_rp100.median():.1f}) m/s")
print(f"  100-year error: mean {gumbel.rp100_error_ms.mean():.1f}, "
      f"worst {gumbel.rp100_error_ms.min():.1f} m/s")
print(f"  10-year value below the location parameter at all of them: "
      f"{bool((gumbel.cadence_rp10 < gumbel.location).all())}")
print(f"  median 100-year value of neighbouring cells: {np.median(neighbor_medians):.1f} m/s")
gumbel_cells = set(zip(gumbel_rows.tolist(), gumbel_cols.tolist()))
pilot_cells = set(zip(lat_idx.tolist(), lon_idx.tolist()))
print(f"  'converged' flag at these cells: "
      f"{set(converged_grid[gumbel_rows, gumbel_cols].tolist())}; "
      f"any of them a pilot cell: {bool(gumbel_cells & pilot_cells)}")

# Sign check. The Gumbel return level is location MINUS scale * ln(-ln p).
p = 0.99
assert np.isclose(10 - 1 * np.log(-np.log(p)), gumbel_r.isf(1 - p, 10, 1))
assert np.isclose(genextreme.isf(1 - p, 0, 10, 1), gumbel_r.isf(1 - p, 10, 1))
print("  sign check against scipy's Gumbel distribution: location - scale * ln(-ln p) is correct")

# Why the shape is exactly 0 at those cells: lmoments3 returns shape 0 when the L-skewness
# of the data is close to the Gumbel value.
gumbel_series = []
for year in WATER_YEARS:
    gumbel_series.append(open_h5(year_raw(year))["SPDUV10MAX"][:][gumbel_rows, gumbel_cols])
gumbel_series = np.array(gumbel_series, np.float64)
refit_shapes = []
l_skewness = []
for k in range(len(gumbel_rows)):
    refit_shapes.append(float(distr.gev.lmom_fit(gumbel_series[:, k])["c"]))
    # index 2 of the L-moment ratios is the L-skewness (0.1699 for a Gumbel distribution)
    l_skewness.append(lm.lmom_ratios(gumbel_series[:, k], nmom=4)[2])
print(f"  re-fitting those cells with lmoments3 gives shape exactly 0 at "
      f"{int(np.sum(np.array(refit_shapes) == 0))} of {len(gumbel_rows)}; "
      f"their L-skewness is {min(l_skewness):.4f}-{max(l_skewness):.4f} (Gumbel = 0.1699)")
print(f"\nwrote {OUT / 'pilot_cadence_vs_baseline.csv'} and cadence_gumbel_branch_cells.csv")
