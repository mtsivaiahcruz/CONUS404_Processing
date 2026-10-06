"""Step 2b. Check that the pipeline's GEV fit works at each pilot cell.

The fit and its 'converged' flag follow 30_fit_gev_parameters.py. The return levels and the
1.1176 gust factor follow 40_calculate_return_periods.py. The annual maxima are read from the
pipeline's own input files (year_raw_data, water years 1980-2024).

Extra checks at each cell:
  - the fit is restarted from several starting points to see whether any of them does better
  - a Kolmogorov-Smirnov test of how well the fitted GEV matches the 45 annual maxima
  - the bounded maximum-likelihood fit that 30a_fit_gev_parameters.py falls back to when
    lmoments3 is not installed, for comparison

Reads:
    output/pilot_cells_snapped.csv (step 01), year_raw_data, data/spduv10max_water_1980-2024.nc
Writes:
    output/pilot_gev_check.csv
    output/pilot_cells_snapped.csv (adds the GEV columns)
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genextreme, kstest

from h5io import open_h5
from paths import GUST_FACTOR, OUT, WATER_NC, WATER_YEARS, require, year_raw

warnings.filterwarnings("ignore")
RETURN_PERIODS = [10, 25, 50, 100, 200, 500]
require(OUT / "pilot_cells_snapped.csv", WATER_NC, *(year_raw(year) for year in WATER_YEARS))

cells = pd.read_csv(OUT / "pilot_cells_snapped.csv")
# Drop the GEV columns if this step has been run before, so it can be run again safely.
cells = cells[[col for col in cells.columns if not col.startswith(("gev_", "rp"))]]
lat_idx, lon_idx = cells.lat_idx.to_numpy(), cells.lon_idx.to_numpy()

# One row per water year, one column per pilot cell.
print("reading the 45 annual-maximum files ...", flush=True)
annual_max = np.empty((len(WATER_YEARS), len(cells)), np.float64)
for k, year in enumerate(WATER_YEARS):
    annual_max[k] = open_h5(year_raw(year))["SPDUV10MAX"][:][lat_idx, lon_idx]

# The pipeline's inputs should equal the annual maxima from my own extraction of the daily data.
print("reading the full-grid annual maxima for the cross-check ...", flush=True)
extraction = open_h5(WATER_NC)
assert list(extraction["year"][:]) == WATER_YEARS
diff_vs_extraction = np.abs(extraction["annual_max"][:][:, lat_idx, lon_idx] - annual_max)


def pipeline_fit(series):
    """The fit in 30_fit_gev_parameters.fit_gev_at_gridpoint, same logic."""
    values = series[~np.isnan(series)]
    if len(values) < 3:
        return np.nan, np.nan, np.nan, False, f"Insufficient valid data: {len(values)} points"
    try:
        shape, loc, scale = genextreme.fit(values)
        if scale <= 0:
            return np.nan, np.nan, np.nan, False, f"Invalid scale parameter: {scale}"
        return float(loc), float(scale), float(shape), True, ""
    except Exception as error:
        return np.nan, np.nan, np.nan, False, f"Fit error: {str(error)[:50]}"


def bounded_mle_fit(values):
    """The fallback fit in 30a_fit_gev_parameters._fit_bounded_mle, same logic."""
    result = minimize(genextreme.nnlf, x0=np.array([0.01, values.mean(), values.std()]),
                      args=(values,), bounds=[(-0.5, 0.5), (None, None), (0.001, None)],
                      method="L-BFGS-B")
    if result.success:
        return float(result.x[1]), float(result.x[2]), float(result.x[0])
    return np.nan, np.nan, np.nan


def pipeline_return_level(R, loc, scale, shape):
    """40_calculate_return_periods.gev_return_level as it was before it was fixed, same logic
    (including the sign error in its Gumbel branch)."""
    if scale <= 0 or np.isnan(loc) or np.isnan(scale) or np.isnan(shape):
        return np.nan
    p = 1.0 - 1.0 / R
    if np.abs(shape) < 1e-6:
        return loc + scale * np.log(-np.log(p))
    return loc + (scale / shape) * (1.0 - (-np.log(p)) ** shape)


print("fitting ...", flush=True)
rows = []
for k, cell in enumerate(cells.itertuples()):
    series = annual_max[:, k]
    loc, scale, shape, converged, warning = pipeline_fit(series)
    values = series[~np.isnan(series)]

    # Restart the fit from five different shapes. If any restart reaches a lower negative
    # log-likelihood (nll) than the pipeline's fit, the pipeline's fit was not the best one.
    # The gap is saved as nll_gap_to_best; 0 means no restart did better.
    nll = genextreme.nnlf((shape, loc, scale), values)
    best_nll = nll
    for shape_start in (-0.3, -0.1, 0.0, 0.1, 0.3):
        restart = minimize(genextreme.nnlf, x0=[shape_start, values.mean(), values.std()],
                           args=(values,), method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
        best_nll = min(best_nll, restart.fun)

    alt_loc, alt_scale, alt_shape = bounded_mle_fit(values)
    native = [pipeline_return_level(R, loc, scale, shape) for R in RETURN_PERIODS]
    alt = [pipeline_return_level(R, alt_loc, alt_scale, alt_shape) for R in RETURN_PERIODS]
    scipy_levels = [genextreme.isf(1 / R, shape, loc, scale) for R in RETURN_PERIODS]
    rows.append({
        "candidate": cell.candidate, "lat_idx": cell.lat_idx, "lon_idx": cell.lon_idx,
        "n_years": len(values), "series_min": values.min(), "series_max": values.max(),
        "max_abs_diff_vs_osn": float(np.nanmax(diff_vs_extraction[:, k])),
        "converged": converged, "warning": warning,
        "location": loc, "scale": scale, "shape_scipy_c": shape, "xi_standard": -shape,
        "nll": nll, "nll_gap_to_best": nll - best_nll,
        "ks_pvalue": kstest(values, genextreme(shape, loc, scale).cdf).pvalue,
        **{f"rp{R}_native_ms": level for R, level in zip(RETURN_PERIODS, native)},
        **{f"rp{R}_3s_ms": level * GUST_FACTOR for R, level in zip(RETURN_PERIODS, native)},
        "rp_formula_matches_scipy": bool(np.allclose(native, scipy_levels, rtol=1e-9)),
        "monotonic": bool(np.all(np.diff(native) > 0)),
        "rp500_over_record": native[-1] / values.max(),
        # index 3 is the 100-year return period
        "rp100_3s_bounded_mle_ms": alt[3] * GUST_FACTOR,
        "rp100_method_diff_pct": 100 * (alt[3] - native[3]) / native[3],
        "bounded_shape": alt_shape,
    })
check = pd.DataFrame(rows)
assert np.nanmax(diff_vs_extraction) == 0, "pipeline inputs differ from the OSN extraction"
check.round(5).to_csv(OUT / "pilot_gev_check.csv", index=False)
shown = ["candidate", "converged", "xi_standard", "nll_gap_to_best", "ks_pvalue", "rp100_3s_ms",
         "rp100_method_diff_pct"]
print(check[shown].round(3).to_string(index=False))

# Add the GEV columns to pilot_cells_snapped.csv.
gev_columns = check[["lat_idx", "lon_idx", "converged", "xi_standard", "rp10_3s_ms", "rp50_3s_ms",
                     "rp100_3s_ms", "rp500_3s_ms", "rp500_over_record"]]
gev_columns = gev_columns.rename(columns={
    "converged": "gev_converged",
    "xi_standard": "gev_shape_xi",
    "rp500_over_record": "rp500_to_record_ratio",
})
snapped = cells.merge(gev_columns, on=["lat_idx", "lon_idx"], how="left")
# a note written by hand from the results; it is not computed
snapped["gev_note"] = ""
snapped.loc[snapped.candidate == "New Orleans, LA", "gev_note"] = (
    "Heavy tail (xi=0.33) driven by Katrina (2005); RP500 is 1.6x the 45-yr record"
)
rounded = ["gev_shape_xi", "rp10_3s_ms", "rp50_3s_ms", "rp100_3s_ms", "rp500_3s_ms",
           "rp500_to_record_ratio"]
for col in rounded:
    snapped[col] = snapped[col].round(3)
assert snapped.gev_converged.all() and len(snapped) == 13
snapped.to_csv(OUT / "pilot_cells_snapped.csv", index=False)
print(f"wrote {OUT / 'pilot_gev_check.csv'} and added GEV columns to pilot_cells_snapped.csv")
