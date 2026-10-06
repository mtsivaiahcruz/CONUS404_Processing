"""Step 3a. GEV baseline at the 13 pilot cells, using the documented maximum-likelihood fit.

1. Take one maximum per water year (October to September), 1980-2024, from the pipeline's own
   input files (year_raw_data). They must equal the maxima from my extraction of the daily data.
2. Fit the documented model: scipy.stats.genextreme.fit (maximum likelihood), as in
   30_fit_gev_parameters.py.
3. Work out return levels for 10, 20, 25, 50, 100, 200 and 500 years with the formula in
   40_calculate_return_periods.py, as native (20-second) wind and as 3-second gust (x 1.1176).
4. Work out two 95% intervals for each return level:
     "delta"     the standard delta method. The variance of the return level is g x C x g, where
                 g holds the derivatives of the return level with respect to the three fitted
                 parameters and C is their covariance matrix (estimated by the inverse of the
                 Hessian of the negative log-likelihood at the fit).
     "pipeline"  the interval CADENCE reports, from gev_return_level_ci in
                 40_calculate_return_periods.py as it was before it was fixed (same logic).

Reads:
    output/pilot_locations_selected.csv (step 03), year_raw_data,
    data/spduv10max_water_1980-2024.nc
Writes:
    output/work/gev_baseline_long.csv, output/work/gev_baseline_params.csv
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genextreme

from h5io import open_h5
from paths import GUST_FACTOR, OUT, WATER_NC, WATER_YEARS, WORK, require, year_raw
from pot import hessian

warnings.filterwarnings("ignore")
RETURN_PERIODS = [10, 20, 25, 50, 100, 200, 500]
Z95 = 1.96    # a 95% interval is the estimate plus or minus 1.96 standard errors
require(OUT / "pilot_locations_selected.csv", WATER_NC,
        *(year_raw(year) for year in WATER_YEARS))

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
lat_idx, lon_idx = cells.lat_idx.to_numpy(), cells.lon_idx.to_numpy()

# Annual maxima: one row per water year, one column per pilot cell.
annual_max = []
for year in WATER_YEARS:
    annual_max.append(open_h5(year_raw(year))["SPDUV10MAX"][:][lat_idx, lon_idx])
annual_max = np.array(annual_max, np.float64)

extraction = open_h5(WATER_NC)
extraction_max = extraction["annual_max"][:][:, lat_idx, lon_idx].astype(np.float64)
assert np.array_equal(extraction_max, annual_max), "water-year maxima differ"


def pipeline_return_level(R, loc, scale, shape):
    """40_calculate_return_periods.gev_return_level as it was before it was fixed, same logic.

    The shape = 0 branch keeps the sign error on purpose; step 07 looks at it.
    """
    if scale <= 0 or np.isnan(loc) or np.isnan(scale) or np.isnan(shape):
        return np.nan
    p = 1.0 - 1.0 / R
    if np.abs(shape) < 1e-6:
        return loc + scale * np.log(-np.log(p))
    return loc + (scale / shape) * (1.0 - (-np.log(p)) ** shape)


def pipeline_interval(R, loc, scale, shape, data_points):
    """40_calculate_return_periods.gev_return_level_ci as it was before it was fixed, same logic.

    Copied as it is, to reproduce the intervals CADENCE reports. It is not the standard delta
    method.
    """
    level = pipeline_return_level(R, loc, scale, shape)
    if np.isnan(level):
        return np.nan, np.nan
    p = 1.0 - 1.0 / R
    y = -np.log(p)
    lly = np.log(y)
    if abs(shape) > 1e-5:
        d_scale = (1.0 - y**shape) / shape
        d_shape = -(1.0 - y**shape) / (shape**2) + (y**shape) * lly / shape
    else:
        d_scale = -lly
        d_shape = -0.5 * (lly**2)
    var_factor = np.sqrt(d_scale**2 + d_shape**2)
    se = (scale / np.sqrt(data_points)) * var_factor * np.log(R + 1)
    return level - Z95 * se, level + Z95 * se


def gev_return_level_grad(R, shape, loc, scale):
    """Derivatives of the return level with respect to (shape, loc, scale).

    The return level is loc + scale * (1 - y**shape) / shape, with y = -ln(1 - 1/R) and
    shape in scipy's sign convention.
    """
    y = -np.log(1 - 1 / R)
    y_shape = y**shape
    return np.array([
        scale * (-(1 - y_shape) / shape**2 - y_shape * np.log(y) / shape),
        1.0,
        (1 - y_shape) / shape,
    ])


rows = []
param_rows = []
for k, cell in enumerate(cells.itertuples()):
    values = annual_max[:, k]
    values = values[~np.isnan(values)]
    # The fit in 30_fit_gev_parameters.py. scipy's shape has the opposite sign to the usual
    # xi, so xi = -shape.
    shape, loc, scale = genextreme.fit(values)
    params = np.array([shape, loc, scale])

    # Covariance of the three parameters, in the order (shape, loc, scale): the inverse of the
    # Hessian of the negative log-likelihood. The Hessian is worked out with two step sizes,
    # and how much the result changes is saved as hessian_step_rel_diff.
    hess = hessian(lambda p: genextreme.nnlf(p, values), params)
    hess_larger_step = hessian(lambda p: genextreme.nnlf(p, values), params, relative_step=1e-3)
    cov = np.linalg.inv(hess)
    eigenvalues = np.linalg.eigvalsh(hess)
    step_diff = np.abs(np.linalg.inv(hess_larger_step) - cov).max() / np.abs(cov).max()
    param_rows.append({
        "cell_id": cell.cell_id, "candidate": cell.candidate, "n_years": len(values),
        "location": loc, "scale": scale, "shape_scipy_c": shape, "xi": -shape,
        "se_location": np.sqrt(cov[1, 1]), "se_scale": np.sqrt(cov[2, 2]),
        "se_xi": np.sqrt(cov[0, 0]),
        "hessian_pos_def": bool((eigenvalues > 0).all()),
        "hessian_step_rel_diff": float(step_diff),
    })

    for R in RETURN_PERIODS:
        estimate = pipeline_return_level(R, loc, scale, shape)
        # isf(1 / R) is scipy's own value for the level exceeded with probability 1/R
        assert np.isclose(estimate, genextreme.isf(1 / R, shape, loc, scale))

        # Check the derivative formulas against a numerical derivative.
        grad = gev_return_level_grad(R, shape, loc, scale)
        eps = 1e-6
        grad_numeric = []
        for direction in np.eye(3):
            above = genextreme.isf(1 / R, *(params + direction * eps))
            below = genextreme.isf(1 / R, *(params - direction * eps))
            grad_numeric.append((above - below) / (2 * eps))
        grad_numeric = np.array(grad_numeric)
        assert np.allclose(grad, grad_numeric, rtol=1e-4, atol=1e-6), (cell.candidate, R)

        se = float(np.sqrt(grad @ cov @ grad))    # delta method
        pipeline_lower, pipeline_upper = pipeline_interval(R, loc, scale, shape, len(values))
        rows.append({
            "cell_id": cell.cell_id, "lat_idx": cell.lat_idx, "lon_idx": cell.lon_idx,
            "latitude": cell.latitude, "longitude": cell.longitude, "candidate": cell.candidate,
            "return_period": R,
            "native_ms": estimate, "delta_se_native_ms": se,
            "delta_lower_native_ms": estimate - Z95 * se,
            "delta_upper_native_ms": estimate + Z95 * se,
            "pipeline_lower_native_ms": pipeline_lower,
            "pipeline_upper_native_ms": pipeline_upper,
            "gust3s_ms": estimate * GUST_FACTOR,
            "delta_lower_3s_ms": (estimate - Z95 * se) * GUST_FACTOR,
            "delta_upper_3s_ms": (estimate + Z95 * se) * GUST_FACTOR,
            "pipeline_lower_3s_ms": pipeline_lower * GUST_FACTOR,
            "pipeline_upper_3s_ms": pipeline_upper * GUST_FACTOR,
        })

levels = pd.DataFrame(rows)
fitted = pd.DataFrame(param_rows)
levels.to_csv(WORK / "gev_baseline_long.csv", index=False)
fitted.to_csv(WORK / "gev_baseline_params.csv", index=False)
print(fitted.round(4).to_string(index=False))
n_negative = int((levels.pipeline_lower_native_ms < 0).sum())
print(f"\npipeline intervals with a negative lower bound: {n_negative} of {len(levels)}")
