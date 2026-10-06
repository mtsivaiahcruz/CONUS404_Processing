"""Step 5a. Bootstrap intervals for the POT fits, and a resampled-years check for GEV and POT.

Part A. Parametric bootstrap of each of the 156 POT fits. This is the same idea as the GEV
bootstrap in step 05, repeated 1000 times per fit:
    1. draw the number of events from a Poisson distribution with the observed count as its mean
    2. draw that many excesses from the fitted GPD
    3. refit by maximum likelihood and recompute the return levels
The threshold is held fixed.

Part B. Resampled-years bootstrap, done the same way for both models, 500 times per cell:
    1. draw 45 water years with replacement (the same draw for GEV and POT, and for every cell)
    2. GEV: fit by maximum likelihood to the annual maxima of the drawn years
    3. POT: join the daily values of the drawn years, recompute the percentile threshold,
       regroup the events and refit
Because the threshold is recomputed each time, this POT interval includes the uncertainty in
the threshold.

The random seeds are fixed (one per fit in part A, one shared by all cells in part B), so
the results do not depend on how many processes run. About 6 + 5 minutes on 2 cores. Set the
ANALYSIS_WORKERS environment variable to change the number of processes.

Reads:
    data/pilot_daily_spduv10max.csv
    output/pilot_locations_selected.csv, output/pilot_pot_results.csv
Writes:
    output/work/boot_param.npz    levels[fit, replicate, return period]
    output/work/boot_years.npz    gev[cell, replicate, return period]
                                  pot[cell, replicate, threshold, separation, return period]
"""
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genextreme, genpareto

import pot
from paths import DAILY_CSV, N_YEARS, OUT, WORK, require

warnings.filterwarnings("ignore")
RETURN_PERIODS = [10, 20, 25, 50, 100, 200, 500]
PERCENTILES = [95, 97, 98, 99]
SEPARATIONS = [1, 2, 3]
N_BOOT_PARAM = 1000
N_BOOT_YEARS = 500
SEED = 20261004


def fit_fast(excesses, start):
    """A quicker GPD maximum-likelihood fit for the bootstraps: two starting points, not six.

    The starting points are `start` and the method-of-moments estimate. test_pot.py checks
    that this gives the same likelihood and return levels as pot.fit_gpd.
    """
    mean = excesses.mean()
    variance = excesses.var()
    starts = [np.asarray(start, float)]
    if variance > 0:
        sigma_start = 0.5 * mean * (mean * mean / variance + 1.0)
        xi_start = 0.5 * (1.0 - mean * mean / variance)
        starts.append(np.array([sigma_start, xi_start]))

    best = None
    for start_point in starts:
        if not np.isfinite(pot.gpd_nll(start_point, excesses)):
            continue
        result = minimize(pot.gpd_nll, start_point, args=(excesses,), method="Nelder-Mead",
                          options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000})
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        # neither start worked, so fall back on the full fit
        full_fit = pot.fit_gpd(excesses)
        return full_fit["sigma"], full_fit["xi"]
    return float(best.x[0]), float(best.x[1])


def decluster_fast(values, threshold, separation):
    """The peak of each event. Same grouping rule as pot.decluster, but returns only the peaks."""
    days_above = np.flatnonzero(values > threshold)
    starts = np.r_[0, np.flatnonzero(np.diff(days_above) > separation) + 1]
    # the highest value from each event start up to the next event start
    return np.maximum.reduceat(values[days_above], starts)


def levels(threshold, rate, sigma, xi):
    """Return levels for all of RETURN_PERIODS."""
    return np.array([pot.return_level(N, threshold, rate, sigma, xi) for N in RETURN_PERIODS])


def param_job(args):
    """Part A for one POT fit. Returns the fit's index and its simulated return levels."""
    fit_index, threshold, n_events, sigma, xi, seed = args
    rng = np.random.default_rng(seed)
    simulated = np.empty((N_BOOT_PARAM, len(RETURN_PERIODS)))
    for b in range(N_BOOT_PARAM):
        n_drawn = max(int(rng.poisson(n_events)), 10)    # never fewer than 10 events to fit
        excesses = genpareto.rvs(xi, 0, sigma, size=n_drawn, random_state=rng)
        sigma_b, xi_b = fit_fast(excesses, (sigma, xi))
        simulated[b] = levels(threshold, n_drawn / N_YEARS, sigma_b, xi_b)
    return fit_index, simulated


def years_job(args):
    """Part B for one cell. Returns the cell's index and its GEV and POT return levels."""
    cell_index, values, water_year, seed = args
    rng = np.random.default_rng(seed)
    years = np.unique(water_year)
    values_by_year = [values[water_year == year] for year in years]
    annual_max = np.array([year_values.max() for year_values in values_by_year])
    gev_levels = np.empty((N_BOOT_YEARS, len(RETURN_PERIODS)))
    pot_levels = np.empty((N_BOOT_YEARS, len(PERCENTILES), len(SEPARATIONS),
                           len(RETURN_PERIODS)))
    for b in range(N_BOOT_YEARS):
        drawn = rng.integers(0, len(years), size=len(years))

        shape, loc, scale = genextreme.fit(annual_max[drawn])
        gev_levels[b] = genextreme.isf(1.0 / np.array(RETURN_PERIODS), shape, loc, scale)

        resampled = np.concatenate([values_by_year[i] for i in drawn])
        for i_pct, percentile in enumerate(PERCENTILES):
            threshold = np.percentile(resampled, percentile)
            for i_sep, separation in enumerate(SEPARATIONS):
                excesses = decluster_fast(resampled, threshold, separation) - threshold
                sigma_b, xi_b = fit_fast(excesses, (excesses.mean(), 0.0))
                rate = excesses.size / N_YEARS
                pot_levels[b, i_pct, i_sep] = levels(threshold, rate, sigma_b, xi_b)
    return cell_index, gev_levels, pot_levels


def main():
    require(DAILY_CSV, OUT / "pilot_locations_selected.csv", OUT / "pilot_pot_results.csv")
    workers = int(os.environ.get("ANALYSIS_WORKERS", min(os.cpu_count() or 2, 8)))
    # "spawn" is the only way Windows starts processes, so it is used on every system.
    ctx = mp.get_context("spawn")
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
    results = pd.read_csv(OUT / "pilot_pot_results.csv")
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])

    # Part A. Parametric bootstrap.
    start = time.time()
    # one independent stream of random numbers per fit, all derived from SEED
    seeds = np.random.SeedSequence(SEED).spawn(len(results))
    jobs = []
    for k, saved in enumerate(results.itertuples()):
        # The results file is rounded, so each fit is redone here at full precision, in the
        # file's row order, and checked against the saved row.
        series = daily[str(saved.cell_id)].to_numpy(np.float64)
        fit = pot.pot_fit(series, saved.threshold_pct, saved.separation_days, N_YEARS, [100])
        assert fit["n_events"] == saved.n_events
        assert abs(fit["gpd_shape_xi"] - saved.gpd_shape_xi) < 1e-4
        assert abs(fit["rp100_ms"] - saved.rp100_ms) < 1e-4
        jobs.append((k, fit["threshold_ms"], fit["n_events"], fit["gpd_scale"],
                     fit["gpd_shape_xi"], seeds[k]))
    param_levels = np.empty((len(results), N_BOOT_PARAM, len(RETURN_PERIODS)))
    with ctx.Pool(workers) as pool:
        # jobs finish in any order; each one says which fit it belongs to
        for n_done, (k, simulated) in enumerate(pool.imap_unordered(param_job, jobs), 1):
            param_levels[k] = simulated
            if n_done % 12 == 0:
                print(f"A: {n_done}/{len(jobs)} fits bootstrapped, {time.time() - start:.0f}s",
                      flush=True)
    np.savez_compressed(WORK / "boot_param.npz", levels=param_levels, rps=RETURN_PERIODS)
    print(f"A done in {(time.time() - start) / 60:.1f} min", flush=True)

    # Part B. Resampled-years bootstrap.
    start = time.time()
    water_year = np.where(daily.date.dt.month >= 10, daily.date.dt.year + 1, daily.date.dt.year)
    water_year = water_year.astype(int)
    # One seed for all cells, so every cell sees the same resampled years.
    seed = np.random.SeedSequence(SEED + 1).spawn(1)[0]
    jobs = []
    for k, cell_id in enumerate(cells.cell_id):
        jobs.append((k, daily[str(cell_id)].to_numpy(np.float64), water_year, seed))
    gev_levels = np.empty((len(cells), N_BOOT_YEARS, len(RETURN_PERIODS)))
    pot_levels = np.empty((len(cells), N_BOOT_YEARS, len(PERCENTILES), len(SEPARATIONS),
                           len(RETURN_PERIODS)))
    with ctx.Pool(workers) as pool:
        for n_done, (k, cell_gev, cell_pot) in enumerate(pool.imap_unordered(years_job, jobs), 1):
            gev_levels[k] = cell_gev
            pot_levels[k] = cell_pot
            print(f"B: {n_done}/{len(jobs)} cells done, {time.time() - start:.0f}s", flush=True)
    np.savez_compressed(WORK / "boot_years.npz", gev=gev_levels, pot=pot_levels,
                        rps=RETURN_PERIODS, pcts=PERCENTILES, seps=SEPARATIONS,
                        cell_id=cells.cell_id.to_numpy())
    print(f"B done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
