"""Stability: how much each version's 100-year estimate moves when one year is left out.

For every pilot cell and each of the six versions (the GEV and POT with the flat, main and
wider priors), the version is fitted to all 45 water years and then refitted 45 times, each
time without one year. For the GEV, that year's annual maximum is removed. For POT, the
threshold is recomputed from the other 44 years' daily values and the events peaking in the
left-out year are removed.

Two measures of the 100-year estimate are reported:
    posterior median   the estimate used in every other step
    posterior mode     the return level at the highest point of the posterior. It has no
                       simulation noise, so it shows cleanly which years matter most.
The posterior median carries some simulation noise from the sampler, so each change also has
its Monte Carlo standard error, worked out from the effective sample size. Changes smaller
than about twice that are noise.

All fits use the same chain lengths as steps 15 and 16. A fit that does not pass the
convergence checks is refitted with chains twice as long. Each fit of a cell and version uses
the same random seed, which keeps the simulation noise in the changes down.

Usage:
    python 19_leave_one_year_out.py [--processes N] [--fresh]
A run that stops part-way resumes where it left off when started again, unless --fresh.
Reads:
    data/pilot_daily_spduv10max.csv, output/pilot_locations_selected.csv (step 03)
Writes:
    output/leave_one_year_out.csv     one row per cell, version and left-out year
    output/stability_summary.csv      one row per cell and version, plus the pooled rows
    output/work/leave_one_year_out_in_progress.csv   only while a run is unfinished
"""
import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import bayes
import pot
from paths import DAILY_CSV, OUT, WATER_YEARS, WORK, require

# Nelder-Mead warns when a starting point is outside the support; those points are skipped.
warnings.filterwarnings("ignore", category=RuntimeWarning)
PERCENTILE = 98
SEPARATION = 2
VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("GEV", "wider"),
            ("POT", "flat"), ("POT", "main"), ("POT", "wider")]
RETURN_PERIOD = 100
RHAT_LIMIT = 1.01
ESS_LIMIT = 400

require(DAILY_CSV, OUT / "pilot_locations_selected.csv")
CELLS = pd.read_csv(OUT / "pilot_locations_selected.csv")
VALUES, ANNUAL_MAX = bayes.pilot_series(CELLS)
DAILY_DATES = pd.read_csv(DAILY_CSV, usecols=["date"], parse_dates=["date"]).date
DAY_WATER_YEAR = np.where(DAILY_DATES.dt.month >= 10, DAILY_DATES.dt.year + 1,
                          DAILY_DATES.dt.year)
YEARS = np.array(WATER_YEARS)
COLUMNS = ["cell_id", "candidate", "model", "prior", "left_out", "rp100_median_ms",
           "rp100_mode_ms", "rp100_mc_se_ms", "xi_median", "n_values", "max_rhat", "min_ess",
           "converged", "longer_chains"]


def starting_points(annual_max, excesses):
    """Rough starting points for the fits: Gumbel moments and GPD moments."""
    scale = np.sqrt(6.0) * annual_max.std() / np.pi
    gev_start = [annual_max.mean() - 0.5772 * scale, np.log(scale), 0.0]
    mean, variance = excesses.mean(), excesses.var()
    xi = 0.5 * (1.0 - mean * mean / variance)
    gpd_start = [np.log(0.5 * mean * (mean * mean / variance + 1.0)), xi]
    return gev_start, gpd_start


def fit_without(task):
    """Fit one version at one cell without one year (left_out = 0 means no year left out)."""
    k, model, prior_name, left_out = task
    cell = CELLS.iloc[k]
    keep_years = YEARS != left_out
    n_years = int(keep_years.sum())
    maxima = ANNUAL_MAX[keep_years, k]
    series = VALUES[:, k]
    in_kept_years = DAY_WATER_YEAR != left_out
    threshold = float(np.percentile(series[in_kept_years], PERCENTILE))
    peak_days, peaks, _ = pot.decluster(series, threshold, SEPARATION)
    excesses = peaks[in_kept_years[peak_days]] - threshold
    gev_start, gpd_start = starting_points(maxima, excesses)
    seed = 1000 * k + VERSIONS.index((model, prior_name))    # the same for every left-out year

    for attempt, chains in enumerate([dict(n_warmup=4000, n_keep=6000),
                                      dict(n_warmup=8000, n_keep=12000)]):
        if model == "GEV":
            fit = bayes.fit_version("GEV", prior_name, maxima, gev_start, seed, [RETURN_PERIOD],
                                    n_years, **chains)
        else:
            fit = bayes.fit_version("POT", prior_name, excesses, gpd_start, seed,
                                    [RETURN_PERIOD], n_years, threshold, **chains)
        max_rhat = max(fit["rhat"][q] for q in ("xi", f"rp{RETURN_PERIOD}"))
        min_ess = min(fit["ess"][q] for q in ("xi", f"rp{RETURN_PERIOD}"))
        if max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT:
            break

    levels = fit["levels"][:, 0]
    median = float(np.median(levels))
    # Monte Carlo standard error of a median: sqrt(p(1-p)) / density at the median / sqrt(ESS),
    # with the density estimated from the draws within 5 percentage points of the median.
    low, high = np.percentile(levels, [45, 55])
    density = 0.10 / (high - low)
    mc_se = 0.5 / density / np.sqrt(fit["ess"][f"rp{RETURN_PERIOD}"])
    mode = fit["mode"]
    if model == "GEV":
        at_mode = bayes.gev_return_levels([RETURN_PERIOD], [mode[0]], [np.exp(mode[1])],
                                          [mode[2]])[0, 0]
    else:
        at_mode = bayes.pot_return_levels([RETURN_PERIOD], threshold, [excesses.size / n_years],
                                          [np.exp(mode[0])], [mode[1]])[0, 0]
    return {
        "cell_id": cell.cell_id, "candidate": cell.candidate, "model": model,
        "prior": prior_name, "left_out": int(left_out), "rp100_median_ms": median,
        "rp100_mode_ms": float(at_mode), "rp100_mc_se_ms": float(mc_se),
        "xi_median": float(np.median(fit["flat"]["xi"])),
        "n_values": maxima.size if model == "GEV" else excesses.size,
        "max_rhat": max_rhat, "min_ess": min_ess,
        "converged": bool(max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT),
        "longer_chains": attempt == 1,
    }


def summarize(fits):
    """Median and largest change per cell and version, the years behind them, and the pools."""
    full = fits[fits.left_out == 0].set_index(["cell_id", "model", "prior"])
    loo = fits[fits.left_out != 0].copy()
    reference = full.reindex(pd.MultiIndex.from_frame(loo[["cell_id", "model", "prior"]]))
    loo["change_ms"] = loo.rp100_median_ms.to_numpy() - reference.rp100_median_ms.to_numpy()
    loo["change_pct"] = 100 * loo.change_ms / reference.rp100_median_ms.to_numpy()
    loo["mode_change_pct"] = 100 * (loo.rp100_mode_ms.to_numpy()
                                    / reference.rp100_mode_ms.to_numpy() - 1)
    # the noise in a change: both medians have their own Monte Carlo error
    loo["change_noise_ms"] = np.sqrt(loo.rp100_mc_se_ms.to_numpy() ** 2
                                     + reference.rp100_mc_se_ms.to_numpy() ** 2)

    rows = []
    for (cell_id, model, prior_name), group in loo.groupby(["cell_id", "model", "prior"],
                                                            sort=False):
        biggest = group.loc[group.change_pct.abs().idxmax()]
        biggest_mode = group.loc[group.mode_change_pct.abs().idxmax()]
        rows.append({
            "cell_id": cell_id, "location": group.candidate.iloc[0], "model": model,
            "prior": prior_name,
            "full_rp100_ms": full.loc[(cell_id, model, prior_name), "rp100_median_ms"],
            "median_abs_change_pct": group.change_pct.abs().median(),
            "largest_abs_change_pct": abs(biggest.change_pct),
            "largest_change_ms": biggest.change_ms,
            "largest_change_year": int(biggest.left_out),
            "largest_mode_change_pct": biggest_mode.mode_change_pct,
            "largest_mode_change_year": int(biggest_mode.left_out),
            "spread_of_estimates_ms": group.rp100_median_ms.max() - group.rp100_median_ms.min(),
            "median_noise_ms": group.change_noise_ms.median(),
        })
    summary = pd.DataFrame(rows)
    pooled = (summary.groupby(["model", "prior"], sort=False)
              .agg(median_abs_change_pct=("median_abs_change_pct", "median"),
                   largest_abs_change_pct=("largest_abs_change_pct", "max"),
                   median_largest_abs_change_pct=("largest_abs_change_pct", "median"),
                   median_noise_ms=("median_noise_ms", "median"))
              .reset_index())
    pooled.insert(0, "location", "All 13 cells (median across cells, largest overall)")
    return loo, pd.concat([summary, pooled], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--fresh", action="store_true", help="ignore an unfinished earlier run")
    args = parser.parse_args()

    progress = WORK / "leave_one_year_out_in_progress.csv"
    rows, done = [], set()
    if progress.exists() and not args.fresh:
        earlier = pd.read_csv(progress)
        rows = earlier.to_dict("records")
        done = set(zip(earlier.cell_id, earlier.model, earlier.prior, earlier.left_out))
        print(f"resuming: {len(done)} fits already done")
    elif progress.exists():
        progress.unlink()

    start = time.time()
    tasks = [(k, model, prior_name, year) for k in range(len(CELLS))
             for model, prior_name in VERSIONS for year in [0] + list(YEARS)
             if (CELLS.cell_id.iloc[k], model, prior_name, year) not in done]
    total = len(CELLS) * len(VERSIONS) * (len(YEARS) + 1)
    with Pool(args.processes) as pool:
        for i, row in enumerate(pool.imap_unordered(fit_without, tasks, chunksize=4)):
            rows.append(row)
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
                progress, mode="a", index=False, header=not progress.exists())
            if (i + 1) % 250 == 0:
                print(f"  {len(done) + i + 1} of {total} fits "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)

    fits = pd.DataFrame(rows).reindex(columns=COLUMNS)
    assert len(fits) == total
    n_longer = int(fits.longer_chains.sum())
    n_bad = int((~fits.converged.astype(bool)).sum())
    print(f"{total} fits; {n_longer} needed the longer chains; {n_bad} still did not pass the "
          f"convergence checks")

    loo, summary = summarize(fits)
    loo = loo.merge(CELLS[["cell_id", "longitude"]], on="cell_id").sort_values(
        ["longitude", "model", "prior", "left_out"]).drop(columns="longitude")
    numbers = loo.select_dtypes("number").columns.difference(["cell_id", "left_out", "n_values"])
    loo[numbers] = loo[numbers].round(4)
    loo.to_csv(OUT / "leave_one_year_out.csv", index=False)
    numbers = summary.select_dtypes("number").columns.difference(
        ["cell_id", "largest_change_year", "largest_mode_change_year"])
    summary[numbers] = summary[numbers].round(3)
    summary.to_csv(OUT / "stability_summary.csv", index=False)
    progress.unlink()

    print("\nAcross the 13 cells (100-year estimate, posterior median):")
    print(summary[summary.location.str.startswith("All")].drop(columns=[
        c for c in summary.columns if summary[summary.location.str.startswith("All")][c].isna()
        .all()]).to_string(index=False))
    print(f"\nwrote {OUT / 'leave_one_year_out.csv'}, {OUT / 'stability_summary.csv'}  "
          f"({(time.time() - start) / 60:.1f} minutes)")
