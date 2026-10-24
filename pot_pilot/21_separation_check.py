"""Storm-separation check: how much the rule for grouping days into independent events matters,
and whether the events it gives are actually independent.

Three parts:

1. The separation rule. The three POT versions (flat, main and wider priors) are refitted at
   every pilot cell at the 98th percentile with separation rules of 1, 2, 3, 5 and 7 days.
   POT with the wider prior, the recommended version, is also fitted on the full grid of
   thresholds (95th to 99th percentile) and rules.

2. Independence of the POT events. POT assumes each event is a separate draw from the GPD. Two
   checks, at every threshold:
   - The extremal index, from the intervals estimator of Ferro & Segers (2003). It is the
     reciprocal of the mean number of days above the threshold per independent cluster, and it
     needs no separation rule. Ferro & Segers also give the run length it implies: the
     shortest rule that splits the days above the threshold into that many clusters. A rule at
     least that long gives independent events. The share of days above the threshold that
     start a new event under each rule (the runs estimate of the extremal index) is reported
     beside it.
   - The Spearman correlation between consecutive event peaks under each rule. Independent
     events should show none.
   The extremal index assumes the series is the same all year round. Strong winds come in
   seasons, so days above a fixed threshold bunch up in the windy months even when the storms
   themselves are separate. To tell the two apart, the extremal index is also worked out with
   a threshold set separately for each calendar month (the same percentile of that month's
   days), which takes the season out.

3. The GEV. Annual maxima are only independent if each comes from a different storm. Water years
   start on October 1, so one storm straddling that date could set the maximum of two years.
   This checks every pair of consecutive water years for maxima less than 3 days apart, and
   gives the Spearman correlation between consecutive annual maxima. Correlated maxima can
   also mean a trend, so it reports Kendall's trend test of the maxima against the year.

Usage:
    python 21_separation_check.py [--processes N]
Reads:
    data/pilot_daily_spduv10max.csv, output/pilot_locations_selected.csv (step 03)
Writes:
    output/separation_check.csv       the POT fits: one row per cell, version, threshold and
                                      rule, with the change from the 2-day rule
    output/independence_check.csv     one row per cell, threshold and rule: events, the
                                      extremal index and its run length, and the correlation
                                      between consecutive peaks
    output/gev_boundary_check.csv     one row per cell: maxima from the same storm, and the
                                      correlation between consecutive annual maxima
"""
import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import bayes
import pot
from paths import DAILY_CSV, N_YEARS, OUT, WATER_YEARS, require

# Nelder-Mead warns when a starting point is outside the support; those points are skipped.
warnings.filterwarnings("ignore", category=RuntimeWarning)
PERCENTILES = [95, 96, 97, 98, 99]
SEPARATIONS = [1, 2, 3, 5, 7]
REFERENCE_PERCENTILE, REFERENCE_SEPARATION = 98, 2
PRIORS = ["flat", "main", "wider"]
RECOMMENDED = "wider"
RETURN_PERIODS = [20, 50, 100]
SAME_STORM_DAYS = 3
RHAT_LIMIT = 1.01
ESS_LIMIT = 400

require(DAILY_CSV, OUT / "pilot_locations_selected.csv")
CELLS = pd.read_csv(OUT / "pilot_locations_selected.csv")
VALUES, ANNUAL_MAX = bayes.pilot_series(CELLS)
DATES = pd.read_csv(DAILY_CSV, usecols=["date"], parse_dates=["date"]).date
WATER_YEAR = np.where(DATES.dt.month >= 10, DATES.dt.year + 1, DATES.dt.year)
MONTH = DATES.dt.month.to_numpy()


def fit_pot(task):
    """Fit one POT version at one cell, threshold and separation rule."""
    k, prior_name, percentile, separation = task
    cell = CELLS.iloc[k]
    series = VALUES[:, k]
    threshold = float(np.percentile(series, percentile))
    _, peaks, _ = pot.decluster(series, threshold, separation)
    excesses = peaks - threshold
    mean, variance = excesses.mean(), excesses.var()
    start = [np.log(0.5 * mean * (mean * mean / variance + 1.0)),
             0.5 * (1.0 - mean * mean / variance)]
    seed = 1000 * k + 100 * PRIORS.index(prior_name) + 10 * PERCENTILES.index(percentile) \
        + SEPARATIONS.index(separation)
    for attempt, chains in enumerate([dict(n_warmup=4000, n_keep=6000),
                                      dict(n_warmup=8000, n_keep=12000)]):
        fit = bayes.fit_version("POT", prior_name, excesses, start, seed + attempt,
                                RETURN_PERIODS, N_YEARS, threshold, **chains)
        max_rhat = max(fit["rhat"].values())
        min_ess = min(fit["ess"].values())
        if max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT:
            break
    row = {"cell_id": cell.cell_id, "candidate": cell.candidate, "longitude": cell.longitude,
           "prior": prior_name, "percentile": percentile, "separation_days": separation,
           "threshold_ms": threshold, "n_events": excesses.size,
           "xi_median": float(np.median(fit["flat"]["xi"])), "max_rhat": max_rhat,
           "min_ess": min_ess, "converged": bool(max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT)}
    for j, period in enumerate(RETURN_PERIODS):
        low, mid, high = np.percentile(fit["levels"][:, j], [2.5, 50, 97.5])
        row.update({f"rp{period}_ms": mid, f"rp{period}_lower_ms": low,
                    f"rp{period}_upper_ms": high})
    return row


def intervals_estimator(days_above):
    """Ferro & Segers (2003) extremal index, and the run length it implies.

    days_above are the day numbers of the days above the threshold. Returns theta (between 0
    and 1) and the shortest separation rule (in days) that splits them into theta x N clusters.
    """
    gaps = np.diff(days_above).astype(float)
    n = days_above.size
    if gaps.max() <= 2:
        theta = 2 * gaps.sum() ** 2 / ((n - 1) * (gaps**2).sum())
    else:
        theta = 2 * (gaps - 1).sum() ** 2 / ((n - 1) * ((gaps - 1) * (gaps - 2)).sum())
    theta = min(1.0, theta)
    n_clusters = int(np.floor(theta * n)) + 1
    # The n_clusters - 1 largest gaps separate clusters. pot.decluster starts a new event when a
    # gap is larger than the rule, so the rule is the largest gap that stays inside a cluster.
    ordered = np.sort(gaps)[::-1]
    run_length = int(ordered[n_clusters - 1]) if n_clusters - 1 < ordered.size else 1
    return theta, run_length


def independence_rows():
    """The extremal index and the peak correlations, per cell, threshold and rule."""
    rows = []
    for k, cell in enumerate(CELLS.itertuples()):
        series = VALUES[:, k]
        for percentile in PERCENTILES:
            threshold = float(np.percentile(series, percentile))
            days_above = np.flatnonzero(series > threshold)
            theta, run_length = intervals_estimator(days_above)
            monthly = np.array([np.percentile(series[MONTH == m], percentile)
                                for m in range(1, 13)])[MONTH - 1]
            theta_monthly, run_length_monthly = intervals_estimator(
                np.flatnonzero(series > monthly))
            for separation in SEPARATIONS:
                peak_days, peaks, n_above = pot.decluster(series, threshold, separation)
                rho, p_value = spearmanr(peaks[:-1], peaks[1:])
                rows.append({
                    "cell_id": cell.cell_id, "candidate": cell.candidate,
                    "longitude": cell.longitude, "percentile": percentile,
                    "separation_days": separation, "n_days_above": n_above,
                    "n_events": peaks.size, "runs_extremal_index": peaks.size / n_above,
                    "intervals_extremal_index": theta, "implied_run_length_days": run_length,
                    "rule_long_enough": separation >= run_length,
                    "monthly_intervals_extremal_index": theta_monthly,
                    "monthly_implied_run_length_days": run_length_monthly,
                    "rule_long_enough_without_season": separation >= run_length_monthly,
                    "peak_correlation": float(rho), "peak_correlation_p": float(p_value),
                    "share_gaps_within_7_days": float((np.diff(peak_days) <= 7).mean()),
                })
    return pd.DataFrame(rows)


def gev_boundary_rows():
    """Consecutive annual maxima from the same storm, and their correlation."""
    rows = []
    for k, cell in enumerate(CELLS.itertuples()):
        series = VALUES[:, k]
        max_day = []
        for year in WATER_YEARS:
            in_year = np.flatnonzero(WATER_YEAR == year)
            max_day.append(in_year[np.argmax(series[in_year])])
        max_day = np.array(max_day)
        days_apart = np.diff(max_day)
        same = np.flatnonzero(days_apart < SAME_STORM_DAYS)
        rho, p_value = spearmanr(ANNUAL_MAX[:-1, k], ANNUAL_MAX[1:, k])
        trend = kendalltau(WATER_YEARS, ANNUAL_MAX[:, k])
        rows.append({
            "cell_id": cell.cell_id, "candidate": cell.candidate,
            "pairs_from_same_storm": int(same.size),
            "same_storm_years": ", ".join(f"{WATER_YEARS[i]}-{WATER_YEARS[i + 1]}" for i in same),
            "closest_days_apart": int(days_apart.min()),
            "annual_max_correlation": float(rho), "annual_max_correlation_p": float(p_value),
            "trend_kendall_tau": float(trend.statistic), "trend_p": float(trend.pvalue),
            "mean_1980_2001_ms": float(ANNUAL_MAX[:22, k].mean()),
            "mean_2003_2024_ms": float(ANNUAL_MAX[23:, k].mean()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()
    start = time.time()

    tasks = set()
    for k in range(len(CELLS)):
        for prior_name in PRIORS:
            for separation in SEPARATIONS:
                tasks.add((k, prior_name, REFERENCE_PERCENTILE, separation))
        for percentile in PERCENTILES:
            for separation in SEPARATIONS:
                tasks.add((k, RECOMMENDED, percentile, separation))
    with Pool(args.processes) as pool:
        fits = pd.DataFrame(pool.map(fit_pot, sorted(tasks), chunksize=2))
    print(f"{len(fits)} POT fits; {int((~fits.converged).sum())} did not pass the convergence "
          f"checks ({(time.time() - start) / 60:.1f} min)")

    # Change from the 2-day rule at the same threshold, and from the reference setting.
    keys = ["cell_id", "prior", "percentile"]
    two_day = fits[fits.separation_days == REFERENCE_SEPARATION][keys + ["rp100_ms"]]
    fits = fits.merge(two_day.rename(columns={"rp100_ms": "rp100_2day_ms"}), on=keys)
    fits["rp100_change_from_2day_pct"] = 100 * (fits.rp100_ms / fits.rp100_2day_ms - 1)
    reference = fits[(fits.percentile == REFERENCE_PERCENTILE)
                     & (fits.separation_days == REFERENCE_SEPARATION)][["cell_id", "prior",
                                                                        "rp100_ms"]]
    fits = fits.merge(reference.rename(columns={"rp100_ms": "rp100_98th_2day_ms"}),
                      on=["cell_id", "prior"])
    fits["rp100_change_from_98th_2day_pct"] = 100 * (fits.rp100_ms / fits.rp100_98th_2day_ms - 1)
    fits = fits.sort_values(["longitude", "prior", "percentile", "separation_days"]).drop(
        columns="longitude")
    numbers = fits.select_dtypes("number").columns.difference(
        ["cell_id", "percentile", "separation_days", "n_events"])
    fits[numbers] = fits[numbers].round(4)
    fits.to_csv(OUT / "separation_check.csv", index=False)

    independence = independence_rows().sort_values(["longitude", "percentile",
                                                    "separation_days"]).drop(columns="longitude")
    independence = independence.round(4)
    independence.to_csv(OUT / "independence_check.csv", index=False)
    boundary = gev_boundary_rows().round(4)
    boundary.to_csv(OUT / "gev_boundary_check.csv", index=False)

    at98 = fits[fits.percentile == REFERENCE_PERCENTILE]
    print("\nChange in the 100-year estimate from the 2-day rule, 98th percentile "
          "(median and largest across the 13 cells):")
    for (prior_name, separation), group in at98.groupby(["prior", "separation_days"]):
        if separation != REFERENCE_SEPARATION:
            change = group.rp100_change_from_2day_pct
            print(f"  POT {prior_name:<6} {separation}-day: median {change.median():+5.1f}%, "
                  f"largest {change.abs().max():4.1f}%")
    run_lengths = independence[independence.separation_days == REFERENCE_SEPARATION]
    print("\nRun length implied by the extremal index (days), by threshold:")
    print(run_lengths.pivot(index="candidate", columns="percentile",
                            values="implied_run_length_days").to_string())
    print("\nThe same, with a threshold for each calendar month (season taken out):")
    print(run_lengths.pivot(index="candidate", columns="percentile",
                            values="monthly_implied_run_length_days").to_string())
    print("\nGEV: consecutive annual maxima from the same storm:",
          int(boundary.pairs_from_same_storm.sum()))
    print(f"\nwrote {OUT / 'separation_check.csv'}, {OUT / 'independence_check.csv'}, "
          f"{OUT / 'gev_boundary_check.csv'}  ({(time.time() - start) / 60:.1f} minutes)")
