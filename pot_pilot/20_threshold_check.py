"""Threshold check for the recommended POT version: POT with the wider prior on the shape.

POT is refitted at every pilot cell with thresholds at the 95th, 96th, 97th, 98th and 99th
percentile of the cell's daily values, always with the 2-day separation rule. The 98th is the
setting used in steps 15 to 19.

The threshold holds up if:
    - the 100-year estimate at the 98th lies inside the 95% interval at every other threshold
    - the shape and the modified scale (scale - shape x threshold) stay steady above the 98th.
      For a GPD, both should stay the same at every threshold that is high enough, so a drift
      above the 98th would mean it is too low.
    - the GPD fits the events: a Kolmogorov-Smirnov test of the excesses against the fitted
      GPD at the posterior median
The spread of the 100-year estimate across thresholds is also compared with the spread for the
maximum-likelihood POT fits in step 08, to see whether the prior makes the threshold matter less.

Reads:
    data/pilot_daily_spduv10max.csv, output/pilot_locations_selected.csv (step 03),
    output/pilot_pot_results.csv (step 08)
Writes:
    output/threshold_check.csv            one row per cell and threshold
    output/threshold_check_summary.csv    one row per cell
"""
import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import genpareto, kstest

import bayes
import pot
from paths import DAILY_CSV, N_YEARS, OUT, require

# Nelder-Mead warns when a starting point is outside the support; those points are skipped.
warnings.filterwarnings("ignore", category=RuntimeWarning)
PRIOR = "wider"
PERCENTILES = [95, 96, 97, 98, 99]
REFERENCE = 98
SEPARATION = 2
RETURN_PERIODS = [20, 50, 100]
RHAT_LIMIT = 1.01
ESS_LIMIT = 400

require(DAILY_CSV, OUT / "pilot_locations_selected.csv", OUT / "pilot_pot_results.csv")
CELLS = pd.read_csv(OUT / "pilot_locations_selected.csv")
VALUES, _ = bayes.pilot_series(CELLS)


def fit_threshold(task):
    """Fit POT with the wider prior at one cell and one threshold."""
    k, percentile = task
    cell = CELLS.iloc[k]
    series = VALUES[:, k]
    threshold = float(np.percentile(series, percentile))
    _, peaks, _ = pot.decluster(series, threshold, SEPARATION)
    excesses = peaks - threshold
    mean, variance = excesses.mean(), excesses.var()
    xi_start = 0.5 * (1.0 - mean * mean / variance)
    start = [np.log(0.5 * mean * (mean * mean / variance + 1.0)), xi_start]
    seed = 1000 * k + percentile

    for attempt, chains in enumerate([dict(n_warmup=4000, n_keep=6000),
                                      dict(n_warmup=8000, n_keep=12000)]):
        fit = bayes.fit_version("POT", PRIOR, excesses, start, seed + attempt, RETURN_PERIODS,
                                N_YEARS, threshold, **chains)
        max_rhat = max(fit["rhat"].values())
        min_ess = min(fit["ess"].values())
        if max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT:
            break

    xi, scale = fit["flat"]["xi"], fit["flat"]["scale"]
    modified_scale = scale - xi * threshold
    row = {
        "cell_id": cell.cell_id, "candidate": cell.candidate, "longitude": cell.longitude,
        "percentile": percentile, "threshold_ms": threshold, "n_events": excesses.size,
        "events_per_year": excesses.size / N_YEARS,
        "xi_median": float(np.median(xi)),
        "xi_lower": float(np.percentile(xi, 2.5)), "xi_upper": float(np.percentile(xi, 97.5)),
        "modified_scale_median": float(np.median(modified_scale)),
        "modified_scale_lower": float(np.percentile(modified_scale, 2.5)),
        "modified_scale_upper": float(np.percentile(modified_scale, 97.5)),
        "ks_pvalue": float(kstest(excesses, genpareto(c=np.median(xi),
                                                      scale=np.median(scale)).cdf).pvalue),
        "max_rhat": max_rhat, "min_ess": min_ess,
        "converged": bool(max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT),
    }
    for j, period in enumerate(RETURN_PERIODS):
        low, mid, high = np.percentile(fit["levels"][:, j], [2.5, 50, 97.5])
        row.update({f"rp{period}_ms": mid, f"rp{period}_lower_ms": low,
                    f"rp{period}_upper_ms": high})
    return row


def summarize(fits):
    """Per cell: does the 98th percentile hold up, and how much does the threshold matter?"""
    step08 = pd.read_csv(OUT / "pilot_pot_results.csv")
    step08 = step08[(step08.separation_days == SEPARATION)
                    & step08.threshold_pct.isin([95, 97, 98, 99])]
    rows = []
    for cell_id, group in fits.groupby("cell_id", sort=False):
        group = group.set_index("percentile")
        ref = group.loc[REFERENCE]
        others = group.drop(index=REFERENCE)
        inside = ((others.rp100_lower_ms <= ref.rp100_ms)
                  & (ref.rp100_ms <= others.rp100_upper_ms))
        four = group.loc[[95, 97, 98, 99]]
        mle = step08[step08.cell_id == cell_id].rp100_ms
        above = group.loc[[REFERENCE, 99]]
        rows.append({
            "cell_id": cell_id, "location": ref.candidate,
            "rp100_at_98th_ms": ref.rp100_ms,
            "inside_every_other_interval": bool(inside.all()),
            "largest_change_from_98th_pct": float(
                (100 * (others.rp100_ms / ref.rp100_ms - 1)).abs().max()),
            "spread_95_to_99_pct": float(100 * (four.rp100_ms.max() - four.rp100_ms.min())
                                         / four.rp100_ms.median()),
            "spread_95_to_99_without_prior_pct": float(100 * (mle.max() - mle.min())
                                                       / mle.median()),
            "xi_change_98th_to_99th": float(above.xi_median.iloc[1] - above.xi_median.iloc[0]),
            "xi_99th_inside_98th_interval": bool(
                ref.xi_lower <= group.loc[99].xi_median <= ref.xi_upper),
            "modified_scale_99th_inside_98th_interval": bool(
                ref.modified_scale_lower <= group.loc[99].modified_scale_median
                <= ref.modified_scale_upper),
            "ks_pvalue_at_98th": ref.ks_pvalue,
            "lowest_ks_pvalue": float(group.ks_pvalue.min()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()
    start = time.time()
    tasks = [(k, p) for k in range(len(CELLS)) for p in PERCENTILES]
    with Pool(args.processes) as pool:
        rows = list(pool.imap_unordered(fit_threshold, tasks))
    fits = pd.DataFrame(rows).sort_values(["longitude", "percentile"]).drop(columns="longitude")
    print(f"{len(fits)} fits; {int((~fits.converged).sum())} did not pass the convergence checks")
    summary = summarize(fits)
    numbers = fits.select_dtypes("number").columns.difference(["cell_id", "percentile",
                                                               "n_events"])
    fits[numbers] = fits[numbers].round(4)
    fits.to_csv(OUT / "threshold_check.csv", index=False)
    numbers = summary.select_dtypes("number").columns.difference(["cell_id"])
    summary[numbers] = summary[numbers].round(3)
    summary.to_csv(OUT / "threshold_check_summary.csv", index=False)
    print(summary.drop(columns="cell_id").to_string(index=False))
    print(f"\nwrote {OUT / 'threshold_check.csv'}, {OUT / 'threshold_check_summary.csv'}  "
          f"({(time.time() - start) / 60:.1f} minutes)")
