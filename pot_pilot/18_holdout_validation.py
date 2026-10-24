"""Held-out validation: how well each version predicts annual maxima it was not fitted to.

The 45 water years are split into nine 5-year blocks (1980-1984, ..., 2020-2024). For each
block, each version is fitted to the other 40 years and then predicts the 5 held-out annual
maxima. Six versions, as in step 17: the GEV and POT with the flat, main and wider priors.

What is scored is the predictive distribution of a held-out annual maximum: the distribution
of next year's maximum, averaged over the posterior. This is a different question from
whether an interval for the 100-year level is right (step 17), so the two are reported
separately.

Scores (lower is better for both CRPS scores):
    twCRPS    the threshold-weighted continuous ranked probability score (Gneiting & Ranjan
              2011). It is the CRPS counted only above a threshold, here each cell's median
              annual maximum, so it judges the upper half of the distribution, where the
              extremes are. It is a proper scoring rule: it rewards accuracy and honest
              uncertainty together, so a forecast cannot gain by being too narrow or too wide.
    twCRPS, upper 10%   the same score counted only above each cell's 90th percentile annual
              maximum (about its 5 largest years), as a check that focuses on the tail.
    CRPS      the same score over the whole distribution, for comparison.
    coverage  the share of held-out maxima inside the central 90% predictive interval.

Score differences are against the flat GEV (the closest to CADENCE's current method) and, for
the constrained versions, against the same model with the flat prior. Each comes with a 95%
range from a bootstrap that resamples the 45 held-out years (2,000 resamples). The pooled
results resample the same years at every cell, because a storm year affects many cells at once.

How POT is fitted for each block: the threshold is the 98th percentile of the 40 training years'
daily values, events are grouped with the 2-day rule, and only events peaking in training
years are used. POT predicts the annual maximum as the largest event in a year:
P(max <= x) = exp(-rate * (1 + xi (x - u) / sigma) ^ (-1 / xi)) above the threshold u.

Usage:
    python 18_holdout_validation.py [--processes N]
Reads:
    data/pilot_daily_spduv10max.csv, output/pilot_locations_selected.csv (step 03)
Writes:
    output/holdout_scores.csv            one row per cell and version, plus the pooled rows
    output/work/holdout_predictions.csv  one row per cell, version and held-out year
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
BLOCK_YEARS = 5
VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("GEV", "wider"),
            ("POT", "flat"), ("POT", "main"), ("POT", "wider")]
PREDICTIVE_DRAWS = 1000
SCORES = ["twcrps", "twcrps_upper", "crps"]
N_BOOTSTRAP = 2000
RHAT_LIMIT = 1.01
ESS_LIMIT = 400
# The grid the predictive distributions are evaluated on (m/s): fine where the data are, coarser
# above 60 m/s, where every predictive distribution is already almost 1.
GRID = np.concatenate([np.linspace(0.0, 60.0, 3001), np.linspace(60.0, 150.0, 451)[1:]])

require(DAILY_CSV, OUT / "pilot_locations_selected.csv")
CELLS = pd.read_csv(OUT / "pilot_locations_selected.csv")
VALUES, ANNUAL_MAX = bayes.pilot_series(CELLS)
DAILY_DATES = pd.read_csv(DAILY_CSV, usecols=["date"], parse_dates=["date"]).date
DAY_WATER_YEAR = np.where(DAILY_DATES.dt.month >= 10, DAILY_DATES.dt.year + 1,
                          DAILY_DATES.dt.year)
YEARS = np.array(WATER_YEARS)
BLOCK = (YEARS - YEARS[0]) // BLOCK_YEARS    # 0 to 8


def starting_points(annual_max, excesses):
    """Rough starting points for the fits: Gumbel moments and GPD moments."""
    scale = np.sqrt(6.0) * annual_max.std() / np.pi
    gev_start = [annual_max.mean() - 0.5772 * scale, np.log(scale), 0.0]
    mean, variance = excesses.mean(), excesses.var()
    xi = 0.5 * (1.0 - mean * mean / variance)
    gpd_start = [np.log(0.5 * mean * (mean * mean / variance + 1.0)), xi]
    return gev_start, gpd_start


def predictive_cdf(model, draws, threshold=None):
    """P(annual maximum <= z) on GRID, averaged over the posterior draws."""
    z = GRID[None, :]
    xi = draws["xi"][:, None]
    scale = draws["scale"][:, None]
    small = np.abs(xi) < bayes.SMALL_XI
    safe_xi = np.where(small, 1.0, xi)
    if model == "GEV":
        y = (z - draws["location"][:, None]) / scale
    else:
        y = (z - threshold) / scale
    t = 1.0 + safe_xi * y
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # the expected number of exceedances of z in a year (GEV form: -log F)
        tail = np.where(t > 0, t ** (-1.0 / safe_xi), np.where(safe_xi > 0, np.inf, 0.0))
        tail = np.where(small, np.exp(-y), tail)
    if model == "GEV":
        cdf = np.exp(-tail)
    else:
        cdf = np.where(z >= threshold, np.exp(-draws["rate"][:, None] * tail), 0.0)
    return cdf.mean(axis=0)


def crps(cdf, observed, weight_from=-np.inf):
    """The CRPS of one observation, counted only above weight_from.

    The integral of F(z)^2 below the observation plus (1 - F(z))^2 above it, by the trapezoid
    rule on GRID, with the observation and weight_from added as exact end points.
    """
    lowest = max(weight_from, GRID[0])

    def integral(a, b, integrand):
        if b <= a:
            return 0.0
        inside = (GRID > a) & (GRID < b)
        z = np.concatenate([[a], GRID[inside], [b]])
        f = np.concatenate([[np.interp(a, GRID, cdf)], cdf[inside], [np.interp(b, GRID, cdf)]])
        return float(np.trapezoid(integrand(f), z))

    return (integral(lowest, observed, lambda f: f**2)
            + integral(max(lowest, observed), GRID[-1], lambda f: (1.0 - f) ** 2))


def quantile(cdf, q):
    """The q quantile of the predictive distribution, by interpolation on GRID."""
    return float(np.interp(q, np.maximum.accumulate(cdf), GRID))


def fit_fold(task):
    """Fit one version at one cell without one block of years; score the held-out years."""
    k, model, prior_name, block = task
    cell = CELLS.iloc[k]
    train_years = YEARS[BLOCK != block]
    test = BLOCK == block
    maxima = ANNUAL_MAX[~test, k]
    series = VALUES[:, k]
    in_training = np.isin(DAY_WATER_YEAR, train_years)
    threshold = float(np.percentile(series[in_training], PERCENTILE))
    peak_days, peaks, _ = pot.decluster(series, threshold, SEPARATION)
    keep = in_training[peak_days]
    excesses = peaks[keep] - threshold
    gev_start, gpd_start = starting_points(maxima, excesses)
    seed = 100000 * k + 1000 * block + 10 * VERSIONS.index((model, prior_name))

    for attempt, chains in enumerate([dict(n_warmup=4000, n_keep=6000),
                                      dict(n_warmup=8000, n_keep=12000)]):
        if model == "GEV":
            fit = bayes.fit_version("GEV", prior_name, maxima, gev_start, seed + attempt,
                                    [100], len(train_years), **chains)
        else:
            fit = bayes.fit_version("POT", prior_name, excesses, gpd_start, seed + attempt,
                                    [100], len(train_years), threshold, **chains)
        max_rhat = max(fit["rhat"][q] for q in ("xi", "rp100"))
        min_ess = min(fit["ess"][q] for q in ("xi", "rp100"))
        if max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT:
            break

    n_draws = fit["flat"]["xi"].size
    keep_draws = np.linspace(0, n_draws - 1, PREDICTIVE_DRAWS).astype(int)
    draws = {name: values[keep_draws] for name, values in fit["flat"].items()}
    cdf = predictive_cdf(model, draws, threshold if model == "POT" else None)
    lower, upper = quantile(cdf, 0.05), quantile(cdf, 0.95)
    weight_from = float(np.median(ANNUAL_MAX[:, k]))
    upper_weight_from = float(np.percentile(ANNUAL_MAX[:, k], 90))

    rows = []
    for year, observed in zip(YEARS[test], ANNUAL_MAX[test, k]):
        rows.append({
            "cell_id": cell.cell_id, "candidate": cell.candidate, "model": model,
            "prior": prior_name, "block": int(block), "water_year": int(year),
            "observed_ms": observed, "twcrps": crps(cdf, observed, weight_from),
            "twcrps_upper": crps(cdf, observed, upper_weight_from),
            "crps": crps(cdf, observed), "weight_from_ms": weight_from,
            "upper_weight_from_ms": upper_weight_from,
            "lower_90_ms": lower, "upper_90_ms": upper,
            "covered_90": bool(lower <= observed <= upper),
            "n_train_values": maxima.size if model == "GEV" else excesses.size,
            "max_rhat": max_rhat, "min_ess": min_ess,
            "converged": bool(max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT),
            "longer_chains": attempt == 1,
        })
    return rows


def bootstrap_range(differences, rng, n=N_BOOTSTRAP):
    """95% range of the mean of differences (years x cells), resampling whole years."""
    differences = np.asarray(differences, float)    # years x cells
    picks = rng.integers(0, differences.shape[0], size=(n, differences.shape[0]))
    means = differences[picks].mean(axis=(1, 2))
    return np.percentile(means, [2.5, 97.5])


def summarize(predictions):
    """Mean scores and score differences with bootstrap ranges, per cell and pooled."""
    rng = np.random.default_rng(18)
    table = predictions.pivot_table(index=["water_year", "cell_id"],
                                    columns=["model", "prior"],
                                    values=SCORES)
    covered = predictions.pivot_table(index=["water_year", "cell_id"],
                                      columns=["model", "prior"], values="covered_90")
    groups = [(cell.cell_id, cell.candidate) for cell in CELLS.itertuples()]
    groups.append((None, "All 13 cells"))
    rows = []
    for cell_id, label in groups:
        if cell_id is None:
            scores, cover = table, covered
        else:
            scores = table.xs(cell_id, level="cell_id", drop_level=False)
            cover = covered.xs(cell_id, level="cell_id", drop_level=False)
        for model, prior_name in VERSIONS:
            row = {"cell_id": cell_id, "location": label, "model": model, "prior": prior_name,
                   "n_years": len(scores),
                   "coverage_90_pct": 100 * cover[(model, prior_name)].mean()}
            for score in SCORES:
                mine = scores[(score, model, prior_name)].unstack("cell_id")
                row[f"mean_{score}"] = float(mine.to_numpy().mean())
                references = [("GEV", "flat", "vs_gev_flat")]
                if prior_name != "flat":
                    references.append((model, "flat", f"vs_{model.lower()}_flat"))
                for ref_model, ref_prior, name in references:
                    if (ref_model, ref_prior) == (model, prior_name):
                        continue
                    ref = scores[(score, ref_model, ref_prior)].unstack("cell_id")
                    difference = (mine - ref).to_numpy()
                    low, high = bootstrap_range(difference, rng)
                    row[f"{score}_diff_{name}"] = float(difference.mean())
                    row[f"{score}_diff_{name}_pct"] = 100 * float(difference.mean()) / float(
                        ref.to_numpy().mean())
                    row[f"{score}_diff_{name}_low"] = low
                    row[f"{score}_diff_{name}_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def check_scoring():
    """The predictive CDF and the CRPS against SciPy, for one known GEV and one known POT."""
    from scipy.integrate import quad
    from scipy.stats import genextreme
    one = {"location": np.array([20.0]), "scale": np.array([2.0]), "xi": np.array([0.15])}
    cdf = predictive_cdf("GEV", one)
    assert np.allclose(cdf, genextreme.cdf(GRID, c=-0.15, loc=20.0, scale=2.0), atol=1e-12)
    exact = (quad(lambda z: genextreme.cdf(z, c=-0.15, loc=20, scale=2) ** 2, 0, 24)[0]
             + quad(lambda z: genextreme.sf(z, c=-0.15, loc=20, scale=2) ** 2, 24, np.inf)[0])
    assert abs(crps(cdf, 24.0) - exact) < 1e-3, (crps(cdf, 24.0), exact)
    weighted = quad(lambda z: genextreme.sf(z, c=-0.15, loc=20, scale=2) ** 2, 24, np.inf)[0]
    assert abs(crps(cdf, 24.0, weight_from=24.0) - weighted) < 1e-3
    # POT with one draw: the annual maximum above u is a GEV with the matching parameters.
    rate, sigma, xi, u = 6.0, 1.3, 0.1, 9.0
    one = {"rate": np.array([rate]), "scale": np.array([sigma]), "xi": np.array([xi])}
    cdf = predictive_cdf("POT", one, threshold=u)
    location = u + sigma / xi * (rate**xi - 1.0)
    scale = sigma * rate**xi
    above = GRID >= u
    assert np.allclose(cdf[above], genextreme.cdf(GRID[above], c=-xi, loc=location, scale=scale),
                       atol=1e-12)


if __name__ == "__main__":
    check_scoring()
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()
    start = time.time()
    tasks = [(k, model, prior_name, block) for k in range(len(CELLS))
             for model, prior_name in VERSIONS for block in range(BLOCK.max() + 1)]
    rows = []
    with Pool(args.processes) as pool:
        for i, result in enumerate(pool.imap_unordered(fit_fold, tasks, chunksize=2)):
            rows.extend(result)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} of {len(tasks)} fits ({(time.time() - start) / 60:.1f} min)",
                      flush=True)
    predictions = pd.DataFrame(rows).sort_values(["cell_id", "model", "prior", "water_year"])
    assert len(predictions) == len(CELLS) * len(VERSIONS) * len(YEARS)
    predictions.to_csv(WORK / "holdout_predictions.csv", index=False)
    n_fits = len(tasks)
    n_longer = int(predictions.groupby(["cell_id", "model", "prior", "block"]).longer_chains
                   .first().sum())
    n_bad = int((~predictions.groupby(["cell_id", "model", "prior", "block"]).converged
                 .first()).sum())
    print(f"{n_fits} fits; {n_longer} needed the longer chains; {n_bad} still did not pass the "
          f"convergence checks")

    summary = summarize(predictions)
    numbers = summary.select_dtypes("number").columns.difference(["cell_id", "n_years"])
    summary[numbers] = summary[numbers].round(5)
    summary.to_csv(OUT / "holdout_scores.csv", index=False)
    pooled = summary[summary.location == "All 13 cells"]
    print("\nAll 13 cells (585 held-out years):")
    print(pooled[["model", "prior", "mean_twcrps", "twcrps_diff_vs_gev_flat_pct",
                  "mean_twcrps_upper", "twcrps_upper_diff_vs_gev_flat_pct", "mean_crps",
                  "crps_diff_vs_gev_flat_pct", "coverage_90_pct"]].to_string(index=False))
    print(f"\nwrote {OUT / 'holdout_scores.csv'}, {WORK / 'holdout_predictions.csv'}  "
          f"({(time.time() - start) / 60:.1f} minutes)")
