"""Simulation study: how often each version's 95% interval contains the true return level, and
how biased its estimates are, when the true model is known.

Three true models, taken from the POT fits in step 08 (98th percentile, 2-day separation):
    light   Atlanta, GA        a bounded tail (xi about -0.11)
    zero    Chicago, IL        close to the Gumbel case (xi about 0)
    heavy   New Orleans, LA    a heavy tail (xi about +0.18)

Each simulated record is 45 years of events. Events arrive at random (a Poisson process) and
each event's peak follows a GPD. To make the annual maxima and the POT events come from the
same record, the simulation runs at a lower level with 20 events a year, where every year has
plenty of events, and uses the GPD's threshold stability to keep the same tail above the pilot
threshold. Then:
    GEV   is fitted to the largest event of each year (45 values)
    POT   is fitted to the events above the pilot cell's own threshold
Under this model the annual maxima follow a GEV exactly, so neither method has the advantage
of being the true model. The true return levels come from the model itself.

The records are idealized: events are independent and there is no seasonality, so this tests
the statistics, not the event grouping.

Six versions are fitted to every record: the GEV and POT with the flat, main and wider priors.
The chains are shorter than in steps 15 and 16 (2,000 warm-up and 2,500 kept draws per chain)
so that the 9,000 fits finish in reasonable time. A fit that does not pass the convergence
checks is refitted with the full-length chains of steps 15 and 16. The first 10 records of
each model are also refitted with full-length chains, to check that the shorter chains give
the same intervals.

Usage:
    python 17_simulation_study.py [--records 500] [--processes N] [--fresh]
A run that stops part-way resumes where it left off when started again, unless --fresh.
Reads:
    output/pilot_pot_results.csv (step 08)
Writes:
    output/simulation_coverage.csv        one row per true model, version and return period
    output/work/simulation_fits.csv       every fit
    output/work/simulation_chain_check.csv   shorter against full-length chains
    output/work/simulation_fits_in_progress.csv   only while a run is unfinished
"""
import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import genpareto

import bayes
from paths import N_YEARS, OUT, WORK, require

# Nelder-Mead warns when a starting point is outside the support; those points are skipped.
warnings.filterwarnings("ignore", category=RuntimeWarning)
PERCENTILE = 98    # the same POT setting as steps 15 and 16
SEPARATION = 2
TRUE_MODELS = {"light": "Atlanta, GA", "zero": "Chicago, IL", "heavy": "New Orleans, LA"}
VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("GEV", "wider"),
            ("POT", "flat"), ("POT", "main"), ("POT", "wider")]
RETURN_PERIODS = [20, 50, 100]
BASE_RATE = 20.0    # events a year at the level the simulation runs at
CHAINS = {"short": dict(n_warmup=2000, n_keep=2500), "full": dict(n_warmup=4000, n_keep=6000)}
CHAIN_CHECK_RECORDS = 10
COLUMNS = ["true_model", "record", "model", "prior", "return_period", "n_values", "true_ms",
           "estimate_ms", "lower_95_ms", "upper_95_ms", "covered", "max_rhat", "min_ess",
           "converged", "error"]
RHAT_LIMIT = 1.01
ESS_LIMIT = 400


def true_models():
    """The three true models: threshold, event rate, GPD scale and shape, and the true levels."""
    require(OUT / "pilot_pot_results.csv")
    fits = pd.read_csv(OUT / "pilot_pot_results.csv")
    fits = fits[(fits.threshold_pct == PERCENTILE) & (fits.separation_days == SEPARATION)]
    models = {}
    for name, candidate in TRUE_MODELS.items():
        fit = fits[fits.candidate == candidate].iloc[0]
        u, rate, sigma, xi = fit.threshold_ms, fit.events_per_year, fit.gpd_scale, fit.gpd_shape_xi
        # The lower level with BASE_RATE events a year, and the GPD scale there. Above u, this
        # gives back exactly the same GPD, because the GPD is stable under a change of threshold.
        if abs(xi) < bayes.SMALL_XI:
            u0 = u - sigma * np.log(BASE_RATE / rate)
        else:
            u0 = u + sigma / xi * ((BASE_RATE / rate) ** (-xi) - 1.0)
        sigma0 = sigma + xi * (u0 - u)
        assert sigma0 > 0
        truth = bayes.pot_return_levels(RETURN_PERIODS, u, [rate], [sigma], [xi])[0]
        models[name] = dict(candidate=candidate, threshold=u, rate=rate, sigma=sigma, xi=xi,
                            base_level=u0, base_sigma=sigma0,
                            truth=dict(zip(RETURN_PERIODS, truth)))
    return models


def simulate_record(model, rng):
    """One 45-year record: the annual maxima, and the excesses over the pilot threshold."""
    counts = rng.poisson(BASE_RATE, N_YEARS)
    assert (counts > 0).all()
    peaks = model["base_level"] + genpareto.rvs(c=model["xi"], scale=model["base_sigma"],
                                                size=counts.sum(), random_state=rng)
    year = np.repeat(np.arange(N_YEARS), counts)
    annual_max = np.full(N_YEARS, -np.inf)
    np.maximum.at(annual_max, year, peaks)
    excesses = peaks[peaks > model["threshold"]] - model["threshold"]
    return annual_max, excesses


def starting_points(annual_max, excesses):
    """Rough starting points for the fits: Gumbel moments and GPD moments."""
    scale = np.sqrt(6.0) * annual_max.std() / np.pi
    gev_start = [annual_max.mean() - 0.5772 * scale, np.log(scale), 0.0]
    mean, variance = excesses.mean(), excesses.var()
    xi = 0.5 * (1.0 - mean * mean / variance)
    gpd_start = [np.log(0.5 * mean * (mean * mean / variance + 1.0)), xi]
    return gev_start, gpd_start


def fit_record(task):
    """Fit the versions to one simulated record. Returns one row per version and period.

    task = (true model, record number, model, "short" or "full" chains, and optionally the
    versions to fit, which defaults to all of them).
    """
    model_name, record, model, chain_length = task[:4]
    only = task[4] if len(task) > 4 else None
    chains = CHAINS[chain_length]
    rng = np.random.default_rng([17, list(TRUE_MODELS).index(model_name), record])
    annual_max, excesses = simulate_record(model, rng)
    gev_start, gpd_start = starting_points(annual_max, excesses)
    rows = []
    for v, (fitted, prior_name) in enumerate(VERSIONS):
        if only is not None and (fitted, prior_name) not in only:
            continue
        seed = 100000 * list(TRUE_MODELS).index(model_name) + 10 * record + v
        if chain_length == "full":
            seed += 5    # different draws from the short run being checked
        try:
            if fitted == "GEV":
                fit = bayes.fit_version("GEV", prior_name, annual_max, gev_start, seed,
                                        RETURN_PERIODS, N_YEARS, **chains)
            else:
                fit = bayes.fit_version("POT", prior_name, excesses, gpd_start, seed,
                                        RETURN_PERIODS, N_YEARS, model["threshold"], **chains)
        except Exception as error:    # a failed fit is recorded, not hidden
            for period in RETURN_PERIODS:
                rows.append({"true_model": model_name, "record": record, "model": fitted,
                             "prior": prior_name, "return_period": period,
                             "error": str(error)[:80]})
            continue
        checked = ["xi"] + [f"rp{p}" for p in RETURN_PERIODS]
        max_rhat = max(fit["rhat"][q] for q in checked)
        min_ess = min(fit["ess"][q] for q in checked)
        for j, period in enumerate(RETURN_PERIODS):
            low, mid, high = np.percentile(fit["levels"][:, j], [2.5, 50, 97.5])
            truth = model["truth"][period]
            rows.append({
                "true_model": model_name, "record": record, "model": fitted, "prior": prior_name,
                "return_period": period, "n_values": annual_max.size if fitted == "GEV"
                else excesses.size, "true_ms": truth, "estimate_ms": mid, "lower_95_ms": low,
                "upper_95_ms": high, "covered": bool(low <= truth <= high),
                "max_rhat": max_rhat, "min_ess": min_ess,
                "converged": bool(max_rhat < RHAT_LIMIT and min_ess >= ESS_LIMIT), "error": "",
            })
    return rows


def summarize(fits):
    """Coverage, bias and error per true model, version and return period."""
    rows = []
    for (true_model, fitted, prior_name, period), group in fits.groupby(
            ["true_model", "model", "prior", "return_period"], sort=False):
        ok = group[group.converged == True]    # noqa: E712 (the column can hold NaN)
        truth = group.true_ms.dropna().iloc[0]
        error = ok.estimate_ms - truth
        coverage = ok.covered.mean()
        rows.append({
            "true_model": true_model, "true_xi": MODELS[true_model]["xi"],
            "model": fitted, "prior": prior_name, "return_period": period, "true_ms": truth,
            "n_records": group.record.nunique(), "n_converged": len(ok),
            "coverage_pct": 100 * coverage,
            "coverage_se_pct": 100 * np.sqrt(coverage * (1 - coverage) / len(ok)),
            "bias_ms": error.mean(), "bias_pct": 100 * error.mean() / truth,
            "median_error_pct": 100 * error.median() / truth,
            "rmse_ms": np.sqrt((error**2).mean()),
            "rmse_pct": 100 * np.sqrt((error**2).mean()) / truth,
            "median_width_ms": (ok.upper_95_ms - ok.lower_95_ms).median(),
            "share_true_above_interval_pct": 100 * (ok.upper_95_ms < truth).mean(),
            "share_true_below_interval_pct": 100 * (ok.lower_95_ms > truth).mean(),
        })
    return pd.DataFrame(rows)


MODELS = true_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--fresh", action="store_true", help="ignore an unfinished earlier run")
    args = parser.parse_args()
    for name, m in MODELS.items():
        print(f"{name:<6} {m['candidate']:<16} threshold {m['threshold']:.2f} m/s, "
              f"{m['rate']:.2f} events/yr, scale {m['sigma']:.3f}, xi {m['xi']:+.3f}; true "
              + ", ".join(f"{p}-yr {m['truth'][p]:.2f}" for p in RETURN_PERIODS))

    # Each finished record is added to a progress file straight away, so a run that stops
    # part-way (a restart, a closed laptop) picks up where it left off. --fresh starts over.
    progress = WORK / "simulation_fits_in_progress.csv"
    done, rows = set(), []
    if progress.exists() and not args.fresh:
        earlier = pd.read_csv(progress)
        earlier["error"] = earlier.error.fillna("")
        rows = earlier.to_dict("records")
        done = set(zip(earlier.true_model, earlier.record))
        print(f"resuming: {len(done)} records already done")
    elif progress.exists():
        progress.unlink()

    start = time.time()
    tasks = [(name, r, MODELS[name], "short") for name in MODELS for r in range(args.records)
             if (name, r) not in done]
    with Pool(args.processes) as pool:
        for i, result in enumerate(pool.imap_unordered(fit_record, tasks, chunksize=4)):
            rows.extend(result)
            pd.DataFrame(result).reindex(columns=COLUMNS).to_csv(
                progress, mode="a", index=False, header=not progress.exists())
            if (i + 1) % 100 == 0:
                print(f"  {len(done) + i + 1} of {len(MODELS) * args.records} records "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)
        # Fits that did not pass the convergence checks are refitted with full-length chains,
        # and the refit replaces them.
        fits_so_far = pd.DataFrame(rows)
        failed = fits_so_far[fits_so_far.converged != True]    # noqa: E712
        to_refit = {}
        for row in failed.itertuples():
            to_refit.setdefault((row.true_model, row.record), set()).add((row.model, row.prior))
        refit_tasks = [(name, r, MODELS[name], "full", versions)
                       for (name, r), versions in to_refit.items()]
        refits = [row for result in pool.imap_unordered(fit_record, refit_tasks)
                  for row in result]
        print(f"refitted {len(refits) // len(RETURN_PERIODS)} fits with full-length chains")
        keys = ["true_model", "record", "model", "prior", "return_period"]
        if refits:
            refitted = pd.DataFrame(refits)
            replaced = fits_so_far.set_index(keys).index.isin(refitted.set_index(keys).index)
            kept = fits_so_far[~replaced]
            refitted["refit"] = True
            rows = pd.concat([kept, refitted], ignore_index=True).to_dict("records")
        # The chain-length check: the first records of each model again, with full chains.
        check_tasks = [(name, r, MODELS[name], "full") for name in MODELS
                       for r in range(min(CHAIN_CHECK_RECORDS, args.records))]
        check_rows = [row for result in pool.imap_unordered(fit_record, check_tasks)
                      for row in result]

    fits = pd.DataFrame(rows)
    fits["refit"] = fits.get("refit", False)
    fits["refit"] = fits.refit.fillna(False).astype(bool)
    fits = fits[fits.record < args.records].sort_values(
        ["true_model", "record", "model", "prior", "return_period"])
    fits.to_csv(WORK / "simulation_fits.csv", index=False)
    progress.unlink()
    n_failed = int((fits.error != "").sum())
    n_unconverged = int((fits.converged == False).sum())    # noqa: E712
    print(f"\n{len(fits) // len(RETURN_PERIODS)} fits; {n_failed // len(RETURN_PERIODS)} failed, "
          f"{n_unconverged // len(RETURN_PERIODS)} did not pass the convergence checks after "
          f"refitting")

    check = pd.DataFrame(check_rows).merge(fits[fits.refit != True], on=keys,    # noqa: E712
                                           suffixes=("_full", "_short"))
    for column in ("lower_95_ms", "estimate_ms", "upper_95_ms"):
        check[f"{column}_diff"] = check[f"{column}_full"] - check[f"{column}_short"]
    check["width_full"] = check.upper_95_ms_full - check.lower_95_ms_full
    check.to_csv(WORK / "simulation_chain_check.csv", index=False)
    worst = check[[c for c in check.columns if c.endswith("_diff")]].abs().max().max()
    worst_rel = (check.upper_95_ms_diff.abs() / check.width_full).max()
    print(f"chain-length check ({len(check) // len(RETURN_PERIODS)} fits): largest change "
          f"{worst:.3f} m/s; largest change in the upper end, as a share of the interval width, "
          f"{100 * worst_rel:.1f}%")

    summary = summarize(fits)
    summary = summary.round(3)
    summary.to_csv(OUT / "simulation_coverage.csv", index=False)
    show = summary[summary.return_period == 100][
        ["true_model", "model", "prior", "n_converged", "coverage_pct", "coverage_se_pct",
         "bias_pct", "rmse_pct", "median_width_ms"]]
    print("\n100-year level:")
    print(show.to_string(index=False))
    print(f"\nwrote {OUT / 'simulation_coverage.csv'}, {WORK / 'simulation_fits.csv'}, "
          f"{WORK / 'simulation_chain_check.csv'}  ({(time.time() - start) / 60:.1f} minutes)")
