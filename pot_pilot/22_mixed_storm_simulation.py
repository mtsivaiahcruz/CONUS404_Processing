"""Mixed-storm simulation: do the six versions stay calibrated when the record mixes hurricanes
with other storms?

Step 17's simulated records have one storm type. Real coastal records do not: the largest
annual maxima at Houston, New Orleans, Miami and Boston are hurricane landfalls, and the shape
of a single GPD keeps rising with the threshold there (step 20). This step builds a simulator
with two storm types at the four coastal cells, checks it against the real records, and then
reruns step 17's coverage test on records from it.

1. Labels. Each event at the 98th percentile threshold with the 2-day separation rule is
   labeled tropical if a tropical or subtropical system in HURDAT2 (status TD, TS, HU, SD or SS)
   passed within 250 km of the cell between the day before the event's first day above the
   threshold and the day after its last. The tracks are given every 6 hours, so they are
   interpolated to every hour first, which keeps a fast storm from slipping between two
   points. The labels at 150 and 400 km are kept as a check on the distance.
2. Two storm types. Each type gets its own event rate and GPD above the threshold. Here
   "hurricanes" means every tropical event, tropical storms included. The other storms have
   248 to 284 events per cell, enough to fit their GPD on its own. The hurricanes have only 5
   to 25, so their shape is fitted to the four cells together, with a scale for each cell,
   and three hurricane tails are tested: the pooled estimate and the two ends of its 95%
   profile-likelihood interval. Each tail keeps its own scale, refitted for that shape.
3. Simulator. Each simulated record is 45 water years. Events of each type arrive at random,
   on days drawn from that type's real event dates at the cell (give or take a week), so the
   hurricane season is kept, and each event lasts a number of days above the threshold drawn
   from that type's real events. Events that fall within 2 days of each other merge into one,
   as they would in the real record. As in step 17, the other storms are simulated at a lower
   level with 20 events a year, so every year has an annual maximum, and the GPD's threshold
   stability keeps the same tail above the real threshold. The true return levels come from
   the two types together:  rate_other * P(other > x) + rate_hurricane * P(hurricane > x)
   = -ln(1 - 1/N).
4. Simulator check. 1,000 simulated records are compared with the real record at each cell
   and hurricane tail: event rates by type, the size of the events, the annual maxima, the
   days above the threshold per event, the share of events from June to November, and how many
   of the 5 largest annual maxima are hurricanes.
5. Coverage. The six versions of step 17 (the GEV and POT, each with the flat, main and wider
   priors) are fitted to 500 records from each cell and hurricane tail, with step 17's chains,
   convergence checks and refits. The POT threshold stays at the real one, as in step 17.

It also checks whether hurricanes explain Miami's rise (step 21a): at each coastal cell, the
number of tropical systems and tropical events before and after 1998, and the annual maxima
with the hurricane days taken out.

Usage:
    python 22_mixed_storm_simulation.py [--records 500] [--processes N] [--fresh] [--check-only]
--check-only stops after the simulator check (about a minute). The coverage test is 36,000
fits. A run that stops part-way resumes where it left off when started again, unless --fresh.
Reads:
    data/pilot_daily_spduv10max.csv, data/hurdat2-1851-2024.txt (NOAA's Atlantic hurricane
    tracks, nhc.noaa.gov/data), output/pilot_locations_selected.csv (step 03), and
    output/pilot_pot_results.csv (step 08, which step 17's code reads)
Writes:
    output/storm_labels.csv            one row per event at the four coastal cells
    output/storm_type_trend.csv        hurricanes before and after 1998, one row per cell
    output/mixed_storm_model.csv       the two-type fit, one row per cell and hurricane tail
    output/mixed_storm_check.csv       the simulator against the real record
    output/mixed_storm_simulation.csv  coverage, bias, error, width and interval score, one row
                                       per cell, hurricane tail, version and return period
    output/work/mixed_storm_fits.csv   every fit
"""
import argparse
import importlib.util
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, genpareto, kendalltau

import bayes
import pot
from hurdat import TROPICAL, distance_km, read_hurdat2
from paths import DAILY_CSV, HURDAT2, N_YEARS, OUT, WATER_YEARS, WORK, require

warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_step(name, file_name):
    """Load another step's code by file name, since the names start with a digit."""
    path = Path(__file__).resolve().parent / file_name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sim = load_step("simulation_study", "17_simulation_study.py")
scores = load_step("tail_scores", "18a_tail_scores.py")
trend = load_step("trend_check", "21a_trend_check.py")

COASTAL = ["Houston, TX", "New Orleans, LA", "Miami, FL", "Boston, MA"]
PERCENTILE = 98
SEPARATION = 2
RADIUS_KM = 250
CHECK_RADII_KM = [150, 400]
TAILS = ["bounded", "central", "heavy"]
DAY_JITTER = 7          # simulated event dates are real ones moved by up to a week
YEAR_DAYS = 366         # days in a simulated water year
CHECK_RECORDS = 1000
CHANGE_YEAR = 1998      # Miami's change year from step 21a
VERSIONS = sim.VERSIONS
RETURN_PERIODS = sim.RETURN_PERIODS
COLUMNS = ["candidate", "tail", "record"] + sim.COLUMNS[2:]


# ---------------------------------------------------------------------------------------------
# The labels (HURDAT2 is read by hurdat.py)

def find_events(values, threshold, separation):
    """The events, grouped exactly as pot.decluster groups them, with their first and last day.

    Returns arrays: first day, last day, peak day, peak value and days above the threshold.
    """
    days_above = np.flatnonzero(values > threshold)
    breaks = np.flatnonzero(np.diff(days_above) > separation)
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks + 1, days_above.size]
    peak_day = np.array([days_above[s:e][np.argmax(values[days_above[s:e]])]
                         for s, e in zip(starts, ends)], dtype=int)
    return {"first": days_above[starts], "last": days_above[ends - 1], "peak_day": peak_day,
            "peak": values[peak_day], "days_above": ends - starts}


def water_year_day(dates):
    """Days since October 1 of the water year (October 1 is day 0)."""
    dates = pd.DatetimeIndex(dates)
    start_year = np.where(dates.month >= 10, dates.year, dates.year - 1)
    starts = pd.to_datetime([f"{y}-10-01" for y in start_year])
    return np.asarray((dates - starts).days, dtype=int)


def label_events(cell, values, dates, hourly):
    """One row per event at one cell, labeled tropical or not."""
    threshold = float(np.percentile(values, PERCENTILE))
    ev = find_events(values, threshold, SEPARATION)
    _, decluster_peaks, _ = pot.decluster(values, threshold, SEPARATION)
    assert np.array_equal(ev["peak"], decluster_peaks)
    near = hourly.assign(km=distance_km(cell.latitude, cell.longitude, hourly.lat, hourly.lon))
    near = near[near.km <= max(CHECK_RADII_KM + [RADIUS_KM])]
    rows = []
    for first, last, peak_day, peak, days_above in zip(*ev.values()):
        window = near[(near.time >= dates[first] - pd.Timedelta(days=1))
                      & (near.time < dates[last] + pd.Timedelta(days=2))]
        tropical = window[window.status.isin(TROPICAL)]
        closest = tropical.loc[tropical.km.idxmin()] if len(tropical) else None
        is_tropical = closest is not None and closest.km <= RADIUS_KM
        rows.append({
            "candidate": cell.candidate, "cell_id": cell.cell_id,
            "water_year": int(dates[peak_day].year + (dates[peak_day].month >= 10)),
            "first_date": dates[first].date(), "peak_date": dates[peak_day].date(),
            "last_date": dates[last].date(), "peak_ms": peak, "excess_ms": peak - threshold,
            "days_above": int(days_above), "tropical": bool(is_tropical),
            "storm_id": closest.storm if is_tropical else "",
            "storm_name": closest["name"].title() if is_tropical else "",
            "status": closest.status if is_tropical else "",
            "closest_tropical_km": round(float(closest.km), 1) if closest is not None else np.nan,
            **{f"tropical_within_{r}km": bool(closest is not None and closest.km <= r)
               for r in CHECK_RADII_KM},
            # a HURDAT2 system nearby that was no longer tropical (extratropical, a low or a wave)
            "only_other_track_nearby": bool(not is_tropical and (
                window[~window.status.isin(TROPICAL)].km <= RADIUS_KM).any()),
            "first_day": int(first), "last_day": int(last), "peak_day": int(peak_day),
        })
    return threshold, pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# The two-type model

def gpd_scale_for_shape(excesses, xi):
    """The maximum-likelihood GPD scale for a fixed shape, and the negative log-likelihood."""
    low = np.log(max(1e-3, -xi * excesses.max() * (1 + 1e-9))) if xi < 0 else np.log(1e-3)
    best = minimize_scalar(lambda s: pot.gpd_nll((np.exp(s), xi), excesses),
                           bounds=(low, np.log(100.0)), method="bounded",
                           options={"xatol": 1e-10})
    return float(np.exp(best.x)), float(best.fun)


def hurricane_tails(tropical_excesses):
    """The pooled hurricane shape and the ends of its 95% profile-likelihood interval.

    tropical_excesses holds each cell's hurricane excesses. The shape is shared and each cell
    has its own scale.
    """
    grid = np.round(np.arange(-0.80, 1.0001, 0.01), 2)
    profile = np.array([sum(gpd_scale_for_shape(x, xi)[1] for x in tropical_excesses)
                        for xi in grid])
    inside = grid[2 * (profile - profile.min()) <= chi2.ppf(0.95, 1)]
    assert grid[0] < inside.min() and inside.max() < grid[-1], "interval reaches the grid edge"
    return {"bounded": float(inside.min()), "central": float(grid[profile.argmin()]),
            "heavy": float(inside.max())}


def true_return_levels(model, periods=RETURN_PERIODS):
    """Return levels of the two types together: the level x with expected exceedances
    rate_o * P(other > x) + rate_h * P(hurricane > x) = -ln(1 - 1/N) a year."""
    u = model["threshold"]

    def exceedances(x):
        return (model["rate_o"] * genpareto.sf(x - u, c=model["xi_o"], scale=model["sigma_o"])
                + model["rate_h"] * genpareto.sf(x - u, c=model["xi_h"], scale=model["sigma_h"]))

    levels = {}
    for period in periods:
        target = -np.log(1.0 - 1.0 / period)
        high = u + 10.0
        while exceedances(high) > target:
            high += 10.0
        levels[period] = brentq(lambda x: exceedances(x) - target, u, high, xtol=1e-10)
    return levels


def check_model():
    """Known cases: the event grouping matches pot.decluster, and with no hurricanes the true
    levels are the usual POT return levels."""
    rng = np.random.default_rng(0)
    values = rng.gamma(2.0, 2.0, 5000)
    _, peaks, n_above = pot.decluster(values, 9.0, SEPARATION)
    ev = find_events(values, 9.0, SEPARATION)
    assert np.array_equal(ev["peak"], peaks) and ev["days_above"].sum() == n_above
    model = {"threshold": 9.0, "rate_o": 6.0, "sigma_o": 1.2, "xi_o": 0.05,
             "rate_h": 1e-12, "sigma_h": 4.0, "xi_h": 0.0}
    expected = bayes.pot_return_levels(RETURN_PERIODS, 9.0, [6.0], [1.2], [0.05])[0]
    assert np.allclose(list(true_return_levels(model).values()), expected, atol=1e-6)


def build_models(labels, thresholds):
    """The two-type model at each coastal cell, for each hurricane tail."""
    tropical_excesses = [labels[c].excess_ms[labels[c].tropical].to_numpy() for c in COASTAL]
    shapes = hurricane_tails(tropical_excesses)
    models, rows = {}, []
    for c, candidate in enumerate(COASTAL):
        events, u = labels[candidate], thresholds[candidate]
        other = events[~events.tropical]
        hurricanes = events[events.tropical]
        fit_o = pot.fit_gpd(other.excess_ms.to_numpy())
        single = pot.fit_gpd(events.excess_ms.to_numpy())
        rate_o, rate_h = len(other) / N_YEARS, len(hurricanes) / N_YEARS
        # the lower level with sim.BASE_RATE other storms a year, as in step 17
        sigma_o, xi_o = fit_o["sigma"], fit_o["xi"]
        if abs(xi_o) < bayes.SMALL_XI:
            base_level = u - sigma_o * np.log(sim.BASE_RATE / rate_o)
        else:
            base_level = u + sigma_o / xi_o * ((sim.BASE_RATE / rate_o) ** (-xi_o) - 1.0)
        base_sigma = sigma_o + xi_o * (base_level - u)
        assert base_sigma > 0
        for tail in TAILS:
            sigma_h, _ = gpd_scale_for_shape(hurricanes.excess_ms.to_numpy(), shapes[tail])
            model = {
                "candidate": candidate, "cell": c, "tail": tail, "tail_index": TAILS.index(tail),
                "threshold": u, "rate_o": rate_o, "sigma_o": sigma_o, "xi_o": xi_o,
                "base_level": base_level, "base_sigma": base_sigma,
                "rate_h": rate_h, "sigma_h": sigma_h, "xi_h": shapes[tail],
                "day_o": water_year_day(other.peak_date),
                "day_h": water_year_day(hurricanes.peak_date),
                "length_o": other.days_above.to_numpy(),
                "length_h": hurricanes.days_above.to_numpy(),
            }
            model["truth"] = true_return_levels(model)
            models[(candidate, tail)] = model
            rows.append({
                "candidate": candidate, "hurricane_tail": tail, "threshold_ms": u,
                "n_events": len(events), "n_other": len(other), "n_hurricane": len(hurricanes),
                "single_gpd_xi": single["xi"], "other_rate": rate_o, "other_scale": sigma_o,
                "other_xi": xi_o, "hurricane_rate": rate_h, "hurricane_scale": sigma_h,
                "hurricane_xi": shapes[tail],
                **{f"true_rp{p}_ms": model["truth"][p] for p in RETURN_PERIODS},
            })
    return models, pd.DataFrame(rows), shapes


# ---------------------------------------------------------------------------------------------
# The simulator and its check

def simulate_record(model, rng):
    """One 45-year record from the two-type model.

    Returns the annual maxima, whether each is a hurricane, and the events above the threshold
    after grouping with the 2-day rule (peak, hurricane or not, days above, day of the year).
    """
    u = model["threshold"]
    counts_o = rng.poisson(sim.BASE_RATE, N_YEARS)
    counts_h = rng.poisson(model["rate_h"], N_YEARS)
    assert (counts_o > 0).all()
    peaks = np.r_[
        model["base_level"] + genpareto.rvs(c=model["xi_o"], scale=model["base_sigma"],
                                            size=counts_o.sum(), random_state=rng),
        u + genpareto.rvs(c=model["xi_h"], scale=model["sigma_h"], size=counts_h.sum(),
                          random_state=rng)]
    year = np.r_[np.repeat(np.arange(N_YEARS), counts_o), np.repeat(np.arange(N_YEARS), counts_h)]
    hurricane = np.r_[np.zeros(counts_o.sum(), bool), np.ones(counts_h.sum(), bool)]

    # annual maxima, and which type each one is
    order = np.lexsort((peaks, year))
    last_of_year = np.r_[np.flatnonzero(np.diff(year[order])), order.size - 1]
    annual_max = peaks[order][last_of_year]
    annual_hurricane = hurricane[order][last_of_year]

    # days above the threshold: a real date and length for each event, by type
    above = np.flatnonzero(peaks > u)
    kind = hurricane[above]
    day = np.where(kind, rng.choice(model["day_h"], above.size),
                   rng.choice(model["day_o"], above.size))
    day = (day + rng.integers(-DAY_JITTER, DAY_JITTER + 1, above.size)) % YEAR_DAYS
    length = np.where(kind, rng.choice(model["length_h"], above.size),
                      rng.choice(model["length_o"], above.size))
    peak_day = year[above] * YEAR_DAYS + day
    first = np.maximum(peak_day - rng.integers(0, length), 0)
    offsets = np.arange(length.sum()) - np.repeat(np.cumsum(length) - length, length)
    days = np.repeat(first, length) + offsets
    event_peak = np.repeat(peaks[above], length)
    value = u + rng.uniform(0.05, 0.95, days.size) * (event_peak - u)
    value[days == np.repeat(peak_day, length)] = event_peak[days == np.repeat(peak_day, length)]
    series = np.zeros(N_YEARS * YEAR_DAYS + 30)
    np.maximum.at(series, days, value)
    series_hurricane = np.zeros(series.size, bool)
    top = value == series[days]
    series_hurricane[days[top]] = np.repeat(kind, length)[top]

    ev = find_events(series, u, SEPARATION)
    return {"annual_max": annual_max, "annual_hurricane": annual_hurricane,
            "peak": ev["peak"], "hurricane": series_hurricane[ev["peak_day"]],
            "days_above": ev["days_above"], "day": ev["peak_day"] % YEAR_DAYS}


def record_metrics(record, u):
    """The numbers the simulator check compares, for one record (real or simulated)."""
    excess = record["peak"] - u
    hurricane = record["hurricane"]
    june_to_november = (record["day"] >= 243) | (record["day"] <= 60)    # June 1 is day 243
    top5 = np.argsort(record["annual_max"])[-5:]
    largest = np.sort(record["annual_max"])
    return {
        "events_per_year": excess.size / N_YEARS,
        "hurricane_events_per_year": hurricane.sum() / N_YEARS,
        "median_excess_ms": np.median(excess),
        "p90_excess_ms": np.percentile(excess, 90),
        "largest_peak_ms": record["peak"].max(),
        "annual_max_mean_ms": record["annual_max"].mean(),
        "annual_max_sd_ms": record["annual_max"].std(ddof=1),
        "second_largest_annual_max_ms": largest[-2],
        "days_above_per_event_other": record["days_above"][~hurricane].mean(),
        "days_above_per_event_hurricane": (record["days_above"][hurricane].mean()
                                           if hurricane.any() else np.nan),
        "share_events_june_to_november_pct": 100 * june_to_november.mean(),
        "hurricanes_in_5_largest_years": int(record["annual_hurricane"][top5].sum()),
    }


def real_record(values, dates, events, u):
    """The real record at one cell, in the same form as simulate_record's output."""
    water_year = np.where(dates.month >= 10, dates.year + 1, dates.year)
    annual_max = np.array([values[water_year == y].max() for y in WATER_YEARS])
    hurricane_days = np.zeros(values.size, bool)
    for row in events[events.tropical].itertuples():
        hurricane_days[row.first_day:row.last_day + 1] = True
    annual_hurricane = np.array([hurricane_days[water_year == y][
        np.argmax(values[water_year == y])] for y in WATER_YEARS])
    return {"annual_max": annual_max, "annual_hurricane": annual_hurricane,
            "peak": events.peak_ms.to_numpy(), "hurricane": events.tropical.to_numpy(),
            "days_above": events.days_above.to_numpy(),
            "day": water_year_day(events.peak_date)}


def check_simulator(models, real_metrics, n_records):
    rows = []
    for (candidate, tail), model in models.items():
        simulated = pd.DataFrame([record_metrics(simulate_record(
            model, np.random.default_rng([22, 1, model["cell"], model["tail_index"], r])),
            model["threshold"]) for r in range(n_records)])
        for metric, real in real_metrics[candidate].items():
            low, mid, high = np.nanpercentile(simulated[metric], [2.5, 50, 97.5])
            rows.append({"candidate": candidate, "hurricane_tail": tail,
                         "hurricane_xi": model["xi_h"],
                         "metric": metric, "real": real, "simulated_median": mid,
                         "simulated_low_95": low, "simulated_high_95": high,
                         "inside": bool(low <= real <= high)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# The coverage test

def fit_record(task):
    """Fit the versions to one simulated record. Returns one row per version and period.

    task = (model, record number, "short" or "full" chains, and the versions to fit).
    """
    model, record, chain_length, versions = task
    rng = np.random.default_rng([22, 2, model["cell"], model["tail_index"], record])
    simulated = simulate_record(model, rng)
    annual_max = simulated["annual_max"]
    excesses = simulated["peak"] - model["threshold"]
    gev_start, gpd_start = sim.starting_points(annual_max, excesses)
    rows = []
    for v, (fitted, prior_name) in enumerate(VERSIONS):
        if (fitted, prior_name) not in versions:
            continue
        key = {"candidate": model["candidate"], "tail": model["tail"], "record": record,
               "model": fitted, "prior": prior_name}
        seed = 1_000_000 * (3 * model["cell"] + model["tail_index"]) + 10 * record + v
        if chain_length == "full":
            seed += 5
        try:
            if fitted == "GEV":
                fit = bayes.fit_version("GEV", prior_name, annual_max, gev_start, seed,
                                        RETURN_PERIODS, N_YEARS, **sim.CHAINS[chain_length])
            else:
                fit = bayes.fit_version("POT", prior_name, excesses, gpd_start, seed,
                                        RETURN_PERIODS, N_YEARS, model["threshold"],
                                        **sim.CHAINS[chain_length])
        except Exception as error:    # a failed fit is recorded, not hidden
            rows.extend({**key, "return_period": p, "error": str(error)[:80]}
                        for p in RETURN_PERIODS)
            continue
        checked = ["xi"] + [f"rp{p}" for p in RETURN_PERIODS]
        max_rhat = max(fit["rhat"][q] for q in checked)
        min_ess = min(fit["ess"][q] for q in checked)
        for j, period in enumerate(RETURN_PERIODS):
            low, mid, high = np.percentile(fit["levels"][:, j], [2.5, 50, 97.5])
            truth = model["truth"][period]
            rows.append({**key, "return_period": period,
                         "n_values": annual_max.size if fitted == "GEV" else excesses.size,
                         "true_ms": truth, "estimate_ms": mid, "lower_95_ms": low,
                         "upper_95_ms": high, "covered": bool(low <= truth <= high),
                         "max_rhat": max_rhat, "min_ess": min_ess,
                         "converged": bool(max_rhat < sim.RHAT_LIMIT and min_ess >= sim.ESS_LIMIT),
                         "error": ""})
    return rows


def summarize(fits, models):
    rows = []
    for (candidate, tail, fitted, prior_name, period), group in fits.groupby(
            ["candidate", "tail", "model", "prior", "return_period"], sort=False):
        ok = group[group.converged == True].copy()    # noqa: E712 (the column can hold NaN)
        truth = models[(candidate, tail)]["truth"][period]
        error = ok.estimate_ms - truth
        coverage = ok.covered.mean()
        ok["score"] = scores.interval_score(ok.lower_95_ms, ok.upper_95_ms, truth, 0.05)
        rows.append({
            "candidate": candidate, "hurricane_tail": tail,
            "hurricane_xi": models[(candidate, tail)]["xi_h"], "model": fitted,
            "prior": prior_name, "return_period": period, "true_ms": truth,
            "n_records": group.record.nunique(), "n_converged": len(ok),
            "coverage_pct": 100 * coverage,
            "coverage_se_pct": 100 * np.sqrt(coverage * (1 - coverage) / len(ok)),
            "bias_pct": 100 * error.mean() / truth,
            "median_error_pct": 100 * error.median() / truth,
            "rmse_pct": 100 * np.sqrt((error**2).mean()) / truth,
            "median_width_ms": (ok.upper_95_ms - ok.lower_95_ms).median(),
            "interval_score_ms": ok.score.mean(),
            "share_true_above_interval_pct": 100 * (ok.upper_95_ms < truth).mean(),
            "share_true_below_interval_pct": 100 * (ok.lower_95_ms > truth).mean(),
        })
    return pd.DataFrame(rows)


def run_coverage(models, n_records, processes, fresh):
    progress = WORK / "mixed_storm_fits_in_progress.csv"
    done, rows = set(), []
    if progress.exists() and not fresh:
        earlier = pd.read_csv(progress)
        earlier["error"] = earlier.error.fillna("")
        rows = earlier.to_dict("records")
        done = set(zip(earlier.candidate, earlier["tail"], earlier.record))
        print(f"resuming: {len(done)} records already done")
    elif progress.exists():
        progress.unlink()

    start = time.time()
    tasks = [(model, r, "short", VERSIONS) for model in models.values() for r in range(n_records)
             if (model["candidate"], model["tail"], r) not in done]
    total = len(models) * n_records
    with Pool(processes) as pool:
        for i, result in enumerate(pool.imap_unordered(fit_record, tasks, chunksize=4)):
            rows.extend(result)
            pd.DataFrame(result).reindex(columns=COLUMNS).to_csv(
                progress, mode="a", index=False, header=not progress.exists())
            if (i + 1) % 200 == 0:
                print(f"  {len(done) + i + 1} of {total} records "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)
        fits = pd.DataFrame(rows)
        to_refit = {}
        for row in fits[fits.converged != True].itertuples():    # noqa: E712
            to_refit.setdefault((row.candidate, row.tail, row.record), set()).add(
                (row.model, row.prior))
        refit_tasks = [(models[(c, t)], r, "full", versions)
                       for (c, t, r), versions in to_refit.items()]
        refits = [row for result in pool.imap_unordered(fit_record, refit_tasks)
                  for row in result]
    print(f"refitted {len(refits) // len(RETURN_PERIODS)} fits with full-length chains")
    keys = ["candidate", "tail", "record", "model", "prior", "return_period"]
    if refits:
        refitted = pd.DataFrame(refits)
        replaced = fits.set_index(keys).index.isin(refitted.set_index(keys).index)
        refitted["refit"] = True
        fits = pd.concat([fits[~replaced], refitted], ignore_index=True)
    fits["refit"] = fits.get("refit", False)
    fits["refit"] = fits.refit.fillna(False).astype(bool)
    fits = fits[fits.record < n_records].sort_values(keys)
    fits.to_csv(WORK / "mixed_storm_fits.csv", index=False)
    progress.unlink()
    n_unconverged = int((fits.converged != True).sum()) // len(RETURN_PERIODS)    # noqa: E712
    print(f"{len(fits) // len(RETURN_PERIODS)} fits; {n_unconverged} did not pass the "
          f"convergence checks after refitting  ({(time.time() - start) / 60:.1f} minutes)")
    return fits


# ---------------------------------------------------------------------------------------------
# Hurricanes before and after 1998

def storm_type_trend(cell, values, dates, events, hourly):
    """Do hurricanes explain a change in the annual maxima? One row for one cell."""
    water_year = np.where(dates.month >= 10, dates.year + 1, dates.year)
    years = np.array(WATER_YEARS)
    before = years < CHANGE_YEAR
    n_before, n_after = before.sum(), (~before).sum()

    near = hourly[hourly.status.isin(TROPICAL) & (distance_km(
        cell.latitude, cell.longitude, hourly.lat, hourly.lon) <= RADIUS_KM)]
    near_year = np.where(near.time.dt.month >= 10, near.time.dt.year + 1, near.time.dt.year)
    systems = near.assign(water_year=near_year).drop_duplicates("storm")
    systems = systems[systems.water_year.between(years[0], years[-1])]

    # the annual maxima with the hurricane events taken out (and a day either side)
    no_hurricane = values.copy()
    for row in events[events.tropical].itertuples():
        no_hurricane[max(row.first_day - 1, 0):row.last_day + 2] = -np.inf
    all_max = np.array([values[water_year == y].max() for y in years])
    other_max = np.array([no_hurricane[water_year == y].max() for y in years])
    hurricane_years = events[events.tropical].water_year.to_numpy()

    row = {"candidate": cell.candidate, "split_year": CHANGE_YEAR,
           "tropical_systems_per_year_before": (systems.water_year < CHANGE_YEAR).sum() / n_before,
           "tropical_systems_per_year_after": (systems.water_year >= CHANGE_YEAR).sum() / n_after,
           "hurricane_events_per_year_before": (hurricane_years < CHANGE_YEAR).sum() / n_before,
           "hurricane_events_per_year_after": (hurricane_years >= CHANGE_YEAR).sum() / n_after}
    for label, maxima in [("annual_max", all_max), ("annual_max_without_hurricanes", other_max)]:
        tau, p = kendalltau(years, maxima)
        index, change_p = trend.pettitt(maxima)
        row.update({f"{label}_before_ms": maxima[before].mean(),
                    f"{label}_after_ms": maxima[~before].mean(),
                    f"{label}_tau": tau, f"{label}_p": p,
                    f"{label}_change_year": int(years[index]), f"{label}_change_p": change_p})
    return row


# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--fresh", action="store_true", help="ignore an unfinished earlier run")
    parser.add_argument("--check-only", action="store_true",
                        help="stop after the simulator check")
    args = parser.parse_args()
    require(DAILY_CSV, HURDAT2, OUT / "pilot_locations_selected.csv")
    trend.check_pettitt()
    check_model()
    start = time.time()

    cells = pd.read_csv(OUT / "pilot_locations_selected.csv").set_index("candidate", drop=False)
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    dates = pd.DatetimeIndex(daily.date)
    hourly = read_hurdat2(HURDAT2, dates[0] - pd.Timedelta(days=2),
                          dates[-1] + pd.Timedelta(days=2))
    print(f"HURDAT2: {hourly.storm.nunique()} systems between {dates[0].date()} and "
          f"{dates[-1].date()}")

    # 1. labels
    labels, thresholds, series = {}, {}, {}
    for candidate in COASTAL:
        cell = cells.loc[candidate]
        values = daily[str(cell.cell_id)].to_numpy(np.float32).astype(np.float64)
        thresholds[candidate], labels[candidate] = label_events(cell, values, dates, hourly)
        series[candidate] = values
    all_labels = pd.concat(labels.values(), ignore_index=True)
    all_labels.drop(columns=["first_day", "last_day", "peak_day"]).to_csv(
        OUT / "storm_labels.csv", index=False)
    print(f"\nEvents labeled tropical (a tropical system within {RADIUS_KM} km; "
          f"{CHECK_RADII_KM[0]} and {CHECK_RADII_KM[1]} km for comparison):")
    for candidate, events in labels.items():
        largest = events.nlargest(5, "peak_ms")
        names = ", ".join(f"{r.storm_name or 'other'} {r.peak_ms:.1f}"
                          for r in largest.itertuples())
        print(f"  {candidate:<16} {events.tropical.sum():>3} of {len(events)} "
              f"({events[f'tropical_within_{CHECK_RADII_KM[0]}km'].sum()} / "
              f"{events[f'tropical_within_{CHECK_RADII_KM[1]}km'].sum()}); "
              f"5 largest: {names}")

    # 2. the two-type model
    models, model_table, shapes = build_models(labels, thresholds)
    model_table.round(4).to_csv(OUT / "mixed_storm_model.csv", index=False)
    print(f"\nHurricane shape, pooled over the four cells: {shapes['central']:+.2f} "
          f"(95% profile interval {shapes['bounded']:+.2f} to {shapes['heavy']:+.2f})")
    print(model_table[model_table.hurricane_tail == "central"][
        ["candidate", "n_other", "n_hurricane", "single_gpd_xi", "other_xi", "other_scale",
         "hurricane_scale"]].round(3).to_string(index=False))
    print("\nTrue 100-year level by hurricane tail (m/s):")
    print(model_table.pivot(index="candidate", columns="hurricane_tail", values="true_rp100_ms")
          .loc[COASTAL, TAILS].round(2).to_string())

    # hurricanes before and after 1998
    trend_table = pd.DataFrame([storm_type_trend(cells.loc[c], series[c], dates, labels[c], hourly)
                                for c in COASTAL])
    trend_table.round(4).to_csv(OUT / "storm_type_trend.csv", index=False)
    print(f"\nBefore and after {CHANGE_YEAR} (per year, and mean annual maxima in m/s):")
    print(trend_table[["candidate", "tropical_systems_per_year_before",
                       "tropical_systems_per_year_after", "annual_max_before_ms",
                       "annual_max_after_ms", "annual_max_without_hurricanes_before_ms",
                       "annual_max_without_hurricanes_after_ms",
                       "annual_max_without_hurricanes_change_p"]].round(3).to_string(index=False))

    # 3 and 4. the simulator check
    real_metrics = {c: record_metrics(real_record(series[c], dates, labels[c], thresholds[c]),
                                      thresholds[c]) for c in COASTAL}
    check = check_simulator(models, real_metrics, CHECK_RECORDS)
    check.round(4).to_csv(OUT / "mixed_storm_check.csv", index=False)
    print(f"\nSimulator check ({CHECK_RECORDS} records each): real values outside the middle 95% "
          f"of the simulated ones")
    for (candidate, tail), group in check.groupby(["candidate", "hurricane_tail"], sort=False):
        outside = group[~group.inside]
        print(f"  {candidate:<16} {tail:<8} {len(outside)} of {len(group)}"
              + (": " + ", ".join(outside.metric) if len(outside) else ""))
    print(f"({(time.time() - start) / 60:.1f} minutes so far)")
    if args.check_only:
        print(f"\nwrote {OUT / 'storm_labels.csv'}, {OUT / 'mixed_storm_model.csv'}, "
              f"{OUT / 'storm_type_trend.csv'}, {OUT / 'mixed_storm_check.csv'}")
        raise SystemExit

    # 5. the coverage test
    fits = run_coverage(models, args.records, args.processes, args.fresh)
    summary = summarize(fits, models).round(3)
    summary.to_csv(OUT / "mixed_storm_simulation.csv", index=False)
    show = summary[summary.return_period == 100]
    print("\nCoverage of the true 100-year level (%), by hurricane tail:")
    print(show.pivot_table(index=["candidate", "model", "prior"], columns="hurricane_tail",
                           values="coverage_pct", sort=False)[TAILS].to_string())
    print("\nBias at the 100-year level (%), by hurricane tail:")
    print(show.pivot_table(index=["candidate", "model", "prior"], columns="hurricane_tail",
                           values="bias_pct", sort=False)[TAILS].to_string())
    print(f"\nwrote {OUT / 'storm_labels.csv'}, {OUT / 'mixed_storm_model.csv'}, "
          f"{OUT / 'storm_type_trend.csv'}, {OUT / 'mixed_storm_check.csv'}, "
          f"{OUT / 'mixed_storm_simulation.csv'}, {WORK / 'mixed_storm_fits.csv'}  "
          f"({(time.time() - start) / 60:.1f} minutes)")
