"""Simulation study for the priors centered at 0: does moving the center from -0.10 to 0 matter?

This reruns the POT half of step 17 with two more priors on the shape:
    zero         normal, mean 0, sd 0.12      the main prior's width
    zero_wider   normal, mean 0, sd 0.20      the wider prior's width
A center of 0 is the tail length NIST's 2015 non-hurricane wind maps use (Pintar et al. 2015,
NIST SP 500-301). Only POT is refitted, because POT is the recommended model.

The simulated records are exactly the ones in step 17: the same three true models, the same
500 records each, built from the same random seeds. So the results here line up record by
record with POT and the main and wider priors in step 17. The chains, the convergence checks
and the refit with full-length chains for any fit that fails them are also the same as step 17.

Usage:
    python 17a_zero_center_simulation.py [--records 500] [--processes N] [--fresh]
A run that stops part-way resumes where it left off when started again, unless --fresh.
Reads:
    output/pilot_pot_results.csv (step 08), output/simulation_coverage.csv and
    output/work/simulation_fits.csv (step 17), for the comparison
Writes:
    output/simulation_zero_center.csv          one row per true model, prior and return period:
                                               coverage, bias, error, width and interval score,
                                               next to POT with the main and wider priors
    output/work/simulation_zero_center_fits.csv   every fit
"""
import argparse
import importlib.util
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from paths import OUT, WORK, require

# Step 17's code, loaded by file name because the name starts with a digit. The new priors go
# after step 17's six versions, so every fit here gets a seed step 17 never used.
_spec = importlib.util.spec_from_file_location(
    "simulation_study", Path(__file__).resolve().parent / "17_simulation_study.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)
NEW_VERSIONS = [("POT", "zero"), ("POT", "zero_wider")]
sim.VERSIONS = sim.VERSIONS + NEW_VERSIONS
COMPARE = [("POT", "main"), ("POT", "wider")]


def fit_new(task):
    """Fit only the new versions to one simulated record."""
    name, record, chain_length, versions = task
    return sim.fit_record((name, record, sim.MODELS[name], chain_length, versions))


def interval_score(lower, upper, observed, alpha=0.05):
    """Width plus 2 / alpha times the miss (Gneiting & Raftery 2007). Lower is better."""
    below = np.clip(lower - observed, 0, None)
    above = np.clip(observed - upper, 0, None)
    return (upper - lower) + 2.0 / alpha * (below + above)


def with_interval_score(summary, fits):
    fits = fits[fits.converged == True].copy()    # noqa: E712
    fits["score"] = interval_score(fits.lower_95_ms, fits.upper_95_ms, fits.true_ms)
    scores = (fits.groupby(["true_model", "model", "prior", "return_period"])
              .score.mean().rename("interval_score_ms").reset_index())
    return summary.merge(scores, on=["true_model", "model", "prior", "return_period"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--fresh", action="store_true", help="ignore an unfinished earlier run")
    args = parser.parse_args()
    require(OUT / "simulation_coverage.csv", WORK / "simulation_fits.csv")

    progress = WORK / "simulation_zero_center_in_progress.csv"
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
    tasks = [(name, r, "short", NEW_VERSIONS) for name in sim.MODELS for r in range(args.records)
             if (name, r) not in done]
    with Pool(args.processes) as pool:
        for i, result in enumerate(pool.imap_unordered(fit_new, tasks, chunksize=4)):
            rows.extend(result)
            pd.DataFrame(result).reindex(columns=sim.COLUMNS).to_csv(
                progress, mode="a", index=False, header=not progress.exists())
            if (i + 1) % 100 == 0:
                print(f"  {len(done) + i + 1} of {len(sim.MODELS) * args.records} records "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)
        fits_so_far = pd.DataFrame(rows)
        failed = fits_so_far[fits_so_far.converged != True]    # noqa: E712
        to_refit = {}
        for row in failed.itertuples():
            to_refit.setdefault((row.true_model, row.record), set()).add((row.model, row.prior))
        refit_tasks = [(name, r, "full", versions) for (name, r), versions in to_refit.items()]
        refits = [row for result in pool.imap_unordered(fit_new, refit_tasks) for row in result]
    print(f"refitted {len(refits) // len(sim.RETURN_PERIODS)} fits with full-length chains")
    keys = ["true_model", "record", "model", "prior", "return_period"]
    fits = pd.DataFrame(rows)
    if refits:
        refitted = pd.DataFrame(refits)
        replaced = fits.set_index(keys).index.isin(refitted.set_index(keys).index)
        refitted["refit"] = True
        fits = pd.concat([fits[~replaced], refitted], ignore_index=True)
    fits["refit"] = fits.get("refit", False)
    fits["refit"] = fits.refit.fillna(False).astype(bool)
    fits = fits[fits.record < args.records].sort_values(keys)
    fits.to_csv(WORK / "simulation_zero_center_fits.csv", index=False)
    progress.unlink()
    n_unconverged = int((fits.converged != True).sum()) // len(sim.RETURN_PERIODS)    # noqa: E712
    print(f"{len(fits) // len(sim.RETURN_PERIODS)} fits; {n_unconverged} did not pass the "
          f"convergence checks after refitting")

    new = with_interval_score(sim.summarize(fits), fits)
    step17_fits = pd.read_csv(WORK / "simulation_fits.csv")
    step17_fits = step17_fits[[(m, p) in COMPARE for m, p in zip(step17_fits.model,
                                                                  step17_fits.prior)]]
    step17_fits = step17_fits[step17_fits.record < args.records]
    old = with_interval_score(sim.summarize(step17_fits), step17_fits)
    table = pd.concat([old, new], ignore_index=True)
    order = {"main": 0, "wider": 1, "zero": 2, "zero_wider": 3}
    table = table.sort_values(["true_model", "return_period", "prior"],
                              key=lambda c: c.map(order) if c.name == "prior" else c)
    table = table.round(3)
    table.to_csv(OUT / "simulation_zero_center.csv", index=False)
    show = table[table.return_period == 100][
        ["true_model", "prior", "coverage_pct", "bias_pct", "rmse_pct", "median_width_ms",
         "interval_score_ms"]]
    print("\nPOT, 100-year level:")
    print(show.to_string(index=False))
    print(f"\nwrote {OUT / 'simulation_zero_center.csv'}, "
          f"{WORK / 'simulation_zero_center_fits.csv'}  ({(time.time() - start) / 60:.1f} minutes)")
