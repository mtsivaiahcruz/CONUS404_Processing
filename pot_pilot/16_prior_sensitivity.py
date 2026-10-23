"""How much the constrained-tail fits depend on the prior: refit them with the wider and the
narrower prior from step 14.

The GEV and POT with a constrained tail are refitted at every pilot cell exactly as in step 15,
but with the wider prior (sd 0.20) and the narrower prior (sd 0.06) on the shape in place of
the main prior (sd 0.12). All three share the same center, -0.10.

Two things are reported:
    - per cell, how far the 20-, 50- and 100-year estimates and intervals move from the main
      prior's
    - whether the comparisons from step 15 still hold with each prior: is the constrained
      version narrower than the unconstrained one, is POT narrower than the GEV, and does the
      constrained tail lower the estimate

Reads:
    data/pilot_daily_spduv10max.csv
    output/pilot_locations_selected.csv (step 03), output/pilot_gev_parameters.csv (step 06),
    output/pilot_pot_results.csv (step 08), output/bayes_fits.csv (step 15)
Writes:
    output/prior_sensitivity.csv           one row per cell, model, prior and return period,
                                           with the change from the main prior
    output/prior_sensitivity_summary.csv   the step 15 comparisons, counted for each prior
"""
import time

import numpy as np
import pandas as pd

import bayes
import pot
from paths import DAILY_CSV, N_YEARS, OUT, require

PERCENTILE = 98    # the same POT setting as step 15
SEPARATION = 2
RETURN_PERIODS = [20, 50, 100]
NEW_PRIORS = ["wider", "narrower"]
RHAT_LIMIT = 1.01
ESS_LIMIT = 400

require(DAILY_CSV, OUT / "pilot_locations_selected.csv", OUT / "pilot_gev_parameters.csv",
        OUT / "pilot_pot_results.csv", OUT / "bayes_fits.csv")

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
values, annual_max = bayes.pilot_series(cells)
mle_gev = pd.read_csv(OUT / "pilot_gev_parameters.csv").set_index("cell_id")
mle_pot = pd.read_csv(OUT / "pilot_pot_results.csv")
mle_pot = mle_pot[(mle_pot.threshold_pct == PERCENTILE)
                  & (mle_pot.separation_days == SEPARATION)].set_index("cell_id")
step15 = pd.read_csv(OUT / "bayes_fits.csv")

rows = []
start_all = time.time()
for k, cell in enumerate(cells.itertuples()):
    series = values[:, k]
    threshold = float(np.percentile(series, PERCENTILE))
    _, peaks, _ = pot.decluster(series, threshold, SEPARATION)
    excesses = peaks - threshold
    for model in ("GEV", "POT"):
        for p, prior_name in enumerate(NEW_PRIORS):
            # Seeds differ from step 15's, so these are fresh draws.
            seed = 1000 * k + 10 + 2 * p + (model == "POT")
            if model == "GEV":
                mle = mle_gev.loc[cell.cell_id]
                start = [mle.location, np.log(mle.scale), mle.xi]
                fit = bayes.fit_version(model, prior_name, annual_max[:, k], start, seed,
                                        RETURN_PERIODS, N_YEARS)
            else:
                mle = mle_pot.loc[cell.cell_id]
                start = [np.log(mle.gpd_scale), mle.gpd_shape_xi]
                fit = bayes.fit_version(model, prior_name, excesses, start, seed,
                                        RETURN_PERIODS, N_YEARS, threshold)
            converged = (max(fit["rhat"].values()) < RHAT_LIMIT
                         and min(fit["ess"].values()) >= ESS_LIMIT)
            xi_median = float(np.median(fit["flat"]["xi"]))
            for j, period in enumerate(RETURN_PERIODS):
                low, mid, high = np.percentile(fit["levels"][:, j], [2.5, 50, 97.5])
                rows.append({
                    "cell_id": cell.cell_id, "candidate": cell.candidate,
                    "longitude": cell.longitude, "model": model, "prior": prior_name,
                    "return_period": period, "estimate_ms": mid, "lower_95_ms": low,
                    "upper_95_ms": high, "width_ms": high - low, "xi_median": xi_median,
                    "max_rhat": max(fit["rhat"].values()), "min_ess": min(fit["ess"].values()),
                    "converged": converged,
                })
            print(f"{cell.candidate:<18} {model}, {prior_name:<9} xi {xi_median:+.3f}  "
                  f"100-yr {rows[-1]['estimate_ms']:5.2f} ({rows[-1]['lower_95_ms']:5.2f} to "
                  f"{rows[-1]['upper_95_ms']:5.2f})  R-hat {rows[-1]['max_rhat']:.4f}  "
                  f"ESS {rows[-1]['min_ess']:6.0f}", flush=True)

new = pd.DataFrame(rows)
assert new.converged.all(), "some fits did not converge"

# Add the main-prior fits from step 15, then compare every prior with the main one.
main = step15[step15.prior == "main"].copy()
main = main.merge(cells[["cell_id", "longitude"]], on="cell_id")
main["converged"] = True
main = main.rename(columns={"rhat": "max_rhat", "ess": "min_ess"})
table = pd.concat([main[new.columns.drop("xi_median")], new], ignore_index=True)
keys = ["cell_id", "model", "return_period"]
reference = table[table.prior == "main"][keys + ["estimate_ms", "width_ms"]].rename(
    columns={"estimate_ms": "main_estimate_ms", "width_ms": "main_width_ms"})
table = table.merge(reference, on=keys)
table["estimate_change_ms"] = table.estimate_ms - table.main_estimate_ms
table["estimate_change_pct"] = 100 * table.estimate_change_ms / table.main_estimate_ms
table["width_change_pct"] = 100 * (table.width_ms / table.main_width_ms - 1)
table = table.drop(columns=["main_estimate_ms", "main_width_ms"])
table["prior_order"] = table.prior.map({"main": 0, "wider": 1, "narrower": 2})
table = table.sort_values(["longitude", "model", "prior_order", "return_period"]).drop(
    columns="prior_order")
numbers = table.select_dtypes("number").columns.difference(
    ["cell_id", "return_period", "longitude"])
table[numbers] = table[numbers].round(4)
table.drop(columns="longitude").to_csv(OUT / "prior_sensitivity.csv", index=False)

# The step 15 comparisons at 100 years, counted with each prior.
flat = step15[(step15.prior == "flat") & (step15.return_period == 100)].set_index(
    ["cell_id", "model"])
summary = []
for prior_name in ["main"] + NEW_PRIORS:
    constrained = table[(table.prior == prior_name) & (table.return_period == 100)].set_index(
        ["cell_id", "model"])
    gev, pot_fit = constrained.xs("GEV", level="model"), constrained.xs("POT", level="model")
    flat_gev, flat_pot = flat.xs("GEV", level="model"), flat.xs("POT", level="model")
    summary.append({
        "prior": prior_name,
        "gev_narrower_than_flat_gev": int((gev.width_ms < flat_gev.width_ms).sum()),
        "pot_narrower_than_flat_pot": int((pot_fit.width_ms < flat_pot.width_ms).sum()),
        "pot_narrower_than_gev": int((pot_fit.width_ms < gev.width_ms).sum()),
        "gev_estimate_below_flat_gev": int((gev.estimate_ms < flat_gev.estimate_ms).sum()),
        "pot_estimate_below_flat_pot": int((pot_fit.estimate_ms < flat_pot.estimate_ms).sum()),
        "median_gev_width_change_vs_flat_gev_pct":
            float(np.median(100 * (gev.width_ms / flat_gev.width_ms - 1))),
        "median_pot_width_change_vs_flat_gev_pct":
            float(np.median(100 * (pot_fit.width_ms / flat_gev.width_ms - 1))),
        "median_gev_estimate_change_vs_flat_gev_pct":
            float(np.median(100 * (gev.estimate_ms / flat_gev.estimate_ms - 1))),
        "median_pot_estimate_change_vs_flat_gev_pct":
            float(np.median(100 * (pot_fit.estimate_ms / flat_gev.estimate_ms - 1))),
        "largest_gev_minus_pot_ms": float((gev.estimate_ms - pot_fit.estimate_ms).abs().max()),
        "n_cells": len(gev),
    })
summary = pd.DataFrame(summary).round(2)
summary.to_csv(OUT / "prior_sensitivity_summary.csv", index=False)

at100 = table[(table.return_period == 100) & (table.prior != "main")]
print("\nChange from the main prior at 100 years (median and largest across the 13 cells):")
for (model, prior_name), group in at100.groupby(["model", "prior"]):
    print(f"  {model}, {prior_name:<9} estimate {group.estimate_change_pct.median():+5.1f}% "
          f"(largest {group.estimate_change_pct.abs().max():4.1f}%), width "
          f"{group.width_change_pct.median():+5.1f}% (largest "
          f"{group.width_change_pct.abs().max():4.1f}%)")
print("\nThe step 15 comparisons at 100 years, with each prior (number of cells out of 13):")
print(summary.to_string(index=False))
print(f"\nwrote {OUT / 'prior_sensitivity.csv'}, {OUT / 'prior_sensitivity_summary.csv'}  "
      f"({(time.time() - start_all) / 60:.1f} minutes)")
