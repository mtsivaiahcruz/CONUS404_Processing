"""Fit four versions at every pilot cell: the GEV and POT, each with and without a constrained
tail.

    GEV                 annual maxima, flat prior on the shape
    GEV, constrained    annual maxima, main prior on the shape
    POT                 event peaks above the 98th percentile, 2-day separation, flat prior
    POT, constrained    the same events, main prior on the shape

All four are fitted the same way (bayes.py), so the only difference between a version with a
constrained tail and one without is the prior on the shape. The estimate is the posterior
median and the interval is the central 95% of the posterior (a credible interval).

Checks:
    - With the flat prior, the highest point of the posterior must equal the maximum-likelihood
      fits from steps 04 and 08.
    - The POT events must be the same ones step 08 found.
    - Convergence: split R-hat below 1.01 and at least 400 effective draws, for every
      parameter and every return level.

Reads:
    data/pilot_daily_spduv10max.csv
    output/pilot_locations_selected.csv (step 03), output/pilot_gev_parameters.csv (step 06),
    output/pilot_pot_results.csv (step 08), output/tail_priors.csv (step 14)
Writes:
    output/bayes_fits.csv          one row per cell, version and return period
    output/bayes_parameters.csv    one row per cell and version: the parameters and the
                                   convergence checks
    output/work/bayes_draws.npz    4,000 posterior draws per cell and version, for later steps
"""
import time

import numpy as np
import pandas as pd

import bayes
import pot
from paths import DAILY_CSV, FIRST_WY, LAST_WY, N_YEARS, OUT, WORK, require

PERCENTILE = 98    # the provisional POT setting from step 08's tests
SEPARATION = 2
RETURN_PERIODS = [20, 50, 100]
VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("POT", "flat"), ("POT", "main")]
SAVED_DRAWS = 4000
RHAT_LIMIT = 1.01
ESS_LIMIT = 400

require(DAILY_CSV, OUT / "pilot_locations_selected.csv", OUT / "pilot_gev_parameters.csv",
        OUT / "pilot_pot_results.csv", OUT / "tail_priors.csv")

# The shape priors must be the ones step 14 wrote.
step14 = pd.read_csv(OUT / "tail_priors.csv").set_index("prior")
for name in ("flat", "main"):
    prior = bayes.SHAPE_PRIORS[name]
    if prior["kind"] == "normal":
        mean, sd = prior["mean"], prior["sd"]
    else:
        mean, sd = (prior["low"] + prior["high"]) / 2, (prior["high"] - prior["low"]) / 12**0.5
    assert abs(step14.loc[name, "mean_xi"] - mean) < 1e-3, f"{name} prior mean differs"
    assert abs(step14.loc[name, "sd_xi"] - sd) < 1e-3, f"{name} prior sd differs"

cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
values, annual_max = bayes.pilot_series(cells)
assert annual_max.shape == (N_YEARS, len(cells))

mle_gev = pd.read_csv(OUT / "pilot_gev_parameters.csv").set_index("cell_id")
mle_pot = pd.read_csv(OUT / "pilot_pot_results.csv")
mle_pot = mle_pot[(mle_pot.threshold_pct == PERCENTILE)
                  & (mle_pot.separation_days == SEPARATION)].set_index("cell_id")

LABELS = {("GEV", "flat"): "GEV", ("GEV", "main"): "GEV, constrained",
          ("POT", "flat"): "POT", ("POT", "main"): "POT, constrained"}
fit_rows, parameter_rows, saved = [], [], {}
start_all = time.time()
for k, cell in enumerate(cells.itertuples()):
    maxima = annual_max[:, k]
    series = values[:, k]
    threshold = float(np.percentile(series, PERCENTILE))
    _, peaks, _ = pot.decluster(series, threshold, SEPARATION)
    excesses = peaks - threshold
    assert excesses.size == mle_pot.loc[cell.cell_id, "n_events"], "POT events differ from step 08"

    for version_number, (model, prior_name) in enumerate(VERSIONS):
        seed = 1000 * k + version_number    # fixed seeds, so every run gives the same draws
        if model == "GEV":
            mle = mle_gev.loc[cell.cell_id]
            start = [mle.location, np.log(mle.scale), mle.xi]
            fit = bayes.fit_version(model, prior_name, maxima, start, seed, RETURN_PERIODS,
                                    N_YEARS)
        else:
            mle = mle_pot.loc[cell.cell_id]
            start = [np.log(mle.gpd_scale), mle.gpd_shape_xi]
            fit = bayes.fit_version(model, prior_name, excesses, start, seed, RETURN_PERIODS,
                                    N_YEARS, threshold)
        if prior_name == "flat":
            # With a flat prior the highest point is the maximum-likelihood fit.
            assert np.allclose(fit["mode"], start, atol=2e-3), (
                f"{cell.candidate} {model}: flat-prior mode {fit['mode']} differs from {start}")
        mode, quantities, levels = fit["mode"], fit["quantities"], fit["levels"]
        flat_draws = fit["flat"]
        rhat, ess = fit["rhat"], fit["ess"]
        converged = max(rhat.values()) < RHAT_LIMIT and min(ess.values()) >= ESS_LIMIT

        row = {"cell_id": cell.cell_id, "candidate": cell.candidate, "model": model,
               "prior": prior_name, "version": LABELS[(model, prior_name)],
               "n_values": maxima.size if model == "GEV" else excesses.size}
        if model == "POT":
            row["threshold_ms"] = threshold
        for name in [n for n in quantities if not n.startswith("rp")]:
            low, mid, high = np.percentile(flat_draws[name], [2.5, 50, 97.5])
            row.update({f"{name}_median": mid, f"{name}_lower": low, f"{name}_upper": high})
        row.update({"mode_xi": mode[-1], "max_rhat": max(rhat.values()),
                    "min_ess": min(ess.values()), "acceptance": fit["acceptance"],
                    "converged": converged})
        parameter_rows.append(row)

        for j, period in enumerate(RETURN_PERIODS):
            low, mid, high = np.percentile(levels[:, j], [2.5, 50, 97.5])
            fit_rows.append({
                "cell_id": cell.cell_id, "candidate": cell.candidate,
                "longitude": cell.longitude, "model": model, "prior": prior_name,
                "version": LABELS[(model, prior_name)], "return_period": period,
                "estimate_ms": mid, "lower_95_ms": low, "upper_95_ms": high,
                "width_ms": high - low,
                "rhat": rhat[f"rp{period}"], "ess": ess[f"rp{period}"],
            })

        keep = np.linspace(0, levels.shape[0] - 1, SAVED_DRAWS).astype(int)
        key = f"{cell.cell_id}_{model}_{prior_name}"
        saved[key + "_xi"] = flat_draws["xi"][keep]
        saved[key + "_scale"] = flat_draws["scale"][keep]
        saved[key + ("_location" if model == "GEV" else "_rate")] = (
            flat_draws["location" if model == "GEV" else "rate"][keep])
        print(f"{cell.candidate:<18} {LABELS[(model, prior_name)]:<17} "
              f"xi {row['xi_median']:+.3f}  100-yr {fit_rows[-1]['estimate_ms']:5.2f} "
              f"({fit_rows[-1]['lower_95_ms']:5.2f} to {fit_rows[-1]['upper_95_ms']:5.2f})  "
              f"R-hat {row['max_rhat']:.4f}  ESS {row['min_ess']:6.0f}", flush=True)

fits = pd.DataFrame(fit_rows).sort_values(["longitude", "model", "prior", "return_period"])
parameters = pd.DataFrame(parameter_rows)
fits.insert(2, "first_water_year", FIRST_WY)
fits.insert(3, "last_water_year", LAST_WY)
not_rounded = ["cell_id", "return_period", "first_water_year", "last_water_year", "longitude"]
to_round = fits.select_dtypes("number").columns.difference(not_rounded)
fits[to_round] = fits[to_round].round(4)
fits.drop(columns="longitude").to_csv(OUT / "bayes_fits.csv", index=False)
numbers = parameters.select_dtypes("number").columns.difference(["cell_id", "n_values"])
parameters[numbers] = parameters[numbers].round(4)
parameters.to_csv(OUT / "bayes_parameters.csv", index=False)
np.savez_compressed(WORK / "bayes_draws.npz", **saved)

assert len(fits) == len(cells) * len(VERSIONS) * len(RETURN_PERIODS)
n_converged = int(parameters.converged.sum())
print(f"\n{n_converged} of {len(parameters)} fits passed the convergence checks "
      f"(R-hat < {RHAT_LIMIT}, ESS >= {ESS_LIMIT}); worst R-hat "
      f"{parameters.max_rhat.max():.4f}, lowest ESS {parameters.min_ess.min():.0f}")
print(f"wrote {OUT / 'bayes_fits.csv'}, {OUT / 'bayes_parameters.csv'}, "
      f"{WORK / 'bayes_draws.npz'}  ({(time.time() - start_all) / 60:.1f} minutes)")
assert n_converged == len(parameters), "some fits did not converge - see bayes_parameters.csv"
