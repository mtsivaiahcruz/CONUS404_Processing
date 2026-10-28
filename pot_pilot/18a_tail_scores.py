"""Tail scores: two more ways to judge the six versions, with the weight on the extremes.

This step refits nothing. It rescores the intervals that steps 17 and 18 already saved.

Simulation study (step 17), the 95% interval for each return level:
    interval score   (Gneiting & Raftery 2007) the interval's width, plus a penalty when the
                     true value falls outside it: 2 / 0.05 = 40 times the distance it misses
                     by. Lower is better. It joins the two things step 17 reports separately,
                     width and coverage, into one number, so an interval cannot score well by
                     being narrow and missing, or by being wide and always covering. The true
                     return level is known here, so this is a direct test of the 100-year
                     level, which the held-out years cannot reach.

Held-out years (step 18), the predictive distribution of a held-out annual maximum:
    quantile score   the pinball loss at the 95th percentile of the predictive distribution.
                     It judges the upper tail of the prediction against every held-out year,
                     so it is less noisy than a yes-or-no score at one high threshold.
    interval score   the same score as above, for the central 90% predictive interval (the
                     5th and 95th percentiles), so the penalty is 2 / 0.10 = 20 times the miss.
Score differences are against the flat GEV, with a 95% range from the same bootstrap as step
18: 2,000 resamples of the 45 held-out years, the same years at every cell.

A Brier score at one high threshold is left out on purpose. The CRPS in step 18 is the Brier
score averaged over every threshold, and the threshold-weighted CRPS is the average over the
high thresholds only, so it already covers this. A single threshold would rest on the few
held-out years above it (about 58 of 585 above each cell's 90th percentile annual maximum).

Reads:
    output/work/simulation_fits.csv (step 17), output/work/holdout_predictions.csv (step 18)
Writes:
    output/simulation_interval_scores.csv   one row per true model, version and return period
    output/holdout_tail_scores.csv          one row per cell and version, plus the pooled rows
"""
import numpy as np
import pandas as pd

from paths import WORK, OUT, require

VERSIONS = [("GEV", "flat"), ("GEV", "main"), ("GEV", "wider"),
            ("POT", "flat"), ("POT", "main"), ("POT", "wider")]
N_BOOTSTRAP = 2000


def interval_score(lower, upper, observed, alpha):
    """Width plus 2 / alpha times the miss, for a central (1 - alpha) interval."""
    lower, upper, observed = (np.asarray(a, float) for a in (lower, upper, observed))
    below = np.clip(lower - observed, 0, None)
    above = np.clip(observed - upper, 0, None)
    return (upper - lower) + 2.0 / alpha * (below + above)


def quantile_score(predicted, observed, level):
    """Pinball loss of a predicted quantile at the given level."""
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    return (observed - predicted) * (level - (observed < predicted))


def check_scores():
    """Known cases worked out by hand."""
    # inside: just the width; below by 1 with alpha 0.05: width + 40
    assert np.allclose(interval_score([10, 10], [14, 14], [12, 9], 0.05), [4, 44])
    # above by 2 with alpha 0.10: width + 20 x 2
    assert np.isclose(interval_score(10, 14, 16, 0.10), 44)
    # pinball loss at 0.95: an observation above the quantile costs 0.95 per m/s, below 0.05
    assert np.allclose(quantile_score([10, 10], [12, 8], 0.95), [1.9, 0.1])
    # the 90% interval score is 20 times the sum of the 5% and 95% quantile scores
    lower, upper, observed = np.array([9.0, 9.0, 9.0]), np.array([15.0] * 3), np.array([8, 12, 17])
    total = quantile_score(lower, observed, 0.05) + quantile_score(upper, observed, 0.95)
    assert np.allclose(interval_score(lower, upper, observed, 0.10), 20 * total)


def simulation_scores():
    fits = pd.read_csv(WORK / "simulation_fits.csv")
    fits = fits[fits.converged.astype(bool)].copy()
    fits["score_ms"] = interval_score(fits.lower_95_ms, fits.upper_95_ms, fits.true_ms, 0.05)
    fits["width_ms"] = fits.upper_95_ms - fits.lower_95_ms
    fits["score_pct"] = 100 * fits.score_ms / fits.true_ms
    rows = []
    for (truth, period), group in fits.groupby(["true_model", "return_period"], sort=False):
        flat_gev = group[(group.model == "GEV") & (group.prior == "flat")].score_ms.mean()
        for model, prior_name in VERSIONS:
            mine = group[(group.model == model) & (group.prior == prior_name)]
            rows.append({
                "true_model": truth, "return_period": period, "model": model,
                "prior": prior_name, "n_records": len(mine), "true_ms": mine.true_ms.iloc[0],
                "interval_score_ms": mine.score_ms.mean(),
                "interval_score_pct_of_true": mine.score_pct.mean(),
                "mean_width_ms": mine.width_ms.mean(),
                "miss_penalty_ms": (mine.score_ms - mine.width_ms).mean(),
                "vs_gev_flat_pct": 100 * (mine.score_ms.mean() / flat_gev - 1),
            })
    return pd.DataFrame(rows)


def bootstrap_range(differences, rng, n=N_BOOTSTRAP):
    """95% range of the mean of differences (years x cells), resampling whole years."""
    differences = np.asarray(differences, float)
    picks = rng.integers(0, differences.shape[0], size=(n, differences.shape[0]))
    return np.percentile(differences[picks].mean(axis=(1, 2)), [2.5, 97.5])


def holdout_scores():
    predictions = pd.read_csv(WORK / "holdout_predictions.csv")
    predictions["quantile_95"] = quantile_score(predictions.upper_90_ms,
                                                predictions.observed_ms, 0.95)
    predictions["interval_90"] = interval_score(predictions.lower_90_ms,
                                                predictions.upper_90_ms,
                                                predictions.observed_ms, 0.10)
    scores = ["quantile_95", "interval_90"]
    table = predictions.pivot_table(index=["water_year", "cell_id"],
                                    columns=["model", "prior"], values=scores)
    cells = predictions[["cell_id", "candidate"]].drop_duplicates()
    groups = [(row.cell_id, row.candidate) for row in cells.itertuples()]
    groups.append((None, "All 13 cells"))
    rng = np.random.default_rng(18)
    rows = []
    for cell_id, label in groups:
        part = table if cell_id is None else table.xs(cell_id, level="cell_id",
                                                     drop_level=False)
        for model, prior_name in VERSIONS:
            row = {"cell_id": cell_id, "location": label, "model": model, "prior": prior_name,
                   "n_years": len(part)}
            for score in scores:
                mine = part[(score, model, prior_name)].unstack("cell_id")
                row[f"mean_{score}"] = float(mine.to_numpy().mean())
                if (model, prior_name) == ("GEV", "flat"):
                    continue
                ref = part[(score, "GEV", "flat")].unstack("cell_id")
                difference = (mine - ref).to_numpy()
                low, high = bootstrap_range(difference, rng)
                ref_mean = float(ref.to_numpy().mean())
                row[f"{score}_diff_vs_gev_flat_pct"] = 100 * float(difference.mean()) / ref_mean
                row[f"{score}_diff_vs_gev_flat_low_pct"] = 100 * low / ref_mean
                row[f"{score}_diff_vs_gev_flat_high_pct"] = 100 * high / ref_mean
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    check_scores()
    require(WORK / "simulation_fits.csv", WORK / "holdout_predictions.csv")

    simulation = simulation_scores()
    numbers = simulation.select_dtypes("number").columns.difference(["return_period",
                                                                     "n_records"])
    simulation[numbers] = simulation[numbers].round(3)
    simulation.to_csv(OUT / "simulation_interval_scores.csv", index=False)
    print("Interval score of the 95% interval for the 100-year level (lower is better):")
    view = simulation[simulation.return_period == 100].pivot_table(
        index=["model", "prior"], columns="true_model", values="interval_score_ms", sort=False)
    print(view[["light", "zero", "heavy"]].round(2).to_string())

    holdout = holdout_scores()
    numbers = holdout.select_dtypes("number").columns.difference(["cell_id", "n_years"])
    holdout[numbers] = holdout[numbers].round(4)
    holdout.to_csv(OUT / "holdout_tail_scores.csv", index=False)
    pooled = holdout[holdout.location == "All 13 cells"]
    print("\nHeld-out years, all 13 cells (585 years):")
    print(pooled.drop(columns=["cell_id", "location", "n_years"]).to_string(index=False))
    print(f"\nwrote {OUT / 'simulation_interval_scores.csv'}, {OUT / 'holdout_tail_scores.csv'}")
