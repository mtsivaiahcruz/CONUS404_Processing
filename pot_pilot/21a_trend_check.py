"""Trend check: where in the distribution each cell's winds change over 1980-2024, and when.

Step 21 found that Miami's annual maxima rise strongly over the 45 water years. This step asks
two questions at every pilot cell that help tell a real change from a change in the model:
    - Is the rise only in the extremes, or in every day? A change in the model's inputs would
      tend to shift the whole distribution. A change in storms would show up mostly at the top.
      So each water year's median, 90th and 99th percentile daily values and its maximum are
      tested for a trend (Kendall's tau).
    - Is it gradual or a step? The Pettitt test (Pettitt 1979) finds the year where the annual
      maxima are most different before and after, and how unlikely that split is by chance.

The daily values are the CONUS404 daily maximum 10 m wind (SPDUV10MAX) from the daily extremes
store, the same values steps 15 to 21 use.

Reads:
    data/pilot_daily_spduv10max.csv, output/pilot_locations_selected.csv (step 03)
Writes:
    output/trend_check.csv    one row per cell
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import bayes
from paths import DAILY_CSV, OUT, WATER_YEARS, require

require(DAILY_CSV, OUT / "pilot_locations_selected.csv")
YEARS = np.array(WATER_YEARS)


def pettitt(x):
    """Pettitt change-point test: the first year after the change, and the approximate p-value."""
    x = np.asarray(x, float)
    n = x.size
    signs = np.sign(x[None, :] - x[:, None])    # signs[i, j] = sign(x[j] - x[i])
    u = np.array([signs[:t + 1, t + 1:].sum() for t in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    statistic = abs(u[k])
    p = min(1.0, 2.0 * np.exp(-6.0 * statistic**2 / (n**3 + n**2)))
    return k + 1, p


def check_pettitt():
    """A clear step is found where it is, and a flat series is not called a change."""
    rng = np.random.default_rng(0)
    step = np.r_[rng.normal(0, 1, 20), rng.normal(3, 1, 25)]
    index, p = pettitt(step)
    assert index == 20 and p < 1e-4, (index, p)
    assert pettitt(rng.normal(0, 1, 45))[1] > 0.05


if __name__ == "__main__":
    check_pettitt()
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
    values, annual_max = bayes.pilot_series(cells)
    dates = pd.read_csv(DAILY_CSV, usecols=["date"], parse_dates=["date"]).date
    water_year = np.where(dates.dt.month >= 10, dates.dt.year + 1, dates.dt.year)
    rows = []
    for k, cell in enumerate(cells.itertuples()):
        series = values[:, k]
        yearly = {q: np.array([np.percentile(series[water_year == y], q) for y in YEARS])
                  for q in (50, 90, 99)}
        yearly["max"] = annual_max[:, k]
        row = {"cell_id": cell.cell_id, "location": cell.candidate, "longitude": cell.longitude}
        for name, label in [(50, "median"), (90, "p90"), (99, "p99"), ("max", "annual_max")]:
            tau, p = kendalltau(YEARS, yearly[name])
            row[f"{label}_tau"], row[f"{label}_p"] = tau, p
        index, p = pettitt(yearly["max"])
        row["change_year"] = int(YEARS[index])
        row["change_p"] = p
        row["annual_max_before_ms"] = yearly["max"][:index].mean()
        row["annual_max_after_ms"] = yearly["max"][index:].mean()
        row["median_before_ms"] = yearly[50][:index].mean()
        row["median_after_ms"] = yearly[50][index:].mean()
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("longitude").drop(columns="longitude")
    numbers = table.select_dtypes("number").columns.difference(["cell_id", "change_year"])
    table[numbers] = table[numbers].round(4)
    table.to_csv(OUT / "trend_check.csv", index=False)
    print(table[["location", "median_tau", "p90_tau", "p99_tau", "annual_max_tau", "change_year",
                 "change_p", "annual_max_before_ms", "annual_max_after_ms", "median_before_ms",
                 "median_after_ms"]].round(3).to_string(index=False))
    print(f"\nwrote {OUT / 'trend_check.csv'}")
