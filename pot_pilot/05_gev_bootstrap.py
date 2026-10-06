"""Step 3b. Check the GEV delta-method intervals with a parametric bootstrap.

For each pilot cell:
  1. simulate N_BOOT records of 45 annual maxima from the fitted GEV
  2. refit each simulated record by maximum likelihood
  3. take the 2.5th and 97.5th percentiles of the refitted return levels as the 95% interval

The cells are done one after another on one core, with one random number generator, so the
random numbers and the results come out the same on every run. About 5 minutes for 1000.

Usage:
    python 05_gev_bootstrap.py [N_BOOT]      (default 1000)
Reads:
    output/work/gev_baseline_params.csv, output/work/gev_baseline_long.csv (step 04)
Writes:
    output/work/gev_baseline_bootstrap.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genextreme

from paths import GUST_FACTOR, WORK, require

warnings.filterwarnings("ignore")
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
RETURN_PERIODS = [10, 20, 25, 50, 100, 200, 500]
SEED = 20260927
require(WORK / "gev_baseline_params.csv", WORK / "gev_baseline_long.csv")

fitted = pd.read_csv(WORK / "gev_baseline_params.csv")
baseline = pd.read_csv(WORK / "gev_baseline_long.csv")
rng = np.random.default_rng(SEED)

rows = []
for cell in fitted.itertuples():
    shape, loc, scale, n_years = cell.shape_scipy_c, cell.location, cell.scale, cell.n_years

    # one row per simulated record, one column per return period
    levels = np.empty((N_BOOT, len(RETURN_PERIODS)))
    for b in range(N_BOOT):
        sample = genextreme.rvs(shape, loc, scale, size=n_years, random_state=rng)
        shape_b, loc_b, scale_b = genextreme.fit(sample)
        levels[b] = genextreme.isf(1 / np.array(RETURN_PERIODS), shape_b, loc_b, scale_b)

    for k, R in enumerate(RETURN_PERIODS):
        lower, upper = np.percentile(levels[:, k], [2.5, 97.5])
        rows.append({
            "cell_id": cell.cell_id, "candidate": cell.candidate, "return_period": R,
            "boot_lower_3s_ms": lower * GUST_FACTOR, "boot_upper_3s_ms": upper * GUST_FACTOR,
            "boot_sd_native_ms": levels[:, k].std(ddof=1),
        })
    print(f"  {cell.candidate}: done", flush=True)

boot = pd.DataFrame(rows)
merged = baseline.merge(boot, on=["cell_id", "candidate", "return_period"])
# close to 1 means the delta method and the bootstrap agree on the size of the uncertainty
merged["sd_ratio_delta_to_boot"] = merged.delta_se_native_ms / merged.boot_sd_native_ms
merged.to_csv(WORK / "gev_baseline_bootstrap.csv", index=False)
print(f"B = {N_BOOT}. Delta-method standard error divided by bootstrap spread, by return period:")
summary = merged.groupby("return_period").sd_ratio_delta_to_boot.agg(["min", "median", "max"])
print(summary.round(2).to_string())
