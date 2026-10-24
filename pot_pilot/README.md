# Pilot study: GEV baseline against peaks-over-threshold (POT)

This folder holds a pilot comparison of two ways to estimate extreme wind speeds from CONUS404
`SPDUV10MAX` (daily maximum 10 m wind speed, m/s), at 13 locations across the lower 48 states:

- **GEV** (generalized extreme value distribution): fit to one maximum per water year. This is
  what the CADENCE pipeline does now.
- **POT**: fit to every event above a high threshold.

The scripts extract the data, choose the 13 locations, reproduce the GEV baseline, compare it
with CADENCE's current files, run the POT tests, compare GEV with POT, make the results table
and the plots, set the priors on the tail shape, fit the GEV and POT with and without a
constrained tail, test how much the constrained fits depend on the prior, check every
version against simulated records where the true return levels are known, score each
version on years it was not fitted to, check how much one year can move each estimate, check the POT threshold, and check the storm-separation rule.

## How to run it

From this folder:

```
python run_all.py
```

This runs steps 01 to 21 in order, then compares the new outputs with the saved results and
prints a verdict for each file. It takes under an hour on my laptop.

- `python run_all.py 8` starts at step 08. The outputs of the earlier steps must already exist.
- Each script can be run on its own, for example `python 04_gev_baseline.py`.
- Steps 01 to 21 write only to `output/` and change nothing in `data/`. To write somewhere
  else, set the `ANALYSIS_OUT` environment variable to a folder.

## What each script does

| Script | What it does | Main output (in `output/`) |
|---|---|---|
| `01_snap_and_screen.py` | Matches each of the 13 cities to its nearest 4 km grid cell and checks the cell: shared cells, daily data quality, land fraction, and whether its winds look like land or water. | `pilot_cells_snapped.csv` |
| `02_gev_convergence_check.py` | Runs the pipeline's GEV fit (`30_fit_gev_parameters.py`) at each cell, checks that it converges, and restarts it from other starting points to see whether any does better. | `pilot_gev_check.csv` |
| `03_select_pilot_locations.py` | Scores every grid cell on complete years and daily completeness, and writes the pilot table with latitude and longitude. | `pilot_locations_selected.csv` |
| `04_gev_baseline.py` | The documented GEV fit (maximum likelihood), return levels for 10 to 500 years, and two kinds of 95% interval: the delta method and the pipeline's own. | `work/gev_baseline_long.csv` |
| `05_gev_bootstrap.py` | Checks the GEV intervals with a bootstrap (1,000 simulated records per cell). | `work/gev_baseline_bootstrap.csv` |
| `06_assemble_baseline.py` | Builds the two GEV baseline tables. | `pilot_gev_baseline.csv`, `pilot_gev_parameters.csv` |
| `07_compare_with_cadence.py` | Reproduces CADENCE's current L-moments fit, compares it with the baseline, and checks CADENCE's intervals and the Gumbel sign error across the whole grid. | `pilot_cadence_vs_baseline.csv`, `cadence_gumbel_branch_cells.csv` |
| `08_pot_tests.py` | POT fits at each cell: 4 thresholds x 3 separation rules. | `pilot_pot_results.csv`, `pilot_pot_events.csv` |
| `09_pot_bootstrap.py` | Bootstrap intervals for the 156 POT fits, and a resampled-years check for both models. | `work/boot_param.npz`, `work/boot_years.npz` |
| `10_gev_vs_pot.py` | GEV against POT at 20, 50 and 100 years: estimates, interval endpoints, widths, and percent change in width. | `pilot_gev_vs_pot.csv` |
| `11_summary_numbers.py` | Prints the headline numbers that the notes quote. | `summary_numbers.txt` |
| `12_results_table.py` | Combines data quality and model results into one table, with one row per location and method setting. | `pilot_results_table.csv` |
| `13_make_plots.py` | Makes the four plots: return-level curves, POT sensitivity, change in interval width, and the map of locations. `--gust3s` also writes 3-second gust versions of the first two. | `plots/fig1_...png` to `plots/fig4_...png` |
| `14_tail_priors.py` | Sets the priors on the tail shape for the constrained-tail fits: a main prior, a wider and a narrower one for sensitivity tests, and a flat one for the unconstrained fits. Checks SciPy's sign conventions and compares the priors with the pilot fits. | `tail_priors.csv`, `plots/fig5_tail_priors.png` |
| `15_bayes_fits.py` | Fits four versions at each cell: the GEV and POT (98th percentile, 2-day separation), each with the flat and the main prior on the shape. All four use the same Bayesian fit (`bayes.py`). Gives the 20-, 50- and 100-year estimates with 95% credible intervals, and checks convergence (split R-hat and effective sample size). | `bayes_fits.csv`, `bayes_parameters.csv`, `work/bayes_draws.npz` |
| `16_prior_sensitivity.py` | Refits the two constrained versions with the wider and the narrower prior, and reports how far the 20-, 50- and 100-year estimates and intervals move from the main prior. Also counts, for each prior, whether the comparisons from step 15 still hold. | `prior_sensitivity.csv`, `prior_sensitivity_summary.csv` |
| `17_simulation_study.py` | Simulates 500 45-year records from each of three known models (light-tailed, near zero and heavy-tailed, from the POT fits at Atlanta, Chicago and New Orleans), with the annual maxima and the POT events taken from the same record. Fits the GEV and POT with the flat, main and wider priors to every record, and reports how often each 95% interval contains the true 20-, 50- and 100-year level, and the bias and error of the estimates. Uses every processor core; `--records` and `--processes` change the defaults. | `simulation_coverage.csv`, `work/simulation_fits.csv`, `work/simulation_chain_check.csv` |
| `18_holdout_validation.py` | Splits the 45 years into nine 5-year blocks, fits each of the six versions to the other 40 years, and scores its prediction of the 5 held-out annual maxima: the threshold-weighted CRPS (above each cell's median annual maximum, and again above its 90th percentile), the plain CRPS, and 90% predictive interval coverage. Score differences come with 95% ranges from a bootstrap over the held-out years. Uses every processor core. | `holdout_scores.csv`, `work/holdout_predictions.csv` |
| `19_leave_one_year_out.py` | Refits each of the six versions at each cell 45 times, each time without one water year, and reports the median and largest change in the 100-year estimate and the years behind the largest changes. Uses every processor core and resumes if it stops part-way. | `leave_one_year_out.csv`, `stability_summary.csv` |
| `20_threshold_check.py` | Refits POT with the wider prior at the 95th to 99th percentile thresholds (2-day separation) and checks whether the 98th holds up: its 100-year estimate against every other threshold's interval, the stability of the shape and modified scale, and a Kolmogorov-Smirnov test of the GPD fit. Also compares the spread across thresholds with the fits without a prior from step 08. | `threshold_check.csv`, `threshold_check_summary.csv` |
| `21_separation_check.py` | Refits the three POT versions with 1- to 7-day separation rules, and POT with the wider prior on the full grid of thresholds and rules. Checks whether the events are independent with the extremal index (Ferro & Segers intervals estimator, with and without the season removed) and the correlation between consecutive peaks. For the GEV, checks for consecutive annual maxima from the same storm, their correlation, and a trend. | `separation_check.csv`, `independence_check.csv`, `gev_boundary_check.csv` |
| `check_against_delivered.py` | Compares `output/` with the saved results in `data/`. | |

Shared code:

- `pot.py` is the POT method itself. Start here to see how a POT fit works. It works in the
  textbook shape xi (xi > 0 is a heavy tail), like the priors in step 14. SciPy's `genpareto`
  uses c = xi, but `genextreme` uses c = -xi, so GEV code has to flip the sign.
- `bayes.py` is the Bayesian fit used from step 15 on: the GEV and GPD likelihoods, the shape
  priors, the sampler, the convergence checks, and `fit_version`, which fits one version at one
  cell. It needs nothing beyond NumPy and SciPy.
- `paths.py` holds the file locations.
- `h5io.py` opens `.nc` and `.h5` files.

The two extraction scripts made the files in `data/`. They download from the public CONUS404
store, so `run_all.py` does not run them.

| Script | What it does | Writes (in `data/`) |
|---|---|---|
| `conus404_spduv10max_extract.py` | Annual maxima and daily quality counts for every grid cell, by water year or calendar year. | `conus404_grid.nc`, `conus404_grid_cells.csv.gz`, `spduv10max_water_1980-2024.nc`, `spduv10max_calendar_1980-2023.nc`, `manifest.json` |
| `conus404_pilot_daily_extract.py` | Daily values at the 13 pilot cells. | `pilot_daily_spduv10max.csv`, `pilot_daily_spduv10max.json` |

Three more scripts check the code itself:

- `python test_pot.py` checks `pot.py` on cases where the right answer is known (about 1 minute).
- `python test_bayes.py` checks `bayes.py` the same way: the likelihoods against SciPy, the
  sampler on a known distribution, the convergence checks, and recovery of known parameters
  (under a minute).
- `python tests/test_extraction_scripts.py` checks the two extraction scripts on small made-up
  stores.

All three are plain scripts, not pytest tests. `conftest.py` tells pytest to skip them.

## What it needs

Python 3.10 or newer, and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Input files:

```
CADENCE/
  gev_parameters.h5              CADENCE's current GEV parameters (from Andrew)
  gev_return_periods.nc          CADENCE's current return periods (from Andrew)
  CONUS404_Processing/           this repository
    year_raw_data/               the pipeline's 45 annual-maximum files
    pot_pilot/                   this folder
      data/
        conus404_grid.nc                    from conus404_spduv10max_extract.py
        spduv10max_calendar_1980-2023.nc
        spduv10max_water_1980-2024.nc
        manifest.json
        pilot_daily_spduv10max.csv          from conus404_pilot_daily_extract.py
        pilot_*.csv, cadence_gumbel_branch_cells.csv, plots/     the saved results
```

`CADENCE` is the folder that holds the repository. If the two GEV files are somewhere else, set
the `CADENCE_DIR` environment variable to that folder.

Git does not track `data/` or `output/`. Two of the files in `data/` are over 230 MB each, which
is more than GitHub accepts, so anyone else who checks out this branch needs their own copy of
`data/` to run the scripts.

`natural_earth/` holds four Natural Earth files, 19 MB in all: land and lakes at 10m, used by
step 01 to work out how much of each cell is land, and state outlines at 50m, used by step 13
for the map. Natural Earth data is in the public domain (naturalearthdata.com).

## The saved results, and checking against them

`data/` holds the result files that the project notes quote (the scripts call them the
"delivered" files). `check_against_delivered.py` compares a new run with them and gives each
file one of three verdicts:

- `IDENTICAL`: every value is the same.
- `MATCH`: the fitted values agree within 0.001 (as a share of the value, for values above 1)
  and bootstrap columns within 10%. Bootstrap columns get more room because a different
  version of SciPy or NumPy can change the random draws.
- `DIFFERENT`: anything else.
- `MISSING`: one of the two copies is not there. This is what you get without `data/`.

The saved results were produced on Linux with Python 3.11.15, NumPy 2.4.4, pandas 3.0.2 and
SciPy 1.17.1. Re-running everything on Windows on 2026-10-04 (Python 3.14, NumPy 2.5.2, pandas
3.0.6, SciPy 1.18.1, pyfive 1.2.1, lmoments3 1.0.8) gave 8 of the 11 files identical and 3 that
matched, with a largest difference of 1.7e-05.

Every bootstrap uses fixed random seeds, so the results do not depend on how many processes
run.

## Script 40 has been fixed since

The pilot found two problems in `40_calculate_return_periods.py`: a sign error in the return
level when the shape is exactly zero, and intervals that were not a delta method. Both are
fixed on this branch.

Steps 02, 04 and 07 keep their own copies of the old formulas (`pipeline_return_level` and
`pipeline_interval`), so they still reproduce `gev_return_periods.nc` as it was when the pilot
ran. Keep that file in the `CADENCE` folder. If it is replaced by one made with the fixed
script, step 07 will no longer match it.

## One thing to know about `pilot_gev_parameters.csv`

A copy written before 2026-10-04 had the right numbers under the wrong column headings. The copy
in `data/` is the corrected one.
