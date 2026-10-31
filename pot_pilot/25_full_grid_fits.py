"""Full-grid run (step 12): the step 11 recommendation at every CONUS404 grid cell, on DCC.

At each of the 1,387,505 grid cells:
    1. Hurricane flag. The cell is a hurricane cell if at least one of its 5 largest water-year
       maxima (1980-2024) falls within a day of a tropical or subtropical system in HURDAT2
       passing within 250 km of the cell.
    2. Fit. Hurricane cells get the GEV with a flat prior on the shape (uniform from -1 to 1),
       fitted to the 45 water-year maxima. Every other cell gets POT at the 98th percentile of
       its daily values, with events separated by 2 days and the wider prior on the GPD shape
       (normal, mean -0.10, sd 0.20). Both are fitted with bayes.fit_version and the chains of
       step 15. A fit that fails the convergence checks (R-hat below 1.01 and an effective
       sample size of at least 400) is refitted once with longer chains.
    3. Return levels. Posterior median and 95% credible interval at 10, 25, 50, 100, 200 and
       500 years, worked out from the draws in the same pass (the draws for every cell would
       be about 67 GB).
The seed of each fit is the cell's number (row x 1,367 + column), the same number the pilot
uses as cell_id, so every cell's result can be reproduced on its own.

Stages, each a separate command:
    download   SPDUV10MAX for water years 1980-2024 from HyTEST's public copy of the CONUS404
               daily extremes store, the store the pilot's daily values came from (the same
               daily maxima as the wrfxtrm files on GDEX). The grid is saved as 12 blocks of
               350 x 350 cells, matching the store's chunks, each about 8 GB, with each cell's
               days stored together. A block is checked against the pilot's daily values for
               any pilot cell inside it.
    fit        One task of a SLURM array. The cells are split into equal runs, and task K fits
               run K, saving its results every 250 cells, so a task that stops resumes.
    combine    Puts the tasks together and writes the return levels in the layout of
               40_calculate_return_periods.py (same file names, variables and return periods,
               so CADENCE reads them unchanged), plus each cell's method, hurricane flag and
               convergence checks. Compares the 100-year level with CADENCE's current file, and
               with the pilot results at the 13 pilot cells.
    check      Runs the same per-cell code on the 13 pilot cells from the pilot's daily values,
               and compares it with step 23's results table. Run it on the laptop first.
    selftest   Builds a small store from the pilot cells, runs download, fit and combine on it,
               and checks that the pipeline gives exactly what the per-cell code gives.

Usage (25_full_grid_fits.sh submits the DCC stages):
    python 25_full_grid_fits.py check
    python 25_full_grid_fits.py selftest
    python 25_full_grid_fits.py download --root ROOT [--block B]
    python 25_full_grid_fits.py fit --root ROOT --task K --tasks N [--processes P]
    python 25_full_grid_fits.py combine --root ROOT --tasks N
ROOT holds everything the run writes (about 100 GB): grid.npz, daily/, fits/, and the outputs.
Reads:
    data/hurdat2-1851-2024.txt; check also reads data/pilot_daily_spduv10max.csv,
    output/pilot_locations_selected.csv and output/checkpoint2_results_table.csv (step 23)
Writes (in ROOT):
    gev_return_periods.nc, gev_return_periods.csv   the return levels, as script 40 writes them
    full_grid_summary.csv                           counts, convergence and the comparisons
    full_grid_pilot_cells.csv                       the 13 pilot cells against step 23
check writes output/full_grid_pilot_check.csv.
"""
import argparse
import os
import shutil
import tempfile
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme

import bayes
import pot
from hurdat import TROPICAL, distance_km, read_hurdat2
from paths import (CADENCE_NC, DAILY_CSV, FIRST_WY, GUST_FACTOR, HURDAT2, LAST_WY, N_YEARS,
                   OUT, WATER_YEARS, require)

warnings.filterwarnings("ignore", category=RuntimeWarning)

STORE = "s3://hytest/conus404/conus404_xtrm_daily.zarr"
OSN_ENDPOINT = "https://usgs.osn.mghpcc.org/"
VARIABLE = "SPDUV10MAX"
FIRST_DAY = f"{FIRST_WY - 1}-10-01"
LAST_DAY = f"{LAST_WY}-09-30"
N_DAYS = 16437
BLOCK = 350
READ_DAYS = 240    # days read at a time, a multiple of the store's 24-day chunks
PERCENTILE = 98
SEPARATION = 2
POT_PRIOR = "wider"
GEV_PRIOR = "flat"
RADIUS_KM = 250
TOP_YEARS = 5
RETURN_PERIODS = [10, 25, 50, 100, 200, 500]
LONGER_CHAINS = dict(n_warmup=8000, n_keep=12000)
RHAT_LIMIT = 1.01
ESS_LIMIT = 400
SAVE_EVERY = 250
TOLERANCE_PCT = 0.01   # check: largest difference from step 23 with the pilot's seeds
METHODS = {0: "POT, wider prior", 1: "GEV, flat prior"}
COASTAL = {"Houston, TX", "New Orleans, LA", "Miami, FL", "Boston, MA"}
FIELDS = (["method", "hurricane_cell", "threshold_ms", "n_events", "rate_median",
           "scale_median", "xi_median", "location_median", "max_rhat", "min_ess", "converged",
           "longer_chains", "failed"]
          + [f"rp{p}_{q}" for p in RETURN_PERIODS for q in ("estimate", "lower", "upper")])


# ---------------------------------------------------------------------------------------------
# One cell

def water_year_starts(day_numbers):
    """The index of the first day of each water year, for days given as days since 1970."""
    dates = pd.to_datetime(np.asarray(day_numbers), unit="D")
    water_year = np.where(dates.month >= 10, dates.year + 1, dates.year)
    assert list(np.unique(water_year)) == WATER_YEARS
    return np.r_[0, np.flatnonzero(np.diff(water_year)) + 1]


def tracks_by_day(first_day, last_day):
    """The tropical track points of each day, as {days since 1970: (latitudes, longitudes)}."""
    hourly = read_hurdat2(HURDAT2, pd.Timestamp(first_day) - pd.Timedelta(days=2),
                          pd.Timestamp(last_day) + pd.Timedelta(days=2))
    hourly = hourly[hourly.status.isin(TROPICAL)]
    day = (hourly.time.dt.floor("D") - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    return {int(d): (hourly.lat.to_numpy()[day == d], hourly.lon.to_numpy()[day == d])
            for d in np.unique(day)}


def hurricane_cell(annual_max, annual_max_day, lat, lon, tracks):
    """The step 11 rule: a tropical system within 250 km, within a day of one of the 5 largest
    water-year maxima."""
    for d in annual_max_day[np.argsort(annual_max)[::-1][:TOP_YEARS]]:
        for near_day in (d - 1, d, d + 1):
            if near_day in tracks:
                track_lat, track_lon = tracks[near_day]
                if (distance_km(lat, lon, track_lat, track_lon) <= RADIUS_KM).any():
                    return True
    return False


def gev_start(maxima):
    """Maximum-likelihood GEV fit as the starting point, as steps 04 and 15 use."""
    c, loc, scale = genextreme.fit(maxima)
    start = [loc, np.log(scale), -c]    # SciPy's c is minus the textbook shape
    if np.isfinite(bayes.gev_log_posterior(start, maxima, bayes.SHAPE_PRIORS[GEV_PRIOR])):
        return start
    scale = np.sqrt(6.0) * maxima.std() / np.pi    # Gumbel moments, if the fit is outside
    return [maxima.mean() - 0.5772 * scale, np.log(scale), 0.0]


def fit_cell(values, starts, days, lat, lon, tracks, seed):
    """The whole method at one cell. Returns a dictionary with the FIELDS."""
    out = dict.fromkeys(FIELDS, np.nan)
    values = np.asarray(values, np.float64)
    annual_max = np.maximum.reduceat(values, starts)
    ends = np.r_[starts[1:], values.size]
    annual_max_day = np.array([days[s + np.argmax(values[s:e])] for s, e in zip(starts, ends)])
    flag = hurricane_cell(annual_max, annual_max_day, lat, lon, tracks)
    out["hurricane_cell"] = flag
    out["method"] = 1 if flag else 0
    names = ["xi"] + [f"rp{p}" for p in RETURN_PERIODS]
    try:
        if flag:
            data, start = annual_max, gev_start(annual_max)
            args = ("GEV", GEV_PRIOR, data, start, seed, RETURN_PERIODS, N_YEARS)
        else:
            threshold = float(np.percentile(values, PERCENTILE))
            _, peaks, _ = pot.decluster(values, threshold, SEPARATION)
            data = peaks - threshold
            fit_ml = pot.fit_gpd(data)
            start = [np.log(fit_ml["sigma"]), fit_ml["xi"]]
            args = ("POT", POT_PRIOR, data, start, seed, RETURN_PERIODS, N_YEARS, threshold)
            out["threshold_ms"], out["n_events"] = threshold, data.size
        fit = bayes.fit_version(*args)
        rhat, ess = max(fit["rhat"][q] for q in names), min(fit["ess"][q] for q in names)
        out["longer_chains"] = False
        if not (rhat < RHAT_LIMIT and ess >= ESS_LIMIT):
            fit = bayes.fit_version(*args[:4], seed + 7, *args[5:], **LONGER_CHAINS)
            rhat, ess = max(fit["rhat"][q] for q in names), min(fit["ess"][q] for q in names)
            out["longer_chains"] = True
    except Exception:    # a failed fit is recorded, not hidden
        out["failed"] = True
        return out
    out["failed"] = False
    out["max_rhat"], out["min_ess"] = rhat, ess
    out["converged"] = bool(rhat < RHAT_LIMIT and ess >= ESS_LIMIT)
    out["xi_median"] = float(np.median(fit["flat"]["xi"]))
    out["scale_median"] = float(np.median(fit["flat"]["scale"]))
    if flag:
        out["location_median"] = float(np.median(fit["flat"]["location"]))
    else:
        out["rate_median"] = float(np.median(fit["flat"]["rate"]))
    low, mid, high = np.percentile(fit["levels"], [2.5, 50, 97.5], axis=0)
    for j, period in enumerate(RETURN_PERIODS):
        out[f"rp{period}_estimate"] = mid[j]
        out[f"rp{period}_lower"] = low[j]
        out[f"rp{period}_upper"] = high[j]
    return out


# ---------------------------------------------------------------------------------------------
# The grid, in blocks

def blocks(ny, nx, size=BLOCK):
    """The blocks as (first row, last row + 1, first column, last column + 1)."""
    return [(r, min(r + size, ny), c, min(c + size, nx))
            for r in range(0, ny, size) for c in range(0, nx, size)]


def block_path(root, b):
    return Path(root) / "daily" / f"block_{b:02d}.npy"


def cell_order(ny, nx, size):
    """Every cell as (block, row, column), block by block, row by row within a block."""
    order = []
    for b, (r0, r1, c0, c1) in enumerate(blocks(ny, nx, size)):
        rows, cols = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        order.append(np.c_[np.full(rows.size, b), rows.ravel(), cols.ravel()])
    return np.concatenate(order)


def task_cells(ny, nx, size, task, tasks):
    order = cell_order(ny, nx, size)
    edges = np.linspace(0, len(order), tasks + 1).astype(int)
    return order[edges[task]:edges[task + 1]]


class ZarrSource:
    """The CONUS404 daily extremes store, read with xarray (as the extraction scripts do)."""

    def __init__(self, url):
        import xarray as xr
        kwargs = {"engine": "zarr", "chunks": {}}
        if url.startswith("s3://"):
            kwargs["storage_options"] = {"anon": True, "requester_pays": False,
                                         "client_kwargs": {"endpoint_url": OSN_ENDPOINT}}
        try:
            ds = xr.open_dataset(url, consolidated=True, **kwargs)
        except (KeyError, FileNotFoundError, ValueError):
            ds = xr.open_dataset(url, consolidated=False, **kwargs)
        da = ds[VARIABLE]
        self.y_dim, self.x_dim = [d for d in da.dims if d != "time"]
        self.time_dim = "time"
        da = da.transpose("time", self.y_dim, self.x_dim).sel(time=slice(FIRST_DAY, LAST_DAY))
        assert da.shape[0] == N_DAYS, f"{da.shape[0]} days, expected {N_DAYS}"
        for lat, lon in (("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")):
            if lat in ds.variables and set(ds[lat].dims) == {self.y_dim, self.x_dim}:
                space = (self.y_dim, self.x_dim)
                self.lat = np.asarray(ds[lat].transpose(*space).values, np.float64)
                self.lon = np.asarray(ds[lon].transpose(*space).values, np.float64)
                break
        else:
            raise SystemExit("no 2-D latitude and longitude in the store")
        times = pd.to_datetime(da[self.time_dim].values).floor("D")
        self.days = np.asarray((times - pd.Timestamp("1970-01-01")).days)
        assert (np.diff(self.days) == 1).all(), "the store's days are not consecutive"
        self.da = da
        self.shape = da.shape

    def read(self, t0, t1, r0, r1, c0, c1, retries=4):
        part = self.da.isel({self.time_dim: slice(t0, t1), self.y_dim: slice(r0, r1),
                             self.x_dim: slice(c0, c1)})
        for attempt in range(retries + 1):
            try:
                return np.asarray(part.load().values, dtype=np.float32)
            except Exception as error:    # network hiccups are common on long reads
                if attempt == retries:
                    raise
                wait = 10 * 2**attempt
                print(f"    read failed ({type(error).__name__}); retrying in {wait} s", flush=True)
                time.sleep(wait)


def pilot_values(root):
    """The pilot's daily values by cell, if this folder has them, for checking the blocks."""
    if not (DAILY_CSV.exists() and (OUT / "pilot_locations_selected.csv").exists()):
        return {}
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
    daily = pd.read_csv(DAILY_CSV)
    return {(c.lat_idx, c.lon_idx): daily[str(c.cell_id)].to_numpy(np.float32)
            for c in cells.itertuples()}


def download(root, store, only_block=None, size=BLOCK, source=None):
    root = Path(root)
    (root / "daily").mkdir(parents=True, exist_ok=True)
    source = source or ZarrSource(store)
    _, ny, nx = source.shape
    if not (root / "grid.npz").exists():    # the array's tasks may all get here at once
        partial = root / f"grid.{os.getpid()}.partial.npz"
        np.savez(partial, lat=source.lat, lon=source.lon, days=source.days, block=size)
        os.replace(partial, root / "grid.npz")
    pilot = pilot_values(root)
    for b, (r0, r1, c0, c1) in enumerate(blocks(ny, nx, size)):
        if only_block is not None and b != only_block:
            continue
        path = block_path(root, b)
        if path.exists():
            print(f"block {b}: already downloaded")
            continue
        start = time.time()
        cells = np.empty((r1 - r0, c1 - c0, N_DAYS), np.float32)    # each cell's days together
        for t0 in range(0, N_DAYS, READ_DAYS):
            t1 = min(t0 + READ_DAYS, N_DAYS)
            cells[:, :, t0:t1] = source.read(t0, t1, r0, r1, c0, c1).transpose(1, 2, 0)
            if (t0 // READ_DAYS) % 10 == 9:    # about every 2,400 days
                print(f"  block {b}: {t1} of {N_DAYS} days ({(time.time() - start) / 60:.1f} min)",
                      flush=True)
        assert np.isfinite(cells).all(), f"block {b} has missing values"
        assert (cells >= 0).all(), f"block {b} has negative values"
        for (row, col), expected in pilot.items():
            if r0 <= row < r1 and c0 <= col < c1:
                assert np.array_equal(cells[row - r0, col - c0], expected), (
                    f"block {b}: cell ({row}, {col}) differs from the pilot's daily values")
                print(f"  block {b}: pilot cell ({row}, {col}) matches the pilot exactly")
        partial = path.with_suffix(".partial.npy")
        np.save(partial, cells)
        os.replace(partial, path)
        print(f"block {b}: rows {r0}-{r1 - 1}, columns {c0}-{c1 - 1}, saved "
              f"({(time.time() - start) / 60:.1f} min)")


# ---------------------------------------------------------------------------------------------
# The fits

_worker = {}


def start_worker(root, tracks, starts, days, lat, lon, origins):
    """Each worker process keeps the inputs, and opens a block file the first time it needs it."""
    _worker.update(root=root, tracks=tracks, starts=starts, days=days, lat=lat, lon=lon,
                   nx=lat.shape[1], origins=origins, blocks={})


def fit_one(cell):
    b, row, col = (int(v) for v in cell)
    w = _worker
    if b not in w["blocks"]:
        w["blocks"][b] = np.load(block_path(w["root"], b), mmap_mode="r")
    r0, c0 = w["origins"][b]
    series = w["blocks"][b][row - r0, col - c0]
    return fit_cell(series, w["starts"], w["days"], w["lat"][row, col], w["lon"][row, col],
                    w["tracks"], seed=row * w["nx"] + col)


def fit(root, task, tasks, processes):
    root = Path(root)
    grid = np.load(root / "grid.npz")
    lat, lon, days, size = grid["lat"], grid["lon"], grid["days"], int(grid["block"])
    ny, nx = lat.shape
    cells = task_cells(ny, nx, size, task, tasks)
    folder = root / "fits" / f"task_{task:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "done").exists():
        print(f"task {task}: already done")
        return
    starts = water_year_starts(days)
    tracks = tracks_by_day(pd.to_datetime(days[0], unit="D"), pd.to_datetime(days[-1], unit="D"))
    origins = {b: (r0, c0) for b, (r0, _, c0, _) in enumerate(blocks(ny, nx, size))}
    start = time.time()
    parts = range(0, len(cells), SAVE_EVERY)
    todo = [k for k in parts if not (folder / f"part_{k // SAVE_EVERY:05d}.npz").exists()]
    print(f"task {task} of {tasks}: {len(cells):,} cells, {len(parts) - len(todo)} of "
          f"{len(parts)} parts already done", flush=True)

    with Pool(processes, initializer=start_worker,
              initargs=(root, tracks, starts, days, lat, lon, origins)) as pool:
        for n, k in enumerate(todo):
            chunk = cells[k:k + SAVE_EVERY]
            results = pool.map(fit_one, chunk, chunksize=4)
            table = {name: np.array([r[name] for r in results], np.float64) for name in FIELDS}
            part = folder / f"part_{k // SAVE_EVERY:05d}.npz"
            np.savez(folder / "part.partial.npz", cells=chunk, **table)
            os.replace(folder / "part.partial.npz", part)    # a stopped task leaves no half part
            per_part = (time.time() - start) / (n + 1)
            print(f"  {min(k + SAVE_EVERY, len(cells)):,} of {len(cells):,} cells "
                  f"({(time.time() - start) / 60:.1f} min, about "
                  f"{per_part * (len(todo) - n - 1) / 60:.0f} min to go)", flush=True)
    (folder / "done").touch()
    print(f"task {task}: done ({(time.time() - start) / 60:.1f} min)")


# ---------------------------------------------------------------------------------------------
# Putting the tasks together

def gather(root, tasks, shape):
    grids = {name: np.full(shape, np.nan) for name in FIELDS}
    seen = np.zeros(shape, int)
    missing = [k for k in range(tasks) if not (Path(root) / "fits" / f"task_{k:04d}" / "done")
               .exists()]
    if missing:
        raise SystemExit(f"{len(missing)} tasks are not finished, for example {missing[:10]}")
    for k in range(tasks):
        for part in sorted((Path(root) / "fits" / f"task_{k:04d}").glob("part_*.npz")):
            data = np.load(part)
            rows, cols = data["cells"][:, 1], data["cells"][:, 2]
            seen[rows, cols] += 1
            for name in FIELDS:
                grids[name][rows, cols] = data[name]
    assert (seen == 1).all(), "some cells were fitted twice or not at all"
    return grids


def write_netcdf(path, grids, lat, lon):
    """The layout of 40_calculate_return_periods.py, with the new method's attributes and four
    more variables (method, hurricane cell, convergence)."""
    import netCDF4 as nc
    # netCDF4 sets array shapes in a way NumPy 2.5 warns about. The values are unaffected.
    warnings.filterwarnings("ignore", message="Setting the shape", category=DeprecationWarning)
    ny, nx = lat.shape
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("return_period", len(RETURN_PERIODS))
    ds.createDimension("south_north", ny)
    ds.createDimension("west_east", nx)
    ds.title = "CONUS404 Return Period Analysis"
    ds.description = ("Native and Durst-converted return periods for SPDUV10MAX: peaks over "
                      "threshold away from hurricane cells, GEV at hurricane cells")
    ds.variable = "SPDUV10MAX (Daily maximum wind speed at 10 meters)"
    ds.units = "m/s"
    ds.method = ("Bayesian. Most cells: generalized Pareto fitted to event peaks above the 98th "
                 "percentile of daily values (2-day separation), Poisson event rate, normal "
                 "prior on the shape (mean -0.10, sd 0.20). Hurricane cells: GEV fitted to the "
                 "water-year maxima, flat prior on the shape (uniform, -1 to 1).")
    ds.hurricane_cell_rule = ("At least one of the 5 largest water-year maxima within a day of a "
                              "HURDAT2 tropical or subtropical system within 250 km")
    ds.num_years = N_YEARS
    ds.water_years = f"{FIRST_WY}-{LAST_WY}"
    ds.confidence_level = "95%"
    ds.ci_method = "Bayesian credible interval: 2.5th and 97.5th percentiles of the posterior"
    ds.estimate = "Posterior median"
    ds.created_date = pd.Timestamp.now().isoformat()
    ds.grid_dimensions = f"{ny} x {nx}"
    ds.Conventions = "CF-1.8"
    ds.coordinate_grid = "curvilinear"
    ds.gust_conversion_factor = GUST_FACTOR
    ds.durst_coefficient_3sec = 1.52
    ds.durst_coefficient_20sec = 1.36
    ds.source_duration_seconds = 20
    ds.target_duration_seconds = 3
    ds.gust_conversion_method = "Durst Curve coefficient ratio: C_Durst(3 s) / C_Durst(20 s)"

    variable = ds.createVariable("return_period", "i4", ("return_period",))
    variable.long_name, variable.units = "Return period", "years"
    variable[:] = RETURN_PERIODS
    space = ("south_north", "west_east")
    for name, values, standard, units in [("lat", lat, "latitude", "degrees_north"),
                                          ("lon", lon, "longitude", "degrees_east")]:
        variable = ds.createVariable(name, "f4", space, zlib=True, complevel=4)
        variable.standard_name, variable.units = standard, units
        variable.long_name = f"CONUS404 grid-cell {standard}"
        variable[:] = values

    stack = {q: np.stack([grids[f"rp{p}_{q}"] for p in RETURN_PERIODS]).astype(np.float32)
             for q in ("estimate", "lower", "upper")}
    names = {"estimate": "", "lower": "_lower_ci", "upper": "_upper_ci"}
    words = {"estimate": "return-level estimate (posterior median)",
             "lower": "return-level 95% credible interval lower bound",
             "upper": "return-level 95% credible interval upper bound"}
    dims = ("return_period",) + space
    for q, suffix in names.items():
        for kind, factor, label in [("native", 1.0, "Native CONUS404 20-second model-time-step"),
                                    ("3sec_gust", GUST_FACTOR, "Durst-converted 3-second gust")]:
            variable = ds.createVariable(f"wind_speed_{kind}{suffix}", "f4", dims, zlib=True,
                                         complevel=4)
            variable.long_name = f"{label} {words[q]}"
            variable.units, variable.coordinates = "m/s", "lat lon"
            if factor != 1.0:
                variable.gust_conversion_factor = GUST_FACTOR
            variable[:] = stack[q] * factor
    # the native per-period variables script 40 keeps for downstream code
    for j, period in enumerate(RETURN_PERIODS):
        for q, suffix in [("estimate", "estimate"), ("lower", "lower_ci"), ("upper", "upper_ci")]:
            variable = ds.createVariable(f"rp_{period}_{suffix}", "f4", space, zlib=True,
                                         complevel=4)
            variable.long_name = f"{period}-year {words[q]}"
            variable.units, variable.coordinates = "m/s", "lat lon"
            native = "wind_speed_native" + names[q]
            variable.compatibility_alias_for = f"{native} at return_period={period}"
            variable[:] = stack[q][j]
    extras = [("method", "i1", "0 = POT with the wider prior, 1 = GEV with a flat prior"),
              ("hurricane_cell", "i1", "1 if the cell meets the hurricane-cell rule"),
              ("converged", "i1", "1 if R-hat < 1.01 and effective sample size >= 400"),
              ("max_rhat", "f4", "Largest split R-hat of the shape and return levels")]
    for name, dtype, description in extras:
        variable = ds.createVariable(name, dtype, space, zlib=True, complevel=4)
        variable.long_name, variable.coordinates = description, "lat lon"
        values = np.nan_to_num(grids[name], nan=-1) if dtype == "i1" else grids[name]
        variable[:] = values.astype(dtype)
    ds.close()


def write_csv(path, grids, lat, lon):
    """The columns of script 40's CSV, plus the method, hurricane flag and convergence."""
    ny, nx = lat.shape
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    table = {"lat_idx": rows.ravel(), "lon_idx": cols.ravel(), "latitude": lat.ravel(),
             "longitude": lon.ravel()}
    for period in RETURN_PERIODS:
        for q, suffix in [("estimate", ""), ("lower", "_lower"), ("upper", "_upper")]:
            table[f"rp_{period}{suffix}"] = grids[f"rp{period}_{q}"].ravel().astype(np.float32)
        for q, suffix in [("estimate", ""), ("lower", "_lower"), ("upper", "_upper")]:
            table[f"rp_{period}_3sec_gust{suffix}"] = (
                grids[f"rp{period}_{q}"].ravel() * GUST_FACTOR).astype(np.float32)
    table["gust_conversion_factor"] = GUST_FACTOR
    table["converged"] = np.nan_to_num(grids["converged"].ravel(), nan=0).astype(int)
    table["method"] = np.nan_to_num(grids["method"].ravel(), nan=-1).astype(int)
    table["hurricane_cell"] = np.nan_to_num(grids["hurricane_cell"].ravel(), nan=-1).astype(int)
    pd.DataFrame(table).to_csv(path, index=False, float_format="%.4f")


def summary_rows(grids, lat, lon):
    rows = []

    def add(what, value):
        rows.append({"item": what, "value": value})

    n = grids["method"].size
    add("grid cells", n)
    add("hurricane cells", int(np.nansum(grids["hurricane_cell"])))
    add("failed fits", int(np.nansum(grids["failed"])))
    add("refitted with longer chains", int(np.nansum(grids["longer_chains"])))
    add("did not pass the convergence checks", int(n - np.nansum(grids["converged"])))
    for method, label in METHODS.items():
        mine = grids["method"] == method
        level = grids["rp100_estimate"][mine]
        width = (grids["rp100_upper"] - grids["rp100_lower"])[mine]
        if level.size:
            add(f"{label}: cells", int(mine.sum()))
            add(f"{label}: median 100-year level (m/s)", float(np.nanmedian(level)))
            add(f"{label}: median 100-year interval width (m/s)", float(np.nanmedian(width)))
    if CADENCE_NC.exists():
        import netCDF4 as nc
        with nc.Dataset(CADENCE_NC) as old:
            current = np.asarray(old["rp_100_estimate"][:], np.float64)
            current_width = (np.asarray(old["rp_100_upper_ci"][:], np.float64)
                             - np.asarray(old["rp_100_lower_ci"][:], np.float64))
        if current.shape == lat.shape:
            ratio = grids["rp100_estimate"] / current
            ok = np.isfinite(ratio) & (current > 0)
            add("100-year level against CADENCE's current file: median ratio",
                float(np.median(ratio[ok])))
            for q in (5, 95):
                add(f"100-year level against CADENCE's current file: {q}th percentile ratio",
                    float(np.percentile(ratio[ok], q)))
            add("share of cells above CADENCE's current 100-year level (%)",
                float(100 * (ratio[ok] > 1).mean()))
            add("CADENCE's current median 100-year interval width (m/s)",
                float(np.nanmedian(current_width)))
    return rows


def pilot_comparison(grids):
    """The 13 pilot cells against step 23's results table, for the version each cell got."""
    table_path = OUT / "checkpoint2_results_table.csv"
    if not (table_path.exists() and (OUT / "pilot_locations_selected.csv").exists()):
        return pd.DataFrame()
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
    ny, nx = grids["method"].shape
    if not ((cells.lat_idx < ny) & (cells.lon_idx < nx)).all():
        return pd.DataFrame()    # not the CONUS404 grid (the self-test)
    table = pd.read_csv(table_path)
    rows = []
    for cell in cells.itertuples():
        method = int(grids["method"][cell.lat_idx, cell.lon_idx])
        version = {0: ("POT", "wider"), 1: ("GEV", "flat")}[method]
        pilot = table[(table.cell_id == cell.cell_id) & (table.model == version[0])
                      & (table.prior == version[1])].iloc[0]
        row = {"location": cell.candidate, "method": METHODS[method]}
        for q, column in [("estimate", "rp100_estimate_ms"), ("lower", "rp100_lower_95_ms"),
                          ("upper", "rp100_upper_95_ms")]:
            row[f"rp100_{q}_ms"] = grids[f"rp100_{q}"][cell.lat_idx, cell.lon_idx]
            row[f"pilot_rp100_{q}_ms"] = pilot[column]
        rows.append(row)
    return pd.DataFrame(rows)


def combine(root, tasks):
    root = Path(root)
    if (root / "gev_return_periods.nc").resolve() == CADENCE_NC.resolve():
        raise SystemExit("ROOT holds CADENCE's current gev_return_periods.nc. Use another folder.")
    grid = np.load(root / "grid.npz")
    lat, lon = grid["lat"], grid["lon"]
    start = time.time()
    grids = gather(root, tasks, lat.shape)
    print(f"gathered {lat.size:,} cells from {tasks} tasks ({time.time() - start:.0f} s)")
    write_netcdf(root / "gev_return_periods.nc", grids, lat, lon)
    write_csv(root / "gev_return_periods.csv", grids, lat, lon)
    summary = pd.DataFrame(summary_rows(grids, lat, lon))
    summary.to_csv(root / "full_grid_summary.csv", index=False)
    print(summary.to_string(index=False, formatters={"value": lambda v: f"{v:,.4g}"}))
    pilot = pilot_comparison(grids)
    if len(pilot):
        pilot.round(3).to_csv(root / "full_grid_pilot_cells.csv", index=False)
        print("\nThe pilot cells, 100-year level (m/s), against step 23's results table:")
        print(pilot.round(2).to_string(index=False))
    print(f"\nwrote {root / 'gev_return_periods.nc'}, {root / 'gev_return_periods.csv'}, "
          f"{root / 'full_grid_summary.csv'}  ({(time.time() - start) / 60:.1f} min)")


# ---------------------------------------------------------------------------------------------
# Checks before the DCC run

def pilot_inputs():
    require(DAILY_CSV, HURDAT2, OUT / "pilot_locations_selected.csv")
    cells = pd.read_csv(OUT / "pilot_locations_selected.csv")
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    days = np.asarray((daily.date - pd.Timestamp("1970-01-01")).dt.days)
    assert days.size == N_DAYS
    return cells, daily, days


def pilot_seed(k, method):
    """The seed step 15 (GEV, flat prior) or step 16 (POT, wider prior) used at pilot cell k."""
    return 1000 * k + (0 if method == 1 else 11)


def check():
    """The per-cell code on the 13 pilot cells, against step 23's results table.

    Each cell is fitted twice. With the seed the pilot used, the result has to match step 23 to
    the 4 decimals it keeps. With the cell's own seed, as on DCC, the difference is the Monte
    Carlo error of the chains.
    """
    require(OUT / "checkpoint2_results_table.csv")
    cells, daily, days = pilot_inputs()
    table = pd.read_csv(OUT / "checkpoint2_results_table.csv")
    starts = water_year_starts(days)
    tracks = tracks_by_day(daily.date.iloc[0], daily.date.iloc[-1])
    rows = []
    start = time.time()
    for k, cell in enumerate(cells.itertuples()):
        series = daily[str(cell.cell_id)].to_numpy(np.float32)
        inputs = (series, starts, days, cell.latitude, cell.longitude, tracks)
        grid = fit_cell(*inputs, seed=int(cell.cell_id))
        same = fit_cell(*inputs, seed=pilot_seed(k, grid["method"]))
        model, prior = ("GEV", "flat") if grid["method"] == 1 else ("POT", "wider")
        pilot = table[(table.cell_id == cell.cell_id) & (table.model == model)
                      & (table.prior == prior)].iloc[0]
        row = {"location": cell.candidate, "hurricane_cell": bool(grid["hurricane_cell"]),
               "version": f"{model}, {prior}",
               "converged": bool(grid["converged"] and same["converged"]),
               "rp100_estimate_ms": grid["rp100_estimate"],
               "rp100_lower_ms": grid["rp100_lower"], "rp100_upper_ms": grid["rp100_upper"],
               "pilot_rp100_estimate_ms": pilot.rp100_estimate_ms}
        for label, result in (("pilot_seed", same), ("cell_seed", grid)):
            for period in (50, 100):
                for q, column in [("estimate", "estimate_ms"), ("lower", "lower_95_ms"),
                                  ("upper", "upper_95_ms")]:
                    got, want = result[f"rp{period}_{q}"], pilot[f"rp{period}_{column}"]
                    row[f"rp{period}_{q}_{label}_change_pct"] = 100 * (got / want - 1)
        rows.append(row)
        print(f"  {cell.candidate:<18} {row['version']:<11} 100-year "
              f"{grid['rp100_estimate']:5.2f} ({grid['rp100_lower']:5.2f} to "
              f"{grid['rp100_upper']:5.2f}), step 23 {pilot.rp100_estimate_ms:5.2f} "
              f"({pilot.rp100_lower_95_ms:5.2f} to {pilot.rp100_upper_95_ms:5.2f})  "
              f"({time.time() - start:.0f} s)", flush=True)
    out = pd.DataFrame(rows)
    hurricane = set(out[out.hurricane_cell].location)
    print(f"\nHurricane cells: {', '.join(sorted(hurricane))}")
    for label, words in (("pilot_seed", "with the pilot's seeds"),
                         ("cell_seed", "with each cell's own seed, as on DCC")):
        change = out.filter(like=f"_{label}_change_pct").abs()
        ends = change.drop(columns=change.filter(like="estimate").columns)
        print(f"Largest difference from step 23 {words}: estimates "
              f"{change.filter(like='estimate').max().max():.3f}%, interval ends "
              f"{ends.max().max():.3f}%")
    assert hurricane == COASTAL, f"hurricane cells {sorted(hurricane)}, expected the four coastal"
    assert out.converged.all(), "a pilot fit did not converge"
    assert (out.filter(like="_pilot_seed_change_pct").abs() < TOLERANCE_PCT).all().all(), (
        f"with the pilot's seeds, a return level differs from step 23 by {TOLERANCE_PCT}% or more")
    out.round(4).to_csv(OUT / "full_grid_pilot_check.csv", index=False)
    print(f"check passed. wrote {OUT / 'full_grid_pilot_check.csv'}")


class PilotSource:
    """A small store-like grid made of the 13 pilot cells and a few made-up cells."""

    def __init__(self, cells, daily, days, ny=4, nx=5, seed=0):
        rng = np.random.default_rng(seed)
        values = rng.gamma(4.0, 1.5, size=(N_DAYS, ny, nx)).astype(np.float32)
        self.lat = np.full((ny, nx), 40.0) + rng.uniform(-2, 2, (ny, nx))    # inland, away
        self.lon = np.full((ny, nx), -100.0) + rng.uniform(-2, 2, (ny, nx))  # from hurricanes
        self.where = {}
        for k, cell in enumerate(cells.itertuples()):
            row, col = divmod(k, nx)
            values[:, row, col] = daily[str(cell.cell_id)].to_numpy(np.float32)
            self.lat[row, col], self.lon[row, col] = cell.latitude, cell.longitude
            self.where[cell.candidate] = (row, col)
        self.values, self.days, self.shape = values, days, values.shape

    def read(self, t0, t1, r0, r1, c0, c1):
        return self.values[t0:t1, r0:r1, c0:c1]


def selftest(keep=False):
    """download, fit and combine on a small grid, against the per-cell code."""
    cells, daily, days = pilot_inputs()
    source = PilotSource(cells, daily, days)
    root = Path(tempfile.mkdtemp(prefix="full_grid_selftest_"))
    print(f"self-test in {root}")
    try:
        download(root, None, size=3, source=source)
        tasks = 3
        for task in range(tasks):
            fit(root, task, tasks, processes=2)
        fit(root, 0, tasks, processes=2)    # a finished task is skipped
        combine(root, tasks)
        grid = np.load(root / "grid.npz")
        grids = gather(root, tasks, grid["lat"].shape)
        starts = water_year_starts(days)
        tracks = tracks_by_day(daily.date.iloc[0], daily.date.iloc[-1])
        for row in range(source.shape[1]):
            for col in range(source.shape[2]):
                direct = fit_cell(source.values[:, row, col], starts, days,
                                  source.lat[row, col], source.lon[row, col], tracks,
                                  seed=row * source.shape[2] + col)
                for name in FIELDS:
                    a, b = grids[name][row, col], float(direct[name])
                    assert (np.isnan(a) and np.isnan(b)) or a == b, (row, col, name, a, b)
        flagged = {name for name, (r, c) in source.where.items() if grids["hurricane_cell"][r, c]}
        assert flagged == COASTAL, f"hurricane cells {sorted(flagged)}"
        import netCDF4 as nc
        with nc.Dataset(root / "gev_return_periods.nc") as ds:
            assert ds["wind_speed_native"].shape == (len(RETURN_PERIODS),) + source.shape[1:]
            assert np.allclose(ds["rp_100_estimate"][:], grids["rp100_estimate"], atol=1e-4)
            assert np.allclose(ds["wind_speed_3sec_gust"][3], grids["rp100_estimate"] *
                               GUST_FACTOR, atol=1e-3)
        csv = pd.read_csv(root / "gev_return_periods.csv")
        assert len(csv) == source.shape[1] * source.shape[2]
        print("\nself-test passed: the pipeline gives exactly the per-cell results, the four "
              "coastal cells are the hurricane cells, and the NetCDF and CSV read back")
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("check")
    test = stages.add_parser("selftest")
    test.add_argument("--keep", action="store_true", help="keep the self-test folder")
    step = stages.add_parser("download")
    step.add_argument("--root", required=True)
    step.add_argument("--store", default=STORE)
    step.add_argument("--block", type=int, default=None, help="one block (0 to 11), or all")
    step = stages.add_parser("fit")
    step.add_argument("--root", required=True)
    step.add_argument("--task", type=int, required=True)
    step.add_argument("--tasks", type=int, required=True)
    step.add_argument("--processes", type=int, default=os.cpu_count())
    step = stages.add_parser("combine")
    step.add_argument("--root", required=True)
    step.add_argument("--tasks", type=int, required=True)
    args = parser.parse_args()
    require(HURDAT2)
    if args.stage == "check":
        check()
    elif args.stage == "selftest":
        selftest(args.keep)
    elif args.stage == "download":
        download(args.root, args.store, args.block)
    elif args.stage == "fit":
        fit(args.root, args.task, args.tasks, args.processes)
    else:
        combine(args.root, args.tasks)
