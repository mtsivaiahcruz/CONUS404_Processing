"""Checks of the two extraction scripts in pot_pilot, on small made-up data stores.

    conus404_spduv10max_extract.py      annual maxima and daily quality counts for every cell
    conus404_pilot_daily_extract.py     daily values at the pilot cells

Each check builds a small store with known values and with problems planted in it (a cell with
no data, negative values, a value that repeats, a spike, a gap, missing dates). It runs the
script on that store and compares the output with a direct calculation. Nothing is downloaded
and no existing file is changed; everything is written to a temporary folder.

Needs xarray, dask, scipy, pandas and numpy.

Usage:
    python tests/test_extraction_scripts.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPTS = Path(__file__).resolve().parents[1]    # pot_pilot

try:
    import netCDF4

    HAVE_NC4 = True
except ImportError:
    HAVE_NC4 = False
    # Without netCDF4, write and read the scripts' outputs as netCDF-3 files through scipy.
    _to_nc, _open_mf, _open = xr.Dataset.to_netcdf, xr.open_mfdataset, xr.open_dataset

    def _patched_to_netcdf(self, path=None, *a, encoding=None, **k):
        k.pop("engine", None)
        return _to_nc(self, path, *a, engine="scipy", **k)

    xr.Dataset.to_netcdf = _patched_to_netcdf
    xr.open_mfdataset = lambda *a, **k: _open_mf(*a, engine="scipy", **k)
    xr.open_dataset = lambda p, *a, **k: _open(p, *a, **{"engine": "scipy", **k})

ok = True


def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")


def load(script):
    """Load one of the extraction scripts as a module, without running its main()."""
    path = SCRIPTS / script
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, argv):
    """Run a script's main() with the given command-line arguments. Returns its exit code."""
    saved_argv = sys.argv
    sys.argv = ["x", *argv]
    try:
        module.main()
    except SystemExit as exit_:
        return exit_.code
    finally:
        sys.argv = saved_argv
    return 0


def made_up_store(ny, nx, drop=(), seed=0):
    """Four water years of random daily values on a small grid, with some dates left out."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("1979-10-01", "1983-09-30", freq="D")
    times = times.drop([pd.Timestamp(date) for date in drop])
    data = rng.gamma(4, 2.5, size=(len(times), ny, nx)).astype(np.float32)
    lat = 25 + np.arange(ny)[:, None] * 0.036 + np.arange(nx)[None, :] * 0.001
    lon = -125 + np.arange(nx)[None, :] * 0.045 + np.arange(ny)[:, None] * 0.002
    return times, data, lat, lon


def open_scipy(url, chunks):
    """Open a made-up store. Stands in for the scripts' own open_store()."""
    opener = _open if not HAVE_NC4 else xr.open_dataset
    return opener(url, engine="scipy", chunks=chunks)


tmp = Path(tempfile.mkdtemp(prefix="cadence_extract_test_"))

# Part 1. The full-grid extraction.
grid_script = load("conus404_spduv10max_extract.py")
ny, nx = 12, 15
times, data, lat, lon = made_up_store(ny, nx, drop=("1981-02-10", "1981-02-11"))
data[:, 0, 0] = np.nan          # a cell with no data at all
data[100:105, 1, 1] = -1.0      # negative values
data[200:240, 2, 2] = 7.25      # the same value for 40 days
data[500, 3, 3] = 95.0          # above 80 m/s
data[600:610, 4, 4] = np.nan    # a 10-day gap
dims = ("time", "y", "x")
source = xr.Dataset(
    {"SPDUV10MAX": (dims, data, {"units": "m s-1"}), "T2MAX": (dims, data)},
    coords={"time": times, "y": np.arange(ny) * 4000.0, "x": np.arange(nx) * 4000.0,
            "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
)
store = tmp / "store_grid.nc"
source.to_netcdf(store, engine="scipy")
grid_script.open_store = lambda url: open_scipy(url, {"time": 20})
out = tmp / "grid_out"
grid_args = ["--store", str(store), "--out-dir", str(out), "--batch-days", "45"]
check("full-grid script runs", run(grid_script, grid_args) == 0)
result = xr.open_dataset(sorted(out.glob("spduv10max_water_*.nc"))[0])
manifest = json.loads((out / "manifest.json").read_text())

# Compare with a direct calculation, year by year.
all_match = True
for year in result.year.values:
    start, end = grid_script.block_bounds(int(year), "water")
    block = source.SPDUV10MAX.sel(time=slice(start, end)).values
    direct_max = np.nanmax(np.where(np.isfinite(block), block, np.nan), axis=0)
    got_max = result.annual_max.sel(year=year).values
    # NaN never equals NaN, so replace it with -1 on both sides before comparing
    all_match &= np.allclose(np.nan_to_num(got_max, nan=-1), np.nan_to_num(direct_max, nan=-1))
    direct_valid = np.isfinite(block).sum(0)
    all_match &= bool((result.n_valid.sel(year=year).values == direct_valid).all())
    # the day recorded for the maximum at cell (5, 6) must really hold that maximum
    day_index = int(result.max_day_index.sel(year=year).values[5, 6])
    day = start + pd.Timedelta(days=day_index)
    value_that_day = source.SPDUV10MAX.sel(time=day).values[5, 6]
    all_match &= bool(np.isclose(value_that_day, got_max[5, 6]))
check("annual maxima, valid-day counts and date of maximum match a direct calculation",
      all_match)
check("dead cell: no valid days and no annual maximum",
      (result.n_valid.values[:, 0, 0] == 0).all()
      and np.isnan(result.annual_max.values[:, 0, 0]).all())
check("negative values counted", int(result.n_negative.sel(year=1980).values[1, 1]) == 5)
check("stuck value: 39 repeats in a row",
      int(result.longest_repeat.sel(year=1980).values[2, 2]) == 39)
check("spike above 80 m/s counted", int(result.n_over_limit.sel(year=1981).values[3, 3]) == 1)
check("10-day gap counted", int(result.n_missing.sel(year=1981).values[4, 4]) == 10)
check("missing dates recorded in the manifest",
      manifest["years"]["1981"]["missing_dates"] == ["1981-02-10", "1981-02-11"])
check("grid table has one row per cell",
      len(pd.read_csv(out / "conus404_grid_cells.csv.gz")) == ny * nx)
result.close()

# Running it again must not redo the years that are already finished.
written_before = {part.name: part.stat().st_mtime for part in (out / "parts").glob("*.nc")}
run(grid_script, grid_args)
written_after = {part.name: part.stat().st_mtime for part in (out / "parts").glob("*.nc")}
check("re-running resumes without redoing finished years", written_before == written_after)

# Part 2. The pilot daily extraction.
daily_script = load("conus404_pilot_daily_extract.py")
ny, nx = 40, 55
times, data, lat, lon = made_up_store(ny, nx, seed=1)
source = xr.Dataset(
    {"SPDUV10MAX": (dims, data, {"units": "m s-1"})},
    coords={"time": times, "y": np.arange(ny) * 4000.0, "x": np.arange(nx) * 4000.0,
            "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
)
store = tmp / "store_daily.nc"
source.to_netcdf(store, engine="scipy")
daily_script.open_store = lambda url: open_scipy(url, {"time": 24, "y": 7, "x": 9})
# six made-up pilot cells
rows, cols = np.array([3, 3, 17, 39, 0, 22]), np.array([4, 50, 17, 54, 0, 9])
cells = pd.DataFrame({"cell_id": rows * nx + cols, "lat_idx": rows, "lon_idx": cols,
                      "latitude": lat[rows, cols].round(5),
                      "longitude": lon[rows, cols].round(5)})
out = tmp / "daily_out"
out.mkdir()
cells.to_csv(out / "pilot_locations_selected.csv", index=False)
daily_args = ["--store", str(store), "--dir", str(out),
              "--first-water-year", "1980", "--last-water-year", "1983"]
check("pilot daily script runs", run(daily_script, daily_args) == 0)
got = pd.read_csv(out / "pilot_daily_spduv10max.csv", parse_dates=["date"])
expected_values = data[:, rows, cols]
check("daily values equal the store's values exactly",
      np.array_equal(got.drop(columns="date").to_numpy(np.float32), expected_values))
check("dates are complete and in order", (pd.DatetimeIndex(got.date) == times).all())


def saved_values():
    saved = pd.read_csv(out / "pilot_daily_spduv10max.csv")
    return saved.drop(columns="date").to_numpy(np.float32)


check("re-running resumes and gives the same file",
      run(daily_script, daily_args) == 0 and np.array_equal(saved_values(), expected_values))

# A cell list whose grid position does not match its coordinates must stop the script.
bad = tmp / "daily_bad"
bad.mkdir()
wrong_cells = cells.assign(lat_idx=cells.lat_idx.where(cells.index != 2, 18))
wrong_cells.to_csv(bad / "pilot_locations_selected.csv", index=False)
exit_code = run(daily_script, ["--store", str(store), "--dir", str(bad),
                               "--first-water-year", "1980", "--last-water-year", "1983"])
check("stops when a cell's grid position does not match its coordinates",
      exit_code not in (0, None) and not (bad / "pilot_daily_spduv10max.csv").exists())

print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
