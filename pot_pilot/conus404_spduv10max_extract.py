#!/usr/bin/env python3
"""Extract CONUS404 daily SPDUV10MAX as annual maxima and data-quality counts for every cell.

Source: HyTEST's public copy of the CONUS404 daily diagnostic ("xtrm") zarr on the USGS Open
Storage Network. It can be read without an account and without download charges.

    s3://hytest/conus404/conus404_xtrm_daily.zarr   endpoint https://usgs.osn.mghpcc.org/

The full daily record is roughly 90 GB uncompressed, so the script never saves the daily
fields. It reads one block-year at a time and keeps, for every grid cell:

    annual_max       highest SPDUV10MAX of the block-year (m/s); this is the input to the GEV
    max_day_index    day within the block-year on which the maximum occurred
    n_valid          days with a finite value
    n_missing        days that are in the store but hold no value
    n_negative       days below 0 (not possible for a speed)
    n_zero           days exactly 0
    n_over_limit     days above --limit-ms (default 80 m/s)
    n_repeat         days whose value is exactly the same as the day before
    longest_repeat   most repeats in a row (40 identical days in a row count as 39)

It also writes the grid itself: lat_idx and lon_idx (the row and column of each cell in the
store), the latitude and longitude of each cell centre, and the projected x and y. This lets
the results be joined to CADENCE's gev_return_periods.csv on (lat_idx, lon_idx).

Each finished year is saved under OUT_DIR/parts/ straight away, so a run that is interrupted
carries on from where it stopped. Run the same command again to continue.

OUT_DIR is the data folder. By default it is the data folder next to this script
(pot_pilot/data), wherever the script is run from. Pass --out-dir to write somewhere else.

Typical use:
    pip install "xarray>=2024.1" zarr s3fs dask netCDF4 pandas numpy
    python conus404_spduv10max_extract.py --inspect          # quick look, no download
    python conus404_spduv10max_extract.py --years 2023       # ~2 GB test run
    python conus404_spduv10max_extract.py                    # full record
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_VERSION = "1.0.0"
DEFAULT_STORE = "s3://hytest/conus404/conus404_xtrm_daily.zarr"
OSN_ENDPOINT = "https://usgs.osn.mghpcc.org/"
VAR_PREFERRED = "SPDUV10MAX"
# rows (lat_idx) x columns (lon_idx) in CADENCE's gev_return_periods.csv
GRID_SHAPE_CADENCE = (1015, 1367)
HERE = Path(__file__).resolve().parent    # CONUS404_Processing/pot_pilot
DEFAULT_OUT_DIR = HERE / "data"


# Opening the store and finding the variable and the grid in it.

def open_store(url):
    """Open the zarr store. The OSN copy is public, so no credentials are needed."""
    kwargs = {"engine": "zarr", "chunks": {}}
    if url.startswith("s3://"):
        kwargs["storage_options"] = {
            "anon": True,
            "requester_pays": False,
            "client_kwargs": {"endpoint_url": OSN_ENDPOINT},
        }
    try:
        return xr.open_dataset(url, consolidated=True, **kwargs)
    except (KeyError, FileNotFoundError, ValueError):
        # the store has no consolidated metadata, or uses the zarr v3 layout
        return xr.open_dataset(url, consolidated=False, **kwargs)


def find_variable(ds, requested):
    """Name of the wind variable to read: the one asked for, or SPDUV10MAX if it is there."""
    if requested:
        if requested not in ds.data_vars:
            sys.exit(f"Variable {requested!r} not in store. "
                     f"Data variables: {sorted(ds.data_vars)}")
        return requested
    if VAR_PREFERRED in ds.data_vars:
        return VAR_PREFERRED
    candidates = [v for v in ds.data_vars if "SPDUV10" in v.upper() and "MAX" in v.upper()]
    if len(candidates) == 1:
        return candidates[0]
    wind_words = ("SPD", "U10", "V10", "WIND")
    wind_like = sorted(v for v in ds.data_vars if any(word in v.upper() for word in wind_words))
    sys.exit(
        f"Could not find {VAR_PREFERRED} in the store.\n"
        f"Wind-like variables present: {wind_like}\n"
        f"All data variables: {sorted(ds.data_vars)}\n"
        "Pass the right one with --var."
    )


def spatial_dims(da):
    """Names of the variable's row and column dimensions."""
    dims = [dim for dim in da.dims if dim != "time"]
    if len(dims) != 2:
        sys.exit(f"Expected (time, y, x); {da.name} has dims {da.dims}")
    return dims[0], dims[1]


def find_latlon(ds, ydim, xdim):
    """Names of the store's 2-D latitude and longitude variables."""
    for lat, lon in (("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")):
        if lat in ds.variables and lon in ds.variables:
            if set(ds[lat].dims) == {ydim, xdim}:
                return lat, lon
    sys.exit(f"No 2-D lat/lon on ({ydim}, {xdim}) found. Variables: {sorted(ds.variables)}")


# Block-years. A water year runs October to September and is named after the year it ends in.

def block_bounds(year, block):
    """First and last day of a water year or a calendar year."""
    if block == "water":
        return pd.Timestamp(year - 1, 10, 1), pd.Timestamp(year, 9, 30)
    return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)


def complete_years(first_time, last_time, block):
    """The block-years that lie wholly inside the store's time range."""
    years = []
    for year in range(first_time.year - 1, last_time.year + 2):
        start, end = block_bounds(year, block)
        if start >= first_time.normalize() and end <= last_time.normalize():
            years.append(year)
    return years


# Reading and summarising one block-year.

def fetch(da, retries):
    """Download the values, trying again after a pause if the read fails."""
    for attempt in range(retries + 1):
        try:
            return np.asarray(da.load().values, dtype=np.float32)
        except Exception as exc:    # short network failures are common on long downloads
            if attempt == retries:
                raise
            wait = 10 * 2**attempt
            print(f"    read failed ({type(exc).__name__}: {exc}); retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def native_time_chunk(da):
    """Number of days in one of the store's chunks, if it can be found."""
    if da.chunks:
        return int(da.chunksizes["time"][0])
    encoding = da.encoding.get("preferred_chunks") or {}
    return encoding.get("time")


def batch_bounds(da, one_year, batch_days):
    """Split the year into batches of days to read.

    The store keeps its data in chunks of days. Batches start and end on chunk boundaries, so
    that no chunk is downloaded twice. A batch holds whole chunks and grows until it has at
    least batch_days days; it is never smaller than one chunk.
    """
    n_days = one_year.sizes["time"]
    chunk_days = native_time_chunk(da)
    if not chunk_days or n_days == 0:
        return [(i, min(i + batch_days, n_days)) for i in range(0, n_days, batch_days)]
    # position of the year's first day in the whole store
    first_position = int(np.searchsorted(da["time"].values, one_year["time"].values[0]))
    chunk_starts = [k for k in range(1, n_days) if (first_position + k) % chunk_days == 0]
    cuts = [0] + chunk_starts + [n_days]
    bounds = []
    start = 0
    for cut in cuts[1:]:
        if cut - start >= batch_days or cut == n_days:
            bounds.append((start, cut))
            start = cut
    return bounds


def process_year(da, year, block, batch_days, limit, retries):
    """Read one block-year and reduce it to per-cell annual maxima and quality counts."""
    start, end = block_bounds(year, block)
    one_year = da.sel(time=slice(start, end + pd.Timedelta(hours=23, minutes=59)))
    times = pd.DatetimeIndex(one_year["time"].values)
    expected = pd.date_range(start, end, freq="D")
    present_days = pd.DatetimeIndex(times.normalize())
    missing_dates = expected.difference(present_days)
    duplicate_days = int(present_days.duplicated().sum())

    # Running totals, one value per grid cell, updated one day at a time.
    ny, nx = one_year.shape[1], one_year.shape[2]
    max_so_far = np.full((ny, nx), -np.inf, dtype=np.float32)
    day_of_max = np.full((ny, nx), -1, dtype=np.int16)
    n_valid = np.zeros((ny, nx), np.int16)
    n_negative = np.zeros((ny, nx), np.int16)
    n_zero = np.zeros((ny, nx), np.int16)
    n_over = np.zeros((ny, nx), np.int16)
    n_repeat = np.zeros((ny, nx), np.int16)
    current_run = np.zeros((ny, nx), np.int16)
    longest_run = np.zeros((ny, nx), np.int16)
    previous_day = None

    # Each day's number counted from the start of the block, so that max_day_index is a true
    # calendar offset even if some days are missing from the store.
    day_offsets = ((present_days - start) / pd.Timedelta(days=1)).astype(int).to_numpy()

    for first, last in batch_bounds(da, one_year, batch_days):
        batch_start = time.time()
        batch = fetch(one_year.isel(time=slice(first, last)), retries)
        for k in range(batch.shape[0]):
            day = batch[k]
            finite = np.isfinite(day)
            n_valid += finite
            n_negative += day < 0
            n_zero += day == 0
            n_over += day > limit
            higher = finite & (day > max_so_far)
            max_so_far[higher] = day[higher]
            day_of_max[higher] = day_offsets[first + k]
            if previous_day is not None:
                same = finite & (day == previous_day)
                n_repeat += same
                current_run = np.where(same, current_run + 1, 0).astype(np.int16)
                np.maximum(longest_run, current_run, out=longest_run)
            previous_day = day
        print(
            f"    {year}: days {first + 1}-{last} of {len(times)} "
            f"({batch.nbytes / 1e6:.0f} MB, {time.time() - batch_start:.0f}s)",
            flush=True,
        )

    # cells with no valid day at all have no annual maximum
    annual_max = np.where(np.isfinite(max_so_far), max_so_far, np.nan).astype(np.float32)
    n_days = len(times)
    cell_dims = ("lat_idx", "lon_idx")
    result = xr.Dataset(
        {
            "annual_max": (cell_dims, annual_max),
            "max_day_index": (cell_dims, day_of_max),
            "n_valid": (cell_dims, n_valid),
            "n_missing": (cell_dims, (n_days - n_valid).astype(np.int16)),
            "n_negative": (cell_dims, n_negative),
            "n_zero": (cell_dims, n_zero),
            "n_over_limit": (cell_dims, n_over),
            "n_repeat": (cell_dims, n_repeat),
            "longest_repeat": (cell_dims, longest_run),
        }
    ).expand_dims(year=[year])
    result["block_start"] = ("year", [np.datetime64(start.date(), "ns")])
    result["n_days_present"] = ("year", np.array([n_days], np.int16))
    result["n_days_expected"] = ("year", np.array([len(expected)], np.int16))

    info = {
        "year": year,
        "block_start": str(start.date()),
        "block_end": str(end.date()),
        "days_expected": len(expected),
        "days_present": n_days,
        "duplicate_days": duplicate_days,
        "missing_dates": [str(day.date()) for day in missing_dates],
        "cells_all_missing": int((n_valid == 0).sum()),
        "cells_any_negative": int((n_negative > 0).sum()),
        "cells_over_limit": int((n_over > 0).sum()),
        "max_value": float(np.nanmax(annual_max)) if np.isfinite(annual_max).any() else None,
    }
    return result, info


def write_grid(ds, var, out_dir):
    """Write the grid's coordinates to conus404_grid.nc and conus404_grid_cells.csv.gz."""
    da = ds[var]
    ydim, xdim = spatial_dims(da)
    lat, lon = find_latlon(ds, ydim, xdim)
    ny, nx = ds.sizes[ydim], ds.sizes[xdim]
    cell_dims = ("lat_idx", "lon_idx")
    grid = xr.Dataset(
        {
            "latitude": (cell_dims, ds[lat].transpose(ydim, xdim).values.astype(np.float64)),
            "longitude": (cell_dims, ds[lon].transpose(ydim, xdim).values.astype(np.float64)),
        },
        coords={
            "lat_idx": np.arange(ny, dtype=np.int32),
            "lon_idx": np.arange(nx, dtype=np.int32),
        },
    )
    # the projected y and x of the rows and columns, if the store has them
    if ydim in ds.coords and ds[ydim].dims == (ydim,):
        grid["proj_y"] = ("lat_idx", ds[ydim].values)
    if xdim in ds.coords and ds[xdim].dims == (xdim,):
        grid["proj_x"] = ("lon_idx", ds[xdim].values)
    grid.attrs.update(
        source_row_dim=ydim,
        source_col_dim=xdim,
        note="lat_idx/lon_idx are the native array indices of the source store, in source order.",
    )
    # keep the description of the map projection, under whichever name the store uses
    for aux in ("crs", "lambert_conformal_conic", "Lambert_Conformal"):
        if aux in ds.variables:
            projection = {key: str(value) for key, value in ds[aux].attrs.items()}
            grid.attrs["projection_" + aux] = json.dumps(projection)
    compression = {name: {"zlib": True, "complevel": 4} for name in grid.data_vars}
    grid.to_netcdf(out_dir / "conus404_grid.nc", encoding=compression)

    csv = out_dir / "conus404_grid_cells.csv.gz"
    if not csv.exists():
        rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        pd.DataFrame(
            {
                "lat_idx": rows.ravel(),
                "lon_idx": cols.ravel(),
                "latitude": grid["latitude"].values.ravel(),
                "longitude": grid["longitude"].values.ravel(),
            }
        ).to_csv(csv, index=False, float_format="%.6f", compression="gzip")
    return grid


def inspect(ds, var, block, url):
    """Print a summary of the store without downloading any data."""
    da = ds[var]
    times = pd.DatetimeIndex(ds["time"].values)
    ydim, xdim = spatial_dims(da)
    years = complete_years(times.min(), times.max(), block)
    chunks = da.encoding.get("chunks") or da.encoding.get("preferred_chunks")
    ny, nx = da.sizes[ydim], da.sizes[xdim]
    print(f"Store:        {url}")
    print(f"Variable:     {var}  dims={da.dims} dtype={da.dtype} chunks={chunks}")
    print(f"Attributes:   {dict(da.attrs)}")
    print(f"Time:         {times.min()} .. {times.max()}  ({len(times)} steps)")
    print(f"Grid:         {ny} x {nx}  "
          f"(CADENCE expects {GRID_SHAPE_CADENCE[0]} x {GRID_SHAPE_CADENCE[1]})")
    print(f"Complete {block}-years: {years[0] if years else '-'}..{years[-1] if years else '-'} "
          f"({len(years)})")
    print(f"Uncompressed size of {var} over those years: "
          f"~{ny * nx * 4 * 365.25 * len(years) / 1e9:.0f} GB")
    others = sorted(v for v in ds.data_vars if v != var)
    print(f"Other variables: {others}")


def parse_years(spec):
    """Turn "2023", "1980-2024" or "1990,2005-2007" into a sorted list of years."""
    if not spec:
        return None
    years = []
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            years.extend(range(int(first), int(last) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=DEFAULT_STORE,
                        help="zarr URL or local path (default: HyTEST OSN xtrm daily)")
    parser.add_argument("--var", default=None,
                        help=f"variable name (default: {VAR_PREFERRED}, auto-detected)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path,
                        help="data folder (default: pot_pilot/data)")
    parser.add_argument("--years", default=None,
                        help="e.g. 2023, 1980-2024 or 1990,2005-2007 "
                             "(default: every complete year)")
    parser.add_argument("--block", choices=("water", "calendar"), default="water",
                        help="block-maximum year: water (Oct-Sep, CONUS404's native year; "
                             "default) or calendar")
    parser.add_argument("--batch-days", type=int, default=32,
                        help="minimum days per read batch; batches snap to the store's time "
                             "chunks (default 32)")
    parser.add_argument("--workers", type=int, default=16,
                        help="parallel chunk downloads (default 16)")
    parser.add_argument("--limit-ms", type=float, default=80.0,
                        help="flag daily values above this (m/s)")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--inspect", action="store_true",
                        help="print store summary and exit (downloads nothing)")
    args = parser.parse_args()

    import dask

    dask.config.set(scheduler="threads", num_workers=args.workers)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"Opening {args.store} ...", flush=True)
    ds = open_store(args.store)
    var = find_variable(ds, args.var)
    if args.inspect:
        inspect(ds, var, args.block, args.store)
        return

    da = ds[var]
    ydim, xdim = spatial_dims(da)
    da = da.transpose("time", ydim, xdim)
    if (da.sizes[ydim], da.sizes[xdim]) != GRID_SHAPE_CADENCE:
        print(f"WARNING: grid is {da.sizes[ydim]} x {da.sizes[xdim]}, CADENCE uses "
              f"{GRID_SHAPE_CADENCE[0]} x {GRID_SHAPE_CADENCE[1]}", flush=True)

    # Work out which years to process: the ones asked for that the store has in full.
    times = pd.DatetimeIndex(ds["time"].values)
    available = complete_years(times.min(), times.max(), args.block)
    requested = parse_years(args.years) or available
    skipped = [year for year in requested if year not in available]
    years = [year for year in requested if year in available]
    if skipped:
        print(f"Skipping incomplete/out-of-range {args.block}-years: {skipped}", flush=True)
    if not years:
        sys.exit(f"No complete {args.block}-years to process. "
                 f"Store covers {times.min()} .. {times.max()}.")

    out_dir = args.out_dir
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print("Writing grid coordinates ...", flush=True)
    write_grid(ds, var, out_dir)

    # The manifest records what was read and what was found, year by year.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"years": {}}
    manifest.update(
        script_version=SCRIPT_VERSION,
        store=args.store,
        variable=var,
        variable_attrs={key: str(value) for key, value in da.attrs.items()},
        source_chunks=str(da.encoding.get("chunks") or da.encoding.get("preferred_chunks")),
        store_time_range=[str(times.min()), str(times.max())],
        block=args.block,
        limit_ms=args.limit_ms,
        source_dims=[ydim, xdim],
        source_y_ascending=bool(
            ydim in ds.coords and ds[ydim].size > 1 and ds[ydim][1] > ds[ydim][0]
        ),
        dataset_attrs={key: str(value)[:500] for key, value in ds.attrs.items()},
        python=platform.python_version(),
        versions={
            module.__name__: getattr(module, "__version__", "?") for module in (np, pd, xr, dask)
        },
    )

    started = time.time()
    for n, year in enumerate(years, 1):
        part = parts_dir / f"{args.block}_{year}.nc"
        if part.exists():
            print(f"[{n}/{len(years)}] {year}: already done, skipping", flush=True)
            continue
        print(f"[{n}/{len(years)}] {args.block}-year {year}", flush=True)
        year_start = time.time()
        result, info = process_year(da, year, args.block, args.batch_days, args.limit_ms,
                                    args.retries)
        # write to a temporary name first, so a part file is never left half written
        tmp = part.with_suffix(".nc.tmp")
        compression = {name: {"zlib": True, "complevel": 4} for name in result.data_vars}
        result.to_netcdf(tmp, encoding=compression)
        os.replace(tmp, part)
        info["seconds"] = round(time.time() - year_start, 1)
        manifest["years"][str(year)] = info
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if info["missing_dates"]:
            note = f", {len(info['missing_dates'])} missing days"
        else:
            note = ""
        print(f"    done in {info['seconds']:.0f}s; max {info['max_value']:.1f} m/s; "
              f"{info['cells_all_missing']} all-missing cells{note}", flush=True)

    print("Combining years ...", flush=True)
    files = [parts_dir / f"{args.block}_{year}.nc" for year in years]
    combined = xr.open_mfdataset(files, combine="nested", concat_dim="year",
                                 data_vars="minimal", coords="minimal", compat="override",
                                 join="exact")
    grid = xr.open_dataset(out_dir / "conus404_grid.nc")
    combined = combined.assign_coords(latitude=grid["latitude"], longitude=grid["longitude"])
    combined["annual_max"].attrs.update(
        units=da.attrs.get("units", "m s-1"),
        long_name=f"{args.block}-year maximum of daily {var}",
        source=args.store,
    )
    combined.attrs.update(
        title=f"CONUS404 {var} {args.block}-year maxima and per-cell QA",
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        script_version=SCRIPT_VERSION,
        block=args.block,
    )
    final_path = out_dir / f"{var.lower()}_{args.block}_{years[0]}-{years[-1]}.nc"
    compression = {name: {"zlib": True, "complevel": 4} for name in combined.data_vars}
    combined.to_netcdf(final_path, encoding=compression)
    grid.close()
    manifest["output"] = final_path.name
    manifest["years_in_output"] = years
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone in {(time.time() - started) / 60:.1f} min.\n  {final_path}\n"
          f"  {out_dir / 'conus404_grid.nc'}\n"
          f"  {out_dir / 'conus404_grid_cells.csv.gz'}\n  {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
