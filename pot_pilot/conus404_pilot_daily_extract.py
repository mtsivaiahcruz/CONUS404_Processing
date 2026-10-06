#!/usr/bin/env python3
"""Extract the daily SPDUV10MAX values at the pilot cells, for the peaks-over-threshold tests.

This reads the same store as conus404_spduv10max_extract.py (HyTEST's public copy of the
CONUS404 daily diagnostic zarr on the USGS Open Storage Network), but keeps every daily value
and only at the pilot cells.

DIR is the data folder. By default it is the data folder next to this script (pot_pilot/data),
wherever the script is run from. Pass --dir if the data folder is somewhere else.

Reads:
    DIR/pilot_locations_selected.csv    columns cell_id, lat_idx, lon_idx, latitude, longitude
Writes:
    DIR/pilot_daily_spduv10max.csv      one row per day, one column per cell (m/s)
    DIR/pilot_daily_spduv10max.json     the source, variable, dates and cells

Water years 1980-2024 (1979-10-01 to 2024-09-30) by default. Each finished water year is saved
under DIR/parts_daily/, so a run that is interrupted carries on from where it stopped.

Run from anywhere:
    python conus404_pilot_daily_extract.py
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_VERSION = "1.0.0"
DEFAULT_STORE = "s3://hytest/conus404/conus404_xtrm_daily.zarr"
OSN_ENDPOINT = "https://usgs.osn.mghpcc.org/"
VAR = "SPDUV10MAX"
HERE = Path(__file__).resolve().parent    # CONUS404_Processing/pot_pilot
DEFAULT_DIR = HERE / "data"


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
        return xr.open_dataset(url, consolidated=False, **kwargs)


def find_latlon(ds, ydim, xdim):
    """Names of the store's 2-D latitude and longitude variables."""
    for lat, lon in (("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")):
        if lat in ds.variables and lon in ds.variables and set(ds[lat].dims) == {ydim, xdim}:
            return lat, lon
    sys.exit(f"No 2-D lat/lon on ({ydim}, {xdim}) found. Variables: {sorted(ds.variables)}")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--var", default=VAR)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help="folder holding pilot_locations_selected.csv")
    parser.add_argument("--cells", type=Path, default=None,
                        help="cell list CSV (default: DIR/pilot_locations_selected.csv)")
    parser.add_argument("--first-water-year", type=int, default=1980)
    parser.add_argument("--last-water-year", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    import dask

    dask.config.set(scheduler="threads", num_workers=args.workers)

    # The list of pilot cells.
    cells_path = args.cells or (args.dir / "pilot_locations_selected.csv")
    if not cells_path.exists():
        sys.exit(f"Cell list not found: {cells_path}")
    cells = pd.read_csv(cells_path)
    required = {"cell_id", "lat_idx", "lon_idx", "latitude", "longitude"}
    if not required.issubset(cells.columns):
        sys.exit(f"{cells_path} must have columns {sorted(required)}")
    if cells.cell_id.duplicated().any():
        sys.exit("Duplicate cell_id in cell list")
    lat_idx = cells.lat_idx.to_numpy(dtype=int)
    lon_idx = cells.lon_idx.to_numpy(dtype=int)
    print(f"{len(cells)} cells from {cells_path}", flush=True)

    print(f"Opening {args.store} ...", flush=True)
    ds = open_store(args.store)
    if args.var not in ds.data_vars:
        sys.exit(f"{args.var} not in store. Data variables: {sorted(ds.data_vars)}")
    da = ds[args.var]
    ydim, xdim = [dim for dim in da.dims if dim != "time"]
    da = da.transpose("time", ydim, xdim)

    # Guard against mixing up the two indices: the store's latitude and longitude at each
    # (lat_idx, lon_idx) must match the cell list.
    lat_name, lon_name = find_latlon(ds, ydim, xdim)
    row_index = xr.DataArray(lat_idx, dims="cell")
    col_index = xr.DataArray(lon_idx, dims="cell")
    store_lat = ds[lat_name].transpose(ydim, xdim).isel({ydim: row_index, xdim: col_index})
    store_lon = ds[lon_name].transpose(ydim, xdim).isel({ydim: row_index, xdim: col_index})
    store_lat = np.asarray(store_lat.values, dtype=float)
    store_lon = np.asarray(store_lon.values, dtype=float)
    lat_diff = np.abs(store_lat - cells.latitude.to_numpy()).max()
    lon_diff = np.abs(store_lon - cells.longitude.to_numpy()).max()
    if lat_diff > 1e-3 or lon_diff > 1e-3:
        sys.exit(f"Grid coordinates do not match the cell list "
                 f"(max diff lat {lat_diff:.5f}, lon {lon_diff:.5f})")
    print(f"Coordinates match the cell list "
          f"(max diff lat {lat_diff:.1e}, lon {lon_diff:.1e} degrees)", flush=True)

    # Daily values at the pilot cells, as (time, cell). Only the parts of the store that hold
    # a pilot cell are downloaded.
    at_cells = da.isel({ydim: row_index, xdim: col_index})

    parts_dir = args.dir / "parts_daily"
    parts_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.first_water_year, args.last_water_year + 1))
    started = time.time()
    year_tables = []
    for n, water_year in enumerate(years, 1):
        part = parts_dir / f"wy_{water_year}.csv"
        start = pd.Timestamp(water_year - 1, 10, 1)
        end = pd.Timestamp(water_year, 9, 30)
        if part.exists():
            year_table = pd.read_csv(part, parse_dates=["date"])
            print(f"[{n}/{len(years)}] water year {water_year}: already done", flush=True)
        else:
            year_start = time.time()
            one_year = at_cells.sel(time=slice(start, end + pd.Timedelta(hours=23, minutes=59)))
            values = fetch(one_year, args.retries)
            dates = pd.DatetimeIndex(one_year["time"].values).normalize()
            year_table = pd.DataFrame(values, columns=[str(cell_id) for cell_id in cells.cell_id])
            year_table.insert(0, "date", dates)
            # write to a temporary name first, so a part file is never left half written
            tmp = part.with_suffix(".tmp")
            year_table.to_csv(tmp, index=False, float_format="%.9g", date_format="%Y-%m-%d")
            tmp.replace(part)
            print(f"[{n}/{len(years)}] water year {water_year}: {len(year_table)} days "
                  f"in {time.time() - year_start:.0f}s", flush=True)
        expected = pd.date_range(start, end, freq="D")
        got = pd.DatetimeIndex(year_table["date"])
        if len(got) != len(expected) or not (got == expected).all():
            missing = expected.difference(got)
            print(f"    WARNING: water year {water_year} has {len(got)} of {len(expected)} days; "
                  f"missing {[str(day.date()) for day in missing[:5]]}", flush=True)
        year_tables.append(year_table)

    daily = pd.concat(year_tables, ignore_index=True)
    daily_csv = args.dir / "pilot_daily_spduv10max.csv"
    daily.to_csv(daily_csv, index=False, float_format="%.9g", date_format="%Y-%m-%d")
    cell_columns = ["cell_id", "lat_idx", "lon_idx", "latitude", "longitude"]
    meta = {
        "script_version": SCRIPT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": args.store,
        "variable": args.var,
        "variable_attrs": {key: str(value) for key, value in da.attrs.items()},
        "first_date": str(pd.Timestamp(daily["date"].iloc[0]).date()),
        "last_date": str(pd.Timestamp(daily["date"].iloc[-1]).date()),
        "n_days": int(len(daily)),
        "n_missing_values": int(daily.drop(columns="date").isna().sum().sum()),
        "water_years": [years[0], years[-1]],
        "cells": cells[cell_columns].to_dict(orient="records"),
        "column_names": "cell_id",
    }
    (args.dir / "pilot_daily_spduv10max.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {(time.time() - started) / 60:.1f} min: "
          f"{len(daily)} days x {len(cells)} cells, "
          f"{meta['n_missing_values']} missing values\n  {daily_csv}", flush=True)


if __name__ == "__main__":
    main()
