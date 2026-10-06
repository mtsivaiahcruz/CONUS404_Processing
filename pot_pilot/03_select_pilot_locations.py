"""Step 2c. Score every grid cell on record length and completeness; write the pilot table.

Every grid cell is scored over water years 1980-2024:
    valid_years  = number of water years with a value on every day the store has
    completeness = days with a value / days the store has
The store has every calendar day of these water years, so this is also the share of expected
days.
The 13 screened pilot cells are then written out with their paired latitude and longitude.

Reads:
    data/spduv10max_water_1980-2024.nc, data/conus404_grid.nc
    year_raw_data (to cross-check the coordinates)
    output/pilot_cells_snapped.csv (steps 01 and 02)
Writes:
    output/pilot_locations_selected.csv
"""
import numpy as np
import pandas as pd

from h5io import open_h5
from paths import GRID_NC, OUT, WATER_NC, require, year_raw

require(WATER_NC, GRID_NC, year_raw(1980), OUT / "pilot_cells_snapped.csv")

# Score every cell on the grid.
water_file = open_h5(WATER_NC)
n_valid = water_file["n_valid"][:].astype(np.int32)           # valid days, per year and cell
days_present = water_file["n_days_present"][:].astype(int)    # days in the store, per year
years = water_file["year"][:]
N_ROWS, N_COLS = n_valid.shape[1:]
valid_years = (n_valid == days_present[:, None, None]).sum(0)
completeness = n_valid.sum(0) / days_present.sum()
print(f"grid cells: {N_ROWS * N_COLS} | "
      f"with {len(years)} complete years: {int((valid_years == len(years)).sum())} | "
      f"at 100% completeness: {int((completeness == 1).sum())}")

grid_file = open_h5(GRID_NC)
grid_lat, grid_lon = grid_file["latitude"][:], grid_file["longitude"][:]
pipeline_file = open_h5(year_raw(1980))
pipeline_lat, pipeline_lon = pipeline_file["lat_2d"][:], pipeline_file["lon_2d"][:]

# Build the pilot table.
snapped = pd.read_csv(OUT / "pilot_cells_snapped.csv")
lat_idx, lon_idx = snapped.lat_idx.to_numpy(), snapped.lon_idx.to_numpy()
table = pd.DataFrame({
    "cell_id": lat_idx * N_COLS + lon_idx, "lat_idx": lat_idx, "lon_idx": lon_idx,
    # latitude and longitude are read at the same (lat_idx, lon_idx), so the pair stays together
    "latitude": grid_lat[lat_idx, lon_idx].round(5),
    "longitude": grid_lon[lat_idx, lon_idx].round(5),
    "candidate": snapped.candidate, "region": snapped.region,
    "valid_years": valid_years[lat_idx, lon_idx],
    "first_year": years[0], "last_year": years[-1],
    "valid_days": n_valid.sum(0)[lat_idx, lon_idx], "expected_days": days_present.sum(),
    "completeness_pct": (100 * completeness[lat_idx, lon_idx]).round(2),
    "gev_converged": snapped.gev_converged,
})

# The coordinates must agree with the GEV pipeline's own input files.
assert np.allclose(table.latitude, pipeline_lat[lat_idx, lon_idx], atol=1e-4)
assert np.allclose(table.longitude, pipeline_lon[lat_idx, lon_idx], atol=1e-4)
# cell_id must be unique and must convert back to the two indices.
assert table.cell_id.is_unique
assert (table.cell_id // N_COLS == table.lat_idx).all()
assert (table.cell_id % N_COLS == table.lon_idx).all()

# How far apart are the two closest pilot cells? (great-circle distance, haversine formula)
EARTH_RADIUS_KM = 6371.0088
lat_rad, lon_rad = np.radians(table.latitude.values), np.radians(table.longitude.values)
haversine = (
    np.sin((lat_rad[:, None] - lat_rad) / 2) ** 2
    + np.cos(lat_rad[:, None]) * np.cos(lat_rad) * np.sin((lon_rad[:, None] - lon_rad) / 2) ** 2
)
distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))
np.fill_diagonal(distance, np.inf)    # ignore each cell's distance to itself
closest = np.unravel_index(distance.argmin(), distance.shape)
print(f"closest pair: {table.candidate[closest[0]]} - {table.candidate[closest[1]]}, "
      f"{distance.min():.0f} km apart")

# West to east.
table = table.sort_values("longitude").reset_index(drop=True)
table.to_csv(OUT / "pilot_locations_selected.csv", index=False)
print(table.to_string(index=False))
