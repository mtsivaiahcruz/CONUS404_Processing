"""Step 2a. Match the 13 candidate cities to CONUS404 4 km grid cells and check each cell.

For each city this finds the nearest grid cell and checks that the cell is usable:
  - the nearest cell centre is found on a sphere, the way CADENCE's attach_nearest_wind_grid
    does it, and confirmed with a second method (WRF's Lambert conformal projection)
  - no two cities share a cell
  - daily data quality at the cell: missing, negative, zero, repeated or above 80 m/s
  - how much of the cell's 4 km footprint is land (Natural Earth 10m land minus lakes)
  - whether the cell's winds look like nearby land cells or nearby water cells

Reads:
    data/conus404_grid.nc, data/spduv10max_calendar_1980-2023.nc, data/manifest.json
    natural_earth/ne_10m_land.geojson, ne_10m_lakes.geojson, ne_10m_lakes_north_america.geojson
Writes:
    output/pilot_cells_snapped.csv                 the screening columns (step 02 adds more)
    output/work/pilot_cells_screen.csv             every screening number, unrounded
    output/work/pilot_cells_landwater_check.csv
"""
import json
import warnings

import numpy as np
import pandas as pd

from h5io import open_h5
from paths import CAL_NC, GRID_NC, MANIFEST, NATURAL_EARTH, OUT, WORK, require

warnings.filterwarnings("ignore")
NE_FILES = ["ne_10m_land.geojson", "ne_10m_lakes.geojson", "ne_10m_lakes_north_america.geojson"]
require(GRID_NC, CAL_NC, MANIFEST, *(NATURAL_EARTH / name for name in NE_FILES))

cities = pd.DataFrame(
    [
        ("Seattle, WA", "Pacific Northwest", 47.6062, -122.3321),
        ("Los Angeles, CA", "Pacific Southwest", 34.0522, -118.2437),
        ("Phoenix, AZ", "Desert Southwest", 33.4484, -112.0740),
        ("Denver, CO", "Rockies / Front Range", 39.7392, -104.9903),
        ("Rapid City, SD", "Northern High Plains", 44.0805, -103.2310),
        ("Oklahoma City, OK", "Southern Plains", 35.4676, -97.5164),
        ("Minneapolis, MN", "Upper Midwest", 44.9778, -93.2650),
        ("Chicago, IL", "Great Lakes", 41.8781, -87.6298),
        ("Houston, TX", "Texas Gulf Coast", 29.7604, -95.3698),
        ("New Orleans, LA", "Lower Mississippi / Gulf Coast", 29.9511, -90.0715),
        ("Atlanta, GA", "Southeast interior", 33.7490, -84.3880),
        ("Miami, FL", "South Florida", 25.7617, -80.1918),
        ("Boston, MA", "Northeast coast", 42.3601, -71.0589),
    ],
    columns=["candidate", "region", "lat", "lon"],
)

print("reading the grid ...", flush=True)
grid_file = open_h5(GRID_NC)
GRID_LAT = grid_file["latitude"][:]
GRID_LON = grid_file["longitude"][:]
N_ROWS, N_COLS = GRID_LAT.shape
EARTH_RADIUS_KM = 6371.0088


def unit_vector(lat, lon):
    """The point on a sphere of radius 1 at this latitude and longitude, as (x, y, z)."""
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)],
        -1,
    )


GRID_XYZ = unit_vector(GRID_LAT, GRID_LON)


def nearest_cell(lat, lon):
    """Row and column of the grid cell whose centre is nearest, and the distance to it in km."""
    chord = np.linalg.norm(GRID_XYZ - unit_vector(lat, lon), axis=-1)
    row, col = np.unravel_index(np.argmin(chord), chord.shape)
    km = float(2 * EARTH_RADIUS_KM * np.arcsin(min(chord[row, col] / 2, 1)))
    return int(row), int(col), km


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points."""
    chord = np.linalg.norm(unit_vector(lat1, lon1) - unit_vector(lat2, lon2))
    return float(2 * EARTH_RADIUS_KM * np.arcsin(chord / 2))


# Land and water polygons.
# Only polygons that reach into this box around the lower 48 states are kept:
# west, south, east, north in degrees.
BBOX = (-127, 22, -65, 51)


class RingSet:
    """A set of closed polygon outlines (rings) with a fast test for points inside them.

    The test is ray casting: draw a line from the point towards the east and count how many
    edges of the ring it crosses. An odd count means the point is inside. To keep it fast, a
    ring is only tested if the points fall inside its bounding box, and only the edges that
    reach the points' latitudes are used.
    """

    def __init__(self, rings):
        self.rings = [np.asarray(ring, float) for ring in rings]
        # bounding box of each ring: lowest x, lowest y, highest x, highest y
        self.boxes = np.array(
            [[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in self.rings]
        ).reshape(-1, 4)
        # lowest and highest latitude of each edge of each ring
        self.edge_low = [np.minimum(r[:-1, 1], r[1:, 1]) for r in self.rings]
        self.edge_high = [np.maximum(r[:-1, 1], r[1:, 1]) for r in self.rings]

    def contains_any(self, px, py):
        """True for each point (px = longitude, py = latitude) inside at least one ring."""
        is_inside = np.zeros(px.shape, bool)
        if not self.rings:
            return is_inside
        candidates = np.flatnonzero(
            (self.boxes[:, 0] <= px.max()) & (self.boxes[:, 2] >= px.min())
            & (self.boxes[:, 1] <= py.max()) & (self.boxes[:, 3] >= py.min())
        )
        for k in candidates:
            ring = self.rings[k]
            near = (self.edge_high[k] > py.min()) & (self.edge_low[k] <= py.max())
            # each edge runs from (x1, y1) to (x2, y2)
            x1, y1 = ring[:-1, 0][near], ring[:-1, 1][near]
            x2, y2 = ring[1:, 0][near], ring[1:, 1][near]
            if x1.size == 0:
                continue
            point_x, point_y = px[:, None], py[:, None]
            # an edge can only be crossed if one end is above the point and the other is not
            straddles = (y1[None, :] > point_y) != (y2[None, :] > point_y)
            with np.errstate(divide="ignore", invalid="ignore"):
                # longitude at which the edge crosses the point's latitude
                crossing_x = (
                    (x2 - x1)[None, :] * (point_y - y1[None, :]) / (y2 - y1)[None, :] + x1[None, :]
                )
            crossings = np.count_nonzero(straddles & (point_x < crossing_x), axis=1)
            is_inside |= (crossings % 2) == 1
        return is_inside


def load_rings(filename):
    """Read a Natural Earth file. Returns its outer rings and its holes as two RingSets."""
    with open(NATURAL_EARTH / filename, encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    outer, holes = [], []
    for feature in features:
        geometry = feature["geometry"]
        if geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            polygons = [geometry["coordinates"]]
        for polygon in polygons:
            ring = np.asarray(polygon[0])
            outside_box = (
                ring[:, 0].max() < BBOX[0] or ring[:, 0].min() > BBOX[2]
                or ring[:, 1].max() < BBOX[1] or ring[:, 1].min() > BBOX[3]
            )
            if outside_box:
                continue
            outer.append(ring)
            holes += [np.asarray(hole) for hole in polygon[1:]]
    assert all(np.array_equal(ring[0], ring[-1]) for ring in outer + holes), "ring not closed"
    return RingSet(outer), RingSet(holes)


land_outer, land_holes = load_rings(NE_FILES[0])
lake_outer, lake_holes = load_rings(NE_FILES[1])
na_lake_outer, na_lake_holes = load_rings(NE_FILES[2])


def inside(px, py, outer, holes):
    """Inside an outer ring and not inside a hole."""
    return outer.contains_any(px, py) & ~holes.contains_any(px, py)


def is_land(lat, lon):
    """True where a point is on land: inside the land polygons and not in a lake."""
    px, py = np.ravel(lon).astype(float), np.ravel(lat).astype(float)
    land = inside(px, py, land_outer, land_holes)
    lake = inside(px, py, lake_outer, lake_holes) | inside(px, py, na_lake_outer, na_lake_holes)
    return (land & ~lake).reshape(np.shape(lat))


def cell_samples(row, col, n=11):
    """Latitude and longitude of an n x n set of points spread across one cell's footprint.

    The footprint runs half a cell either side of the centre. Positions between cell centres
    are found by bilinear interpolation of the grid's latitude and longitude.
    """
    offsets = np.linspace(-0.5, 0.5, n)
    frac_row, frac_col = np.meshgrid(row + offsets, col + offsets, indexing="ij")
    row0 = np.clip(np.floor(frac_row).astype(int), 0, N_ROWS - 2)
    col0 = np.clip(np.floor(frac_col).astype(int), 0, N_COLS - 2)
    d_row, d_col = frac_row - row0, frac_col - col0

    def interpolate(field):
        return (field[row0, col0] * (1 - d_row) * (1 - d_col)
                + field[row0 + 1, col0] * d_row * (1 - d_col)
                + field[row0, col0 + 1] * (1 - d_row) * d_col
                + field[row0 + 1, col0 + 1] * d_row * d_col)

    return interpolate(GRID_LAT), interpolate(GRID_LON)


def land_fraction(row, col, n=11):
    """Share of a cell's footprint that is land, from n x n sample points."""
    lat, lon = cell_samples(row, col, n)
    return float(is_land(lat, lon).mean())


# Daily data quality counts, calendar years 1980-2023, from the full-grid extraction.
print("reading the daily quality counts (8 large arrays; the slowest part of this step) ...",
      flush=True)
quality_file = open_h5(CAL_NC)
years = quality_file["year"][:]
QA_NAMES = ["annual_max", "n_valid", "n_missing", "n_negative", "n_zero",
            "n_over_limit", "n_repeat", "longest_repeat"]
QA = {name: quality_file[name][:] for name in QA_NAMES}
# days that are missing from the store altogether, for every cell
store_missing_days = int(
    (quality_file["n_days_expected"][:] - quality_file["n_days_present"][:]).sum()
)
ANNUAL_MAX = QA["annual_max"]
MEDIAN_ANNUAL_MAX = np.nanmedian(ANNUAL_MAX, axis=0)


def cell_quality(row, col):
    """Data quality numbers for one cell, over the 44 calendar years 1980-2023."""
    annual_max = ANNUAL_MAX[:, row, col]
    # The cell's eight neighbours. The cell itself is blanked out: it sits at index 1 of the
    # 3 x 3 window, or at index 0 if the cell is on the edge of the grid.
    neighbors = MEDIAN_ANNUAL_MAX[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].copy()
    neighbors[min(row, 1), min(col, 1)] = np.nan
    return {
        "missing_days": int(QA["n_missing"][:, row, col].sum()) + store_missing_days,
        "valid_days": int(QA["n_valid"][:, row, col].sum()),
        "negative_days": int(QA["n_negative"][:, row, col].sum()),
        "zero_days": int(QA["n_zero"][:, row, col].sum()),
        "over_80ms_days": int(QA["n_over_limit"][:, row, col].sum()),
        "repeat_days": int(QA["n_repeat"][:, row, col].sum()),
        "longest_repeat": int(QA["longest_repeat"][:, row, col].max()),
        "years_with_max": int(np.isfinite(annual_max).sum()),
        "annmax_min_ms": float(np.nanmin(annual_max)),
        "annmax_median_ms": float(np.nanmedian(annual_max)),
        "annmax_max_ms": float(np.nanmax(annual_max)),
        "annmax_max_year": int(years[np.nanargmax(annual_max)]),
        "neighbor_ratio": float(MEDIAN_ANNUAL_MAX[row, col] / np.nanmedian(neighbors)),
    }


def nearest_land_cell(lat, lon, row, col, window=6, min_fraction=0.9):
    """The closest cell to the city, within `window` cells, that is mostly land."""
    best = None
    for near_row in range(row - window, row + window + 1):
        for near_col in range(col - window, col + window + 1):
            if not (0 <= near_row < N_ROWS and 0 <= near_col < N_COLS):
                continue
            near_lat, near_lon = GRID_LAT[near_row, near_col], GRID_LON[near_row, near_col]
            km = distance_km(lat, lon, near_lat, near_lon)
            if best is not None and km >= best[2]:
                continue
            if land_fraction(near_row, near_col) >= min_fraction:
                best = (near_row, near_col, km)
    return best


print("snapping and screening the 13 cities ...", flush=True)
rows = []
for city in cities.itertuples():
    row, col, km = nearest_cell(city.lat, city.lon)
    fraction = land_fraction(row, col)
    record = {
        "candidate": city.candidate, "region": city.region, "lat": city.lat, "lon": city.lon,
        "lat_idx": row, "lon_idx": col,
        "cell_lat": GRID_LAT[row, col], "cell_lon": GRID_LON[row, col],
        "snap_km": km, "city_point_on_land": bool(is_land(city.lat, city.lon)),
        "cell_land_fraction": fraction, **cell_quality(row, col),
    }
    # If much of the cell is water by the coastline data, also record the nearest cell that
    # is mostly land, as an alternative.
    if fraction < 0.9:
        alternative = nearest_land_cell(city.lat, city.lon, row, col)
        if alternative:
            alt_row, alt_col, alt_km = alternative
            alt_quality = cell_quality(alt_row, alt_col)
            record.update(
                alt_lat_idx=alt_row, alt_lon_idx=alt_col,
                alt_cell_lat=GRID_LAT[alt_row, alt_col], alt_cell_lon=GRID_LON[alt_row, alt_col],
                alt_km_from_city=alt_km, alt_land_fraction=land_fraction(alt_row, alt_col),
                alt_missing_days=alt_quality["missing_days"],
                alt_annmax_median_ms=alt_quality["annmax_median_ms"],
                alt_neighbor_ratio=alt_quality["neighbor_ratio"],
                alt_repeat_days=alt_quality["repeat_days"],
            )
    rows.append(record)
screen = pd.DataFrame(rows)
screen["duplicate_cell"] = screen.duplicated(["lat_idx", "lon_idx"], keep=False)
screen.to_csv(WORK / "pilot_cells_screen.csv", index=False)
print(f"grid-wide missing daily values: {int(QA['n_missing'].sum())}; "
      f"cities sharing a cell: {int(screen.duplicate_cell.sum())}")

# Second way of finding the cell: WRF's Lambert conformal projection.
# In the projection the grid is a regular 4 km lattice, so a point's cell follows directly
# from its projected x and y.
with open(MANIFEST, encoding="utf-8") as fh:
    attrs = json.load(fh)["dataset_attrs"]
WRF_RADIUS_M = 6370000.0
TRUELAT1 = np.radians(float(attrs["TRUELAT1"]))
TRUELAT2 = np.radians(float(attrs["TRUELAT2"]))
STAND_LON = float(attrs["STAND_LON"])
# the two constants of the projection (standard Lambert conformal formulas)
CONE = (np.log(np.cos(TRUELAT1) / np.cos(TRUELAT2))
        / np.log(np.tan(np.pi / 4 + TRUELAT2 / 2) / np.tan(np.pi / 4 + TRUELAT1 / 2)))
F = np.cos(TRUELAT1) * np.tan(np.pi / 4 + TRUELAT1 / 2) ** CONE / CONE


def project(lat, lon):
    """Projected x and y, in metres, of a latitude and longitude."""
    rho = WRF_RADIUS_M * F / np.tan(np.pi / 4 + np.radians(lat) / 2) ** CONE
    theta = CONE * np.radians(np.asarray(lon) - STAND_LON)
    return rho * np.sin(theta), -rho * np.cos(theta)


GRID_X, GRID_Y = project(GRID_LAT, GRID_LON)
cell_size_m = float(attrs["DX"])
print(f"grid spacing in the projection: dx {np.diff(GRID_X, axis=1).mean():.1f} m, "
      f"dy {np.diff(GRID_Y, axis=0).mean():.1f} m")
n_agree = 0
for city in screen.itertuples():
    x, y = project(city.lat, city.lon)
    col = int(np.floor((x - GRID_X[0, 0]) / cell_size_m + 0.5))
    row = int(np.floor((y - GRID_Y[0, 0]) / cell_size_m + 0.5))
    n_agree += (row, col) == (city.lat_idx, city.lon_idx)
print(f"projection method picks the same cell as nearest-centre: {n_agree}/{len(screen)}")
assert n_agree == len(screen)

# Do the cell's winds look like land or like water?
# Compare the cell's median annual maximum with the all-land and all-water cells within
# 8 cells of it.
print("comparing each cell with nearby land and water cells ...", flush=True)
comparisons = []
for city in screen.itertuples():
    row, col = city.lat_idx, city.lon_idx
    land, water = [], []
    for near_row in range(row - 8, row + 9):
        for near_col in range(col - 8, col + 9):
            if (near_row, near_col) == (row, col):
                continue
            fraction = land_fraction(near_row, near_col, n=5)
            if fraction == 1:
                land.append(MEDIAN_ANNUAL_MAX[near_row, near_col])
            elif fraction == 0:
                water.append(MEDIAN_ANNUAL_MAX[near_row, near_col])
    land, water = np.array(land), np.array(water)
    cell_median = MEDIAN_ANNUAL_MAX[row, col]
    comparisons.append({
        "candidate": city.candidate, "cell_med": cell_median,
        "land_med": np.median(land) if land.size else np.nan,
        "water_med": np.median(water) if water.size else np.nan,
        "n_land": land.size, "n_water": water.size,
        # share of the nearby land cells that are calmer than this cell
        "cell_pct_in_land": (land < cell_median).mean() * 100 if land.size else np.nan,
    })
landwater = pd.DataFrame(comparisons)
landwater.to_csv(WORK / "pilot_cells_landwater_check.csv", index=False)
print(landwater.round(2).to_string(index=False))

# Build pilot_cells_snapped.csv from the screening columns.
merged = screen.merge(landwater[["candidate", "land_med", "water_med", "cell_pct_in_land"]],
                      on="candidate")
# These notes were written by hand after looking at the screening numbers. They are not
# computed by this script.
NOTES = {
    "Seattle, WA": (
        "Cell includes Lake Union/Elliott Bay per coastline data, but winds behave like land"
    ),
    "Chicago, IL": (
        "~20% Lake Michigan in footprint; windier than 94% of nearby land cells "
        "(lake exposure), far below lake cells"
    ),
    "Boston, MA": (
        "Harbor edge; windier than 98% of nearby land cells (ocean exposure), "
        "far below water cells"
    ),
    "Miami, FL": (
        "Coastline data puts much of the footprint in Biscayne Bay, but winds are land-like, "
        "so the model treats it as land"
    ),
    "New Orleans, LA": "Land-like winds despite nearby lake/Gulf cells",
    "Rapid City, SD": (
        "Valley cell; ~15% calmer than its immediate neighbours (Black Hills terrain)"
    ),
    "Los Angeles, CA": "Calmer than most land nearby (mountains in window are much windier)",
}
URBAN = {"Phoenix, AZ", "Denver, CO", "Oklahoma City, OK", "Minneapolis, MN", "Houston, TX"}
URBAN_NOTE = "Urban cell; calmer than surrounding land (expected urban roughness)"


def note_for(candidate):
    if candidate in NOTES:
        return NOTES[candidate]
    if candidate in URBAN:
        return URBAN_NOTE
    return ""


snapped = pd.DataFrame({
    "candidate": merged.candidate, "region": merged.region,
    "city_lat": merged.lat, "city_lon": merged.lon,
    "lat_idx": merged.lat_idx, "lon_idx": merged.lon_idx,
    "cell_lat": merged.cell_lat.round(4), "cell_lon": merged.cell_lon.round(4),
    "snap_km": merged.snap_km.round(2), "years": merged.years_with_max,
    "valid_days": merged.valid_days, "missing_days": merged.missing_days,
    "negative_days": merged.negative_days, "repeat_days": merged.repeat_days,
    "over_80ms_days": merged.over_80ms_days,
    "annmax_median_ms": merged.annmax_median_ms.round(2),
    "annmax_max_ms": merged.annmax_max_ms.round(2), "annmax_max_year": merged.annmax_max_year,
    "coastline_land_fraction": merged.cell_land_fraction.round(2),
    "nearby_land_median_ms": merged.land_med.round(2),
    "nearby_water_median_ms": merged.water_med.round(2),
    # all 13 cells pass the land-or-water check below, so they all get the same two values
    "wind_behaves_like": "land", "status": "keep",
    "alt_land_lat_idx": merged.alt_lat_idx.astype("Int64"),
    "alt_land_lon_idx": merged.alt_lon_idx.astype("Int64"),
    "alt_land_km_from_city": merged.alt_km_from_city.round(2),
    "note": [note_for(candidate) for candidate in merged.candidate],
})
assert not snapped.duplicated(["lat_idx", "lon_idx"]).any(), "two cities share a cell"
assert (snapped.missing_days == 0).all(), "a pilot cell has missing days"
# Open water is windier than land. Each cell's median annual maximum must be below the median
# of the nearby all-water cells, which is what a land cell looks like.
assert (merged.annmax_median_ms < merged.water_med.fillna(np.inf)).all()
snapped.to_csv(OUT / "pilot_cells_snapped.csv", index=False)
print(f"wrote {OUT / 'pilot_cells_snapped.csv'}")
