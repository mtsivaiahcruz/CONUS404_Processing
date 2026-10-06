"""Where the pilot scripts find their input files and write their outputs.

Folder layout. CADENCE is the folder that holds the CONUS404_Processing repository.

    CADENCE/
        gev_parameters.h5         CADENCE's current GEV parameters (from Andrew)
        gev_return_periods.nc     CADENCE's current return periods (from Andrew)
        CONUS404_Processing/
            year_raw_data/        the pipeline's annual-maximum files, one per water year
            pot_pilot/            this folder
                natural_earth/    coastline and state outline files
                data/             extraction outputs and the delivered result files
                output/           everything these scripts write
                    work/         intermediate files

The scripts read from data/ and write to output/, so the delivered result files in data/ are
never overwritten. Git tracks neither folder: the repository's .gitignore leaves out every
folder named output, and pot_pilot/.gitignore leaves out data because two of its files are
over 230 MB each.

Two environment variables change the defaults:
    CADENCE_DIR     the folder that holds the two GEV files, if they are somewhere else
    ANALYSIS_OUT    a folder to write the outputs to, instead of output/
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

if os.environ.get("CADENCE_DIR"):
    CADENCE = Path(os.environ["CADENCE_DIR"]).resolve()
else:
    CADENCE = REPO.parent

if os.environ.get("ANALYSIS_OUT"):
    OUT = Path(os.environ["ANALYSIS_OUT"]).resolve()
else:
    OUT = HERE / "output"

DATA = HERE / "data"
WORK = OUT / "work"
WORK.mkdir(parents=True, exist_ok=True)

# Input files
GRID_NC = DATA / "conus404_grid.nc"
CAL_NC = DATA / "spduv10max_calendar_1980-2023.nc"
WATER_NC = DATA / "spduv10max_water_1980-2024.nc"
MANIFEST = DATA / "manifest.json"
DAILY_CSV = DATA / "pilot_daily_spduv10max.csv"
YEAR_RAW = REPO / "year_raw_data"
CADENCE_H5 = CADENCE / "gev_parameters.h5"
CADENCE_NC = CADENCE / "gev_return_periods.nc"
NATURAL_EARTH = HERE / "natural_earth"

# Water years run October to September. Water year 1980 is October 1979 to September 1980.
FIRST_WY = 1980
LAST_WY = 2024
WATER_YEARS = list(range(FIRST_WY, LAST_WY + 1))
N_YEARS = len(WATER_YEARS)    # 45

# The pipeline's gust factor, from the Durst curve. The pipeline treats the native wind speed as
# a 20-second average and multiplies it by 1.52 / 1.36 = 1.1176 to get a 3-second gust.
GUST_FACTOR = round(1.52 / 1.36, 4)


def year_raw(year):
    """The pipeline's annual-maximum file for one water year."""
    return YEAR_RAW / f"water_year_{year}" / f"wrfxtrm_d01_max_spduv10max_{year}.nc"


def require(*paths):
    """Stop with a clear message if any of the input files is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing input file(s):\n  " + "\n  ".join(missing))
