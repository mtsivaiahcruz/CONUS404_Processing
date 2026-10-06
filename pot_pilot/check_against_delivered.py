"""Compare the files the scripts just wrote (output/) with the delivered result files (data/).

For each result file this says whether the two copies are identical and, if not, how far
apart they are. Columns that come from a bootstrap (a random simulation) are reported
separately, because a different version of SciPy or NumPy can change the random draws a little.

Verdicts:
    IDENTICAL   every value is the same
    MATCH       fitted values agree within 0.001 (absolute, or relative for values above 1)
                and bootstrap columns within 10%; text columns may differ only where they
                depend on a bootstrap
    DIFFERENT   anything else
    MISSING     one of the two copies is not there

Usage:
    python check_against_delivered.py
"""
import sys

import numpy as np
import pandas as pd

from paths import DATA, OUT

# Each result file, with the columns that identify a row in it.
FILES = {
    "pilot_cells_snapped.csv": ["candidate"],
    "pilot_gev_check.csv": ["candidate"],
    "pilot_locations_selected.csv": ["cell_id"],
    "pilot_gev_parameters.csv": ["cell_id"],
    "pilot_gev_baseline.csv": ["cell_id", "return_period"],
    "pilot_cadence_vs_baseline.csv": ["cell_id", "return_period"],
    "cadence_gumbel_branch_cells.csv": ["lat_idx", "lon_idx"],
    "pilot_pot_results.csv": ["cell_id", "threshold_pct", "separation_days"],
    "pilot_pot_events.csv": ["cell_id", "threshold_pct", "separation_days", "peak_date"],
    "pilot_gev_vs_pot.csv": ["cell_id", "threshold_pct", "separation_days", "return_period"],
    "pilot_results_table.csv": ["cell_id", "method", "threshold_pct", "separation_days"],
}
# A column is treated as depending on a bootstrap, and compared with the looser tolerance, if
# its name has one of these words in it.
BOOT_WORDS = ("boot", "years_", "_years", "narrower", "flags")
FIT_TOLERANCE = 1e-3
BOOT_TOLERANCE = 0.10


def is_boot(column):
    if column in ("valid_years", "n_years", "n_water_years"):
        return False    # these three have "years" in the name but are plain counts
    return any(word in column for word in BOOT_WORDS)


def largest_difference(new, old):
    """Largest gap between two numeric columns.

    The gap is the plain difference where the old value is below 1, and the difference as a
    share of the old value above that.
    """
    if np.isfinite(new).any():
        diff = np.nanmax(np.abs(new - old) / np.maximum(1.0, np.abs(old)))
    else:
        diff = 0.0
    # Infinite values (a fit with no upper bound, for example) must sit in the same places.
    infinite = np.isinf(new) | np.isinf(old)
    if infinite.any():
        diff = 0.0 if np.array_equal(new[infinite], old[infinite]) else np.inf
        finite = np.isfinite(new) & np.isfinite(old)
        if finite.any():
            gap = np.abs(new[finite] - old[finite]) / np.maximum(1.0, np.abs(old[finite]))
            diff = max(diff, float(np.max(gap)))
    return diff


def compare(name, keys):
    """Compare one result file. Returns the verdict and a line of detail."""
    new_path = OUT / name
    old_path = DATA / name
    if not new_path.exists() or not old_path.exists():
        return "MISSING", f"{'output' if not new_path.exists() else 'delivered'} copy not found"
    new = pd.read_csv(new_path)
    old = pd.read_csv(old_path)
    if list(new.columns) != list(old.columns):
        return "DIFFERENT", f"columns differ: {sorted(set(new.columns) ^ set(old.columns))}"
    if len(new) != len(old):
        return "DIFFERENT", f"row counts differ: {len(new)} vs {len(old)}"
    # put the rows of both copies in the same order before comparing
    new = new.sort_values(keys).reset_index(drop=True)
    old = old.sort_values(keys).reset_index(drop=True)
    if not new[keys].equals(old[keys]):
        return "DIFFERENT", "row keys differ"
    if new.equals(old):
        return "IDENTICAL", f"{len(new)} rows x {new.shape[1]} columns"

    worst = {"fit": 0.0, "boot": 0.0}
    other_differences = []      # differences that are not allowed
    boot_text_differences = []  # text that differs only because a bootstrap value differs
    for column in new.columns:
        a, b = new[column], old[column]
        both_numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        if both_numeric and a.dtype != bool:
            a, b = a.to_numpy(float), b.to_numpy(float)
            if not (np.isnan(a) == np.isnan(b)).all():
                other_differences.append(column)    # empty cells in different places
                continue
            kind = "boot" if is_boot(column) else "fit"
            worst[kind] = max(worst[kind], float(largest_difference(a, b)))
        else:
            n_different = int((a.fillna("").astype(str) != b.fillna("").astype(str)).sum())
            if n_different:
                where = boot_text_differences if is_boot(column) else other_differences
                where.append(f"{column} ({n_different} rows)")

    detail = (f"largest difference: fitted values {worst['fit']:.1e}, "
              f"bootstrap values {worst['boot']:.1e}")
    if boot_text_differences:
        detail += f"; bootstrap-dependent text differs in {', '.join(boot_text_differences)}"
    if other_differences:
        return "DIFFERENT", detail + f"; other columns differ: {', '.join(other_differences)}"
    if worst["fit"] <= FIT_TOLERANCE and worst["boot"] <= BOOT_TOLERANCE:
        return "MATCH", detail
    return "DIFFERENT", detail


def main():
    print(f"new outputs : {OUT}\ndelivered   : {DATA}\n")
    verdicts = []
    for name, keys in FILES.items():
        verdict, detail = compare(name, keys)
        verdicts.append(verdict)
        print(f"{verdict:<10} {name:<34} {detail}")
    n_identical = verdicts.count("IDENTICAL")
    n_match = verdicts.count("MATCH")
    print(f"\n{n_identical} of {len(FILES)} files identical; {n_match} match within tolerance; "
          f"{len(FILES) - n_identical - n_match} different or missing")
    all_ok = all(verdict in ("IDENTICAL", "MATCH") for verdict in verdicts)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
