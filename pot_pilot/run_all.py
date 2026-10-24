"""Run every step in order, then compare the new outputs with the delivered result files.

    python run_all.py        every step (under an hour on my laptop)
    python run_all.py 8      start at step 08 (the earlier outputs must already exist)

Everything is written to pot_pilot/output. The delivered result files in pot_pilot/data are
only read, never changed.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_snap_and_screen.py",
    "02_gev_convergence_check.py",
    "03_select_pilot_locations.py",
    "04_gev_baseline.py",
    "05_gev_bootstrap.py",
    "06_assemble_baseline.py",
    "07_compare_with_cadence.py",
    "08_pot_tests.py",
    "09_pot_bootstrap.py",
    "10_gev_vs_pot.py",
    "11_summary_numbers.py",
    "12_results_table.py",
    "13_make_plots.py",
    "14_tail_priors.py",
    "15_bayes_fits.py",
    "16_prior_sensitivity.py",
    "17_simulation_study.py",
    "check_against_delivered.py",
]


def main():
    first_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_all = time.time()
    for step in STEPS:
        # the numbered scripts can be skipped; the final check always runs
        if step[:2].isdigit() and int(step[:2]) < first_step:
            continue
        print(f"\n{'=' * 78}\n{step}\n{'=' * 78}", flush=True)
        start = time.time()
        exit_code = subprocess.run([sys.executable, str(HERE / step)], cwd=HERE).returncode
        print(f"[{step}: {time.time() - start:.0f}s, exit code {exit_code}]", flush=True)
        if exit_code != 0:
            print(f"\nStopped: {step} failed.")
            sys.exit(exit_code)
    print(f"\nAll steps finished in {(time.time() - start_all) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
