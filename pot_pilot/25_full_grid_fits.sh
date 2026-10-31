#!/bin/bash
# Submits step 12, the full-grid run of 25_full_grid_fits.py, to DCC's SLURM queue.
#
#   bash 25_full_grid_fits.sh submit      download, fit and combine, each waiting for the last
#   bash 25_full_grid_fits.sh status      how far the run has got
#
# Run it from pot_pilot on a DCC login node. Before the first run:
#   - the Python environment named in SETUP needs xarray, zarr, s3fs, netCDF4, scipy and pandas
#   - data/hurdat2-1851-2024.txt has to be copied over (git doesn't track data/)
#   - CADENCE_DIR (see paths.py) can point at CADENCE's current gev_return_periods.nc, for
#     the comparison in combine
# Every stage resumes, so a stage that stops can be submitted again on its own, for example
#   sbatch --partition=common --array=3 --mem=24G --time=12:00:00 \
#       --export=ALL,STAGE=download 25_full_grid_fits.sh
#
# ROOT needs about 95 GB for the daily values. DCC purges old files from /work, so copy
# gev_return_periods.nc, gev_return_periods.csv and full_grid_summary.csv somewhere safe.

ROOT=${ROOT:-/work/$USER/cadence_full_grid}
PARTITION=${PARTITION:-common}
TASKS=${TASKS:-100}          # fit tasks, about 14,000 cells each
CPUS=${CPUS:-8}              # processes per fit task
SETUP=${SETUP:-"source ~/.bashrc && conda activate cadence"}    # the Python environment

set -eo pipefail    # not -u, which conda activate trips over
mkdir -p logs

if [ "${1:-}" = "submit" ]; then
    mkdir -p "$ROOT"
    common="--partition=$PARTITION"
    download=$(sbatch --parsable $common --job-name=fg_download --array=0-11 \
        --cpus-per-task=2 --mem=24G --time=12:00:00 --output=logs/%x_%A_%a.out \
        --export=ALL,STAGE=download,ROOT=$ROOT "$0")
    fit=$(sbatch --parsable $common --job-name=fg_fit --array=0-$((TASKS - 1)) \
        --cpus-per-task=$CPUS --mem=16G --time=6:00:00 --output=logs/%x_%A_%a.out \
        --dependency=afterok:$download --export=ALL,STAGE=fit,ROOT=$ROOT,TASKS=$TASKS,CPUS=$CPUS "$0")
    combine=$(sbatch --parsable $common --job-name=fg_combine --cpus-per-task=1 --mem=48G \
        --time=4:00:00 --output=logs/%x_%j.out --dependency=afterok:$fit \
        --export=ALL,STAGE=combine,ROOT=$ROOT,TASKS=$TASKS "$0")
    echo "download $download, fit $fit, combine $combine; outputs in $ROOT"
    exit 0
fi

if [ "${1:-}" = "status" ]; then
    echo "blocks downloaded: $(ls "$ROOT"/daily/block_??.npy 2>/dev/null | wc -l) of 12"
    echo "fit tasks done:    $(ls "$ROOT"/fits/task_*/done 2>/dev/null | wc -l) of $TASKS"
    [ -f "$ROOT/gev_return_periods.nc" ] && echo "combined:          $ROOT/gev_return_periods.nc"
    squeue -u "$USER" -n fg_download,fg_fit,fg_combine 2>/dev/null || true
    exit 0
fi

eval "$SETUP"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1    # one thread per process
case "${STAGE:-}" in
    download) python 25_full_grid_fits.py download --root "$ROOT" --block "$SLURM_ARRAY_TASK_ID" ;;
    fit)      python 25_full_grid_fits.py fit --root "$ROOT" --task "$SLURM_ARRAY_TASK_ID" \
                  --tasks "$TASKS" --processes "$CPUS" ;;
    combine)  python 25_full_grid_fits.py combine --root "$ROOT" --tasks "$TASKS" ;;
    *)        echo "usage: bash $0 submit | status"; exit 1 ;;
esac
