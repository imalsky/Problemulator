#!/usr/bin/env bash

# Single-task SLURM script that generates the shared synthetic-profiles HDF5
# used by the picaso job array (gen.sh).
#
# Submission flow from the repo root:
#     GEN=$(sbatch --parsable Problemulator/supercomputer_cmds/gen_profiles.sh)
#     sbatch --dependency=afterok:$GEN --kill-on-invalid-dep=yes \
#         Problemulator/supercomputer_cmds/gen.sh
#
# This script writes:
#   - gen_data/output/synthetic_profiles_5M.h5
#   - gen_data/output/synthetic_profiles_5M.h5.done  (completion sentinel)
# gen.sh hard-fails if either is missing, so the array tasks can rely on the
# input being fully written before they start. This replaces the in-array
# flock-based generation that did not provide cross-node mutual exclusion on
# GPFS and produced torn / partially-written HDF5 files.

#SBATCH -J picaso_gen_profiles
#SBATCH -o picaso_gen_profiles.o%j
#SBATCH -e picaso_gen_profiles.e%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH -t 6:00:00
#SBATCH --hint=nomultithread
#SBATCH -p compute
#SBATCH -A exoweather
#SBATCH --clusters=edge

set -euo pipefail

resolve_project_root() {
    local start_dir="$1"
    if [[ -d "$start_dir/gen_data" && -d "$start_dir/Problemulator" ]]; then
        cd "$start_dir" && pwd
    elif [[ -d "$start_dir/src" && -d "$start_dir/config" && -d "$start_dir/../gen_data" ]]; then
        cd "$start_dir/.." && pwd
    elif [[ -d "$start_dir/../gen_data" && -d "$start_dir/../Problemulator" ]]; then
        cd "$start_dir/.." && pwd
    elif [[ -d "$start_dir/../src" && -d "$start_dir/../config" && -d "$start_dir/../../gen_data" ]]; then
        cd "$start_dir/../.." && pwd
    else
        echo "Could not resolve project root from: $start_dir" >&2
        return 1
    fi
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$(resolve_project_root "$SLURM_SUBMIT_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CONDA_ENV="${CONDA_ENV:-nn}"
INPUT_PATH="${INPUT_PATH:-output/synthetic_profiles_5M.h5}"
PROFILE_CONFIG="${PROFILE_CONFIG:-config/paper_comparable_profiles.jsonc}"
TOTAL_PROFILES="${TOTAL_PROFILES:-5000000}"
PROFILE_SEED="${PROFILE_SEED:-20260423}"

NPROC_FALLBACK="$(nproc 2>/dev/null || echo 1)"
if [[ -n "${NCPUS:-}" ]]; then
    NCPUS_SOURCE="NCPUS env"
elif [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    NCPUS="$SLURM_CPUS_PER_TASK"
    NCPUS_SOURCE="SLURM_CPUS_PER_TASK"
else
    NCPUS="$NPROC_FALLBACK"
    NCPUS_SOURCE="nproc"
fi
ALLOC_CPUS="${SLURM_CPUS_PER_TASK:-$NPROC_FALLBACK}"
if (( NCPUS < 1 )); then
    echo "NCPUS must be >= 1" >&2
    exit 1
fi
if (( NCPUS > ALLOC_CPUS )); then
    echo "NCPUS ($NCPUS) exceeds allocated cpus-per-task ($ALLOC_CPUS)." >&2
    exit 1
fi
if (( NCPUS < 4 )); then
    echo "WARNING: running with NCPUS=$NCPUS (source: $NCPUS_SOURCE)." >&2
fi

CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" ]]; then
    echo "Conda executable not found." >&2
    exit 1
fi
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ----- One-time PICASO dependency install (idempotent, race-safe) -----
# Mirrors gen.sh so that gen_profiles.sh can also run on a fresh env.
# The flock here protects against a re-queue of this same single-task job;
# since this script is a singleton (not an array), cross-node races are not
# an issue.
DEPS_CACHE_DIR="$HOME/.cache/picaso_deps"
DEPS_OK_SENTINEL="$DEPS_CACHE_DIR/${CONDA_ENV}_v2.ok"
DEPS_LOCK="$DEPS_CACHE_DIR/${CONDA_ENV}_v2.lock"
mkdir -p "$DEPS_CACHE_DIR"

if [[ ! -f "$DEPS_OK_SENTINEL" ]]; then
    (
        flock -x 9
        if [[ ! -f "$DEPS_OK_SENTINEL" ]]; then
            echo "Installing PICASO dependencies into env '$CONDA_ENV' (one-time setup)..."
            python -m pip install --no-input --quiet \
                "bokeh>=2.3.0,<3.8.2" \
                "numpy>=2.0" \
                numba \
                pandas \
                joblib \
                photutils \
                astropy \
                matplotlib \
                stsynphot \
                synphot \
                "scipy<1.14" \
                h5py \
                virga-exo \
                xarray \
                pooch \
                tqdm \
                netcdf4 \
                h5netcdf
            touch "$DEPS_OK_SENTINEL"
            echo "Dependency install complete. Sentinel: $DEPS_OK_SENTINEL"
        fi
    ) 9>"$DEPS_LOCK"
fi

INPUT_FULL_PATH="gen_data/$INPUT_PATH"
DONE_SENTINEL="${INPUT_FULL_PATH}.done"
INPROGRESS_PATH="${INPUT_FULL_PATH}.inprogress"

mkdir -p "$(dirname "$INPUT_FULL_PATH")"

echo "------------------------------------------------"
echo "Project root:        $PROJECT_ROOT"
echo "Conda env:           $CONDA_ENV"
echo "Output HDF5:         $INPUT_FULL_PATH"
echo "Completion sentinel: $DONE_SENTINEL"
echo "Profile config:      gen_data/$PROFILE_CONFIG"
echo "Total profiles:      $TOTAL_PROFILES"
echo "Profile seed:        $PROFILE_SEED"
echo "Worker processes:    $NCPUS (source: $NCPUS_SOURCE)"
echo "------------------------------------------------"

if [[ -f "$INPUT_FULL_PATH" && -f "$DONE_SENTINEL" ]]; then
    echo "Input file and completion sentinel already present; nothing to do."
    exit 0
fi

# Stale state from a prior failed/interrupted run: drop everything so we
# start clean. create_profiles.py also handles its own .inprogress cleanup.
rm -f "$INPUT_FULL_PATH" "$DONE_SENTINEL" "$INPROGRESS_PATH"

srun --ntasks=1 --cpu-bind=cores python -u gen_data/create_profiles.py \
    --n-profiles "$TOTAL_PROFILES" \
    --config "$PROFILE_CONFIG" \
    --ncpus "$NCPUS" \
    --seed "$PROFILE_SEED" \
    --output "$INPUT_PATH"

# Touched only after create_profiles.py returns 0 (set -e aborts on failure
# above, so we never reach this line on error).
touch "$DONE_SENTINEL"
echo "Profile generation complete: $INPUT_FULL_PATH (sentinel: $DONE_SENTINEL)"
