#!/usr/bin/env bash

# SLURM job-array script for large profile sweeps (gen.sh).
# Each array task:
#   - runs ONE Slurm task with many CPUs
#   - reads a [START_INDEX, END_INDEX) slice of the shared synthetic-profiles HDF5
#   - writes its own shard file: output/shards/picaso_results_shard_${SLURM_ARRAY_TASK_ID}.h5
#
# Default sizing targets a 5,000,000-profile run in 50 shards of 100,000 profiles each.
# Adjust TOTAL_PROFILES, SHARD_SIZE, and the --array directive after a one-shard benchmark.

#SBATCH -J picaso_rt_array
#SBATCH -o picaso_%A_%a.out
#SBATCH -e picaso_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-49%25

## Cluster-specific settings. On `edge` the default CPU partition is `compute`
## (64 cores/node). `--clusters` is most reliable on the sbatch command line,
## but also works as an #SBATCH directive here.
#SBATCH -p compute
#SBATCH -A exoweather
#SBATCH --clusters=edge

# Must come AFTER the #SBATCH block — any executable line before #SBATCH
# directives causes sbatch to stop parsing them (resource requests ignored).
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

# SLURM copies the script to its spool dir before executing, so BASH_SOURCE[0]
# points to the spool path under sbatch. Use SLURM_SUBMIT_DIR when available.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$(resolve_project_root "$SLURM_SUBMIT_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Prevent BLAS/OpenMP oversubscription inside each worker process.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
CONDA_ENV="${CONDA_ENV:-nn}"
INPUT_PATH="${INPUT_PATH:-output/synthetic_profiles_5M.h5}"
OPACITY_PATH="${OPACITY_PATH:-}"
PROFILE_CONFIG="${PROFILE_CONFIG:-config/paper_comparable_profiles.jsonc}"

TOTAL_PROFILES="${TOTAL_PROFILES:-5000000}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

START_INDEX=$(( SHARD_INDEX * SHARD_SIZE ))
END_INDEX=$(( START_INDEX + SHARD_SIZE ))
if (( END_INDEX > TOTAL_PROFILES )); then
    END_INDEX=$TOTAL_PROFILES
fi
if (( START_INDEX >= TOTAL_PROFILES )); then
    echo "SHARD_INDEX $SHARD_INDEX (start=$START_INDEX) is past TOTAL_PROFILES ($TOTAL_PROFILES); nothing to do." >&2
    exit 0
fi

SHARD_ID_PADDED=$(printf "%04d" "$SHARD_INDEX")
OUTPUT_PATH="${OUTPUT_PATH:-output/shards/picaso_results_shard_${SHARD_ID_PADDED}.h5}"

# Resolve worker count: explicit NCPUS wins, then SLURM_CPUS_PER_TASK, then nproc
# (cgroup-aware inside a SLURM allocation), finally 1. Print the source so it's
# obvious when something unexpected happens.
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
READ_CHUNK_SIZE="${READ_CHUNK_SIZE:-$(( NCPUS * 8 ))}"
WORKER_CHUNKSIZE="${WORKER_CHUNKSIZE:-1}"
HDF5_CHUNK_ROWS="${HDF5_CHUNK_ROWS:-$(( NCPUS * 8 ))}"

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
    echo "         SLURM_CPUS_PER_TASK='${SLURM_CPUS_PER_TASK:-<unset>}', nproc=$NPROC_FALLBACK." >&2
    echo "         If this is a production run you likely want --cpus-per-task=64 in the allocation." >&2
fi

if (( READ_CHUNK_SIZE < NCPUS )); then
    READ_CHUNK_SIZE="$NCPUS"
fi

if (( HDF5_CHUNK_ROWS < 1 )); then
    HDF5_CHUNK_ROWS=256
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
# Installs the packages from picaso/pyproject.toml into the active conda env.
# Array tasks start concurrently, so flock serializes the first-time install.
# A sentinel file short-circuits subsequent runs. Delete it to force a reinstall.
DEPS_CACHE_DIR="$HOME/.cache/picaso_deps"
# Bump the suffix (v1 -> v2 -> ...) whenever the pip list below changes so the
# install re-runs once. v2: pinned scipy<1.14 (newer scipy dropped trapz/simps,
# which virga + PyMieScatt still call).
DEPS_OK_SENTINEL="$DEPS_CACHE_DIR/${CONDA_ENV}_v2.ok"
DEPS_LOCK="$DEPS_CACHE_DIR/${CONDA_ENV}_v2.lock"
mkdir -p "$DEPS_CACHE_DIR"

if [[ ! -f "$DEPS_OK_SENTINEL" ]]; then
    (
        flock -x 9
        if [[ ! -f "$DEPS_OK_SENTINEL" ]]; then
            echo "Installing PICASO dependencies into env '$CONDA_ENV' (one-time setup)..."
            # virga-exo==2.0.2 (PICASO 4's pin) needs Python >=3.11; we leave it
            # unpinned so pip picks the latest version compatible with the conda env's Python.
            # If PICASO import/runtime ever fails on virga API changes, build a dedicated
            # Python 3.11+ env and set CONDA_ENV to that instead.
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

mkdir -p "gen_data/$(dirname "$OUTPUT_PATH")"

# ----- Ensure the synthetic-profiles input file exists -----
# Generates gen_data/$INPUT_PATH with $TOTAL_PROFILES profiles if it's missing.
# Serialized via flock so concurrent array tasks don't double-generate.
# Override the seed via PROFILE_SEED env var; skip this step entirely by pre-generating
# the file or setting SKIP_PROFILE_GEN=1.
PROFILE_SEED="${PROFILE_SEED:-20260423}"
INPUT_FULL_PATH="gen_data/$INPUT_PATH"
INPUT_LOCK_PATH="${INPUT_FULL_PATH}.lock"

mkdir -p "$(dirname "$INPUT_FULL_PATH")"

if [[ ! -f "$INPUT_FULL_PATH" && "${SKIP_PROFILE_GEN:-0}" != "1" ]]; then
    (
        flock -x 7
        if [[ ! -f "$INPUT_FULL_PATH" ]]; then
            echo "Input file missing — generating $TOTAL_PROFILES synthetic profiles (seed=$PROFILE_SEED)..."
            python -u gen_data/create_profiles.py \
                --n-profiles "$TOTAL_PROFILES" \
                --config "$PROFILE_CONFIG" \
                --ncpus "$NCPUS" \
                --seed "$PROFILE_SEED" \
                --output "$INPUT_PATH"
            echo "Profile generation complete: $INPUT_FULL_PATH"
        fi
    ) 7>"$INPUT_LOCK_PATH"
fi

if [[ ! -f "$INPUT_FULL_PATH" ]]; then
    echo "Input file still missing after generation attempt: $INPUT_FULL_PATH" >&2
    exit 1
fi

echo "------------------------------------------------"
echo "Project root:        $PROJECT_ROOT"
echo "Conda env:           $CONDA_ENV"
echo "Input HDF5:          gen_data/$INPUT_PATH"
echo "Profile config:      gen_data/$PROFILE_CONFIG"
echo "Output HDF5:         gen_data/$OUTPUT_PATH"
echo "Shard index:         $SHARD_INDEX"
echo "Shard range:         [$START_INDEX, $END_INDEX)"
echo "Total profiles:      $TOTAL_PROFILES"
echo "Shard size:          $SHARD_SIZE"
echo "Profile seed:        $PROFILE_SEED"
echo "Worker processes:    $NCPUS (source: $NCPUS_SOURCE)"
echo "Read chunk size:     $READ_CHUNK_SIZE"
echo "Worker chunksize:    $WORKER_CHUNKSIZE"
echo "HDF5 chunk rows:     $HDF5_CHUNK_ROWS"
echo "------------------------------------------------"

CMD=(
    python -u gen_data/run_picaso.py
    --ncpus "$NCPUS"
    --input "$INPUT_PATH"
    --output "$OUTPUT_PATH"
    --start-index "$START_INDEX"
    --end-index "$END_INDEX"
    --read-chunk-size "$READ_CHUNK_SIZE"
    --worker-chunksize "$WORKER_CHUNKSIZE"
    --hdf5-chunk-rows "$HDF5_CHUNK_ROWS"
)

if [[ -n "$OPACITY_PATH" ]]; then
    CMD+=(--opacity-path "$OPACITY_PATH")
fi

srun --ntasks=1 --cpu-bind=cores "${CMD[@]}"
