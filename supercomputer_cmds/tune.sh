#!/bin/bash
#SBATCH -J rt_tune
#SBATCH -o rt_tune.o%j
#SBATCH -e rt_tune.e%j
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Drive the Optuna hyperparameter search defined in src/tune.py.
# By default, tunes the model family declared by BASE_CONFIG. Set
# MODEL_FAMILY=both for a fair transformer-vs-LSTM ablation with no
# cross-architecture pruning. Only the top 5 checkpoints are kept.
#
# Pre-requisite: at least one successful train.sh run, so that
# data/raw/$TRAIN_RAW_NAME exists. tune.sh will not merge
# shards itself.
#
# Submit:
#     sbatch Problemulator/supercomputer_cmds/tune.sh
#     # or, from Problemulator/:
#     sbatch supercomputer_cmds/tune.sh
#
# Resume after preemption / time limit (re-uses SQLite study by name):
#     STUDY_NAME=<existing_name> sbatch Problemulator/supercomputer_cmds/tune.sh
#
# Optional overrides:
#     STUDY_NAME=rt_tune_2026 sbatch Problemulator/supercomputer_cmds/tune.sh
#     N_TRIALS=500 TIMEOUT_SECONDS=86400 sbatch Problemulator/supercomputer_cmds/tune.sh
#     EPOCHS=100 PATIENCE=15 DATA_FRACTION=1.0 sbatch Problemulator/supercomputer_cmds/tune.sh
#     BASE_CONFIG=config/lstm_v2.jsonc sbatch Problemulator/supercomputer_cmds/tune.sh
#     MODEL_FAMILY=both sbatch Problemulator/supercomputer_cmds/tune.sh

set -euo pipefail

resolve_problemulator_root() {
    local start_dir="$1"
    if [[ -d "$start_dir/src" && -d "$start_dir/config" ]]; then
        cd "$start_dir" && pwd
    elif [[ -d "$start_dir/Problemulator/src" && -d "$start_dir/Problemulator/config" ]]; then
        cd "$start_dir/Problemulator" && pwd
    elif [[ -d "$start_dir/../src" && -d "$start_dir/../config" ]]; then
        cd "$start_dir/.." && pwd
    elif [[ -d "$start_dir/../Problemulator/src" && -d "$start_dir/../Problemulator/config" ]]; then
        cd "$start_dir/../Problemulator" && pwd
    else
        echo "Could not resolve Problemulator root from: $start_dir" >&2
        return 1
    fi
}

resolve_problemulator_path() {
    local path_value="$1"
    if [[ "$path_value" = /* ]]; then
        [[ -f "$path_value" ]] && printf '%s\n' "$path_value"
    elif [[ -f "$PROBLEMULATOR_ROOT/$path_value" ]]; then
        cd "$(dirname "$PROBLEMULATOR_ROOT/$path_value")" && printf '%s/%s\n' "$(pwd)" "$(basename "$path_value")"
    elif [[ -f "$PROJECT_ROOT/$path_value" ]]; then
        cd "$(dirname "$PROJECT_ROOT/$path_value")" && printf '%s/%s\n' "$(pwd)" "$(basename "$path_value")"
    else
        return 1
    fi
}

# SLURM copies the script to its spool dir before executing, so BASH_SOURCE[0]
# points to the spool path, not the repo. Use SLURM_SUBMIT_DIR when available.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROBLEMULATOR_ROOT="$(resolve_problemulator_root "$SLURM_SUBMIT_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROBLEMULATOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="$(cd "$PROBLEMULATOR_ROOT/.." && pwd)"
cd "$PROBLEMULATOR_ROOT"
CONDA_ENV="${CONDA_ENV:-nn}"
MERGED_NAME="${MERGED_NAME:-picaso_results_5M.h5}"
TRAIN_RAW_NAME="${TRAIN_RAW_NAME:-$MERGED_NAME}"
BASE_CONFIG="${BASE_CONFIG:-config/transformer_v2.jsonc}"
MODEL_FAMILY="${MODEL_FAMILY:-config}"
STUDY_NAME="${STUDY_NAME:-rt_tune_$(date +%Y%m%d_%H%M%S)}"
# With MODEL_FAMILY=both, 64 trials = 32 transformer + 32 LSTM.
N_TRIALS="${N_TRIALS:-64}"
# ~69h leaves a 3h margin under the 72h walltime so the leaderboard flush
# can complete before SLURM hard-kills the job.
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-248400}"
EPOCHS="${EPOCHS:-30}"
DATA_FRACTION="${DATA_FRACTION:-0.1}"
PATIENCE="${PATIENCE:-5}"

# Mitigate fragmentation when many trials cycle the GPU allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----- Conda env -----
CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" ]]; then
    echo "Conda executable not found." >&2
    exit 1
fi
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate Conda env '$CONDA_ENV'." >&2; exit 1; }

# ----- CUDA module (best-effort, matches train.sh) -----
echo "Looking for available CUDA modules..."
module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found via module system"
for cuda_version in cuda12.6/toolkit/12.6.3 cuda12.6/12.6 cuda11.8/toolkit/11.8.0 cuda/12.7 cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.0 cuda/11.8 cuda/11.7 cuda; do
    if module load "$cuda_version" 2>/dev/null; then
        echo "Successfully loaded $cuda_version"
        break
    fi
done

# ----- Idempotent optuna install -----
python -c "import optuna" 2>/dev/null || python -m pip install --no-input --quiet "optuna>=3.6"

# ----- Pre-flight: base config + staged raw HDF5 -----
if ! BASE_CONFIG_PATH="$(resolve_problemulator_path "$BASE_CONFIG")"; then
    echo "Base config not found: $BASE_CONFIG" >&2
    exit 1
fi
if ! grep -q "\"$TRAIN_RAW_NAME\"" "$BASE_CONFIG_PATH"; then
    echo "Base config $BASE_CONFIG does not list \"$TRAIN_RAW_NAME\" in hdf5_dataset_filename." >&2
    exit 1
fi
if [[ ! -f "$PROBLEMULATOR_ROOT/data/raw/$TRAIN_RAW_NAME" ]]; then
    echo "Raw HDF5 missing: $PROBLEMULATOR_ROOT/data/raw/$TRAIN_RAW_NAME" >&2
    echo "Run 'sbatch Problemulator/supercomputer_cmds/train.sh' once to merge shards and stage the raw file." >&2
    exit 1
fi

echo "------------------------------------------------"
echo "Project root:    $PROJECT_ROOT"
echo "Problemulator:   $PROBLEMULATOR_ROOT"
echo "Conda env:       $CONDA_ENV"
echo "Base config:     $BASE_CONFIG_PATH"
echo "Model family:    $MODEL_FAMILY"
echo "Study name:      $STUDY_NAME"
echo "Trials cap:      $N_TRIALS"
echo "Timeout (s):     $TIMEOUT_SECONDS"
echo "Epochs / trial:  $EPOCHS"
echo "Data fraction:   $DATA_FRACTION"
echo "Patience:        $PATIENCE"
echo "------------------------------------------------"

python -u src/tune.py \
    --base-config "$BASE_CONFIG_PATH" \
    --study-name  "$STUDY_NAME" \
    --n-trials    "$N_TRIALS" \
    --timeout     "$TIMEOUT_SECONDS" \
    --model-family "$MODEL_FAMILY" \
    --epochs      "$EPOCHS" \
    --data-fraction "$DATA_FRACTION" \
    --patience    "$PATIENCE"

echo "=== Tuning job completed ==="
