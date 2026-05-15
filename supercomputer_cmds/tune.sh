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
# Default layout: MODEL_FAMILY=sequential with N_TRIALS=32 -> 16 LSTM trials
# (trials 0-15), followed by 16 transformer trials (trials 16-31). Both phases
# run with NopPruner so no cross-architecture median-pruning bias creeps in.
# Each trial is given EPOCHS=60, DATA_FRACTION=0.2, PATIENCE=12 by default --
# a deliberately heavier per-trial budget than the prior 30-epoch/10%-data
# screening sweep, intended to actually converge each trial so the architecture
# comparison is meaningful rather than budget-limited.
# Override MODEL_FAMILY for other layouts: 'both' (interleaved by trial number),
# 'transformer'/'lstm' (single architecture), or 'config' (whatever the base
# config declares). Only the top 5 checkpoints are kept.
#
# Pre-requisite: at least one successful train.sh run, so that
# data/raw/$TRAIN_RAW_NAME exists AND data/processed/ has a current fingerprint.
# tune.sh will not merge shards itself, and will hard-fail if the processed
# cache is missing or stale (rebuild_processed_data is forced false to avoid
# clobbering a parallel train.sh). If you don't have a current cache yet, run
# normalize.sh first (CPU-only, ~minutes).
#
# Side-by-side with train.sh:
#     # 1. Bootstrap data/processed/ if needed (skip if already current):
#     sbatch Problemulator/supercomputer_cmds/normalize.sh
#     # 2. Submit both jobs in parallel:
#     sbatch Problemulator/supercomputer_cmds/train.sh    # baseline transformer
#     sbatch Problemulator/supercomputer_cmds/tune.sh     # 16 LSTM + 16 transformer
#     # tune.sh writes to models/tune_<study>/; train.sh writes to
#     # models/transformer_main/. Both read data/processed/ read-only.
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
#     BASE_CONFIG=config/transformer_main_v3.jsonc sbatch Problemulator/supercomputer_cmds/tune.sh  # base config (architecture-agnostic for sequential/both)
#     MODEL_FAMILY=both sbatch Problemulator/supercomputer_cmds/tune.sh                          # alternate transformer/LSTM trials by trial number
#     MODEL_FAMILY=lstm sbatch Problemulator/supercomputer_cmds/tune.sh                          # single-architecture sweep
#     N_TRIALS_LSTM=16 N_TRIALS=48 MODEL_FAMILY=sequential sbatch ...                            # asymmetric split (16 LSTM + 32 transformer)

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
BASE_CONFIG="${BASE_CONFIG:-config/lstm_main_v3.jsonc}"
MODEL_FAMILY="${MODEL_FAMILY:-sequential}"
STUDY_NAME="${STUDY_NAME:-rt_tune_$(date +%Y%m%d_%H%M%S)}"
# Defaults tuned for a "good comparison" sweep that fits in ~72h SLURM
# walltime. With MODEL_FAMILY=sequential, 32 trials = 16 LSTM (trials 0-15)
# followed by 16 transformer (trials 16-31). Override N_TRIALS_LSTM to change
# the split. Estimated budget (post search-space expansion in src/tune.py):
#   ~22.7 min/trial at the prior 30-epoch/10%-data config × 2 (epochs)
#                                                       × 2 (data)
#                                                       × ~1.3 (wider space)
#     = ~118 min/trial avg => 32 * 1.97h = ~63h. Fits 69h timeout with margin.
N_TRIALS="${N_TRIALS:-32}"
N_TRIALS_LSTM="${N_TRIALS_LSTM:-}"
# ~69h leaves a 3h margin under the 72h walltime so the leaderboard flush
# can complete before SLURM hard-kills the job.
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-248400}"
# 60 epochs lets both architectures escape the early-plateau region that the
# 30-epoch budget revealed in tune_rt_tune_20260510_104818 (transformer
# trial 63 plateaued by epoch 25; LSTM trial 20 was still descending at
# epoch 30 because cosine LR had already collapsed to 1e-8).
EPOCHS="${EPOCHS:-60}"
# 20% of the training set (~1M profiles) gives noticeably less per-epoch
# noise than 10% without quadrupling per-epoch cost.
DATA_FRACTION="${DATA_FRACTION:-0.2}"
# Patience scales with the longer epoch budget so well-converged trials can
# early-stop without being cut short by interim noise.
PATIENCE="${PATIENCE:-12}"

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
echo "LSTM split:      ${N_TRIALS_LSTM:-<auto: N_TRIALS/2 if sequential>}"
echo "Study name:      $STUDY_NAME"
echo "Trials cap:      $N_TRIALS"
echo "Timeout (s):     $TIMEOUT_SECONDS"
echo "Epochs / trial:  $EPOCHS"
echo "Data fraction:   $DATA_FRACTION"
echo "Patience:        $PATIENCE"
echo "------------------------------------------------"

TUNE_ARGS=(
    --base-config "$BASE_CONFIG_PATH"
    --study-name  "$STUDY_NAME"
    --n-trials    "$N_TRIALS"
    --timeout     "$TIMEOUT_SECONDS"
    --model-family "$MODEL_FAMILY"
    --epochs      "$EPOCHS"
    --data-fraction "$DATA_FRACTION"
    --patience    "$PATIENCE"
)
if [[ -n "$N_TRIALS_LSTM" ]]; then
    TUNE_ARGS+=(--n-trials-lstm "$N_TRIALS_LSTM")
fi

python -u src/tune.py "${TUNE_ARGS[@]}"

echo "=== Tuning job completed ==="
