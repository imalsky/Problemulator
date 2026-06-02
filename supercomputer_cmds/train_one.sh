#!/bin/bash
#SBATCH -J rt_train_one
#SBATCH -o rt_train_one.o%j
#SBATCH -e rt_train_one.e%j
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH -t 48:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov
#
# Train EXACTLY ONE model config on ONE GPU. Use this to run several models as
# independent, parallel jobs (one each) — e.g. the matched v6 pair:
#
#     CONFIG_NAME=transformer_main_v6 sbatch -J tfm_v6  supercomputer_cmds/train_one.sh
#     CONFIG_NAME=lstm_main_v6        sbatch -J lstm_v6 supercomputer_cmds/train_one.sh
#
# WHY a separate script (vs train.sh):
#   train.sh bundles merge -> stage -> (optional) preprocess -> train in one job
#   and trains its CONFIG_NAMES list SEQUENTIALLY. This script does ONLY the
#   train step for a single config, so N of them run truly in parallel.
#
# PARALLEL SAFETY:
#   - Each job writes only to its own models/<fixed_model_foldername>/.
#   - main.py reuses data/processed/ READ-ONLY when its fingerprint matches
#     (the v6 configs do NOT change any preprocessing-relevant field, so the
#     existing cache is reused — no rebuild). Jobs cannot clobber each other.
#   - This script never merges, never (re)stages data/raw, and never rebuilds
#     the processed cache. If the cache is missing/stale, main.py hard-fails by
#     design; build it once first (see below), then resubmit.
#
# PREREQUISITES (already true in the current repo state):
#   - data/raw/<raw>.h5 is staged
#   - data/dataset_splits.json exists
#   - data/processed/{train,val,test} exist and match the config fingerprint
#   If not, build once with:  sbatch supercomputer_cmds/normalize.sh

set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME=<config stem>, e.g. CONFIG_NAME=transformer_main_v6}"
RAW_NAME="${RAW_NAME:-picaso_results_5M.h5}"
CONDA_ENV="${CONDA_ENV:-nn}"

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

# SLURM copies the script to its spool dir, so BASH_SOURCE[0] is not the repo.
# Use SLURM_SUBMIT_DIR when available (matches train.sh).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROBLEMULATOR_ROOT="$(resolve_problemulator_root "$SLURM_SUBMIT_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROBLEMULATOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROBLEMULATOR_ROOT"

CONFIG_PATH="$PROBLEMULATOR_ROOT/config/${CONFIG_NAME}.jsonc"
RAW_FILE="$PROBLEMULATOR_ROOT/data/raw/$RAW_NAME"
SPLITS_FILE="$PROBLEMULATOR_ROOT/data/dataset_splits.json"
PROCESSED_DIR="$PROBLEMULATOR_ROOT/data/processed"

# ----- Fail-fast prerequisites (prevents half-runs / parallel rebuild races) -----
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 1; }
if ! grep -q "\"$RAW_NAME\"" "$CONFIG_PATH"; then
    echo "Config $CONFIG_PATH does not list \"$RAW_NAME\" in hdf5_dataset_filename." >&2
    exit 1
fi
[[ -f "$RAW_FILE" ]]    || { echo "Staged raw file missing: $RAW_FILE  (run train.sh once or stage it, then retry)" >&2; exit 1; }
[[ -f "$SPLITS_FILE" ]] || { echo "Splits file missing: $SPLITS_FILE  (run normalize.sh once first)" >&2; exit 1; }
for split in train val test; do
    [[ -d "$PROCESSED_DIR/$split" ]] || { echo "Processed cache missing: $PROCESSED_DIR/$split  (run normalize.sh once first)" >&2; exit 1; }
done

# ----- Conda env -----
CONDA_EXE="$(command -v conda || true)"
[[ -n "$CONDA_EXE" ]] || { echo "Conda executable not found." >&2; exit 1; }
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate Conda env '$CONDA_ENV'." >&2; exit 1; }

# ----- CUDA module (best-effort, matches train.sh) -----
for cuda_version in cuda12.6/toolkit/12.6.3 cuda12.6/12.6 cuda11.8/toolkit/11.8.0 cuda/12.7 cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.0 cuda/11.8 cuda/11.7 cuda; do
    if module load "$cuda_version" 2>/dev/null; then
        echo "Successfully loaded $cuda_version"
        break
    fi
done

echo "------------------------------------------------"
echo "Config:        $CONFIG_PATH"
echo "Conda env:     $CONDA_ENV"
echo "Raw (staged):  $RAW_FILE"
echo "Processed dir: $PROCESSED_DIR  (reused read-only; never rebuilt here)"
echo "------------------------------------------------"

cd "$PROBLEMULATOR_ROOT"
python -u src/main.py --config "$CONFIG_PATH"
echo "=== Finished '$CONFIG_NAME' ==="
