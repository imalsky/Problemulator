#!/bin/bash
#SBATCH -J rt_normalize
#SBATCH -o rt_normalize.o%j
#SBATCH -e rt_normalize.e%j
#SBATCH -p compute
#SBATCH --mem=60G
#SBATCH -t 0-02:00:00
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# One-shot job that regenerates Problemulator/data/processed/ from the staged
# raw HDF5(s). CPU-only — normalization does not need a GPU. Use this whenever:
#   - the loss formula or normalization spec changes
#   - the raw HDF5 mtime/size changes
#   - the configured normalization methods diverge from what is on disk
#
# This wrapper does not edit the checked-in config. It loads BASE_CONFIG, flips
# rebuild_processed_data=true and execution_mode="normalize" in memory, writes
# a temp JSON config, and runs main.py against it. The temp file is cleaned up
# on exit.
#
# Submit:
#     sbatch Problemulator/supercomputer_cmds/normalize.sh
#     # or, from Problemulator/:
#     sbatch supercomputer_cmds/normalize.sh
#
# Optional overrides:
#     BASE_CONFIG=config/lstm_v2.jsonc sbatch Problemulator/supercomputer_cmds/normalize.sh
#     TRAIN_RAW_NAME=picaso_results_5M.h5 sbatch Problemulator/supercomputer_cmds/normalize.sh
#
# Pre-requisite: Problemulator/data/raw/$TRAIN_RAW_NAME exists. Run train.sh
# at least once first to merge shards and stage the file (or stage it manually).

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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROBLEMULATOR_ROOT="$(resolve_problemulator_root "$SLURM_SUBMIT_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROBLEMULATOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="$(cd "$PROBLEMULATOR_ROOT/.." && pwd)"
cd "$PROBLEMULATOR_ROOT"
CONDA_ENV="${CONDA_ENV:-nn}"
BASE_CONFIG="${BASE_CONFIG:-config/transformer_v2.jsonc}"
MERGED_NAME="${MERGED_NAME:-picaso_results_5M.h5}"
TRAIN_RAW_NAME="${TRAIN_RAW_NAME:-$MERGED_NAME}"

# ----- Conda env -----
CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" ]]; then
    echo "Conda executable not found." >&2
    exit 1
fi
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate Conda env '$CONDA_ENV'." >&2; exit 1; }

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
echo "Raw HDF5 name:   $TRAIN_RAW_NAME"
echo "------------------------------------------------"

# ----- Build a temp config that overrides the two flags in-memory -----
TMP_CFG="$(mktemp -t rt_normalize_cfg.XXXXXX.json)"
trap 'rm -f "$TMP_CFG"' EXIT

cd "$PROBLEMULATOR_ROOT"
python - "$BASE_CONFIG_PATH" "$TMP_CFG" <<'PY'
import json
import sys
sys.path.insert(0, "src")
from utils import load_config

src, dst = sys.argv[1], sys.argv[2]
cfg = load_config(src)
cfg["miscellaneous_settings"]["rebuild_processed_data"] = True
cfg["miscellaneous_settings"]["execution_mode"] = "normalize"
# Normalization is CPU-only logic, but main.py initializes the device before
# dispatching to run_normalize. Override the backend so this can run on the
# compute (no-GPU) SLURM partition without tripping the cuda availability check.
cfg["miscellaneous_settings"]["device_backend"] = "cpu"
# AMP requires cuda; the validator rejects use_amp=True with device_backend=cpu.
# Normalization never trains, so disabling AMP here is safe and only relevant
# to this temp config. The precision validator additionally requires
# amp_autocast_dtype='none' whenever use_amp is False, so flip both.
cfg["training_hyperparameters"]["use_amp"] = False
cfg["precision"]["amp_autocast_dtype"] = "none"
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote temp normalize config to {dst}")
PY

echo "=== Running normalization (rebuild=true, execution_mode=normalize) ==="
python -u src/main.py --config "$TMP_CFG"

echo "=== Normalization complete ==="
ls -lh data/processed/normalization_metadata.json data/processed/processed_fingerprint.json 2>&1 || true
