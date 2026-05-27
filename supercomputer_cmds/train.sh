#!/bin/bash
#SBATCH -J rt_train
#SBATCH -o rt_train.o%j
#SBATCH -e rt_train.e%j
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH -t 72:00:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Merge PICASO shards, stage the merged HDF5 into Problemulator/data/raw/,
# and run model training on one GPU. Submit this AFTER gen.sh has finished
# successfully:
#
#     sbatch Problemulator/supercomputer_cmds/train.sh
#     # or, from Problemulator/:
#     sbatch supercomputer_cmds/train.sh
#
# Each step is idempotent:
#   - Merge is skipped automatically when gen_data/output/$MERGED_NAME already
#     reflects the current shards (same source_shard_count / successful_profiles
#     attrs and output mtime >= every shard mtime).
#   - Staging re-uses the existing symlink if it already points at $MERGED_PATH.
#   - The processed/ cache is reused when main.py's fingerprint check matches
#     (see rebuild_processed_data in the selected config). It is only wiped if
#     REBUILD_PROCESSED=1 is set explicitly.
#
# By default this trains both production models in sequence:
#   1. transformer_main_v5.jsonc — ~8.0M params; d_model=256, 6-layer encoder,
#      gelu FFN, sweep-informed architecture.
#   2. lstm_main_v5.jsonc        — ~8.0M params; d_model=464, 2-layer biLSTM,
#      matched parameter count for apples-to-apples comparison.
# Both use symlog flux normalization, pure float32, no EMA, no AMP, and an
# identical training schedule.
# Models in CONFIG_NAMES train sequentially in the listed order, sharing the
# processed data cache.
#
# Side-by-side with tune.sh:
#     # 1. Make sure data/processed/ exists and is current. If unsure, run
#     #    normalize.sh first (CPU-only, ~minutes):
#     sbatch Problemulator/supercomputer_cmds/normalize.sh
#     # 2. Then submit both jobs in parallel:
#     sbatch Problemulator/supercomputer_cmds/train.sh
#     sbatch Problemulator/supercomputer_cmds/tune.sh
#     # train.sh writes to models/transformer_main/, tune.sh writes to
#     # models/tune_<study>/. Both read data/processed/ read-only when the
#     # fingerprint matches, so they cannot clobber each other.
#
# Overrides (optional):
#     MERGED_NAME=picaso_results_10M.h5 sbatch Problemulator/supercomputer_cmds/train.sh
#     SKIP_MERGE=1 sbatch Problemulator/supercomputer_cmds/train.sh             # skip the merge step entirely
#     SKIP_MERGE=0 sbatch Problemulator/supercomputer_cmds/train.sh             # always run the merge freshness check
#     REBUILD_PROCESSED=1 sbatch Problemulator/supercomputer_cmds/train.sh      # force a full preprocessing rebuild
#     CONFIG_NAMES="lstm_main_v5" sbatch Problemulator/supercomputer_cmds/train.sh                          # LSTM only
#     CONFIG_NAMES="transformer_main_v5" sbatch Problemulator/supercomputer_cmds/train.sh                   # transformer only

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

# SLURM copies the script to its spool dir before executing, so BASH_SOURCE[0]
# points to the spool path, not the repo. Use SLURM_SUBMIT_DIR when available.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROBLEMULATOR_ROOT="$(resolve_problemulator_root "$SLURM_SUBMIT_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROBLEMULATOR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="$(cd "$PROBLEMULATOR_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
CONDA_ENV="${CONDA_ENV:-nn}"
SHARDS_GLOB="${SHARDS_GLOB:-output/shards/picaso_results_shard_*.h5}"
MERGED_NAME="${MERGED_NAME:-picaso_results_5M.h5}"
TRAIN_RAW_NAME="${TRAIN_RAW_NAME:-$MERGED_NAME}"
REBUILD_PROCESSED="${REBUILD_PROCESSED:-0}"
SKIP_MERGE="${SKIP_MERGE:-auto}"
# Space-separated list of config stems under config/<name>.jsonc.
# Models train sequentially in the listed order; the first one that fails
# stops the job (set -e). Override with CONFIG_NAMES="lstm_main_v5" etc.
CONFIG_NAMES="${CONFIG_NAMES:-transformer_main_v5 lstm_main_v5}"

MERGED_PATH="$PROJECT_ROOT/gen_data/output/$MERGED_NAME"
TRAIN_RAW_DIR="$PROBLEMULATOR_ROOT/data/raw"
TRAIN_RAW_PATH="$TRAIN_RAW_DIR/$TRAIN_RAW_NAME"
PROCESSED_DIR="$PROBLEMULATOR_ROOT/data/processed"

# ----- Conda env -----
CONDA_EXE="$(command -v conda || true)"
if [[ -z "$CONDA_EXE" ]]; then
    echo "Conda executable not found." >&2
    exit 1
fi
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate Conda env '$CONDA_ENV'." >&2; exit 1; }

# ----- CUDA module (best-effort, matches Problemulator/run.sh) -----
echo "Looking for available CUDA modules..."
module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found via module system"
for cuda_version in cuda12.6/toolkit/12.6.3 cuda12.6/12.6 cuda11.8/toolkit/11.8.0 cuda/12.7 cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.0 cuda/11.8 cuda/11.7 cuda; do
    if module load "$cuda_version" 2>/dev/null; then
        echo "Successfully loaded $cuda_version"
        break
    fi
done

# Resolve and validate every config up front so the job fails fast if a
# stem is mistyped or missing — no point running the long merge/preprocess
# steps just to crash before the second model.
read -r -a CONFIG_NAME_ARRAY <<< "$CONFIG_NAMES"
if [[ ${#CONFIG_NAME_ARRAY[@]} -eq 0 ]]; then
    echo "CONFIG_NAMES is empty; nothing to train." >&2
    exit 1
fi
CONFIG_PATHS=()
for name in "${CONFIG_NAME_ARRAY[@]}"; do
    path="$PROBLEMULATOR_ROOT/config/${name}.jsonc"
    if [[ ! -f "$path" ]]; then
        echo "Training config not found: $path" >&2
        exit 1
    fi
    if ! grep -q "\"$TRAIN_RAW_NAME\"" "$path"; then
        echo "Config $path does not list \"$TRAIN_RAW_NAME\" in hdf5_dataset_filename." >&2
        echo "Update data_paths_config.hdf5_dataset_filename to include \"$TRAIN_RAW_NAME\" and resubmit." >&2
        exit 1
    fi
    CONFIG_PATHS+=("$path")
done

echo "------------------------------------------------"
echo "Project root:        $PROJECT_ROOT"
echo "Problemulator root:  $PROBLEMULATOR_ROOT"
echo "Conda env:           $CONDA_ENV"
echo "Shards glob:         gen_data/$SHARDS_GLOB"
echo "Merged output:       $MERGED_PATH"
echo "Staged raw file:     $TRAIN_RAW_PATH"
echo "Processed dir:       $PROCESSED_DIR"
echo "Configs to train:    ${CONFIG_NAME_ARRAY[*]}"
echo "Skip merge:          $SKIP_MERGE"
echo "Rebuild processed:   $REBUILD_PROCESSED"
echo "------------------------------------------------"

# ----- 1. Merge shards into a single training-ready HDF5 -----
merge_is_current() {
    local merge_status
    if ! merge_status="$(
        python - "$PROJECT_ROOT" "$MERGED_PATH" "$SHARDS_GLOB" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
merged_path = Path(sys.argv[2])
shards_glob = sys.argv[3]
gen_data_dir = repo_root / "gen_data"
sys.path.insert(0, str(gen_data_dir))

import merge_picaso_shards as merger

input_paths = merger._expand_inputs([shards_glob], gen_data_dir)
if not input_paths:
    raise SystemExit(2)

shards = merger._validate_and_sort(input_paths)
total_rows = sum(row_count for _, _, _, row_count in shards)
is_current = merged_path.exists() and merger._existing_output_is_current(
    merged_path,
    shards,
    total_rows,
)
print("1" if is_current else "0")
PY
    )"; then
        return 1
    fi

    [[ "$merge_status" == "1" ]]
}

should_skip_merge=0
case "$SKIP_MERGE" in
    1|true|TRUE|yes|YES)
        should_skip_merge=1
        ;;
    0|false|FALSE|no|NO)
        should_skip_merge=0
        ;;
    auto|AUTO)
        if [[ -f "$MERGED_PATH" ]] && merge_is_current; then
            should_skip_merge=1
        fi
        ;;
    *)
        echo "Invalid SKIP_MERGE value '$SKIP_MERGE'. Use 0, 1, or auto." >&2
        exit 1
        ;;
esac

if [[ "$should_skip_merge" != "1" ]]; then
    echo "=== Merging PICASO shards ==="
    mkdir -p "$(dirname "$MERGED_PATH")"
    python -u gen_data/merge_picaso_shards.py \
        --inputs "$SHARDS_GLOB" \
        --output "output/$MERGED_NAME"

    if [[ ! -f "$MERGED_PATH" ]]; then
        echo "Merged output missing after merge step: $MERGED_PATH" >&2
        exit 1
    fi
else
    echo "=== Skipping merge (SKIP_MERGE=$SKIP_MERGE) ==="
    if [[ ! -f "$MERGED_PATH" ]]; then
        echo "Merge was skipped but merged file is missing: $MERGED_PATH" >&2
        exit 1
    fi
fi

# ----- 2. Stage merged file into Problemulator/data/raw/ as a symlink -----
mkdir -p "$TRAIN_RAW_DIR"
current_target=""
if [[ -L "$TRAIN_RAW_PATH" ]]; then
    current_target="$(readlink "$TRAIN_RAW_PATH")"
fi
if [[ "$current_target" == "$MERGED_PATH" ]]; then
    echo "=== Raw symlink already points at $MERGED_PATH; skipping restage ==="
else
    echo "=== Staging merged file into Problemulator/data/raw/ ==="
    if [[ -e "$TRAIN_RAW_PATH" || -L "$TRAIN_RAW_PATH" ]]; then
        backup="${TRAIN_RAW_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
        echo "Existing entry at $TRAIN_RAW_PATH; moving to $backup"
        mv "$TRAIN_RAW_PATH" "$backup"
    fi
    ln -s "$MERGED_PATH" "$TRAIN_RAW_PATH"
    echo "Staged: $TRAIN_RAW_PATH -> $MERGED_PATH"
fi

# ----- 3. Optionally force a processed cache rebuild -----
# main.py's fingerprint check already reuses processed artifacts when the raw
# HDF5 path/size/mtime, splits file, and preprocessing-relevant config all match
# (see _can_reuse_processed_data). Only wipe the cache when the user explicitly
# asks for a rebuild via REBUILD_PROCESSED=1.
if [[ "$REBUILD_PROCESSED" == "1" && -d "$PROCESSED_DIR" ]]; then
    echo "=== REBUILD_PROCESSED=1: clearing $PROCESSED_DIR ==="
    rm -rf "$PROCESSED_DIR"
fi

# ----- 4. Train each configured model in order -----
cd "$PROBLEMULATOR_ROOT"
total_models=${#CONFIG_PATHS[@]}
for i in "${!CONFIG_PATHS[@]}"; do
    cfg_path="${CONFIG_PATHS[$i]}"
    cfg_name="${CONFIG_NAME_ARRAY[$i]}"
    echo "=== [$((i + 1))/${total_models}] Training '$cfg_name' (config: $cfg_path) ==="
    python -u src/main.py --config "$cfg_path"
    echo "=== [$((i + 1))/${total_models}] Finished '$cfg_name' ==="
done
echo "=== All training jobs completed ==="
