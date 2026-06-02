#!/bin/bash
#SBATCH -J rt_bench
#SBATCH -o rt_bench.o%j
#SBATCH -e rt_bench.e%j
#SBATCH -p gpu
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --gpus=1
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Benchmark emulator inference latency/throughput on one GPU and write
# models/<dir>/inference_benchmark_cuda.json. This fills the GPU/CUDA timing
# placeholders in the manuscript (Section "Inference Time") and the reviewer
# response (R1.2 table). Submit from the repo root or from Problemulator/:
#
#     sbatch Problemulator/supercomputer_cmds/benchmark.sh
#     # or, from Problemulator/:
#     sbatch supercomputer_cmds/benchmark.sh
#
# Overrides (optional):
#     MODEL_DIRS="models/transformer_main models/lstm_main" sbatch ... benchmark.sh
#     BATCH_SIZES="1,8,128,256,512,1024"                    sbatch ... benchmark.sh
#     DEVICE=cuda                                           sbatch ... benchmark.sh
#
# The methodology matches testing/paper_revision.ipynb cell 39 (10 warmup
# calls, 5 repeats x 3 calls, batch sizes 1..256), so the resulting
# inference_benchmark_cuda.json is directly comparable to the MPS/CPU JSONs.

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
cd "$PROBLEMULATOR_ROOT"

CONDA_ENV="${CONDA_ENV:-nn}"
DEVICE="${DEVICE:-cuda}"
MODEL_DIRS="${MODEL_DIRS:-models/transformer_main}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64,128,256}"

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

echo "------------------------------------------------"
echo "Problemulator root:  $PROBLEMULATOR_ROOT"
echo "Conda env:           $CONDA_ENV"
echo "Device:              $DEVICE"
echo "Model dirs:          $MODEL_DIRS"
echo "Batch sizes:         $BATCH_SIZES"
echo "------------------------------------------------"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu'))"

read -r -a MODEL_DIR_ARRAY <<< "$MODEL_DIRS"
for mdir in "${MODEL_DIR_ARRAY[@]}"; do
    if [[ ! -f "$PROBLEMULATOR_ROOT/$mdir/best_model.pt" ]]; then
        echo "WARNING: $mdir/best_model.pt not found; skipping." >&2
        continue
    fi
    echo "=== Benchmarking $mdir on $DEVICE ==="
    python -u testing/benchmark_inference_cuda.py \
        --device "$DEVICE" \
        --model-dir "$mdir" \
        --batch-sizes "$BATCH_SIZES"
done
echo "=== Benchmark complete ==="
