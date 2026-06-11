#!/usr/bin/env python3
"""Build the CPU/GPU inference benchmark figure (latency + throughput vs batch size).

Two-step, git-friendly workflow (models/ and testing/outputs/ are git-ignored):

  1. On the machine with the hardware (e.g. a GH200 node after `git pull`):
         conda run -n nn python testing/make_benchmark_figure.py --run cpu,cuda
     This invokes testing/benchmark_inference_cuda.py once per device with
     batch sizes 1..2048, which writes models/<dir>/inference_benchmark_<device>.json;
     each result is then mirrored to testing/benchmarks/inference_benchmark_<device>.json
     (a tracked dir; models/ and testing/outputs/ are git-ignored) so a plain
     `git add testing/benchmarks && git push` carries the numbers home.

  2. Anywhere (after `git pull`), build the figure from whatever device JSONs exist:
         conda run -n nn python testing/make_benchmark_figure.py
     Output: testing/figs/reviewer_fig_benchmark.png (the name expected by ms.tex).

Both steps in one go also works (step 1 plots at the end). Device JSONs are read
from models/<dir>/ first, then testing/benchmarks/ as the git-synced fallback.

Notes:
  - The timing methodology lives in benchmark_inference_cuda.py (10 warmup calls,
    5 repeats x 3 calls, device sync around every timed region); this script only
    orchestrates it, so figure and manuscript table numbers stay comparable.
  - The CPU sweep is slow at large batches (a B=2048 forward is seconds per call
    on a laptop CPU); use --batch-sizes to shorten it when iterating.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
RUNNER = THIS_DIR / "benchmark_inference_cuda.py"
# Mirror dir for benchmark JSONs. models/ and testing/outputs/ are git-ignored,
# so results are copied here to travel with git push/pull.
BENCH_DIR = THIS_DIR / "benchmarks"
FIG_DIR = THIS_DIR / "figs"
STYLE = THIS_DIR / "science.mplstyle"

# Powers of two through 2048: the manuscript quotes B=1 and B=128, and 2048 is
# comfortably past the throughput plateau on both CPU and GPU.
DEFAULT_BATCH_SIZES = [2**k for k in range(12)]  # 1 .. 2048

# Order also sets plot order; the first available entry of each class is used.
KNOWN_DEVICES = ["cpu", "cuda", "mps"]
DEVICE_COLORS = {"cpu": "#444444", "cuda": "#1f77b4", "mps": "#1f77b4"}
DEVICE_MARKERS = {"cpu": "o", "cuda": "s", "mps": "s"}


def device_json_paths(model_dir: Path, device: str) -> list[Path]:
    """Candidate locations for one device's benchmark JSON, most-fresh first."""
    name = f"inference_benchmark_{device}.json"
    return [model_dir / name, BENCH_DIR / name]


def load_device_results(model_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for device in KNOWN_DEVICES:
        for path in device_json_paths(model_dir, device):
            if path.is_file():
                results[device] = json.loads(path.read_text())
                print(f"[fig] {device}: using {path.relative_to(PROJECT_ROOT)}")
                break
    return results


def device_label(device: str, data: dict) -> str:
    if device == "cpu":
        return "CPU"
    if device == "cuda":
        return f"GPU ({data.get('gpu_name', 'CUDA')})"
    return "GPU (Apple silicon, MPS)"


def run_benchmarks(devices: list[str], model_dir: Path, batch_sizes: list[int],
                   extra_args: list[str]) -> None:
    for device in devices:
        cmd = [sys.executable, str(RUNNER),
               "--device", device,
               "--model-dir", str(model_dir.relative_to(PROJECT_ROOT)),
               "--batch-sizes", ",".join(str(b) for b in batch_sizes),
               *extra_args]
        print(f"[fig] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        src = model_dir / f"inference_benchmark_{device}.json"
        if not src.is_file():
            raise FileNotFoundError(f"benchmark runner did not produce {src}")
        BENCH_DIR.mkdir(exist_ok=True)
        dst = BENCH_DIR / src.name
        shutil.copy2(src, dst)
        print(f"[fig] mirrored {src.name} -> {dst.relative_to(PROJECT_ROOT)} (tracked)")


def make_figure(results: dict[str, dict], out_path: Path) -> None:
    if not results:
        raise FileNotFoundError(
            "No inference_benchmark_<device>.json found in the model dir or "
            f"{BENCH_DIR}. Run with --run cpu,cuda (or cpu,mps) first.")
    if STYLE.is_file():
        plt.style.use(str(STYLE))

    fig, (ax_lat, ax_tput) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for device, data in results.items():
        b = np.asarray(data["batch_sizes"], dtype=float)
        mean_ms = np.asarray(data["mean_ms"], dtype=float)
        std_ms = np.asarray(data["std_ms"], dtype=float)
        tput = np.asarray(data["throughput_samples_per_s"], dtype=float)
        label = device_label(device, data)
        color, marker = DEVICE_COLORS[device], DEVICE_MARKERS[device]
        ax_lat.errorbar(b, mean_ms, yerr=std_ms, color=color, marker=marker,
                        ms=4, lw=1.4, capsize=2, label=label)
        ax_tput.plot(b, tput, color=color, marker=marker, ms=4, lw=1.4, label=label)

    for ax in (ax_lat, ax_tput):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.legend(frameon=False)
    ax_lat.set_ylabel("Latency per batch (ms)")
    ax_tput.set_ylabel("Throughput (samples s$^{-1}$)")

    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, default=None,
                    help="comma-separated devices to benchmark before plotting, "
                         "e.g. 'cpu,cuda' on a GPU node or 'cpu,mps' locally")
    ap.add_argument("--model-dir", type=str, default="models/transformer_main")
    ap.add_argument("--batch-sizes", type=str,
                    default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
                    help="batch sizes for --run (default: powers of 2, 1..2048)")
    ap.add_argument("--out", type=str, default=str(FIG_DIR / "reviewer_fig_benchmark.png"))
    ap.add_argument("--warmup", type=int, default=None,
                    help="override runner warmup calls (for quick iteration)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="override runner timed repeats (for quick iteration)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    if args.run:
        devices = [d.strip() for d in args.run.split(",") if d.strip()]
        unknown = sorted(set(devices) - set(KNOWN_DEVICES))
        if unknown:
            raise ValueError(f"unknown device(s) {unknown}; expected {KNOWN_DEVICES}")
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        extra: list[str] = []
        if args.warmup is not None:
            extra += ["--warmup", str(args.warmup)]
        if args.repeats is not None:
            extra += ["--repeats", str(args.repeats)]
        run_benchmarks(devices, model_dir, batch_sizes, extra)

    make_figure(load_device_results(model_dir), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
