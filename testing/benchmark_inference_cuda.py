#!/usr/bin/env python3
"""Inference-latency / throughput benchmark for the emulator on a GPU.

Same methodology as paper_revision.ipynb cell 39 (which produced the MPS/CPU
numbers in inference_benchmark_*.json), so the results are directly comparable
to the values reported in the manuscript -- with the one addition required for
correct GPU timing: torch.cuda.synchronize() around every timed region (CUDA
kernel launches are asynchronous, so wall-clock timing without a sync measures
launch overhead, not compute).

It benchmarks the base model forward (create_prediction_model + best_model.pt),
which dominates inference cost; the StandaloneModel's normalization/denorm wrap
is a handful of cheap elementwise ops and does not change the latency story.

Run on a SLURM GPU node via supercomputer_cmds/benchmark.sh, or directly:
    conda run -n nn python testing/benchmark_inference_cuda.py
    conda run -n nn python testing/benchmark_inference_cuda.py --device cuda \
        --model-dir models/transformer_main --batch-sizes 1,8,128,256,512

Writes models/<model_dir>/inference_benchmark_<device>.json (e.g.
inference_benchmark_cuda.json) and prints a latency/throughput table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config                       # noqa: E402
from model import create_prediction_model           # noqa: E402

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP_CALLS = 10
REPEATS_BENCH = 5
CALLS_PER = 3


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_sync(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps" and hasattr(torch, "mps"):
        return torch.mps.synchronize
    return lambda: None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda | mps | cpu | auto (default: cuda)")
    ap.add_argument("--model-dir", type=str, default="models/transformer_main")
    ap.add_argument("--batch-sizes", type=str, default=None,
                    help="comma-separated, e.g. 1,8,128,256,512 (default: 1..256 powers of 2)")
    ap.add_argument("--warmup", type=int, default=WARMUP_CALLS)
    ap.add_argument("--repeats", type=int, default=REPEATS_BENCH)
    ap.add_argument("--calls-per", type=int, default=CALLS_PER)
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.device in ("cuda", "auto") and device.type != "cuda":
        print(f"[bench] WARNING: requested cuda but it is unavailable; running on {device}.", flush=True)
    sync = make_sync(device)

    batch_sizes = ([int(x) for x in args.batch_sizes.split(",")]
                   if args.batch_sizes else list(DEFAULT_BATCH_SIZES))

    model_dir = (PROJECT_ROOT / args.model_dir) if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    config = load_config(model_dir / "train_config.json")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"[bench] device={device}  gpu_name={gpu_name}  model_dir={model_dir.name}", flush=True)

    # Build base model and load the published min-val checkpoint.
    base = create_prediction_model(config=config, device=device, compile_model=False).eval()
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    base.load_state_dict(state, strict=True)
    base = base.to(device=device, dtype=torch.float32).eval()
    n_params = sum(p.numel() for p in base.parameters())

    seq_len = int(config["model_hyperparameters"]["max_sequence_length"])
    in_dim = len(config["data_specification"]["input_variables"])
    g_dim = len(config["data_specification"]["global_variables"])

    mean_ms, std_ms = [], []
    with torch.inference_mode():
        for B in batch_sizes:
            seq = torch.randn(B, seq_len, in_dim, dtype=torch.float32, device=device)
            msk = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
            glob = torch.randn(B, g_dim, dtype=torch.float32, device=device) if g_dim > 0 else None

            for _ in range(args.warmup):
                base(seq, glob, msk)
            sync()

            ts = []
            for _ in range(args.repeats):
                sync()
                t0 = time.perf_counter_ns()
                for _ in range(args.calls_per):
                    base(seq, glob, msk)
                sync()
                dt_ms = (time.perf_counter_ns() - t0) / 1e6
                ts.append(dt_ms / args.calls_per)
            a = np.asarray(ts)
            mean_ms.append(float(a.mean()))
            std_ms.append(float(a.std(ddof=1) if a.size > 1 else 0.0))
            print(f"[bench]   B={B:<4d} mean={a.mean():8.3f} ms  "
                  f"std={a.std(ddof=1) if a.size > 1 else 0.0:6.3f}  "
                  f"throughput={B / (a.mean() / 1000.0):10.1f} samples/s", flush=True)

    tput = [B / (m / 1000.0) for B, m in zip(batch_sizes, mean_ms)]
    bench_data = {
        "device": device.type,
        "gpu_name": gpu_name,
        "model_dir": model_dir.name,
        "num_parameters": int(n_params),
        "seq_len": seq_len,
        "warmup_calls": args.warmup,
        "repeats": args.repeats,
        "calls_per": args.calls_per,
        "batch_sizes": batch_sizes,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "throughput_samples_per_s": tput,
    }
    out = model_dir / f"inference_benchmark_{device.type}.json"
    out.write_text(json.dumps(bench_data, indent=2))
    print(f"\n[bench] wrote {out}", flush=True)
    if 1 in batch_sizes:
        print(f"[bench] B=1 latency      : {mean_ms[batch_sizes.index(1)]:.3f} ms", flush=True)
    if 128 in batch_sizes:
        i = batch_sizes.index(128)
        print(f"[bench] B=128 per-sample : {mean_ms[i] / 128:.4f} ms/sample", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
