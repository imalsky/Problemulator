#!/usr/bin/env python3
"""Compute per-channel percent errors of the transformer emulator on the test set.

Verifies the manuscript's "~1%" accuracy claim against the full held-out test
split. Mirrors the notebook's error pipeline (paper_revision.ipynb cells 4/22):
the processed test loader yields normalized inputs/targets; we run the base
model (the same forward whose masked-MSE is reported as test_loss in
test_metrics.json), then denormalize predictions and targets to physical units
and form percent errors with the manuscript's epsilon=1 erg/cm^2/s floor.

Usage (from Problemulator/):
    conda run -n nn python testing/verify_percent_error.py            # full test set
    conda run -n nn python testing/verify_percent_error.py --max-batches 50
    conda run -n nn python testing/verify_percent_error.py --device cpu

Writes the percent-error statistics into testing/percent_error_full.json and
augments the "transformer" block of testing/per_channel_test_errors.json with
full-test-set MAE/RMSE/R^2 and the percent-error summary.
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
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import load_config                                   # noqa: E402
from dataset import create_collate_fn, create_dataset          # noqa: E402
from normalizer import DataNormalizer                          # noqa: E402
from model import create_prediction_model                      # noqa: E402
from torch.utils.data import DataLoader                         # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "transformer_main"
PROC_TEST = PROJECT_ROOT / "data" / "processed" / "test"
NORM_META_PATH = PROJECT_ROOT / "data" / "processed" / "normalization_metadata.json"
OUTPUTS_DIR = THIS_DIR / "outputs"  # consolidated home for small JSON/CSV artifacts
OUTPUTS_DIR.mkdir(exist_ok=True)
PER_CHANNEL_JSON = OUTPUTS_DIR / "per_channel_test_errors.json"
PERCENT_JSON = OUTPUTS_DIR / "percent_error_full.json"

PCT_DENOM_FLOOR = 1.0  # epsilon = 1 erg cm^-2 s^-1, per the manuscript


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denorm_channel(values: torch.Tensor, var_name: str, meta: dict) -> torch.Tensor:
    method = meta["normalization_methods"][var_name]
    stats = meta["per_key_stats"][var_name]
    if method != "bool" and stats:
        return DataNormalizer.denormalize_tensor(values.to(torch.float32), method, stats)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = full test set")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--model-dir", type=str, default="transformer_main",
                    help="model folder name under models/ (must share this project's normalization)")
    args = ap.parse_args()

    global MODEL_DIR, PERCENT_JSON
    MODEL_DIR = PROJECT_ROOT / "models" / args.model_dir
    block_name = "transformer" if args.model_dir == "transformer_main" else args.model_dir
    if args.model_dir != "transformer_main":
        PERCENT_JSON = OUTPUTS_DIR / f"percent_error_full_{args.model_dir}.json"

    device = pick_device(args.device)
    print(f"[verify] device={device}  batch_size={args.batch_size}  "
          f"max_batches={'all' if args.max_batches == 0 else args.max_batches}", flush=True)

    config = load_config(MODEL_DIR / "train_config.json")
    norm_meta = json.load(open(NORM_META_PATH))
    ds_spec = config["data_specification"]
    target_vars = list(ds_spec["target_variables"])
    flux_vars = [n for n in ("net_thermal_flux", "net_reflected_flux") if n in target_vars]
    padding_value = float(ds_spec["padding_value"])
    padding_epsilon = float(config["normalization"]["padding_comparison_epsilon"])

    # Build base model and load the published min-val checkpoint.
    base = create_prediction_model(config=config, device=device, compile_model=False).eval()
    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    base.load_state_dict(state, strict=True)
    base = base.to(device=device, dtype=torch.float32).eval()
    print(f"[verify] loaded best_model.pt (epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})", flush=True)

    total_samples = int(json.load(open(PROC_TEST / "metadata.json"))["total_samples"])
    ds_full = create_dataset(PROC_TEST, config, list(range(total_samples)))
    loader = DataLoader(
        ds_full, batch_size=args.batch_size, shuffle=False,
        collate_fn=create_collate_fn(padding_value, padding_epsilon), drop_last=False,
    )
    n_batches = len(loader)
    cap = args.max_batches if args.max_batches > 0 else n_batches
    print(f"[verify] test profiles={total_samples:,}  batches={n_batches}  running={min(cap, n_batches)}", flush=True)

    # Accumulators (physical-space) per flux channel. MAE/RMSE/R^2 are streamed
    # (running sums) so we never hold the full diff/true arrays in memory; only
    # the percent-error arrays are kept, for quantiles.
    acc = {n: {"abs_pct": [], "signed_pct": [],
               "n": 0, "sum_abs_diff": 0.0, "sum_diff2": 0.0,
               "sum_true": 0.0, "sum_true2": 0.0} for n in flux_vars}
    glob_vars = ds_spec["global_variables"]
    t0 = time.perf_counter()
    seen = 0
    with torch.inference_mode():
        for b_idx, (inputs_norm, masks, targets_norm, target_masks) in enumerate(loader):
            if b_idx >= cap:
                break
            seq_norm = inputs_norm["sequence"].to(device=device, dtype=torch.float32)
            seq_mask = masks["sequence"].bool().to(device=device)            # True == padding
            glob_norm = None
            if glob_vars and "global_features" in inputs_norm:
                glob_norm = inputs_norm["global_features"].to(device=device, dtype=torch.float32)
            out_norm = base(seq_norm, glob_norm, seq_mask).detach().to("cpu", torch.float32)

            tgt_norm = targets_norm.to(torch.float32)
            valid = ~target_masks.bool()                                     # True == real layer
            for n in flux_vars:
                j = target_vars.index(n)
                pred = denorm_channel(out_norm[..., j], n, norm_meta)[valid]
                true = denorm_channel(tgt_norm[..., j], n, norm_meta)[valid]
                if pred.numel() == 0:
                    continue
                diff = pred - true
                abs_pct = 100.0 * diff.abs() / true.abs().clamp_min(PCT_DENOM_FLOOR)
                signed_pct = 100.0 * diff / true.clamp_min(PCT_DENOM_FLOOR)   # manuscript denom: max(true, eps)
                acc[n]["abs_pct"].append(abs_pct.numpy().astype(np.float32))
                acc[n]["signed_pct"].append(signed_pct.numpy().astype(np.float32))
                d = diff.to(torch.float64)
                t = true.to(torch.float64)
                acc[n]["n"] += int(d.numel())
                acc[n]["sum_abs_diff"] += float(d.abs().sum())
                acc[n]["sum_diff2"] += float((d * d).sum())
                acc[n]["sum_true"] += float(t.sum())
                acc[n]["sum_true2"] += float((t * t).sum())
            seen += seq_norm.shape[0]
            if (b_idx + 1) % 25 == 0 or b_idx + 1 == min(cap, n_batches):
                rate = seen / max(time.perf_counter() - t0, 1e-9)
                print(f"[verify]   batch {b_idx + 1}/{min(cap, n_batches)}  "
                      f"{seen:,} profiles  ({rate:,.0f}/s)", flush=True)

    # Summaries.
    results = {}
    for n in flux_vars:
        if not acc[n]["abs_pct"]:
            continue
        abs_pct = np.concatenate(acc[n]["abs_pct"])
        signed_pct = np.concatenate(acc[n]["signed_pct"])
        nn = acc[n]["n"]
        sse = acc[n]["sum_diff2"]
        sst = acc[n]["sum_true2"] - (acc[n]["sum_true"] ** 2) / nn if nn else 0.0
        results[n] = {
            "n_points": int(nn),
            "mae": acc[n]["sum_abs_diff"] / nn,
            "rmse": (acc[n]["sum_diff2"] / nn) ** 0.5,
            "r2": float(1.0 - sse / max(sst, 1e-30)),
            "abs_pct_median": float(np.median(abs_pct)),
            "abs_pct_mean": float(np.mean(abs_pct)),
            "abs_pct_p5": float(np.percentile(abs_pct, 5)),
            "abs_pct_p25": float(np.percentile(abs_pct, 25)),
            "abs_pct_p75": float(np.percentile(abs_pct, 75)),
            "abs_pct_p95": float(np.percentile(abs_pct, 95)),
            "signed_pct_median": float(np.median(signed_pct)),
            "signed_pct_mean": float(np.mean(signed_pct)),
        }

    meta = {
        "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "n_profiles_evaluated": seen,
        "n_profiles_total": total_samples,
        "full_test_set": (cap >= n_batches),
        "epsilon_denominator_floor": PCT_DENOM_FLOOR,
        "abs_pct_def": "100*|pred-true|/max(|true|,eps)",
        "signed_pct_def": "100*(pred-true)/max(true,eps)  [manuscript Eq.]",
        "device": str(device),
    }
    payload = {"meta": meta, "channels": results}
    PERCENT_JSON.write_text(json.dumps(payload, indent=2))

    # Augment per_channel_test_errors.json model block (full-set values).
    pcj = json.loads(PER_CHANNEL_JSON.read_text()) if PER_CHANNEL_JSON.exists() else {}
    pcj.setdefault(block_name, {})
    for n, s in results.items():
        pcj[block_name][n] = s
    pcj.setdefault("_meta", {})[f"{block_name}_percent_error"] = meta
    PER_CHANNEL_JSON.write_text(json.dumps(pcj, indent=2))

    # Report.
    print("\n========== PER-CHANNEL TEST ERROR (physical space) ==========")
    print(f"profiles evaluated: {seen:,} / {total_samples:,}"
          f"  ({'FULL test set' if meta['full_test_set'] else 'subset'})")
    for n, s in results.items():
        print(f"\n  {n}  (n={s['n_points']:,} layer-points)")
        print(f"    R^2 = {s['r2']:.6f}   MAE = {s['mae']:.4e}   RMSE = {s['rmse']:.4e}")
        print(f"    abs %% error : median={s['abs_pct_median']:.3f}%  mean={s['abs_pct_mean']:.3f}%  "
              f"[p5={s['abs_pct_p5']:.3f}, p25={s['abs_pct_p25']:.3f}, "
              f"p75={s['abs_pct_p75']:.3f}, p95={s['abs_pct_p95']:.3f}]")
        print(f"    signed %% err: median={s['signed_pct_median']:.3f}%  mean={s['signed_pct_mean']:.3f}%")
    print(f"\nWrote {PERCENT_JSON.name} and updated {PER_CHANNEL_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
