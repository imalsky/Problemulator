#!/usr/bin/env python3
"""Violin plot of the hero transformer's signed percent error by pressure decade.

Evaluates models/transformer_main on the FULL processed test split (same loader,
checkpoint, and denormalization path as testing/verify_percent_error.py), bins
every valid layer point by pressure decade, and draws one violin per bin and
flux channel. Violin bodies are exact mirrored histograms of ALL points in the
bin (no KDE, no subsampling); overlaid markers give the median and 25-75%
range. Signed percent error follows the manuscript definition:
    100 * (pred - true) / max(|true|, eps),  eps = 1 erg cm^-2 s^-1.

Usage (from Problemulator/):
    conda run -n nn python testing/_make_error_violin.py
    conda run -n nn python testing/_make_error_violin.py --device cpu --max-batches 8

Writes testing/figs/error_violin_pressure.png and
testing/outputs/error_violin_pressure_stats.json.
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

from utils import load_config                                   # noqa: E402
from dataset import create_collate_fn, create_dataset          # noqa: E402
from normalizer import DataNormalizer                          # noqa: E402
from model import create_prediction_model                      # noqa: E402
from torch.utils.data import DataLoader                         # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "transformer_main"
PROC_TEST = PROJECT_ROOT / "data" / "processed" / "test"
NORM_META_PATH = PROJECT_ROOT / "data" / "processed" / "normalization_metadata.json"
FIGS_DIR = THIS_DIR / "figs"
OUTPUTS_DIR = THIS_DIR / "outputs"

PCT_DENOM_FLOOR = 1.0          # eps = 1 erg cm^-2 s^-1, per the manuscript Eq.
# Pressure-decade bin edges in bar; profiles span ~1e-5 to ~1e2 bar.
PRESSURE_EDGES = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
# Fine log-pressure grid for the percentile-band figure; 10 bins per decade so
# each violin decade aggregates exactly 10 fine bins.
FINE_BINS_PER_DECADE = 10
FINE_EDGES = np.logspace(-5, 2, (len(PRESSURE_EDGES) - 1) * FINE_BINS_PER_DECADE + 1)
# Violin bodies span the central 1-99% of each bin; histogram resolution below.
VIOLIN_CLIP_LO, VIOLIN_CLIP_HI = 1.0, 99.0
VIOLIN_HIST_BINS = 160
T_COLOR = "#1f77b4"

CHANNEL_LABELS = {"net_thermal_flux": "Thermal", "net_reflected_flux": "Reflected"}


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
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[violin] device={device}  batch_size={args.batch_size}", flush=True)

    config = load_config(MODEL_DIR / "train_config.json")
    norm_meta = json.load(open(NORM_META_PATH))
    ds_spec = config["data_specification"]
    target_vars = list(ds_spec["target_variables"])
    input_vars = list(ds_spec["input_variables"])
    flux_vars = [n for n in ("net_thermal_flux", "net_reflected_flux") if n in target_vars]
    p_idx = input_vars.index("pressure_bar")
    padding_value = float(ds_spec["padding_value"])
    padding_epsilon = float(config["normalization"]["padding_comparison_epsilon"])

    base = create_prediction_model(config=config, device=device, compile_model=False).eval()
    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    base.load_state_dict(state, strict=True)
    base = base.to(device=device, dtype=torch.float32).eval()

    total_samples = int(json.load(open(PROC_TEST / "metadata.json"))["total_samples"])
    ds_full = create_dataset(PROC_TEST, config, list(range(total_samples)))
    loader = DataLoader(
        ds_full, batch_size=args.batch_size, shuffle=False,
        collate_fn=create_collate_fn(padding_value, padding_epsilon), drop_last=False,
    )
    n_batches = len(loader)
    cap = args.max_batches if args.max_batches > 0 else n_batches
    print(f"[violin] test profiles={total_samples:,}  batches={n_batches}  running={min(cap, n_batches)}", flush=True)

    n_bins = len(PRESSURE_EDGES) - 1
    n_fine = len(FINE_EDGES) - 1
    # per channel: list of per-fine-bin float32 chunks; decade bin i aggregates
    # fine bins [10*i, 10*i+9]
    store = {n: [[] for _ in range(n_fine)] for n in flux_vars}
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
            pres = denorm_channel(inputs_norm["sequence"][..., p_idx].to(torch.float32),
                                  "pressure_bar", norm_meta)[valid].numpy()
            # boundary points sitting exactly on the last edge belong to the
            # closing bin; everything else is in range by construction
            pres = np.clip(pres, FINE_EDGES[0], np.nextafter(FINE_EDGES[-1], 0.0))
            bin_idx = np.digitize(pres, FINE_EDGES, right=False) - 1
            for n in flux_vars:
                j = target_vars.index(n)
                pred = denorm_channel(out_norm[..., j], n, norm_meta)[valid].numpy()
                true = denorm_channel(tgt_norm[..., j], n, norm_meta)[valid].numpy()
                signed_pct = (100.0 * (pred - true)
                              / np.maximum(np.abs(true), PCT_DENOM_FLOOR)).astype(np.float32)
                for i in range(n_fine):
                    sel = bin_idx == i
                    if sel.any():
                        store[n][i].append(signed_pct[sel])
            seen += seq_norm.shape[0]
            if (b_idx + 1) % 25 == 0 or b_idx + 1 == min(cap, n_batches):
                rate = seen / max(time.perf_counter() - t0, 1e-9)
                print(f"[violin]   batch {b_idx + 1}/{min(cap, n_batches)}  "
                      f"{seen:,} profiles  ({rate:,.0f}/s)", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = THIS_DIR / "science.mplstyle"
    if style.is_file():
        plt.style.use(str(style))

    FIGS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(flux_vars), figsize=(13, 5.4), sharey=True)
    if len(flux_vars) == 1:
        axes = [axes]

    stats_out = {"meta": {
        "model_dir": "models/transformer_main",
        "n_profiles_evaluated": seen,
        "n_profiles_total": total_samples,
        "full_test_set": cap >= n_batches,
        "signed_pct_def": "100*(pred-true)/max(|true|,eps), eps=1 erg cm^-2 s^-1",
        "pressure_bin_edges_bar": PRESSURE_EDGES.tolist(),
        "violin_body": f"exact mirrored histogram of all points, clipped to {VIOLIN_CLIP_LO}-{VIOLIN_CLIP_HI} percentiles",
    }, "channels": {}}

    for ax, n in zip(axes, flux_vars):
        chan_stats = []
        for i in range(n_bins):
            fine_slice = store[n][i * FINE_BINS_PER_DECADE:(i + 1) * FINE_BINS_PER_DECADE]
            chunks = [c for fb in fine_slice for c in fb]
            vals = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
            xc = i  # violin center on a categorical axis
            if vals.size < 10:
                chan_stats.append({"n": int(vals.size)})
                continue
            lo, med, hi = np.percentile(vals, [25.0, 50.0, 75.0])
            clo, chi = np.percentile(vals, [VIOLIN_CLIP_LO, VIOLIN_CLIP_HI])
            body = vals[(vals >= clo) & (vals <= chi)]
            hist, edges = np.histogram(body, bins=VIOLIN_HIST_BINS, range=(clo, chi), density=True)
            # light smoothing so the outline is readable at 160 bins
            kernel = np.ones(5) / 5.0
            hist = np.convolve(hist, kernel, mode="same")
            half_w = 0.42 * hist / hist.max()
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.fill_betweenx(centers, xc - half_w, xc + half_w,
                             color=T_COLOR, alpha=0.55, linewidth=0.6,
                             edgecolor=T_COLOR, zorder=3)
            ax.plot([xc - 0.18, xc + 0.18], [med, med], color="k", lw=1.4, zorder=5)
            ax.plot([xc, xc], [lo, hi], color="k", lw=1.0, zorder=4)
            chan_stats.append({
                "n": int(vals.size), "median": float(med),
                "p25": float(lo), "p75": float(hi),
                "p1": float(clo), "p99": float(chi),
            })
        ax.axhline(0.0, color="gray", ls="--", lw=1.0, alpha=0.8, zorder=1)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f"$10^{{{int(np.log10(PRESSURE_EDGES[i]))}}}$–$10^{{{int(np.log10(PRESSURE_EDGES[i+1]))}}}$"
                            for i in range(n_bins)], fontsize=9)
        ax.set_xlabel("Pressure bin (bar)")
        ax.set_title(CHANNEL_LABELS.get(n, n))
        stats_out["channels"][n] = chan_stats
    axes[0].set_ylabel("Signed % error")

    fig.tight_layout()
    out = FIGS_DIR / "error_violin_pressure.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    (OUTPUTS_DIR / "error_violin_pressure_stats.json").write_text(json.dumps(stats_out, indent=2))
    print(f"[violin] wrote {out}")
    print(f"[violin] wrote {OUTPUTS_DIR / 'error_violin_pressure_stats.json'}")

    # ---- Percentile-band figure (same data, fine pressure grid). Mirrors the
    # notebook's pressure_error_pastel_bands cell, but over the full test set.
    BAND_1, BAND_5, BAND_25, MEAN_LINE = "#D4E8F7", "#C8E6C9", "#FFE0B2", "#2C3E50"
    ERROR_XLIM = 10.0
    figb, axesb = plt.subplots(1, len(flux_vars), figsize=(14, 7), sharey=True, sharex=True)
    if len(flux_vars) == 1:
        axesb = [axesb]
    fine_centers_all = np.sqrt(FINE_EDGES[:-1] * FINE_EDGES[1:])
    for ax, n in zip(axesb, flux_vars):
        centers, means, q25, q75, q05, q95, q01, q99 = [], [], [], [], [], [], [], []
        for i in range(n_fine):
            vals = np.concatenate(store[n][i]) if store[n][i] else np.empty(0, np.float32)
            if vals.size < 3:
                continue
            centers.append(fine_centers_all[i])
            means.append(float(np.mean(vals)))
            p = np.percentile(vals, [25, 75, 5, 95, 1, 99])
            q25.append(p[0])
            q75.append(p[1])
            q05.append(p[2])
            q95.append(p[3])
            q01.append(p[4])
            q99.append(p[5])
        centers = np.array(centers)
        ax.fill_betweenx(centers, q01, q99, color=BAND_1, alpha=0.9, edgecolor="none",
                         label="1–99%", zorder=2)
        ax.fill_betweenx(centers, q05, q95, color=BAND_5, alpha=0.9, edgecolor="none",
                         label="5–95%", zorder=3)
        ax.fill_betweenx(centers, q25, q75, color=BAND_25, alpha=0.9, edgecolor="none",
                         label="25–75% (IQR)", zorder=4)
        ax.plot(means, centers, color=MEAN_LINE, linewidth=2.5, label="Mean", zorder=5)
        ax.axvline(0.0, color="gray", linestyle="dashed", linewidth=3.0, alpha=0.7, zorder=10)
        label = CHANNEL_LABELS.get(n, n)
        ax.set_xlabel(f"{label} Signed % Error")
        ax.set_xlim(-ERROR_XLIM, ERROR_XLIM)
        ax.set_yscale("log")
        ax.grid(False, which="both")
        if ax is axesb[0]:
            ax.legend(loc="upper left", framealpha=0.95, fancybox=True, shadow=True)
    axesb[0].set_ylabel("Pressure (bar)")
    axesb[0].set_ylim(FINE_EDGES[-1], FINE_EDGES[0])  # pressure decreasing upward
    figb.tight_layout()
    outb = FIGS_DIR / "pressure_error_pastel_bands_fulltest.png"
    figb.savefig(outb, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[violin] wrote {outb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
