#!/usr/bin/env python3
"""Synthesis figure: when does the transformer win vs the LSTM?

Uses the held-out test loss (masked MSE) from each model's test_metrics.json.
Comparisons are made WITHIN matched pairs (transformer vs LSTM trained under
identical conditions and the SAME normalization), so the within-pair loss ratio
LSTM/Transformer is normalization-independent -- it is the robust headline metric
that sidesteps the cross-folder symlog-threshold difference (see when_each_wins.md).

All numbers are verified from the model dirs (see _gather check in the report):
  - extra_runs (5M data): transformer_4m/lstm_4m, transformer_8m_300ep/lstm_8m_compare,
                          transformer_8m_fp32/lstm_8m_fp32, transformer_final/lstm_final
  - main (5M data):        transformer_main/lstm_main
  - main (2M / 40% data):  transformer_main_v8_fastwin/lstm_main_v8_fastwin
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TESTING_DIR = Path(__file__).resolve().parent
OUT_DIR = TESTING_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
STYLE = TESTING_DIR / "science.mplstyle"
if STYLE.is_file():
    plt.style.use(str(STYLE))

T_COLOR = "#1f77b4"   # transformer
L_COLOR = "#d62728"   # lstm

# regime label, data desc, params(M), transformer test_loss, lstm test_loss
REGIMES = [
    ("2M data\n(40% subset)",            "2M",  5.3, 2.762200e-05, 5.212543e-05),
    ("4M params\nbf16 AMP, warmup 2",    "5M",  4.0, 2.2284617e-05, 6.2173582e-05),
    ("8M params\nbf16 AMP, batch 512",   "5M",  8.0, 5.2428702e-05, 8.4000254e-05),
    ("8M params\nfull data, tuned",      "5M",  8.0, 6.3240459e-06, 4.4157464e-06),
    ("8M params\nfp32 + EMA, batch 512", "5M",  8.0, 8.5377205e-06, 4.5442962e-06),
    ("10M params\nfull data, tuned",     "5M",  9.7, 9.9478029e-06, 6.5983772e-06),
]
labels = [r[0] for r in REGIMES]
t_loss = np.array([r[3] for r in REGIMES])
l_loss = np.array([r[4] for r in REGIMES])
ratio = l_loss / t_loss              # >1 transformer better; <1 lstm better

# Top-3 trials per architecture from the 64-trial tuning sweep
# (validation loss; 60 epochs on a 20% subsample). Source:
# Problemulator/testing/outputs/tuning_all_trials.csv (trials 59/32/60 and 4/41/58).
TUNE_T = np.array([3.124680e-05, 3.130445e-05, 3.150489e-05])
TUNE_L = np.array([5.132835e-05, 5.375416e-05, 5.694755e-05])
TUNE_LABEL = "Tuning sweep\ntop 3 trials each"

fig, ax = plt.subplots(figsize=(9.5, 5.2))

# Single panel: absolute loss, grouped bars, LINEAR y. Leftmost group shows the
# top-3 tuning-sweep trials of each architecture; the remaining groups are the
# matched full-training pairs.
LOSS_SCALE = 1e-5
x_tune = 0.0
x = np.arange(len(REGIMES)) + 1.5
w = 0.38
wt = 0.12
for k in range(3):
    ax.bar(x_tune - 0.31 + k * wt, TUNE_T[k] / LOSS_SCALE, wt, color=T_COLOR)
    ax.bar(x_tune + 0.07 + k * wt, TUNE_L[k] / LOSS_SCALE, wt, color=L_COLOR)
ax.bar(x - w/2, t_loss / LOSS_SCALE, w, color=T_COLOR, label="Transformer")
ax.bar(x + w/2, l_loss / LOSS_SCALE, w, color=L_COLOR, label="LSTM")
ax.set_ylabel(r"Held-out loss (masked MSE, $\times 10^{-5}$)")
ax.set_xticks(np.concatenate(([x_tune], x)))
ax.set_xticklabels([TUNE_LABEL] + labels, fontsize=8)
ymax = max(t_loss.max(), l_loss.max(), TUNE_L.max()) / LOSS_SCALE
ax.set_ylim(0, ymax * 1.08)
ax.set_xlim(-0.6, x[-1] + 0.6)
ax.legend(frameon=False, loc="upper right")

fig.tight_layout()
for ext in ("png",):
    p = OUT_DIR / f"when_each_wins.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)

# Print the table for the report / sanity.
print("\nregime, params(M), T_loss, L_loss, ratio L/T, winner")
for (lab, _, pm, tl, ll), r in zip(REGIMES, ratio):
    win = "Transformer %.2fx" % r if r > 1 else "LSTM %.2fx" % (1/r)
    print(f"  {lab.splitlines()[0]:12s} {pm:>4} {tl:.4e} {ll:.4e} {r:.3f}  {win}")
