#!/usr/bin/env python3
"""Plot training progression from log."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "trained_model"
STYLE_PATH = THIS_FILE.with_name("science.mplstyle")
if not STYLE_PATH.is_file():
    raise FileNotFoundError(f"Missing matplotlib style file: {STYLE_PATH}")
plt.style.use(str(STYLE_PATH))

def plot_training():
    """Plot training curves."""
    log_path = MODEL_DIR / "training_log.csv"
    if not log_path.exists():
        print(f"No log found: {log_path}")
        return
    
    df = pd.read_csv(log_path)
    required_cols = {"epoch", "train_loss", "val_loss", "lr"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"training_log.csv missing required columns: {missing}")

    df = df.dropna(subset=["epoch", "train_loss", "val_loss", "lr"]).copy()
    df = df[np.isfinite(df["epoch"]) & np.isfinite(df["train_loss"]) & np.isfinite(df["val_loss"]) & np.isfinite(df["lr"])]
    if df.empty:
        raise RuntimeError("No finite rows available in training_log.csv for plotting.")
    df = df.sort_values("epoch")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.scatter(df["epoch"], df["train_loss"], color='black', label="Train", alpha=0.7)
    ax1.scatter(df["epoch"], df["val_loss"], color='red',label="Validation", alpha=0.7)

    
    best_idx = df["val_loss"].idxmin()
    ax1.axvline(df.loc[best_idx, "epoch"], color="gray", linestyle=":", alpha=0.6, label="Best epoch")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_yscale('log')
    ax1.legend()
    
    # Learning rate
    ax2.plot(df["epoch"], df["lr"], color="black")
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")

    fig.tight_layout()
    save_path = MODEL_DIR / "plots" / "training_curves.png"
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Total epochs: {len(df)}")
    print(f"Best val loss: {df.loc[best_idx, 'val_loss']:.3e} (epoch {df.loc[best_idx, 'epoch']})")
    print(f"Final train loss: {df.iloc[-1]['train_loss']:.3e}")
    print(f"Final val loss: {df.iloc[-1]['val_loss']:.3e}")

if __name__ == "__main__":
    plot_training()
