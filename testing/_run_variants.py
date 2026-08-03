#!/usr/bin/env python3
"""Execute paper_revision.ipynb for each new model pair, writing per-variant figs.

Runs the canonical evaluation notebook once per (MODEL_VARIANT, TRANSFORMER_DIR,
LSTM_DIR) triple. Uses allow_errors=True so a single failing cell (e.g. a
sweep-only or benchmark cell) does not abort the whole figure set. Each run
writes figures into testing/figs<VARIANT>/.

Invoke from anywhere; it chdir's into its own testing/ dir so the notebook's
PROJECT_ROOT resolves to Problemulator_extra_runs.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TESTING_DIR = Path(__file__).resolve().parent
NB_PATH = TESTING_DIR / "paper_revision.ipynb"

# (variant label, transformer dir, lstm dir)
PAIRS = [
    ("_final", "transformer_final", "lstm_final"),
    ("_4m", "transformer_4m_full_compare", "lstm_4m_full_compare"),
]

os.chdir(TESTING_DIR)

for variant, tf_dir, lstm_dir in PAIRS:
    os.environ["MODEL_VARIANT"] = variant
    os.environ["TRANSFORMER_DIR"] = tf_dir
    os.environ["LSTM_DIR"] = lstm_dir
    print(f"\n{'='*70}\n[run] variant={variant}  TF={tf_dir}  LSTM={lstm_dir}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=2400,                 # per-cell cap (40 min)
        kernel_name="python3",
        allow_errors=True,            # keep going past a failing cell
        resources={"metadata": {"path": str(TESTING_DIR)}},
    )
    client.execute()
    out_path = TESTING_DIR / f"_executed_paper_revision{variant}.ipynb"
    nbformat.write(nb, out_path)
    # Surface any cell errors for the log.
    n_err = 0
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                n_err += 1
                print(f"[run]   CELL {i} ERROR: {o.get('ename')}: {o.get('evalue')}", flush=True)
    figs_dir = TESTING_DIR / f"figs{variant}"
    n_figs = len(list(figs_dir.glob('*.png'))) if figs_dir.is_dir() else 0
    print(f"[run] variant={variant} DONE in {time.perf_counter()-t0:.0f}s  "
          f"cell_errors={n_err}  figs={n_figs} -> {figs_dir}", flush=True)

print("\n[run] ALL VARIANTS COMPLETE", flush=True)
