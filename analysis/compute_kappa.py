"""Compute Cohen's kappa from the filled-in labeling sheet.

Run:  .venv/bin/python analysis/compute_kappa.py

Reads analysis/kappa_labeling_sheet.csv (with human_label column filled in).
Reports Cohen's kappa, agreement rate, and confusion matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

SHEET_PATH = ROOT / "analysis" / "kappa_labeling_sheet.csv"


def cohens_kappa(y1, y2, labels=None):
    """Compute Cohen's kappa between two label arrays."""
    if labels is None:
        labels = sorted(set(y1) | set(y2))
    n = len(y1)
    assert n == len(y2), "Arrays must be same length"

    # Confusion matrix
    k = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(y1, y2):
        matrix[label_idx[a]][label_idx[b]] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = (row_sums * col_sums).sum()

    # Kappa
    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return kappa, p_o, matrix, labels


def main():
    if not SHEET_PATH.exists():
        print(f"Error: {SHEET_PATH} not found.")
        print(f"Run: .venv/bin/python analysis/generate_kappa_sheet.py first.")
        return

    df = pd.read_csv(SHEET_PATH, index_col="id")

    # Check human labels are filled
    empty = df["human_label"].isna() | (df["human_label"].str.strip() == "")
    if empty.any():
        n_empty = empty.sum()
        print(f"Warning: {n_empty} of {len(df)} rows have empty human_label.")
        df = df[~empty]
        print(f"Computing kappa on {len(df)} labeled rows.\n")

    if len(df) < 5:
        print("Too few labeled rows for meaningful kappa.")
        return

    auto = df["auto_label"].str.strip().str.upper().values
    human = df["human_label"].str.strip().str.upper().values

    valid_labels = {"Y", "P", "N"}
    bad_auto = [v for v in auto if v not in valid_labels]
    bad_human = [v for v in human if v not in valid_labels]
    if bad_auto:
        print(f"Error: auto_label contains unexpected values: {set(bad_auto)}")
        return
    if bad_human:
        print(f"Error: human_label contains unexpected values: {set(bad_human)}")
        print("Expected Y, P, or N. Fix the sheet and re-run.")
        return

    labels = ["Y", "P", "N"]
    kappa, agreement, matrix, _ = cohens_kappa(auto, human, labels=labels)

    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Raw agreement: {agreement:.1%} ({int(agreement * len(df))}/{len(df)})")
    print()
    print("Confusion matrix (rows=auto, cols=human):")
    print(f"         {'  '.join(labels)}")
    for i, label in enumerate(labels):
        row = "  ".join(f"{matrix[i][j]:3d}" for j in range(len(labels)))
        print(f"  {label}:   {row}")
    print()

    # Interpretation
    if kappa >= 0.81:
        interp = "almost perfect"
    elif kappa >= 0.61:
        interp = "substantial"
    elif kappa >= 0.41:
        interp = "moderate"
    elif kappa >= 0.21:
        interp = "fair"
    else:
        interp = "slight"
    print(f"Interpretation (Landis & Koch): {interp} agreement")
    print()
    print(f"For paper: \"Cohen's κ = {kappa:.2f} ({interp} agreement, n={len(df)})\"")


if __name__ == "__main__":
    main()
