"""Generate a stratified sample of ~50 commitments for Cohen's kappa validation.

Run:  .venv/bin/python analysis/generate_kappa_sheet.py

Outputs analysis/kappa_labeling_sheet.csv — a sheet with columns:
  id, run_id, model, turn, commitment, evidence, auto_label, human_label

The human_label column is empty for you to fill in (Y/P/N).
After filling in, run: .venv/bin/python analysis/compute_kappa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

CSV_PATH = ROOT / "analysis" / "rag_auto_results.csv"
OUTPUT_PATH = ROOT / "analysis" / "kappa_labeling_sheet.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total commitments: {len(df)}")
    print(f"Distribution: {df['executed'].value_counts().to_dict()}")

    # Stratified sample: ~50 total, proportional by model x executed,
    # with minimum 2 per stratum where possible
    TARGET = 50
    rng = np.random.RandomState(42)

    # Build strata
    strata = df.groupby(["model", "executed"])
    n_strata = len(strata)
    per_stratum_base = max(2, TARGET // n_strata)

    samples = []
    for (model, label), group in strata:
        n_sample = min(len(group), per_stratum_base)
        samples.append(group.sample(n=n_sample, random_state=rng))

    sampled_orig = pd.concat(samples)  # preserve original df indices for exclusion
    sampled_orig_idx = set(sampled_orig.index)
    sampled = sampled_orig.reset_index(drop=True)

    # If under target, add more from largest strata
    remaining = TARGET - len(sampled)
    if remaining > 0:
        unsampled = df[~df.index.isin(sampled_orig_idx)]
        if len(unsampled) >= remaining:
            extras = unsampled.sample(n=remaining, random_state=rng)
            sampled = pd.concat([sampled, extras], ignore_index=True)

    # Shuffle
    sampled = sampled.sample(frac=1, random_state=rng).reset_index(drop=True)
    sampled.index = range(1, len(sampled) + 1)
    sampled.index.name = "id"

    # Build output sheet
    sheet = sampled[["run_id", "model", "turn", "commitment", "evidence"]].copy()
    sheet["auto_label"] = sampled["executed"]
    sheet["human_label"] = ""

    sheet.to_csv(OUTPUT_PATH)
    print(f"\nGenerated {len(sheet)} items → {OUTPUT_PATH}")
    print(f"\nDistribution in sample:")
    print(sheet["auto_label"].value_counts().to_string())
    print(f"\nPer model:")
    print(sheet.groupby("model")["auto_label"].value_counts().to_string())
    print(f"\nInstructions:")
    print(f"  1. Open {OUTPUT_PATH}")
    print(f"  2. For each row, read commitment + evidence")
    print(f"  3. Fill in human_label with Y, P, or N")
    print(f"  4. Save and run: .venv/bin/python analysis/compute_kappa.py")


if __name__ == "__main__":
    main()
