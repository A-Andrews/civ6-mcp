"""Generate a SECOND stratified sample of ~50 commitments for Cohen's kappa validation.

This extends the original 50-item validation (analysis/kappa_labeling_sheet.csv)
with 50 fresh commitments drawn from the remaining, not-yet-labeled pool. Use it to
check whether the Haiku auto-labels keep lining up with human perception at n≈100.

Run:  .venv/bin/python analysis/generate_kappa_sheet_batch2.py

Outputs analysis/kappa_labeling_sheet_batch2.csv — same columns as batch 1:
  id, run_id, model, turn, commitment, evidence, auto_label, human_label

The human_label column is empty for you to fill in (Y/P/N).
After filling in, run: .venv/bin/python analysis/compute_kappa.py --pool
(the --pool flag scores batch 1 + batch 2 together).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

CSV_PATH = ROOT / "analysis" / "rag_auto_results.csv"
BATCH1_PATH = ROOT / "analysis" / "kappa_labeling_sheet.csv"
OUTPUT_PATH = ROOT / "analysis" / "kappa_labeling_sheet_batch2.csv"

KEY = ["run_id", "turn", "commitment"]
TARGET = 50


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total commitments in source: {len(df)}")

    # Exclude anything already in batch 1 (match on run_id + turn + commitment)
    b1 = pd.read_csv(BATCH1_PATH)
    b1_keys = set(map(tuple, b1[KEY].astype(str).values))
    df_key = df[KEY].astype(str).apply(tuple, axis=1)
    remaining = df[~df_key.isin(b1_keys)].copy()
    print(f"Already labeled in batch 1: {len(b1_keys)} keys")
    print(f"Remaining pool for batch 2: {len(remaining)}")
    print(f"Remaining distribution: {remaining['executed'].value_counts().to_dict()}")

    rng = np.random.RandomState(43)  # distinct seed from batch 1 (42)

    # Stratified by model x executed. Proportional allocation of TARGET across
    # strata, but take everything in a stratum if it is smaller than its share.
    strata = list(remaining.groupby(["model", "executed"]))
    total = len(remaining)

    # First pass: floor of proportional share, min 1 where the stratum exists.
    alloc = {}
    for (model, label), group in strata:
        share = TARGET * len(group) / total
        alloc[(model, label)] = min(len(group), max(1, int(np.floor(share))))

    # Distribute any leftover slots to the strata with the most headroom.
    group_size = {k: len(g) for k, g in strata}
    while sum(alloc.values()) < min(TARGET, total):
        # pick stratum with largest remaining headroom
        cand = max(alloc, key=lambda k: group_size[k] - alloc[k])
        if group_size[cand] - alloc[cand] <= 0:
            break
        alloc[cand] += 1

    samples = []
    for (model, label), group in strata:
        n = alloc[(model, label)]
        if n > 0:
            samples.append(group.sample(n=n, random_state=rng))

    sampled = pd.concat(samples)
    sampled = sampled.sample(frac=1, random_state=rng).reset_index(drop=True)
    sampled.index = range(1, len(sampled) + 1)
    sampled.index.name = "id"

    sheet = sampled[["run_id", "model", "turn", "commitment", "evidence"]].copy()
    sheet["auto_label"] = sampled["executed"]
    sheet["human_label"] = ""

    sheet.to_csv(OUTPUT_PATH)
    print(f"\nGenerated {len(sheet)} items → {OUTPUT_PATH}")
    print(f"\nDistribution in batch-2 sample:")
    print(sheet["auto_label"].value_counts().to_string())
    print(f"\nPer model:")
    print(sheet.groupby("model")["auto_label"].value_counts().to_string())
    print(f"\nInstructions:")
    print(f"  1. Open {OUTPUT_PATH}")
    print(f"  2. For each row, read commitment + evidence")
    print(f"  3. Fill in human_label with Y, P, or N")
    print(f"  4. Save and run: .venv/bin/python analysis/compute_kappa.py --pool")


if __name__ == "__main__":
    main()
