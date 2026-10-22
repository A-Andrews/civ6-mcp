"""Aggregate the fixed-commitment-set RAG sweep.

Reads per-model verdict files rag_verdicts_<slug>.json where each commitment
carries three verdicts (exec_k5/exec_k10/exec_k20) judged against the T+1..T+K
tool windows. Because the commitment set is frozen per turn, n is identical
across K and only the verdict varies — isolating the K-effect from judge/count
variance.

Verdict schema (list of turns):
  [{"run_id","model","scenario","turn",
    "commitments":[{"text","exec_k5","exec_k10","exec_k20","evidence"}]}]

Usage: .venv/bin/python analysis/aggregate_rag_sweep.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KS = [5, 10, 20]


def _rag(verdicts: list[str]) -> tuple[float, int, int, int]:
    c = Counter(v.strip().upper()[:1] for v in verdicts)
    y, p, n = c.get("Y", 0), c.get("P", 0), c.get("N", 0)
    tot = y + p + n
    return ((y + 0.5 * p) / tot if tot else float("nan")), y, p, n


def main() -> None:
    files = sorted((ROOT / "analysis").glob("rag_verdicts_*.json"))
    # exclude the old per-K single-verdict files from the failed sweep
    files = [f for f in files if not any(f.name.endswith(f"_k{k}.json") for k in KS)]
    if not files:
        print("No rag_verdicts_<slug>.json sweep files found.")
        return

    turns = []
    for f in files:
        turns.extend(json.loads(f.read_text()))

    by_model: dict[str, list[dict]] = {}
    for t in turns:
        by_model.setdefault(t["model"], []).append(t)

    print(f"Fixed-commitment-set RAG sweep ({len(turns)} turns, {len(files)} chunks)\n")
    header = f"{'model':28s} " + "  ".join(f"RAG@{k}" for k in KS) + "     n"
    print(header)
    print("-" * len(header))

    for model in sorted(by_model):
        commits = [c for t in by_model[model] for c in t.get("commitments", [])]
        cells = []
        for k in KS:
            rag, y, p, n = _rag([c.get(f"exec_k{k}", "N") for c in commits])
            cells.append(f"{rag:5.1%}")
        print(f"{model:28s} " + "  ".join(cells) + f"   {len(commits):4d}")

    # overall
    allc = [c for t in turns for c in t.get("commitments", [])]
    cells = []
    for k in KS:
        rag, y, p, n = _rag([c.get(f"exec_k{k}", "N") for c in allc])
        cells.append(f"{rag:5.1%}")
    print("-" * len(header))
    print(f"{'OVERALL':28s} " + "  ".join(cells) + f"   {len(allc):4d}")

    # per-K N-rate as a consistency check (should be non-zero and decreasing with K)
    print("\nN-rate (not-done fraction) by K — sanity check:")
    for k in KS:
        _, y, p, n = _rag([c.get(f"exec_k{k}", "N") for c in allc])
        tot = y + p + n
        print(f"  @{k}: N={n}/{tot} = {n/tot:.0%}   (Y={y} P={p})")


if __name__ == "__main__":
    main()
