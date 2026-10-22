"""Aggregate Haiku-subagent RAG verdicts into per-model scores + CSV.

Reads analysis/rag_verdicts_k{K}.json (written by the Haiku judge subagent) and
produces analysis/rag_auto_results_k{K}.csv, using the identical RAG formula as
rag_extraction.run_auto_rag:  RAG = (Y + 0.5*P) / n_commitments.

Verdict file schema (list of turns):
  [{"run_id","model","scenario","turn",
    "commitments":[{"text","executed":"Y|P|N","evidence"}]}, ...]

Usage: .venv/bin/python analysis/aggregate_rag_verdicts.py 5 10 20
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def aggregate(k: int) -> None:
    verdict_path = ROOT / "analysis" / f"rag_verdicts_k{k}.json"
    if not verdict_path.exists():
        print(f"MISSING: {verdict_path} — run the judge subagent for k={k} first")
        return
    turns = json.loads(verdict_path.read_text())

    rows = []
    for t in turns:
        for c in t.get("commitments", []):
            rows.append({
                "run_id": t["run_id"],
                "model": t.get("model", "unknown"),
                "scenario": t.get("scenario", "unknown"),
                "turn": t["turn"],
                "commitment": c.get("text", ""),
                "executed": (c.get("executed") or "N").strip().upper()[:1],
                "evidence": c.get("evidence", ""),
            })

    suffix = "" if k == 10 else f"_k{k}"
    # subagent variant kept separate from the script's canonical @10 file
    out = ROOT / "analysis" / f"rag_auto_results_subagent_k{k}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "model", "scenario", "turn",
                                          "commitment", "executed", "evidence"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== RAG@{k} (Haiku subagent judge, {len(rows)} commitments) ===")
    for model in sorted(set(r["model"] for r in rows)):
        m = [r for r in rows if r["model"] == model]
        n = len(m)
        y = sum(1 for r in m if r["executed"] == "Y")
        p = sum(1 for r in m if r["executed"] == "P")
        rag = (y + 0.5 * p) / n if n else float("nan")
        print(f"  {model:28s} RAG={rag:5.1%}  ({y}Y + {p}P + {n-y-p}N / {n})")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    for k in ([int(a) for a in sys.argv[1:]] or [10]):
        aggregate(k)
