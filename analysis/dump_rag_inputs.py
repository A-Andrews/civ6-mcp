"""Dump RAG judge inputs (planning text + tools summary) per sample turn.

Reuses rag_extraction.fetch_planning + summarise_tools so the inputs are
byte-identical to what the Haiku judge would see in run_auto_rag. Writes one
JSON per K to analysis/rag_inputs_k{K}.json. No ANTHROPIC key needed — this is
the extraction half of the pipeline, before any messages.create call.

Usage: .venv/bin/python analysis/dump_rag_inputs.py 5 10 20
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.rag_extraction import (  # noqa: E402
    SAMPLE_TURNS,
    fetch_planning,
    summarise_tools,
)
from analysis.admissible_games import ADMISSIBLE_GAMES  # noqa: E402


def dump(k: int) -> Path:
    meta_by_run = {g["run_id"]: g for g in ADMISSIBLE_GAMES}
    rows = []
    total = len(SAMPLE_TURNS)
    for i, (run_id, turn) in enumerate(SAMPLE_TURNS):
        print(f"  [{i+1}/{total}] k={k} {run_id} T{turn}", flush=True)
        planning = fetch_planning(run_id, turn)
        if not planning.strip():
            print("    (no planning text — skipping)")
            continue
        tools = summarise_tools(run_id, turn, k=k)
        meta = meta_by_run.get(run_id, {})
        rows.append({
            "run_id": run_id,
            "model": meta.get("model", "unknown"),
            "scenario": meta.get("scenario", "unknown"),
            "turn": turn,
            "planning": planning,
            "tools": tools,
        })
    out = ROOT / "analysis" / f"rag_inputs_k{k}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out} ({len(rows)} turns)")
    return out


if __name__ == "__main__":
    ks = [int(a) for a in sys.argv[1:]] or [10]
    for k in ks:
        dump(k)
