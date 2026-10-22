"""Nested RAG@K sensitivity on the fixed Figure 4 set — NO post-hoc clamping.

The 142 Figure 4 commitments and their original K=10 Haiku labels are the fixed
anchor. K=5 and K=20 are derived by a NESTED annotation in which the nesting
constraint is part of the task, not enforced afterward:

K=5 (window can only show <= K=10 execution):
  K10=No  -> K5=No           (no judge call)
  K10=Partial -> judge T+1..T+5, allowed {Partial, No}
  K10=Yes -> judge T+1..T+5, allowed {Yes, Partial, No}

K=20 (window can only show >= K=10 execution):
  K10=Yes -> K20=Yes         (no judge call)
  K10=Partial -> judge full T+1..T+20 (anchored: already Partial by T+10),
                 allowed {Yes, Partial}
  K10=No  -> judge full T+1..T+20 (anchored: unexecuted by T+10),
             allowed {Yes, Partial, No}

Each judge call is TOLD the K=10 label and the permitted labels. If the model
returns a label outside the permitted set, it is NOT silently coerced: the row
is flagged `needs_review=disallowed` and left for manual resolution.

Scoring: RAG@K = (Y + 0.5*P)/n, denominator = all commitments for that model.
Outputs analysis/rag_nested_sweep_results.csv with columns preserving:
  exec_k10, raw_k5, raw_k20 (judge output or AUTO), final_k5, final_k20,
  needs_review, human_corrected, correction_reason.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.rag_extraction import fetch_planning, summarise_tools, CACHE_DIR  # noqa: E402

FIXED_SET = ROOT / "analysis" / "rag_fig4_fixed_set.csv"
OUT = ROOT / "analysis" / "rag_nested_sweep_results.csv"
MODELS = ["claude-opus-4-6", "gemini-3.1-pro-preview", "gpt-5.4"]

RUBRIC = ("Execution rubric — Yes: the committed action is clearly carried out in the log; "
          "Partial: partially carried out (different city/tile than stated, only part of a "
          "multi-part commitment, or started but not clearly completed); No: no evidence. "
          "Reason about meaning, not literal tokens (a settler production satisfies 'settle "
          "to expand south'; a different tech than named is Partial).")


def _call(system: str, user: str, allowed: set[str], api_key: str, cache_tag: str) -> tuple[str, str]:
    """Return (raw_label, evidence). raw_label may be outside `allowed` (caller flags it)."""
    key = hashlib.sha1(f"{cache_tag}|{system}|{user}".encode("utf-8")).hexdigest()[:16]
    cache = CACHE_DIR / f"nested_{key}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["raw"], d.get("evidence", "")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=200,
                                 system=system, messages=[{"role": "user", "content": user}])
    raw = msg.content[0].text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        d = json.loads(raw)
        lab = str(d.get("executed", "")).strip().upper()[:1]
        ev = d.get("evidence", "")
    except Exception:
        lab, ev = "?", "unparseable"
    cache.write_text(json.dumps({"raw": lab, "evidence": ev}))
    return lab, ev


def judge_k5(commitment, planning, tools5, k10, api_key):
    if k10 == "N":
        return "AUTO", "N", "K10=No => K5=No by nesting", ""
    allowed = {"P", "N"} if k10 == "P" else {"Y", "P", "N"}
    permit = "Partial or No (it cannot be Yes, since within 10 turns it was only Partial)" if k10 == "P" \
        else "Yes, Partial, or No"
    system = (f"You judge whether ONE fixed commitment from a Civ VI agent's turn-T plan was executed "
              f"within a SHORT window of only turns T+1..T+5. {RUBRIC}\n"
              f"ANCHOR: within the first 10 turns this commitment was judged "
              f"{'PARTIALLY executed' if k10=='P' else 'FULLY executed (Yes)'}. You are now looking at "
              f"the strictly shorter 5-turn window to see how much had happened by T+5. "
              f"Permitted labels: {permit}. Do not re-extract; judge only this commitment.\n"
              f'Return ONLY JSON: {{"executed":"Y|P|N","evidence":"one line"}}')
    user = f"COMMITMENT: {commitment}\n\nPLANNING NOTE (turn T):\n{planning}\n\nTOOLS T+1..T+5:\n{tools5}"
    raw, ev = _call(system, user, allowed, api_key, "k5")
    return raw, raw, ("" if raw in allowed else "disallowed"), ev


def judge_k20(commitment, planning, tools20, k10, api_key):
    if k10 == "Y":
        return "AUTO", "Y", "K10=Yes => K20=Yes by nesting", ""
    allowed = {"Y", "P"} if k10 == "P" else {"Y", "P", "N"}
    permit = "Yes or Partial (it cannot be No, since it was already Partial by T+10)" if k10 == "P" \
        else "Yes, Partial, or No"
    system = (f"You judge whether ONE fixed commitment from a Civ VI agent's turn-T plan was executed "
              f"within a window of turns T+1..T+20. {RUBRIC}\n"
              f"ANCHOR: within the first 10 turns this commitment was judged "
              f"{'PARTIALLY executed' if k10=='P' else 'NOT executed (No)'}. The full 20-turn log below "
              f"CONTAINS those first 10 turns; focus on whether the LATER turns (T+11..T+20) add "
              f"execution. Permitted labels: {permit}. Do not re-extract; judge only this commitment.\n"
              f'Return ONLY JSON: {{"executed":"Y|P|N","evidence":"one line"}}')
    user = f"COMMITMENT: {commitment}\n\nPLANNING NOTE (turn T):\n{planning}\n\nTOOLS T+1..T+20:\n{tools20}"
    raw, ev = _call(system, user, allowed, api_key, "k20")
    return raw, raw, ("" if raw in allowed else "disallowed"), ev


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY"); sys.exit(1)
    rows = list(csv.DictReader(open(FIXED_SET)))
    assert len(rows) == 142

    pc, tc = {}, {}
    n_call_k5 = n_call_k20 = 0
    out = []
    for i, r in enumerate(rows):
        run_id, turn, k10 = r["run_id"], int(r["turn"]), r["exec_k10"]
        if (run_id, turn) not in pc:
            pc[(run_id, turn)] = fetch_planning(run_id, turn)
        planning = pc[(run_id, turn)]
        def win(k):
            if (run_id, turn, k) not in tc:
                tc[(run_id, turn, k)] = summarise_tools(run_id, turn, k=k)
            return tc[(run_id, turn, k)]

        raw5, fin5, flag5, ev5 = judge_k5(r["commitment"], planning, win(5), k10, api_key)
        raw20, fin20, flag20, ev20 = judge_k20(r["commitment"], planning, win(20), k10, api_key)
        n_call_k5 += raw5 != "AUTO"
        n_call_k20 += raw20 != "AUTO"
        review = ";".join(x for x in [f"k5:{flag5}" if flag5 else "", f"k20:{flag20}" if flag20 else ""] if x)
        out.append({**r, "raw_k5": raw5, "final_k5": fin5, "raw_k20": raw20, "final_k20": fin20,
                    "ev_k5": ev5, "ev_k20": ev20, "needs_review": review,
                    "human_corrected": "", "correction_reason": ""})
        print(f"  [{i+1}/142] {r['model'][:8]:8s} T{turn} k10={k10} -> k5={fin5} k20={fin20}"
              f"{'  <REVIEW '+review+'>' if review else ''}", flush=True)

    fields = ["cid", "run_id", "turn", "model", "commitment", "exec_k10",
              "raw_k5", "final_k5", "ev_k5", "raw_k20", "final_k20", "ev_k20",
              "needs_review", "human_corrected", "correction_reason"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in out:
            w.writerow({k: o.get(k, "") for k in fields})

    print(f"\nHaiku calls: K5={n_call_k5}, K20={n_call_k20} (auto: K5 N={sum(1 for o in out if o['exec_k10']=='N')}, "
          f"K20 Y={sum(1 for o in out if o['exec_k10']=='Y')})")
    print(f"Rows flagged needs_review: {sum(1 for o in out if o['needs_review'])}")
    print(f"Wrote {OUT}")
    _report(out)


def _report(out):
    def stat(rs, col):
        c = Counter(o[col] for o in rs); y, p, n = c["Y"], c["P"], c["N"]; t = y + p + n
        return (y + 0.5 * p) / t if t else 0.0, y, p, n, t
    print("\n=== RAG@K (nested, no clamping) ===")
    print(f"{'Model':22s} {'n':>3s} {'RAG@5':>7s} {'RAG@10':>7s} {'RAG@20':>7s}")
    for m in MODELS:
        rs = [o for o in out if o["model"] == m]
        a, b, c = stat(rs, "final_k5"), stat(rs, "exec_k10"), stat(rs, "final_k20")
        print(f"{m:22s} {a[4]:>3d} {a[0]:>6.1%} {b[0]:>6.1%} {c[0]:>6.1%}")
        print(f"      @5  Y={a[1]:2d} P={a[2]:2d} N={a[3]:2d}   @10 Y={b[1]:2d} P={b[2]:2d} N={b[3]:2d}   @20 Y={c[1]:2d} P={c[2]:2d} N={c[3]:2d}")
    a, b, c = stat(out, "final_k5"), stat(out, "exec_k10"), stat(out, "final_k20")
    print(f"{'OVERALL pooled':22s} {a[4]:>3d} {a[0]:>6.1%} {b[0]:>6.1%} {c[0]:>6.1%}")
    print(f"      @5  Y={a[1]:3d} P={a[2]:2d} N={a[3]:3d}   @10 Y={b[1]:3d} P={b[2]:2d} N={b[3]:3d}   @20 Y={c[1]:3d} P={c[2]:2d} N={c[3]:3d}")
    # validation
    rank = {"N": 0, "P": 1, "Y": 2, "AUTO": -1}
    v5 = sum(1 for o in out if o["final_k5"] in rank and rank[o["final_k5"]] > rank[o["exec_k10"]])
    v20 = sum(1 for o in out if o["final_k20"] in rank and rank[o["final_k20"]] < rank[o["exec_k10"]])
    print(f"\nValidation: K5>K10 violations={v5}, K20<K10 violations={v20} (both must be 0)")
    print("Fig4 K10 check:", {m: round(stat([o for o in out if o['model']==m],'exec_k10')[0],3) for m in MODELS})


if __name__ == "__main__":
    main()
