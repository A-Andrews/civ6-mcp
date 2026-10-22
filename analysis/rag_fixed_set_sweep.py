"""Fixed-commitment-set RAG@K sensitivity (K=5,10,20) on the exact Figure 4 set.

Methodology (per rebuttal requirements):
- The commitment set is FIXED = the 142 commitments underlying Figure 4 of the
  submitted paper (recovered from the original Haiku run's cache; persisted in
  analysis/rag_fig4_fixed_set.csv with per-commitment K=10 labels).
- K=10 uses the paper's ORIGINAL labels verbatim (reproduces Figure 4 exactly).
- K=5 and K=20 re-judge each SAME commitment's *execution* against the 5- and
  20-turn tool windows, using the same Haiku model (claude-haiku-4-5-20251001)
  and the same Y/P/N execution rubric, in an execution-only framing (the
  commitment text is given; nothing is re-extracted).
- Nested-window property enforced: rank(K5) <= rank(K10) <= rank(K20). Because
  the windows are nested, a commitment cannot be "more executed" in a smaller
  window; any raw judge output violating this is clamped to the K=10 anchor and
  counted (QA metric).
- Scoring: RAG@K = (Y + 0.5*P)/n. Denominator = all commitments for that model
  (Y+P+N). Overall = pooled over all 142.

Usage: ANTHROPIC_API_KEY=... .venv/bin/python analysis/rag_fixed_set_sweep.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.rag_extraction import fetch_planning, summarise_tools, CACHE_DIR  # noqa: E402

FIXED_SET = ROOT / "analysis" / "rag_fig4_fixed_set.csv"
MODELS = ["claude-opus-4-6", "gemini-3.1-pro-preview", "gpt-5.4"]
RANK = {"N": 0, "P": 1, "Y": 2}
INV = {0: "N", 1: "P", 2: "Y"}

EXEC_JUDGE_SYSTEM = """You are given ONE specific commitment that a Civilization VI agent stated in its turn-T planning note, plus a log of the tool calls it actually made over the next {k} turns. Determine whether THAT ONE commitment was executed within this window.

Judge execution against the tool log:
- Y = clearly done (the committed action plainly appears in the log)
- P = partially done (done in a different city/tile than stated, only part of a multi-part commitment carried out, or started but not clearly completed)
- N = not done (no evidence in the log)

Reason about meaning, not literal tokens: "build a settler to expand south" is satisfied by a settler production; a research commitment fulfilled by a different tech than the one named is P, not Y; a movement/attack near the intended target counts.

Do NOT invent or extract new commitments. Judge ONLY the single commitment given.
Return ONLY JSON: {"executed":"Y|P|N","evidence":"one-line reason"}"""


def judge(commitment: str, planning: str, tools: str, k: int, api_key: str) -> str:
    key = hashlib.sha1(f"execjudge|k{k}|{commitment}|{tools}".encode("utf-8")).hexdigest()[:16]
    cache = CACHE_DIR / f"execjudge_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())["executed"]

    import anthropic, re
    client = anthropic.Anthropic(api_key=api_key)
    user = f"COMMITMENT: {commitment}\n\nPLANNING NOTE (turn T, context):\n{planning}\n\nTOOLS (turns T+1..T+{k}):\n{tools}"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        system=EXEC_JUDGE_SYSTEM.replace("{k}", str(k)),
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        v = json.loads(raw).get("executed", "N").strip().upper()[:1]
        if v not in RANK:
            v = "N"
    except Exception:
        v = "N"
    cache.write_text(json.dumps({"executed": v}))
    return v


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY")
        sys.exit(1)

    rows = list(csv.DictReader(open(FIXED_SET)))
    assert len(rows) == 142, f"expected 142, got {len(rows)}"

    # per-turn planning + windows (cached; no network needed if warm)
    planning_cache: dict[tuple[str, int], str] = {}
    tools_cache: dict[tuple[str, int, int], str] = {}

    clamp5 = clamp20 = 0
    out = []
    for i, r in enumerate(rows):
        run_id, turn = r["run_id"], int(r["turn"])
        pk = (run_id, turn)
        if pk not in planning_cache:
            planning_cache[pk] = fetch_planning(run_id, turn)
        planning = planning_cache[pk]
        k10 = r["exec_k10"]

        def win(k):
            tk = (run_id, turn, k)
            if tk not in tools_cache:
                tools_cache[tk] = summarise_tools(run_id, turn, k=k)
            return tools_cache[tk]

        raw5 = judge(r["commitment"], planning, win(5), 5, api_key)
        raw20 = judge(r["commitment"], planning, win(20), 20, api_key)
        # enforce nesting against the K=10 anchor
        e5 = raw5
        if RANK[e5] > RANK[k10]:
            e5 = k10; clamp5 += 1
        e20 = raw20
        if RANK[e20] < RANK[k10]:
            e20 = k10; clamp20 += 1

        out.append({**r, "exec_k5": e5, "exec_k20": e20,
                    "raw_k5": raw5, "raw_k20": raw20})
        print(f"  [{i+1}/142] {r['model'][:10]:10s} T{turn} k5={e5} k10={k10} k20={e20}", flush=True)

    # write full labels
    outp = ROOT / "analysis" / "rag_fixed_set_sweep_results.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cid", "run_id", "turn", "model", "commitment",
                                          "exec_k5", "exec_k10", "exec_k20", "raw_k5", "raw_k20",
                                          "evidence_k10"])
        w.writeheader()
        for o in out:
            w.writerow({k: o.get(k, "") for k in w.fieldnames})

    def stats(rs, kcol):
        c = Counter(o[kcol] for o in rs); y, p, n = c["Y"], c["P"], c["N"]; t = y + p + n
        return (y + 0.5 * p) / t if t else 0.0, y, p, n, t

    print(f"\nClamped to enforce nesting: K5 {clamp5}/142, K20 {clamp20}/142\n")
    print(f"{'Model':22s} {'n':>3s}  {'RAG@5':>7s} {'RAG@10':>7s} {'RAG@20':>7s}")
    for m in MODELS:
        rs = [o for o in out if o["model"] == m]
        r5 = stats(rs, "exec_k5"); r10 = stats(rs, "exec_k10"); r20 = stats(rs, "exec_k20")
        print(f"{m:22s} {r5[4]:>3d}  {r5[0]:>6.1%} {r10[0]:>6.1%} {r20[0]:>6.1%}")
        for lbl, s in [("@5", r5), ("@10", r10), ("@20", r20)]:
            print(f"      {lbl:>3s}  Y={s[1]:2d} P={s[2]:2d} N={s[3]:2d}")
    r5 = stats(out, "exec_k5"); r10 = stats(out, "exec_k10"); r20 = stats(out, "exec_k20")
    print(f"{'OVERALL pooled':22s} {r5[4]:>3d}  {r5[0]:>6.1%} {r10[0]:>6.1%} {r20[0]:>6.1%}")
    for lbl, s in [("@5", r5), ("@10", r10), ("@20", r20)]:
        print(f"      {lbl:>3s}  Y={s[1]:3d} P={s[2]:2d} N={s[3]:3d}")

    # reproduction check
    print("\nReproduction check (K=10 must equal Figure 4):")
    ok = True
    for m, target in [("claude-opus-4-6", 0.482), ("gemini-3.1-pro-preview", 0.658), ("gpt-5.4", 0.632)]:
        r = stats([o for o in out if o["model"] == m], "exec_k10")[0]
        good = abs(r - target) < 0.005
        ok &= good
        print(f"  {m:24s} {r:.1%} vs {target:.1%}  {'OK' if good else 'MISMATCH'}")
    print("  => Figure 4 reproduced" if ok else "  => STOP: K=10 does not match Figure 4")
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
