"""Reflection-Action Gap (RAG) extraction pipeline.

Steps:
  1. sample_turns()       — pick ~25 turns stratified across admissible games/phases
  2. fetch_planning()     — pull planning text from Convex for each sampled turn
  3. extract_commitments() — use Claude Haiku to parse commitments from planning text
  4. summarise_tools()    — format the next K turns of tool calls as readable text
  5. build_labeling_sheet() — write CSV for human labeling
  6. compute_agreement()  — compare human labels vs auto-match (after sheet is filled)
  7. compute_rag_scores() — full RAG@K scores across all admissible games

Usage:
    # Build the labeling sheet (run once):
    .venv/bin/python analysis/rag_extraction.py sheet

    # After filling in the sheet, check agreement:
    .venv/bin/python analysis/rag_extraction.py agreement analysis/rag_labeling_sheet.csv
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.tool_taxonomy import SKIP_TOOLS  # noqa: E402

CACHE_DIR = ROOT / "analysis" / "rag_cache"
CACHE_DIR.mkdir(exist_ok=True)

SHEET_PATH = ROOT / "analysis" / "rag_labeling_sheet.csv"

# Stratified sample: 3 per ground_control game (early/mid/late), 1-2 per snowflake.
# Last refreshed 2026-05-01: 35 turns across 13 admissible games (added 3 turns
# from gilded-teal-relic-32 / gpt-5.4 GC). Including the new game shifted GPT
# RAG@10 from 69.8% → 63.2% (15 new commitments: 5Y/2P/8N).
SAMPLE_PLAN = {
    # (run_id, turn)
    # Claude GC — 3 games × 3 turns
    "claude-opus-4-6-gc": [
        ("steep-vermil-trebuchet-27", 30),
        ("steep-vermil-trebuchet-27", 120),
        ("steep-vermil-trebuchet-27", 220),
        ("solar-flax-forum-10", 25),
        ("solar-flax-forum-10", 100),
        ("solar-flax-forum-10", 180),
        ("wandering-carmine-ballista-57", 30),
        ("wandering-carmine-ballista-57", 130),
        ("wandering-carmine-ballista-57", 250),
    ],
    # Claude snowflake — 2 games × 2 turns
    "claude-opus-4-6-sf": [
        ("regal-sepia-corsair-87", 50),
        ("regal-sepia-corsair-87", 200),
        ("hidden-lilac-atlas-39", 50),
        ("hidden-lilac-atlas-39", 200),
    ],
    # Gemini GC — 2 games (defeat + victory)
    "gemini-3.1-pro-preview-gc": [
        ("remnant-khaki-oracle-28", 30),
        ("remnant-khaki-oracle-28", 150),
        ("volcanic-vermil-ember-37", 50),
        ("volcanic-vermil-ember-37", 150),
        ("volcanic-vermil-ember-37", 250),
    ],
    # GPT GC — 4 games × 3 turns (skip fierce which starts at T131)
    "gpt-5.4-gc": [
        ("veiled-lapis-flagship-44", 30),
        ("veiled-lapis-flagship-44", 150),
        ("veiled-lapis-flagship-44", 260),
        ("dread-cobalt-chariot-34", 25),
        ("dread-cobalt-chariot-34", 130),
        ("dread-cobalt-chariot-34", 240),
        ("flint-olive-phalanx-07", 60),
        ("flint-olive-phalanx-07", 150),
        ("fierce-umber-garrison-15", 140),
        ("fierce-umber-garrison-15", 230),
        ("gilded-teal-relic-32", 30),
        ("gilded-teal-relic-32", 120),
        ("gilded-teal-relic-32", 220),
    ],
    # GPT snowflake — 2 games × 2 turns
    "gpt-5.4-sf": [
        ("celestial-ochre-monument-04", 60),
        ("celestial-ochre-monument-04", 200),
        ("fierce-rust-horseman-79", 60),
        ("fierce-rust-horseman-79", 200),
    ],
}

SAMPLE_TURNS: list[tuple[str, int]] = [
    item for group in SAMPLE_PLAN.values() for item in group
]


def _run_id_to_game_id(run_id: str) -> str | None:
    from scripts.analyze import convex_query
    games = convex_query("diary:listGames")
    for g in games:
        if g.get("runId") == run_id:
            return g["gameId"]
    return None


def fetch_planning(run_id: str, turn: int) -> str:
    """Fetch the planning reflection for a specific run_id + turn."""
    cache_path = CACHE_DIR / f"planning_{run_id}_t{turn:04d}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    from scripts.analyze import convex_query
    game_id = _run_id_to_game_id(run_id)
    if not game_id:
        return ""
    detail = convex_query("diary:getGameTurnDetail", {"gameId": game_id, "turn": turn})
    for row in detail.get("playerRows", []):
        if row.get("is_agent"):
            text = (row.get("reflections") or {}).get("planning") or ""
            cache_path.write_text(text)
            return text
    cache_path.write_text("")
    return ""


EXTRACT_SYSTEM = """You extract concrete action commitments from a Civilization VI agent's planning note.

A commitment is a specific, observable action the agent says it will take in the next few turns.
Examples of commitments:
  - "build settler in Babylon" → COMMITMENT: produce settler in Babylon
  - "research Iron Working" → COMMITMENT: set research to Iron Working
  - "check victory progress" → COMMITMENT: call get_victory_progress
  - "move warrior to (44, 18)" → COMMITMENT: move unit to tile

NOT commitments (too vague):
  - "continue expanding" (no specific action)
  - "focus on science" (no specific action)
  - "be careful of barbarians" (no specific action)

Return ONLY a JSON array of strings, each a short (<10 word) description of one commitment.
If there are no concrete commitments, return [].
Return nothing but the JSON array."""


def extract_commitments(planning_text: str) -> list[str]:
    """Use Claude Haiku to extract concrete commitments from planning text."""
    if not planning_text.strip():
        return []

    cache_key = hashlib.sha1(planning_text.encode("utf-8")).hexdigest()[:12]
    cache_path = CACHE_DIR / f"commitments_{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    import anthropic
    import os
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": planning_text}],
    )
    raw = msg.content[0].text.strip()
    try:
        commitments = json.loads(raw)
        if not isinstance(commitments, list):
            commitments = []
    except Exception:
        commitments = []

    cache_path.write_text(json.dumps(commitments))
    return commitments


_CITY_NAME_RE = re.compile(r'(\w[\w\s\-\']+?) \(pop \d+\).+?\[id:(\d+)\]')


def _build_city_name_map(entries: list[dict]) -> dict[int, str]:
    """Build city_id -> city_name from get_cities result summaries in the log."""
    mapping: dict[int, str] = {}
    for e in entries:
        if e.get("tool") == "get_cities":
            rs = e.get("result_summary") or ""
            for m in _CITY_NAME_RE.finditer(rs):
                mapping[int(m.group(2))] = m.group(1).strip()
    return mapping


def _city_suffix(params: dict, city_names: dict[int, str] | None) -> str:
    """Return '@ CityName' if city_id is in params and known, else ''."""
    city_id = params.get("city_id")
    if not city_id:
        return ""
    city = (city_names or {}).get(city_id, "")
    return f" @ {city}" if city else ""


# --- _fmt_call dispatch table ---
# Each formatter takes (params, result, city_names) and returns the arg string
# to render inside tool(...). Missing tools fall back to _fmt_generic.

def _fmt_research(params, result, city_names):
    arg = params.get("tech_or_civic", "")
    return arg.replace("TECH_", "").replace("CIVIC_", "").replace("_", " ").title()


def _fmt_city_production(params, result, city_names):
    item = params.get("item_name", "")
    item = item.split("_", 1)[-1].replace("_", " ").title() if item else ""
    return item + _city_suffix(params, city_names)


def _fmt_purchase_item(params, result, city_names):
    item = params.get("item_name", "").split("_", 1)[-1].replace("_", " ").title()
    return item + _city_suffix(params, city_names)


def _fmt_city_attack(params, result, city_names):
    tx, ty = params.get("x"), params.get("y")
    # City name is in result: "CITY_RANGE_ATTACK|Babylon -> ..."
    city = ""
    if "|" in result:
        parts = result.split("|")
        if len(parts) > 1:
            city = parts[1].split("->")[0].strip()
    if not city:
        city = (city_names or {}).get(params.get("city_id"), "")
    return (city + " " if city else "") + (f"→ ({tx},{ty})" if tx is not None else "")


def _fmt_assign_governor(params, result, city_names):
    gov = params.get("governor_type", "").replace("GOVERNOR_THE_", "").replace("_", " ").title()
    return gov + _city_suffix(params, city_names)


def _fmt_city_focus(params, result, city_names):
    focus = params.get("focus", "").replace("CITY_FOCUS_", "").replace("_", " ").title()
    return focus + _city_suffix(params, city_names)


def _fmt_unit_action(params, result, city_names):
    action = params.get("action", "")
    tx, ty = params.get("target_x"), params.get("target_y")
    return action + (f" → ({tx},{ty})" if tx is not None else "")


def _fmt_propose_trade(params, result, city_names):
    return result[:60] if result else ""


def _fmt_diplomatic_action(params, result, city_names):
    return params.get("action", "").replace("_", " ").title()


def _fmt_empty(params, result, city_names):
    # Monitoring calls — name alone is informative
    return ""


def _fmt_generic(params, result, city_names):
    skip = {"unit_id", "city_id", "other_player_id", "player_id"}
    vals = [str(v) for k, v in params.items() if k not in skip and v is not None]
    return vals[0][:40] if vals else ""


_FMT_TABLE = {
    "set_research": _fmt_research,
    "set_city_production": _fmt_city_production,
    "purchase_item": _fmt_purchase_item,
    "city_attack": _fmt_city_attack,
    "assign_governor": _fmt_assign_governor,
    "set_city_focus": _fmt_city_focus,
    "unit_action": _fmt_unit_action,
    "propose_trade": _fmt_propose_trade,
    "send_diplomatic_action": _fmt_diplomatic_action,
    "get_victory_progress": _fmt_empty,
    "get_religion_spread": _fmt_empty,
    "get_diplomacy": _fmt_empty,
    "get_empire_resources": _fmt_empty,
    "get_great_people": _fmt_empty,
}


def _fmt_call(entry: dict, city_names: dict[int, str] | None = None) -> str:
    """Format one log entry as 'tool_name(key_param) → result_snippet'."""
    tool = entry.get("tool", "?")
    params = entry.get("params") or {}
    result = (entry.get("result_summary") or "")[:80].split("\n")[0]

    formatter = _FMT_TABLE.get(tool, _fmt_generic)
    arg = formatter(params, result, city_names)

    call = f"{tool}({arg})" if arg else tool
    snippet = result[:60] if result and result not in ("", "None") else ""
    return f"{call} → {snippet}" if snippet else call


def summarise_tools(run_id: str, from_turn: int, k: int = 10) -> str:
    """Return a readable summary of tool calls for turns [from_turn+1, from_turn+k].

    Loads directly from Azure log.jsonl so params and results are available.
    Skips game_management and pure state queries to keep it scannable.
    """
    from scripts.analyze import cloud_log

    cache_path = CACHE_DIR / f"tools_v2_{run_id}_t{from_turn:04d}_k{k}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    entries = cloud_log(run_id)
    city_names = _build_city_name_map(entries)

    window = [
        e for e in entries
        if from_turn < (e.get("turn") or 0) <= from_turn + k
        and e.get("tool") not in SKIP_TOOLS
        and e.get("success", True)
    ]

    if not window:
        result = f"(no notable actions in next {k} turns)"
        cache_path.write_text(result)
        return result

    parts = []
    for entry in window:
        t = entry.get("turn", 0)
        parts.append(f"T+{t - from_turn}: {_fmt_call(entry, city_names)}")

    text = " | ".join(parts)
    cache_path.write_text(text)
    return text


def build_labeling_sheet(k: int = 10) -> Path:
    """Sample turns and write CSV for human labeling.

    One row per sampled turn. You fill in:
      - commitment_1 … commitment_5: concrete actions stated in the planning text
      - executed_1  … executed_5:   Y / N / P for each commitment
    """
    import csv
    from analysis.admissible_games import ADMISSIBLE_GAMES

    meta_by_run = {g["run_id"]: g for g in ADMISSIBLE_GAMES}

    rows = []
    total = len(SAMPLE_TURNS)
    print(f"Building labeling sheet from {total} sampled turns...")

    for i, (run_id, turn) in enumerate(SAMPLE_TURNS):
        print(f"  [{i+1}/{total}] {run_id} T{turn}", flush=True)

        planning = fetch_planning(run_id, turn).replace("\n", " ").replace("\r", "")
        if not planning:
            print(f"    (no planning text — skipping)")
            continue

        tools_summary = summarise_tools(run_id, turn, k=k)
        meta = meta_by_run.get(run_id, {})

        rows.append({
            "run_id": run_id,
            "model": meta.get("model", "unknown"),
            "scenario": meta.get("scenario", "unknown"),
            "turn": turn,
            "planning_text": planning,
            "tools_next_10_turns": tools_summary,
            # Human fills these in — one commitment + executed verdict per slot
            "commitment_1": "", "executed_1": "",
            "commitment_2": "", "executed_2": "",
            "commitment_3": "", "executed_3": "",
            "commitment_4": "", "executed_4": "",
            "commitment_5": "", "executed_5": "",
            "notes": "",
        })

    fieldnames = [
        "run_id", "model", "scenario", "turn",
        "planning_text", "tools_next_10_turns",
        "commitment_1", "executed_1",
        "commitment_2", "executed_2",
        "commitment_3", "executed_3",
        "commitment_4", "executed_4",
        "commitment_5", "executed_5",
        "notes",
    ]

    with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSheet written: {SHEET_PATH}")
    print(f"  {len(rows)} turns sampled")
    print()
    print("Instructions:")
    print("  Read the planning_text and tools_next_10_turns for each row.")
    print("  For each concrete commitment in the planning text, fill in:")
    print("    commitment_N: short description (e.g. 'build settler in Babylon')")
    print("    executed_N:   Y = done in next 10 turns, N = not done, P = partial")
    print("  Leave unused commitment slots blank.")
    return SHEET_PATH


AUTO_SYSTEM = """You analyze a Civilization VI agent's planning note and determine which specific commitments it made and whether those commitments were executed in the following turns.

Given:
- PLANNING: the agent's planning reflection for turn T
- TOOLS: a summary of tool calls made in turns T+1 through T+10

For each concrete commitment in the planning text (a specific, observable action — not vague goals):
1. Name the commitment briefly (<10 words)
2. Judge whether it was executed: Y (clearly done), N (not done), P (partially done)

Evidence for execution:
- set_city_production(X @ City) → commitment to build X in that city → Y
- set_research(Tech) → commitment to research that tech → Y
- unit_action(move/found_city) → commitment to move/settle → Y
- get_victory_progress → commitment to check victory → Y
- If the action appears in a DIFFERENT city than specified → P
- If the action never appears → N

Return ONLY valid JSON in this exact format:
{"commitments": [{"text": "short description", "executed": "Y|N|P", "evidence": "one-line reason"}, ...]}
If no concrete commitments exist, return {"commitments": []}"""


def auto_annotate(planning_text: str, tools_summary: str, api_key: str) -> list[dict]:
    """Use Claude Haiku to extract commitments and judge execution in one call.

    Returns list of dicts: [{text, executed, evidence}, ...]
    """
    if not planning_text.strip():
        return []

    cache_key = hashlib.sha1((planning_text + "\x00" + tools_summary).encode("utf-8")).hexdigest()[:12]
    cache_path = CACHE_DIR / f"auto_{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = f"PLANNING:\n{planning_text}\n\nTOOLS:\n{tools_summary}"
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=AUTO_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        data = json.loads(raw)
        result = data.get("commitments", [])
        if not isinstance(result, list):
            result = []
    except Exception:
        result = []

    cache_path.write_text(json.dumps(result))
    return result


def run_auto_rag(api_key: str, k: int = 10) -> dict:
    """Run full automated RAG pipeline. Returns scores by model.

    Writes results to analysis/rag_auto_results.csv for spot-checking.
    """
    import csv
    from analysis.admissible_games import ADMISSIBLE_GAMES

    meta_by_run = {g["run_id"]: g for g in ADMISSIBLE_GAMES}
    results_path = ROOT / "analysis" / "rag_auto_results.csv"

    all_commitments = []
    total = len(SAMPLE_TURNS)
    print(f"Auto-annotating {total} sampled turns...")

    for i, (run_id, turn) in enumerate(SAMPLE_TURNS):
        print(f"  [{i+1}/{total}] {run_id} T{turn}", flush=True)

        planning = fetch_planning(run_id, turn)
        if not planning.strip():
            print(f"    (no planning text — skipping)")
            continue

        tools_summary = summarise_tools(run_id, turn, k=k)
        meta = meta_by_run.get(run_id, {})
        model = meta.get("model", "unknown")

        commitments = auto_annotate(planning, tools_summary, api_key)
        for c in commitments:
            all_commitments.append({
                "run_id": run_id,
                "model": model,
                "scenario": meta.get("scenario", "unknown"),
                "turn": turn,
                "commitment": c.get("text", ""),
                "executed": c.get("executed", "N"),
                "evidence": c.get("evidence", ""),
                "planning_snippet": planning[:200].replace("\n", " "),
                "tools_snippet": tools_summary[:200],
            })

        time.sleep(0.3)  # stay within rate limits

    if not all_commitments:
        print("No commitments extracted.")
        return {}

    # Write CSV for spot-checking
    fieldnames = ["run_id", "model", "scenario", "turn", "commitment",
                  "executed", "evidence", "planning_snippet", "tools_snippet"]
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_commitments)
    print(f"\nResults written: {results_path} ({len(all_commitments)} commitments)")

    # Compute RAG scores
    scores: dict[str, dict] = {}
    for model in set(c["model"] for c in all_commitments):
        m_rows = [c for c in all_commitments if c["model"] == model]
        n_total = len(m_rows)
        n_executed = sum(1 for c in m_rows if c["executed"] == "Y")
        n_partial = sum(1 for c in m_rows if c["executed"] == "P")
        rag = (n_executed + 0.5 * n_partial) / n_total if n_total else float("nan")
        scores[model] = {"n_commitments": n_total, "rag_score": round(rag, 3),
                         "n_executed": n_executed, "n_partial": n_partial, "n_not": n_total - n_executed - n_partial}
        print(f"  {model}: RAG={rag:.1%}  ({n_executed}Y + {n_partial}P + {n_total-n_executed-n_partial}N / {n_total} commitments)")

    return scores


def compute_agreement(sheet_path: Path = ROOT / "analysis" / "rag_auto_results.csv") -> dict:
    """Compare human-validated 'executed' labels against auto-match.

    Expects rag_auto_results.csv (from run_auto_rag) with two human-added columns:
      commitment_correct: Y/N — was the auto-extracted commitment text correct?
      executed:           Y/P/N — human judgment of whether it was executed
                          (overwrite the auto-assigned value for spot-check rows)
    """
    import csv
    import numpy as np

    rows = []
    with open(sheet_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only use rows where human has labeled both columns
    labeled = [
        r for r in rows
        if r["commitment_correct"].strip().upper() in ("Y", "N")
        and r["executed"].strip().upper() in ("Y", "N", "P")
        and r["commitment"] != "(none extracted)"
    ]

    if not labeled:
        print("No labeled rows found. Fill in commitment_correct and executed columns first.")
        return {}

    # Extraction accuracy: what fraction of Haiku commitments were judged correct?
    correct = sum(1 for r in labeled if r["commitment_correct"].strip().upper() == "Y")
    extraction_acc = correct / len(labeled)

    # Execution rate (human labels, correct commitments only)
    # Uses same formula as RAG score: (Y + 0.5*P) / n
    valid = [r for r in labeled if r["commitment_correct"].strip().upper() == "Y"]
    def _rag_val(label: str) -> float:
        u = label.strip().upper()
        return 1.0 if u == "Y" else (0.5 if u == "P" else 0.0)
    executed_human = [_rag_val(r["executed"]) for r in valid]
    execution_rate = np.mean(executed_human) if valid else float("nan")

    # By model
    by_model = {}
    for model in set(r["model"] for r in valid):
        m_rows = [r for r in valid if r["model"] == model]
        m_exec = [_rag_val(r["executed"]) for r in m_rows]
        by_model[model] = {"n": len(m_rows), "execution_rate": np.mean(m_exec)}

    print(f"Labeled rows: {len(labeled)}")
    print(f"Haiku extraction accuracy: {extraction_acc:.1%} ({correct}/{len(labeled)} commitments judged correct)")
    print(f"Execution rate (valid commitments only): {execution_rate:.1%}")
    print()
    print("By model:")
    for model, stats in by_model.items():
        print(f"  {model}: {stats['execution_rate']:.1%} ({stats['n']} commitments)")

    return {
        "n_labeled": len(labeled),
        "extraction_accuracy": extraction_acc,
        "execution_rate": execution_rate,
        "by_model": by_model,
    }


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheet"

    if cmd == "sheet":
        build_labeling_sheet(k=10)

    elif cmd == "auto":
        import os
        key = os.environ.get("ANTHROPIC_API_KEY") or (sys.argv[2] if len(sys.argv) > 2 else "")
        if not key:
            print("Error: ANTHROPIC_API_KEY not set.")
            print("Usage: ANTHROPIC_API_KEY=sk-... python rag_extraction.py auto")
            sys.exit(1)
        run_auto_rag(api_key=key)

    elif cmd == "agreement":
        path = Path(sys.argv[2]) if len(sys.argv) > 2 else SHEET_PATH
        compute_agreement(path)

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python rag_extraction.py [sheet|auto|agreement [path]]")
