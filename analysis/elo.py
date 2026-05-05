"""ELO rating computation for CivBench.

Mirrors the algorithm in web/src/lib/elo.ts (FFA pairwise, K=32/sqrt(N-1),
base 1500), pulling admissible-only game data via diary:getEloData. The
Convex query already filters to admissible=True; this module additionally
normalises winnerCiv to handle the one-off "CIVILIZATION_JAPAN" anomaly in
the rusted-indigo-oracle-11 (Kimi-K2.5) record that the production frontend
silently drops.

Run:  .venv/bin/python analysis/elo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze import convex_query  # noqa: E402

BASE_ELO = 1500
K = 32


def _normalise_civ(s: str | None) -> str:
    """Normalise civ identifiers so 'Japan' and 'CIVILIZATION_JAPAN' compare equal."""
    if not s:
        return ""
    return s.upper().replace("CIVILIZATION_", "").replace(" ", "_")


def compute_elo() -> dict:
    """Return a dict with `ratings` (list of entries) and `n_games` (int)."""
    games = convex_query("diary:getEloData")
    ratings: dict[str, dict] = {}

    def get_or_create(pid: str, name: str, ptype: str) -> dict:
        if pid not in ratings:
            ratings[pid] = {
                "id": pid, "name": name, "type": ptype,
                "elo": BASE_ELO, "games": 0, "wins": 0, "losses": 0,
            }
        return ratings[pid]

    n_used = 0
    for g in games:
        win_norm = _normalise_civ(g.get("winnerCiv"))
        parts = g.get("players", [])
        enriched = []
        for p in parts:
            if p.get("is_agent") and p.get("agent_model"):
                pid, name, ptype = f"model:{p['agent_model']}", p["agent_model"], "model"
            else:
                pid, name, ptype = f"ai:{p['leader']}", p["leader"], "ai_leader"
            won = _normalise_civ(p.get("civ")) == win_norm
            enriched.append((pid, name, ptype, won))
        if not any(w for _, _, _, w in enriched):
            continue
        n = len(enriched)
        if n < 2:
            continue
        n_used += 1
        k_eff = K / math.sqrt(n - 1)
        winner = next((pn for pn in enriched if pn[3]))
        losers = [pn for pn in enriched if not pn[3]]

        we = get_or_create(winner[0], winner[1], winner[2])
        we["games"] += 1
        we["wins"] += 1
        pre_w = we["elo"]
        delta_w = 0.0
        delta_l: dict[str, float] = {}
        for lpid, lname, ltype, _ in losers:
            le = get_or_create(lpid, lname, ltype)
            le["games"] += 1
            le["losses"] += 1
            pre_l = le["elo"]
            exp_w = 1.0 / (1.0 + 10 ** ((pre_l - pre_w) / 400.0))
            delta_w += k_eff * (1 - exp_w)
            delta_l[lpid] = delta_l.get(lpid, 0.0) + k_eff * (0 - (1 - exp_w))
        we["elo"] += delta_w
        for lpid, d in delta_l.items():
            ratings[lpid]["elo"] += d

    sorted_ratings = sorted(
        ({**r, "elo": round(r["elo"])} for r in ratings.values()),
        key=lambda x: -x["elo"],
    )
    return {"ratings": sorted_ratings, "n_games": n_used}


def split(ratings: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (model_ratings, ai_leader_ratings) preserving order."""
    return (
        [r for r in ratings if r["type"] == "model"],
        [r for r in ratings if r["type"] == "ai_leader"],
    )


if __name__ == "__main__":
    out = compute_elo()
    models, ais = split(out["ratings"])
    print(f"ELO computed over {out['n_games']} admissible games.")
    print()
    print("=== LLM MODELS ===")
    print(f"{'Model':<28} {'ELO':>5}  {'Games':>5}  {'W':>3}  {'L':>3}")
    for r in models:
        print(f"{r['name']:<28} {r['elo']:>5}  {r['games']:>5}  {r['wins']:>3}  {r['losses']:>3}")
    print()
    print("=== TOP AI LEADERS ===")
    print(f"{'Leader':<28} {'ELO':>5}  {'Games':>5}  {'W':>3}  {'L':>3}")
    for r in ais[:12]:
        print(f"AI {r['name']:<25} {r['elo']:>5}  {r['games']:>5}  {r['wins']:>3}  {r['losses']:>3}")
