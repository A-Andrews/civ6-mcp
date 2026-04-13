"""CivBench data loader — fetches and caches all game data from Convex.

Primary data source: Convex cloud backend.
  - listGames: game metadata (model, scenario, outcome, victory type)
  - getGameSummary: per-turn metric trajectories for all players
  - getGameTurnDetail: rich per-turn player state including diary reflections

Azure blob (log.jsonl, diary.jsonl) is a secondary source for tool call
analysis. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME
in evals/.env to enable it.

All results are cached to /tmp/civbench_cache/ on first fetch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze import convex_query, cloud_log, CACHE_DIR  # noqa: E402
from analysis.admissible_games import ADMISSIBLE_IDS  # noqa: E402

CACHE_DIR.mkdir(exist_ok=True)
SUMMARY_CACHE = CACHE_DIR / "summaries"
SUMMARY_CACHE.mkdir(exist_ok=True)
TURN_DETAIL_CACHE = CACHE_DIR / "turn_details"
TURN_DETAIL_CACHE.mkdir(exist_ok=True)


# Trajectory metric keys pulled out of turnSeries.players[*].metrics
# per (game, turn, player) row in load_diary_df.
TRAJECTORY_METRICS = (
    "score",
    "cities",
    "pop",
    "science",
    "culture",
    "gold",
    "faith",
    "military",
    "exploration_pct",
    "territory",
    "tourism",
    "spatial_actions",
    "spatial_cumulative",
)


def _iter_games(
    admissible_only: bool = True,
    game_ids: list[str] | None = None,
) -> list[dict]:
    """Fetch game list from Convex with admissible + game_id filters applied.

    Single source of truth for game filtering used by every loader.
    """
    games = convex_query("diary:listGames")
    if admissible_only:
        games = [g for g in games if g.get("runId") in ADMISSIBLE_IDS]
    if game_ids is not None:
        gid_set = set(game_ids)
        games = [g for g in games if g["gameId"] in gid_set]
    return games


def _metric_at(metrics: dict, i: int, key: str):
    vals = metrics.get(key, [])
    return vals[i] if i < len(vals) else None


def _series(metrics: dict, key: str) -> list:
    """Raw metric series for a key (empty list if absent)."""
    return metrics.get(key, [])


def _at_turn(metrics: dict, turns: list, key: str, target_turn: int):
    """First value at or after `target_turn`, or None if game ended earlier."""
    series = _series(metrics, key)
    for i, t in enumerate(turns):
        if t >= target_turn and i < len(series):
            return series[i]
    return None  # game ended before target_turn — do not carry forward


def _final(metrics: dict, key: str):
    """Last value of a metric series, or None if empty."""
    series = _series(metrics, key)
    return series[-1] if series else None


def _fetch_summary(game_id: str) -> dict:
    """Fetch and cache getGameSummary for a game."""
    path = SUMMARY_CACHE / f"{game_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = convex_query("diary:getGameSummary", {"gameId": game_id})
    path.write_text(json.dumps(data))
    return data


def _fetch_turn_detail(game_id: str, turn: int) -> dict:
    """Fetch and cache getGameTurnDetail for a specific turn."""
    path = TURN_DETAIL_CACHE / f"{game_id}_t{turn:04d}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = convex_query("diary:getGameTurnDetail", {"gameId": game_id, "turn": turn})
    path.write_text(json.dumps(data))
    return data


def load_games_df(admissible_only: bool = True):
    """Load game metadata as a pandas DataFrame.

    admissible_only: if True (default), restrict to games in ADMISSIBLE_IDS.

    Columns:
        run_id, game_id, model, scenario, turns_played, raw_score,
        normalised_score, outcome, victory_type, winner_civ, difficulty,
        map_type, map_size, game_speed, git_describe

    normalised_score: agent_score / winner_score_at_game_end.
    For an agent that won: 1.0.
    For an eliminated/defeated agent: raw_score / winner_score_at_last_turn.
    Winner score is taken from getGameSummary turnSeries (all-player scores).
    """
    import pandas as pd

    games = _iter_games(admissible_only=admissible_only)
    rows = []
    for g in games:
        game_id = g["gameId"]
        outcome = g.get("outcome") or {}

        # Fetch summary for winner score computation
        try:
            summary = _fetch_summary(game_id)
        except Exception as e:
            print(f"Warning: could not fetch summary for {game_id}: {e}", file=sys.stderr)
            summary = {}

        raw_score = g.get("score") or 0.0
        normalised_score = _compute_normalised_score(raw_score, summary, outcome)

        rows.append({
            "run_id": g.get("runId"),
            "game_id": game_id,
            "model": g.get("agentModel") or "unknown",
            "scenario": g.get("scenarioId") or "unknown",
            "turns_played": int(g.get("count") or 0),
            "raw_score": raw_score,
            "normalised_score": normalised_score,
            "outcome": outcome.get("result") or g.get("status") or "unknown",
            "victory_type": outcome.get("victoryType") or "",
            "winner_civ": outcome.get("winnerCiv") or "",
            "difficulty": g.get("difficulty") or "",
            "map_type": summary.get("mapType") or g.get("mapType") or "",
            "map_size": summary.get("mapSize") or g.get("mapSize") or "",
            "game_speed": summary.get("gameSpeed") or "",
            "git_describe": summary.get("gitDescribe") or g.get("gitDescribe") or "",
        })

    df = pd.DataFrame(rows)
    # Normalise model names for consistency
    if not df.empty:
        df["model"] = df["model"].str.strip()
    return df


def _compute_normalised_score(
    agent_raw_score: float,
    summary: dict,
    outcome: dict,
) -> float:
    """Compute normalised score = agent_score / max_player_score_at_last_turn.

    Uses the turnSeries from getGameSummary to find the highest score
    among all players at the final recorded turn.

    If summary data is unavailable, returns NaN.
    """
    ts = summary.get("turnSeries")
    if not ts:
        return float("nan")

    players = ts.get("players", {})
    # Max final-turn score across all players. Each player's scores[-1] is
    # their last recorded score; eliminated players contribute their
    # elimination score, survivors contribute their game-end score.
    final_scores = []
    for pdata in players.values():
        scores = pdata.get("metrics", {}).get("score", [])
        last = next((s for s in reversed(scores) if s is not None), None)
        if last is not None:
            final_scores.append(last)

    max_score = max(final_scores, default=0.0)
    if max_score <= 0:
        return float("nan")

    # If agent won, normalised = 1.0 (agent_raw_score == max_score)
    return min(1.0, agent_raw_score / max_score)


def load_diary_df(game_ids: list[str] | None = None, include_reflections: bool = False, admissible_only: bool = True):
    """Load per-turn trajectory data from getGameSummary turnSeries.

    One row per (game_id, turn, player_id).

    Columns: game_id, run_id, model, scenario, turn, player_id, is_agent,
             civ, leader, score, cities, pop, science, culture, gold,
             faith, military, exploration_pct, territory, tourism,
             spatial_actions, spatial_cumulative.

    If include_reflections=True, also fetches getGameTurnDetail for agent
    rows and adds: reflection_tactical, reflection_strategic,
    reflection_planning, reflection_hypothesis, reflection_tooling.
    This is slow (one API call per turn per game) — use sparingly or cache.
    """
    import pandas as pd

    all_games = _iter_games(admissible_only=admissible_only, game_ids=game_ids)

    rows = []
    for g in all_games:
        game_id = g["gameId"]
        run_id = g.get("runId")
        model = g.get("agentModel") or "unknown"
        scenario = g.get("scenarioId") or "unknown"

        try:
            summary = _fetch_summary(game_id)
        except Exception as e:
            print(f"Warning: skipping {game_id}: {e}", file=sys.stderr)
            continue

        ts = summary.get("turnSeries", {})
        turns = ts.get("turns", [])
        players = ts.get("players", {})

        for pid, pdata in players.items():
            is_agent = pdata.get("is_agent", False)
            civ = pdata.get("civ", "")
            leader = pdata.get("leader", "")
            metrics = pdata.get("metrics", {})

            for i, turn in enumerate(turns):
                rows.append({
                    "game_id": game_id,
                    "run_id": run_id,
                    "model": model if is_agent else f"AI_{civ}",
                    "scenario": scenario,
                    "turn": int(turn),
                    "player_id": int(pid),
                    "is_agent": is_agent,
                    "civ": civ,
                    "leader": leader,
                    **{k: _metric_at(metrics, i, k) for k in TRAJECTORY_METRICS},
                })

    df = pd.DataFrame(rows)

    if include_reflections and not df.empty:
        refl_df = load_reflections(
            game_ids=sorted(df["game_id"].unique().tolist()),
            admissible_only=False,  # already filtered above
        )
        if not refl_df.empty:
            refl_cols = [c for c in refl_df.columns if c.startswith("reflection_")]
            df = df.merge(
                refl_df[["game_id", "turn", *refl_cols]],
                on=["game_id", "turn"],
                how="left",
            )
            # Reflections only exist for agent rows; blank them out for AI rows
            for col in refl_cols:
                df.loc[~df["is_agent"], col] = ""

    return df


def load_reflections(
    game_ids: list[str] | None = None,
    every_n_turns: int = 1,
    admissible_only: bool = True,
):
    """Fetch agent diary reflection text per turn from getGameTurnDetail.

    One row per (game_id, turn). Slow — one API call per turn per game on
    first run, then served from TURN_DETAIL_CACHE.

    Columns: game_id, run_id, model, scenario, turn,
             reflection_planning, reflection_tactical, reflection_strategic,
             reflection_hypothesis, reflection_tooling
    """
    import pandas as pd

    all_games = _iter_games(admissible_only=admissible_only, game_ids=game_ids)

    step = max(1, int(every_n_turns))
    rows = []
    for g in all_games:
        game_id = g["gameId"]
        run_id = g.get("runId")
        model = g.get("agentModel") or "unknown"
        scenario = g.get("scenarioId") or "unknown"

        try:
            summary = _fetch_summary(game_id)
        except Exception as e:
            print(f"Warning: skipping {game_id}: {e}", file=sys.stderr)
            continue

        turns = summary.get("turnSeries", {}).get("turns", [])

        for turn in turns[::step]:
            turn_int = int(turn)
            try:
                detail = _fetch_turn_detail(game_id, turn_int)
            except Exception as e:
                print(
                    f"Warning: turn detail {game_id} t{turn_int}: {e}",
                    file=sys.stderr,
                )
                continue

            agent_row = next(
                (r for r in detail.get("playerRows", []) if r.get("is_agent")),
                None,
            )
            if not agent_row:
                continue
            refl = agent_row.get("reflections") or {}
            rows.append({
                "game_id": game_id,
                "run_id": run_id,
                "model": model,
                "scenario": scenario,
                "turn": turn_int,
                "reflection_planning": refl.get("planning") or "",
                "reflection_tactical": refl.get("tactical") or "",
                "reflection_strategic": refl.get("strategic") or "",
                "reflection_hypothesis": refl.get("hypothesis") or "",
                "reflection_tooling": refl.get("tooling") or "",
            })

    return pd.DataFrame(rows)


def load_metrics_df(admissible_only: bool = True):
    """Load per-game aggregate metrics from evals/metrics.py.

    Requires Azure blob access (log.jsonl) for tool call metrics.
    Returns a DataFrame with game_id, model, scenario, and all metrics
    that can be computed. Tool-call-based metrics will be NaN if Azure
    is unavailable.

    Columns computed from Convex trajectories (no Azure needed):
        final_score, final_cities, final_exploration_pct,
        city_t50, city_t100, city_t150, city_t200,
        final_science, final_gold, final_military

    Columns requiring Azure log.jsonl:
        victory_check_freq, great_people_check_count, spaceport_turn,
        space_project_count, map_scan_freq, military_unit_count, etc.
    """
    import pandas as pd

    all_games = _iter_games(admissible_only=admissible_only)
    rows = []
    for g in all_games:
        game_id = g["gameId"]
        try:
            summary = _fetch_summary(game_id)
        except Exception as e:
            print(f"Warning: skipping {game_id}: {e}", file=sys.stderr)
            continue

        ts = summary.get("turnSeries", {})
        turns = ts.get("turns", [])
        players = ts.get("players", {})

        # Find agent player
        agent = next(
            (pdata for pdata in players.values() if pdata.get("is_agent")),
            None,
        )
        if agent is None:
            continue

        metrics = agent.get("metrics", {})

        row = {
            "game_id": game_id,
            "run_id": g.get("runId"),
            "model": g.get("agentModel") or "unknown",
            "scenario": g.get("scenarioId") or "unknown",
            "turns_played": int(g.get("count") or 0),
            "raw_score": g.get("score") or 0.0,
            # Trajectory-based metrics
            "final_cities": _final(metrics, "cities"),
            "final_exploration_pct": _final(metrics, "exploration_pct"),
            "final_science": _final(metrics, "science"),
            "final_gold": _final(metrics, "gold"),
            "final_military": _final(metrics, "military"),
            "final_culture": _final(metrics, "culture"),
            "city_t50": _at_turn(metrics, turns, "cities", 50),
            "city_t100": _at_turn(metrics, turns, "cities", 100),
            "city_t150": _at_turn(metrics, turns, "cities", 150),
            "city_t200": _at_turn(metrics, turns, "cities", 200),
            "exploration_t50": _at_turn(metrics, turns, "exploration_pct", 50),
            "exploration_t100": _at_turn(metrics, turns, "exploration_pct", 100),
        }

        # Tool-call metrics from Azure (optional)
        try:
            from scripts.analyze import _load_env
            env = _load_env()
            if not (env.get("AZURE_STORAGE_CONNECTION_STRING")
                    or (env.get("AZURE_STORAGE_ACCOUNT_NAME")
                        and (env.get("AZURE_STORAGE_SAS_TOKEN")
                             or env.get("AZURE_STORAGE_ACCOUNT_KEY")))):
                raise RuntimeError("No Azure credentials")
            log_entries = cloud_log(g["runId"])
            if log_entries:
                from evals.metrics import (
                    score_ground_control,
                    score_snowflake,
                    score_cry_havoc,
                )
                # Convert log entries to ToolCall objects for metrics.py
                from evals.scorer import ToolCall
                tool_calls = [
                    ToolCall(
                        name=e.get("tool", ""),
                        arguments=e.get("args") or {},
                        result=e.get("result") or "",
                        inspect_error=not e.get("success", True),
                    )
                    for e in log_entries
                    if e.get("type") == "tool_call"
                ]
                scenario_id = g.get("scenarioId", "")
                if scenario_id == "ground_control":
                    row.update(score_ground_control(tool_calls))
                elif scenario_id == "snowflake":
                    row.update(score_snowflake(tool_calls))
                elif scenario_id == "cry_havoc":
                    row.update(score_cry_havoc(tool_calls))
        except Exception as e:
            print(
                f"Warning: Azure tool metrics failed for {game_id}: {e}",
                file=sys.stderr,
            )  # tool metrics will be NaN for this game

        rows.append(row)

    return pd.DataFrame(rows)


def load_founding_df(admissible_only: bool = True):
    """Build a DataFrame of city-founding events.

    Inferred from the cities metric series: whenever cities[t] > cities[t-1],
    a new city was founded that turn.

    Columns: game_id, run_id, model, scenario, city_number, turn_founded
    """
    import pandas as pd

    all_games = _iter_games(admissible_only=admissible_only)
    rows = []
    for g in all_games:
        game_id = g["gameId"]
        try:
            summary = _fetch_summary(game_id)
        except Exception:
            continue

        ts = summary.get("turnSeries", {})
        turns = ts.get("turns", [])
        players = ts.get("players", {})

        for pid, pdata in players.items():
            if not pdata.get("is_agent"):
                continue
            cities_series = _series(pdata.get("metrics", {}), "cities")
            prev = 0
            for i, (turn, count) in enumerate(zip(turns, cities_series)):
                count = int(count or 0)
                while count > prev:
                    prev += 1
                    rows.append({
                        "game_id": game_id,
                        "run_id": g.get("runId"),
                        "model": g.get("agentModel") or "unknown",
                        "scenario": g.get("scenarioId") or "unknown",
                        "city_number": prev,
                        "turn_founded": int(turn),
                    })

    return pd.DataFrame(rows)


def load_tools_df(game_ids: list[str] | None = None, admissible_only: bool = True):
    """Load per-tool-call data from Azure log.jsonl files.

    One row per tool call across all games.

    Columns:
        game_id, run_id, model, scenario, turn, tool, category,
        pmr_subcategory, is_strategic, is_pmr_denominator,
        success, duration_ms, seq

    Requires Azure credentials. Raises RuntimeError if unavailable.
    """
    import pandas as pd
    from analysis.tool_taxonomy import categorise, pmr_denominator

    all_games = _iter_games(admissible_only=admissible_only, game_ids=game_ids)

    rows = []
    errors = []
    for g in all_games:
        run_id = g.get("runId")
        if not run_id:
            continue
        game_id = g["gameId"]
        model = g.get("agentModel") or "unknown"
        scenario = g.get("scenarioId") or "unknown"

        try:
            entries = cloud_log(run_id)
        except Exception as e:
            errors.append(f"{game_id}: {e}")
            continue

        for e in entries:
            tool = e.get("tool") or e.get("name") or ""
            if not tool:
                continue
            cat, pmr_sub = categorise(tool)
            rows.append({
                "game_id": game_id,
                "run_id": run_id,
                "model": model,
                "scenario": scenario,
                "turn": e.get("turn"),
                "tool": tool,
                "category": cat,
                "pmr_subcategory": pmr_sub,
                "is_strategic": cat == "strategic_monitoring",
                "is_pmr_denominator": pmr_denominator(tool),
                "success": e.get("success", True),
                "duration_ms": e.get("duration_ms"),
                "seq": e.get("seq"),
            })

    if errors:
        print(f"Warning: {len(errors)} games failed Azure fetch: {errors[:3]}", file=sys.stderr)

    df = pd.DataFrame(rows)
    return df


def load_all(verbose: bool = True) -> dict:
    """Load all DataFrames and return as a dict.

    Returns:
        {
            "games": games_df,
            "diary": diary_df,      # agent rows only
            "metrics": metrics_df,
            "founding": founding_df,
        }
    """
    log = (lambda msg: print(msg, flush=True)) if verbose else (lambda _: None)

    log("Loading games metadata...")
    games = load_games_df()
    log(f"  {len(games)} games loaded")

    log("Loading diary trajectories...")
    diary = load_diary_df()
    diary_agent = diary[diary["is_agent"]].copy()
    log(f"  {len(diary_agent)} agent-turn rows")

    log("Loading per-game metrics...")
    metrics = load_metrics_df()
    log(f"  {len(metrics)} metrics rows")

    log("Loading city founding events...")
    founding = load_founding_df()
    log(f"  {len(founding)} founding events")

    return {
        "games": games,
        "diary": diary_agent,
        "diary_all": diary,
        "metrics": metrics,
        "founding": founding,
    }


if __name__ == "__main__":
    data = load_all()
    games = data["games"]
    print("\n=== Dataset Summary ===")
    print(games.groupby(["model", "scenario"]).size().to_string())
    print()
    print("Outcome distribution:")
    print(games.groupby(["model", "outcome"]).size().to_string())
    print()
    print("Normalised score by model:")
    print(games.groupby("model")["normalised_score"].describe().round(3).to_string())
