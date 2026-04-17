"""Pre-harness runs — hex-ID runs collected before the eval harness was stable.

These runs used mcp_version 0.1.x or early 1.0.x and did NOT go through the
harness that enforces auto-resume, structured telemetry, and run naming.
They are identified by their short 8-character hex run IDs (as opposed to the
word-based IDs of harness runs, e.g. "solar-flax-forum-10").

Runs that collapsed at T=0 (infrastructure crash before any gameplay) are
excluded. Only runs that produced at least one diary turn are included.
Of the 42 total hex-ID runs, 23 crashed at T=0 and are omitted here; the
remaining 19 are listed below.

Fields:
    run_id          — Azure blob run identifier
    mcp_version     — MCP server version at time of run
    model           — model_id from manifest
    scenario        — scenario_id from manifest
    turns_played    — max(turn) from diary.jsonl
    end_condition   — "victory", "defeat", "hung", "diplomacy_loop",
                      or "truncated"
    normalised_score — agent_score / max_player_score at final turn;
                       None for non-completed runs
"""

from __future__ import annotations

PRE_HARNESS_RUNS: list[dict] = [
    # ------------------------------------------------------------------ #
    # Claude Opus 4.6 — Ground Control
    # ------------------------------------------------------------------ #
    {
        "run_id": "88ab0918", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 268, "end_condition": "victory",
        "normalised_score": 0.899,
    },
    {
        "run_id": "4fee9865", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 236, "end_condition": "victory",
        "normalised_score": 0.978,
    },
    {
        "run_id": "5042fdae", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 211, "end_condition": "hung",
        "normalised_score": None,
    },
    {
        "run_id": "0af0e204", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 93, "end_condition": "truncated",
        "normalised_score": None,
    },
    {
        "run_id": "21b5d5d8", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 88, "end_condition": "truncated",
        "normalised_score": None,
    },
    {
        "run_id": "f20e5b66", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 57, "end_condition": "truncated",
        "normalised_score": None,
    },
    {
        "run_id": "6650b168", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 36, "end_condition": "truncated",
        "normalised_score": None,
    },
    # ------------------------------------------------------------------ #
    # Claude Opus 4.6 — Snowflake
    # ------------------------------------------------------------------ #
    {
        "run_id": "cdf5af07", "mcp_version": "0.1.0",
        "model": "claude-opus-4-6", "scenario": "snowflake",
        "turns_played": 129, "end_condition": "hung",
        "normalised_score": None,
    },
    # ------------------------------------------------------------------ #
    # Claude Opus 4.6 — Ground Control (early 1.0.x, still hex ID)
    # ------------------------------------------------------------------ #
    {
        "run_id": "246e5821", "mcp_version": "1.0.4",
        "model": "claude-opus-4-6", "scenario": "ground_control",
        "turns_played": 156, "end_condition": "hung",
        "normalised_score": None,
    },
    # ------------------------------------------------------------------ #
    # GPT-5.4 — Ground Control
    # ------------------------------------------------------------------ #
    {
        "run_id": "2e378a4c", "mcp_version": "1.0.1",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 321, "end_condition": "defeat",
        "normalised_score": 0.426,
    },
    {
        "run_id": "98beeba1", "mcp_version": "0.1.0",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 306, "end_condition": "defeat",
        "normalised_score": 0.540,
    },
    {
        "run_id": "b4293a14", "mcp_version": "0.1.0",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 304, "end_condition": "diplomacy_loop",
        "normalised_score": None,
    },
    {
        "run_id": "e8265dc1", "mcp_version": "0.1.0",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 299, "end_condition": "diplomacy_loop",
        "normalised_score": None,
    },
    {
        "run_id": "90b9d28e", "mcp_version": "1.0.4",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 198, "end_condition": "truncated",
        "normalised_score": None,
    },
    {
        "run_id": "e696fafa", "mcp_version": "1.0.4",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 197, "end_condition": "truncated",
        "normalised_score": None,
    },
    {
        "run_id": "7e619159", "mcp_version": "1.0.1",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 119, "end_condition": "truncated",
        "normalised_score": None,
    },
    {
        "run_id": "0290368e", "mcp_version": "1.0.4",
        "model": "gpt-5.4", "scenario": "ground_control",
        "turns_played": 32, "end_condition": "truncated",
        "normalised_score": None,
    },
    # ------------------------------------------------------------------ #
    # GPT-5.4 — Snowflake
    # ------------------------------------------------------------------ #
    {
        "run_id": "a62353e9", "mcp_version": "0.1.0",
        "model": "gpt-5.4", "scenario": "snowflake",
        "turns_played": 331, "end_condition": "hung",
        "normalised_score": None,
    },
    # ------------------------------------------------------------------ #
    # Gemini 3.1 Pro Preview — Ground Control (early 1.0.x, hex ID)
    # ------------------------------------------------------------------ #
    {
        "run_id": "8ce6d00f", "mcp_version": "1.0.4",
        "model": "gemini-3.1-pro-preview", "scenario": "ground_control",
        "turns_played": 80, "end_condition": "truncated",
        "normalised_score": None,
    },
]

PRE_HARNESS_IDS: set[str] = {r["run_id"] for r in PRE_HARNESS_RUNS}

# Subset that reached a natural game conclusion (victory or defeat)
PRE_HARNESS_COMPLETED = [r for r in PRE_HARNESS_RUNS if r["end_condition"] in ("victory", "defeat")]


def pre_harness_df():
    """Return PRE_HARNESS_RUNS as a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(PRE_HARNESS_RUNS)


if __name__ == "__main__":
    df = pre_harness_df()
    print(f"Pre-harness runs: {len(df)}")
    print()
    print("End condition breakdown:")
    print(df["end_condition"].value_counts().to_string())
    print()
    print(f"Completed (victory+defeat): {len(PRE_HARNESS_COMPLETED)}")
    for r in PRE_HARNESS_COMPLETED:
        print(f"  {r['run_id']} ({r['model']}, T{r['turns_played']}, {r['end_condition']}, norm={r['normalised_score']:.3f})")
