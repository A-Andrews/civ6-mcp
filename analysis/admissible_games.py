"""Admissible games list for CivBench analysis.

Admissibility criteria:
  - Code version v1.1.5 or later (consistent telemetry schema, reliable game-over detection)
  - Natural game completion (turn limit reached or a winner declared — no infrastructure exits)
  - turns_played >= 50

Games are added here manually as runs are validated. Import ADMISSIBLE_IDS
anywhere you need to filter to valid games only.
"""

from __future__ import annotations

# Each entry: run_id, model, scenario, final_turn, end_condition, notes
ADMISSIBLE_GAMES: list[dict] = [
    # ------------------------------------------------------------------ #
    # Claude Opus 4.6 — Ground Control
    # ------------------------------------------------------------------ #
    {
        "run_id": "solar-flax-forum-10",
        "model": "claude-opus-4-6",
        "scenario": "ground_control",
        "final_turn": 238,
        "end_condition": "defeat",
        "notes": "AI Culture victory (Japan/Hojo Tokimune)",
    },
    {
        "run_id": "steep-vermil-trebuchet-27",
        "model": "claude-opus-4-6",
        "scenario": "ground_control",
        "final_turn": 308,
        "end_condition": "defeat",
        "notes": "AI Technology victory (Scotland/Robert the Bruce)",
    },
    {
        "run_id": "wandering-carmine-ballista-57",
        "model": "claude-opus-4-6",
        "scenario": "ground_control",
        "final_turn": 326,
        "end_condition": "victory",
        "notes": "Agent Technology victory (Babylon)",
    },
    # ------------------------------------------------------------------ #
    # Claude Opus 4.6 — Snowflake
    # ------------------------------------------------------------------ #
    {
        "run_id": "hidden-lilac-atlas-39",
        "model": "claude-opus-4-6",
        "scenario": "snowflake",
        "final_turn": 329,
        "end_condition": "defeat",
        "notes": "AI Score victory (Scythia)",
    },
    {
        "run_id": "regal-sepia-corsair-87",
        "model": "claude-opus-4-6",
        "scenario": "snowflake",
        "final_turn": 330,
        "end_condition": "defeat",
        "notes": "AI Score victory (Scythia)",
    },
    # ------------------------------------------------------------------ #
    # Gemini 3.1 Pro Preview — Ground Control
    # ------------------------------------------------------------------ #
    {
        "run_id": "remnant-khaki-oracle-28",
        "model": "gemini-3.1-pro-preview",
        "scenario": "ground_control",
        "final_turn": 263,
        "end_condition": "defeat",
        "notes": "AI Culture victory (Japan/Hojo Tokimune)",
    },
    {
        "run_id": "volcanic-vermil-ember-37",
        "model": "gemini-3.1-pro-preview",
        "scenario": "ground_control",
        "final_turn": 300,
        "end_condition": "victory",
        "notes": "Agent Technology victory (Babylon)",
    },
    {
        "run_id": "vast-cedar-chronicle-40",
        "model": "gemini-3.1-pro-preview",
        "scenario": "ground_control",
        "final_turn": 305,
        "end_condition": "defeat",
        "notes": "AI Technology victory (Scotland/Robert the Bruce)",
    },
    # ------------------------------------------------------------------ #
    # GPT-5.4 — Ground Control
    # ------------------------------------------------------------------ #
    {
        "run_id": "flint-olive-phalanx-07",
        "model": "gpt-5.4",
        "scenario": "ground_control",
        "final_turn": 230,
        "end_condition": "defeat",
        "notes": "AI Culture victory (Japan/Hojo Tokimune)",
    },
    {
        "run_id": "dread-cobalt-chariot-34",
        "model": "gpt-5.4",
        "scenario": "ground_control",
        "final_turn": 298,
        "end_condition": "defeat",
        "notes": "AI Score victory (Japan/Hojo Tokimune)",
    },
    {
        "run_id": "fierce-umber-garrison-15",
        "model": "gpt-5.4",
        "scenario": "ground_control",
        "final_turn": 300,
        "end_condition": "defeat",
        "notes": "AI Technology victory (Korea/Seondeok)",
    },
    {
        "run_id": "veiled-lapis-flagship-44",
        "model": "gpt-5.4",
        "scenario": "ground_control",
        "final_turn": 315,
        "end_condition": "defeat",
        "notes": "AI Technology victory (Japan/Hojo Tokimune)",
    },
    # ------------------------------------------------------------------ #
    # GPT-5.4 — Snowflake
    # ------------------------------------------------------------------ #
    {
        "run_id": "fierce-rust-horseman-79",
        "model": "gpt-5.4",
        "scenario": "snowflake",
        "final_turn": 312,
        "end_condition": "defeat",
        "notes": "AI Domination victory (Kongo)",
    },
    {
        "run_id": "celestial-ochre-monument-04",
        "model": "gpt-5.4",
        "scenario": "snowflake",
        "final_turn": 331,
        "end_condition": "defeat",
        "notes": "AI Score victory (Scythia)",
    },
]

# Flat set of run IDs for fast membership testing
ADMISSIBLE_IDS: set[str] = {g["run_id"] for g in ADMISSIBLE_GAMES}


def admissible_df():
    """Return ADMISSIBLE_GAMES as a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(ADMISSIBLE_GAMES)


if __name__ == "__main__":
    df = admissible_df()
    print(f"Admissible games: {len(df)}")
    print()
    print(df.groupby(["model", "scenario"]).size().to_string())
    print()
    print(df[["run_id", "model", "scenario", "final_turn", "end_condition"]].to_string(index=False))
