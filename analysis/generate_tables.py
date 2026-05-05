"""Generate LaTeX table files for all paper tables.

Run:  .venv/bin/python analysis/generate_tables.py

Outputs .tex files to analysis/figures/ alongside the existing icc_table.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIGURES = ROOT / "analysis" / "figures"
FIGURES.mkdir(exist_ok=True)

from analysis.admissible_games import ADMISSIBLE_GAMES, admissible_df


MODEL_LABELS = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.4": "GPT-5.4",
    "Kimi-K2.5": "Kimi-K2.5",
}

MODEL_ORDER = ["claude-opus-4-6", "gemini-3.1-pro-preview", "gpt-5.4", "Kimi-K2.5"]


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _esc(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_")


# ---------------------------------------------------------------------------
# Table 2: Aggregate Performance
# ---------------------------------------------------------------------------
def generate_table2():
    """Aggregate performance table (model, games, victories, norm score)."""
    import pandas as pd

    try:
        from analysis.data_loader import load_games_df
        games = load_games_df()
    except Exception as e:
        print(f"  Skipped table2 (data unavailable): {e}")
        return

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Aggregate performance across " + str(len(games)) + r" admissible games.}",
        r"  \label{tab:aggregate}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Model & Games & Victories & Defeats & Norm.\ score (mean $\pm$ SD) \\",
        r"    \midrule",
    ]

    for model in MODEL_ORDER:
        sub = games[games["model"] == model]
        if sub.empty:
            continue
        n = len(sub)
        wins = (sub["outcome"] == "victory").sum() if "outcome" in sub.columns else 0
        # fallback: check normalised_score == 1.0
        if wins == 0 and "normalised_score" in sub.columns:
            wins = (sub["normalised_score"] >= 0.999).sum()
        defeats = n - wins
        win_pct = 100 * wins / n if n > 0 else 0
        mean_ns = sub["normalised_score"].mean()
        sd_ns = sub["normalised_score"].std()

        if n == 1:
            score_str = f"{mean_ns:.3f}"
        else:
            score_str = f"{mean_ns:.3f} $\\pm$ {sd_ns:.3f}"

        lines.append(
            f"    {_label(model)} & {n} & {wins} ({win_pct:.0f}\\%) & {defeats} & {score_str} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_aggregate.tex").write_text(tex)
    print(f"  ✓ table_aggregate.tex")


# ---------------------------------------------------------------------------
# Table 2a: Model × Scenario Outcomes
# ---------------------------------------------------------------------------
def generate_table2a():
    """Model × scenario outcome breakdown."""
    df = admissible_df()
    df["won"] = (df["end_condition"] == "victory").astype(int)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Outcome breakdown by model and scenario.}",
        r"  \label{tab:model-scenario}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Model & Scenario & Games & Victories & Defeats \\",
        r"    \midrule",
    ]

    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue
        for scenario in ["ground_control", "snowflake"]:
            sub = model_df[model_df["scenario"] == scenario]
            if sub.empty:
                continue
            n = len(sub)
            wins = sub["won"].sum()
            defeats = n - wins
            scenario_label = "Ground Control" if scenario == "ground_control" else "Snowflake"
            lines.append(
                f"    {_label(model)} & {scenario_label} & {n} & {wins} & {defeats} \\\\"
            )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_model_scenario.tex").write_text(tex)
    print(f"  ✓ table_model_scenario.tex")


# ---------------------------------------------------------------------------
# PMR Table
# ---------------------------------------------------------------------------
def generate_pmr_table():
    """PMR by subcategory per model."""
    try:
        from analysis.data_loader import load_tools_df
        tools_df = load_tools_df()
    except Exception as e:
        print(f"  Skipped pmr table (Azure unavailable): {e}")
        return

    if tools_df.empty:
        print("  Skipped pmr table (no data)")
        return

    import pandas as pd

    subcats = ["victory_monitoring", "strategic_map", "resource_monitoring"]
    subcat_labels = {
        "victory_monitoring": "Victory mon.",
        "strategic_map": "Strategic map",
        "resource_monitoring": "Resource mon.",
    }

    rows = {}
    for model in MODEL_ORDER:
        model_tools = tools_df[tools_df["model"] == model]
        if model_tools.empty:
            continue

        per_game = []
        for gid, gdf in model_tools.groupby("game_id"):
            denom = gdf["is_pmr_denominator"].sum()
            if denom == 0:
                continue
            overall = (gdf["category"] == "strategic_monitoring").sum() / denom
            sub_rates = {}
            for sub in subcats:
                sub_rates[sub] = (gdf["pmr_subcategory"] == sub).sum() / denom
            per_game.append({"overall": overall, **sub_rates})

        if not per_game:
            continue
        pg_df = pd.DataFrame(per_game)
        rows[model] = {k: pg_df[k].mean() * 100 for k in ["overall"] + subcats}

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Proactive Monitoring Rate (\%) by subcategory.}",
        r"  \label{tab:pmr}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Model & Overall PMR & Victory mon. & Strategic map & Resource mon. \\",
        r"    \midrule",
    ]

    for model in MODEL_ORDER:
        if model not in rows:
            continue
        r = rows[model]
        lines.append(
            f"    {_label(model)} & {r['overall']:.2f} & {r['victory_monitoring']:.2f} "
            f"& {r['strategic_map']:.2f} & {r['resource_monitoring']:.2f} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_pmr.tex").write_text(tex)
    print(f"  ✓ table_pmr.tex")


# ---------------------------------------------------------------------------
# RAG Table
# ---------------------------------------------------------------------------
def generate_rag_table():
    """RAG@10 results per model."""
    import pandas as pd

    csv_path = ROOT / "analysis" / "rag_auto_results.csv"
    if not csv_path.exists():
        print("  Skipped rag table (rag_auto_results.csv not found)")
        return

    rag_df = pd.read_csv(csv_path)

    try:
        from analysis.data_loader import load_games_df
        games_df = load_games_df()
        model_map = dict(zip(games_df["run_id"], games_df["model"]))
        rag_df["model"] = rag_df["run_id"].map(model_map)
    except Exception:
        pass

    import numpy as np
    score_map = {"Y": 1.0, "P": 0.5, "N": 0.0}
    rng = np.random.default_rng(0)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Reflection-Action Gap results (RAG@10). 95\% CIs from per-commitment bootstrap (10,000 resamples).}",
        r"  \label{tab:rag}",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Model & Commitments & Executed (Y) & Partial (P) & Not done (N) & RAG@10 [95\% CI] \\",
        r"    \midrule",
    ]

    for model in MODEL_ORDER:
        sub = rag_df[rag_df["model"] == model] if "model" in rag_df.columns else pd.DataFrame()
        if sub.empty:
            continue
        total = len(sub)
        y = (sub["executed"] == "Y").sum()
        p = (sub["executed"] == "P").sum()
        n = (sub["executed"] == "N").sum()
        rag_score = (y + 0.5 * p) / total * 100 if total > 0 else 0
        y_pct = 100 * y / total
        p_pct = 100 * p / total
        n_pct = 100 * n / total
        scores = sub["executed"].map(score_map).to_numpy()
        boots = rng.choice(scores, size=(10000, total), replace=True).mean(axis=1) * 100
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        lines.append(
            f"    {_label(model)} & {total} & {y} ({y_pct:.0f}\\%) "
            f"& {p} ({p_pct:.0f}\\%) & {n} ({n_pct:.0f}\\%) "
            f"& {rag_score:.1f}\\% [{ci_lo:.1f}, {ci_hi:.1f}] \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_rag.tex").write_text(tex)
    print(f"  ✓ table_rag.tex")


# ---------------------------------------------------------------------------
# Capability Profiles Table (calls/turn with SDs)
# ---------------------------------------------------------------------------
def generate_capability_table():
    """Tool calls per turn, mean ± SD, per model."""
    try:
        from analysis.data_loader import load_tools_df
        tools_df = load_tools_df()
    except Exception as e:
        print(f"  Skipped capability table (Azure unavailable): {e}")
        return

    if tools_df.empty:
        print("  Skipped capability table (no data)")
        return

    import numpy as np

    per_game_turn = tools_df.groupby(["game_id", "turn"]).size().reset_index(name="calls")
    model_map = tools_df.groupby("game_id")["model"].first()
    per_game_turn["model"] = per_game_turn["game_id"].map(model_map)
    per_game_mean = per_game_turn.groupby(["game_id", "model"])["calls"].mean().reset_index()

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Tool calls per turn (mean $\pm$ SD across games).}",
        r"  \label{tab:calls-per-turn}",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        r"    Model & Calls/turn & Games \\",
        r"    \midrule",
    ]

    for model in MODEL_ORDER:
        sub = per_game_mean[per_game_mean["model"] == model]
        if sub.empty:
            continue
        mean = sub["calls"].mean()
        sd = sub["calls"].std()
        n = len(sub)
        if n == 1 or np.isnan(sd):
            val_str = f"{mean:.1f}"
        else:
            val_str = f"{mean:.1f} $\\pm$ {sd:.1f}"
        lines.append(f"    {_label(model)} & {val_str} & {n} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_calls_per_turn.tex").write_text(tex)
    print(f"  ✓ table_calls_per_turn.tex")


# ---------------------------------------------------------------------------
# Feature Comparison Table (CivRealm vs CivBench)
# ---------------------------------------------------------------------------
def generate_feature_comparison():
    """CivRealm vs CivBench feature comparison (Table 3 in paper)."""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Feature comparison: CivRealm (FreeCiv) vs.\ CivBench (Civ~VI).}",
        r"  \label{tab:feature-comparison}",
        r"  \begin{tabular}{lll}",
        r"    \toprule",
        r"    Feature & CivRealm (FreeCiv) & CivBench (Civ~VI) \\",
        r"    \midrule",
        r"    Game engine & FreeCiv (open-source, Civ~II-era) & Civilization~VI (commercial, 2016+) \\",
        r"    Grid type & Square, 8-directional & Hex, 6-directional \\",
        r"    Unit stacking & Allowed & One-unit-per-tile \\",
        r"    Districts & No & Yes (13 types + unique) \\",
        r"    Governors & No & Yes (7 governors, promotions) \\",
        r"    Loyalty/amenities & No & Yes \\",
        r"    Espionage & Basic & Full (missions, counterintel) \\",
        r"    World Congress & No & Yes (resolutions, diplomatic VP) \\",
        r"    Great People & Basic & Full (recruitment race, activation) \\",
        r"    Government & Tax sliders & Policy card slots + types \\",
        r"    Religion & Basic & Full (pantheon, found, spread, inquisition) \\",
        r"    Victory conditions & 3 & 6 \\",
        r"    Agent interface & Gymnasium / LangChain & MCP + narration layer \\",
        r"    LLM training data & Minimal (niche OSS) & Extensive (decades of guides) \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_feature_comparison.tex").write_text(tex)
    print(f"  ✓ table_feature_comparison.tex")


# ---------------------------------------------------------------------------
# Tool Inventory Table
# ---------------------------------------------------------------------------
def generate_tool_inventory():
    """Tool inventory by category (Table 4 in paper)."""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{MCP tool inventory by category.}",
        r"  \label{tab:tool-inventory}",
        r"  \begin{tabular}{lrl}",
        r"    \toprule",
        r"    Category & Count & Representative tools \\",
        r"    \midrule",
        r"    State queries & 27 & \texttt{get\_game\_overview}, \texttt{get\_units}, \texttt{get\_map\_area} \\",
        r"    Unit actions & 10 & \texttt{unit\_action}, \texttt{upgrade\_unit}, \texttt{promote\_unit} \\",
        r"    City management & 9 & \texttt{set\_city\_production}, \texttt{purchase\_item} \\",
        r"    Diplomacy & 7 & \texttt{respond\_to\_diplomacy}, \texttt{propose\_trade} \\",
        r"    Governance & 10 & \texttt{set\_research}, \texttt{set\_policies}, \texttt{change\_government} \\",
        r"    Religion \& culture & 5 & \texttt{choose\_pantheon}, \texttt{choose\_dedication} \\",
        r"    Game lifecycle & 8 & \texttt{end\_turn}, \texttt{get\_diary}, \texttt{load\_save} \\",
        r"    \midrule",
        r"    \textbf{Total} & \textbf{76} & \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_tool_inventory.tex").write_text(tex)
    print(f"  ✓ table_tool_inventory.tex")


# ---------------------------------------------------------------------------
# Scenario Suite Table
# ---------------------------------------------------------------------------
def generate_scenario_table():
    """Scenario suite configuration (Table 1 in paper)."""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Standardised scenario suite.}",
        r"  \label{tab:scenarios}",
        r"  \begin{tabular}{lllll}",
        r"    \toprule",
        r"    Scenario & Map Type & Speed & Difficulty & Purpose \\",
        r"    \midrule",
        r"    Ground Control & Pangaea, standard & Quick & Prince & Baseline strategic competence \\",
        r"    Snowflake & Six arm snowflake, small & Quick & King & Military under constraint \\",
        r"    Cry Havoc & Pangaea, tiny & Quick & Immortal & Stress-test at extreme AI advantage \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_scenarios.tex").write_text(tex)
    print(f"  ✓ table_scenarios.tex")


# ---------------------------------------------------------------------------
# District Adjacency Table
# ---------------------------------------------------------------------------
def generate_district_table():
    """District adjacency bonuses (referenced in CLAUDE.md / paper appendix)."""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{District adjacency bonuses.}",
        r"  \label{tab:districts}",
        r"  \begin{tabular}{ll}",
        r"    \toprule",
        r"    District & Adjacency bonuses \\",
        r"    \midrule",
        r"    Campus & +1 per mountain, +1 per 2 jungles, +2 geothermal/reef \\",
        r"    Holy Site & +1 per mountain, +1 per 2 forests, +2 natural wonder \\",
        r"    Industrial Zone & +1 per mine/quarry, +2 aqueduct \\",
        r"    Commercial Hub & +2 adjacent river, +2 harbor \\",
        r"    Theater Square & +1 per wonder, +2 Entertainment Complex \\",
        r"    Encampment & Cannot be adjacent to city center \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (FIGURES / "table_districts.tex").write_text(tex)
    print(f"  ✓ table_districts.tex")


# ---------------------------------------------------------------------------
# Missed-warning analysis (sensorium effect, §6.1)
# ---------------------------------------------------------------------------
def generate_missed_warnings_table():
    """Per-model missed-warning analysis for defeated games.

    A "missed warning" is a defeat where get_victory_progress was NOT called in
    the 20-turn window before game end despite the game ending in a (detectable)
    rival victory. Aligns with the §6.1 claim about the sensorium effect.
    """
    try:
        from analysis.data_loader import load_games_df, load_tools_df
        games = load_games_df()
        tools = load_tools_df()
    except Exception as e:
        print(f"  Skipped missed_warnings (data unavailable): {e}")
        return

    defeats = games[games["outcome"] == "defeat"].copy()
    vp = tools[tools["tool"] == "get_victory_progress"].copy()

    rows = []
    for _, g in defeats.iterrows():
        last_turn = g["turns_played"]
        window_start = last_turn - 19  # inclusive: [last-19, last] = exactly 20 turns
        sub = vp[vp["game_id"] == g["game_id"]]
        in_window = sub[
            (sub["turn"].fillna(-1) >= window_start)
            & (sub["turn"].fillna(-1) <= last_turn)
        ]
        rows.append({
            "model": g["model"],
            "detectable": 1,  # all victory_types in this dataset are detectable via get_victory_progress
            "queried": 1 if len(in_window) > 0 else 0,
            "missed": 0 if len(in_window) > 0 else 1,
            "calls_last_20": len(in_window),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = df.groupby("model").agg(
        defeats=("detectable", "count"),
        detectable=("detectable", "sum"),
        queried=("queried", "sum"),
        missed=("missed", "sum"),
    )

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Missed-warning analysis for defeated runs. A warning is counted as detectable if \texttt{get\_victory\_progress} would have exposed a rival victory threat at least 20 turns before game end. Queried = at least one \texttt{get\_victory\_progress} call in the 20-turn window before game end. Missed = detectable defeats with no query in window.}",
        r"  \label{tab:missed-warnings}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Model & Defeats & Detectable warnings & Queried in window & Missed warnings \\",
        r"    \midrule",
    ]
    for model in MODEL_ORDER:
        if model not in agg.index:
            continue
        r = agg.loc[model]
        lines.append(
            f"    {_label(model)} & {int(r['defeats'])} & {int(r['detectable'])} "
            f"& {int(r['queried'])} & {int(r['missed'])} \\\\"
        )
    # Totals row
    total_d = int(df["detectable"].sum())
    total_q = int(df["queried"].sum())
    total_m = int(df["missed"].sum())
    lines.append(r"    \midrule")
    lines.append(
        f"    \\textbf{{Total}} & \\textbf{{{len(df)}}} & \\textbf{{{total_d}}} "
        f"& \\textbf{{{total_q}}} & \\textbf{{{total_m}}} \\\\"
    )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    (FIGURES / "table_missed_warnings.tex").write_text("\n".join(lines))
    print("  ✓ table_missed_warnings.tex")
    return df


# ---------------------------------------------------------------------------
# ELO leaderboard (appendix)
# ---------------------------------------------------------------------------
def generate_elo_table():
    """Two-part ELO leaderboard: LLM models + top AI leaders."""
    try:
        from analysis.elo import compute_elo, split
        out = compute_elo()
    except Exception as e:
        print(f"  Skipped elo (data unavailable): {e}")
        return
    models, ais = split(out["ratings"])

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{ELO leaderboard across " + str(out["n_games"]) + r" admissible games. "
        r"Free-for-all pairwise scoring with $K=32/\sqrt{N-1}$ and base 1500; "
        r"the winner of each game beats every loser pairwise. AI leaders shown are the top eight by ELO.}",
        r"  \label{tab:elo}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Type & Participant & ELO & Games & Wins & Losses \\",
        r"    \midrule",
    ]
    for r in models:
        label = MODEL_LABELS.get(r["name"], r["name"])
        lines.append(
            f"    Model & {_esc(label)} & {r['elo']} & "
            f"{r['games']} & {r['wins']} & {r['losses']} \\\\"
        )
    lines.append(r"    \midrule")
    for r in ais[:8]:
        lines.append(
            f"    AI & {_esc(r['name'])} & {r['elo']} & "
            f"{r['games']} & {r['wins']} & {r['losses']} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    (FIGURES / "table_elo.tex").write_text("\n".join(lines))
    print("  ✓ table_elo.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating LaTeX tables...\n")

    # Data-driven tables
    generate_table2()
    generate_table2a()
    generate_pmr_table()
    generate_rag_table()
    generate_capability_table()
    generate_missed_warnings_table()
    generate_elo_table()

    # Static tables from paper
    generate_feature_comparison()
    generate_tool_inventory()
    generate_scenario_table()

    print("\nAll tables saved to analysis/figures/")
