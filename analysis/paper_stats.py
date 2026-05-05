"""Compute all missing statistics for the CivBench paper.

Run:  .venv/bin/python analysis/paper_stats.py

Prints computed values for insertion into civbench_paper.md and macros.tex.
Does not modify any files. Requires Convex + Azure credentials for full output.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from analysis.admissible_games import ADMISSIBLE_GAMES, admissible_df


# ---------------------------------------------------------------------------
# 1a. Fisher's exact test on win rates
# ---------------------------------------------------------------------------
def fisher_exact_win_rates():
    """Fisher-Freeman-Halton test on 4x2 model x {victory, defeat} table."""
    print("=" * 70)
    print("1a. FISHER'S EXACT TEST ON WIN RATES")
    print("=" * 70)

    df = admissible_df()
    df["won"] = (df["end_condition"] == "victory").astype(int)
    df["lost"] = 1 - df["won"]

    # Contingency table
    ct = df.groupby("model")[["won", "lost"]].sum()
    print("\nContingency table (model x {victory, defeat}):")
    print(ct.to_string())
    print()

    # Fisher-Freeman-Halton via chi2_contingency with simulation for small counts
    # scipy >= 1.7 supports fisher_exact only for 2x2; use chi2 with correction
    # or Monte Carlo approach
    table = ct.values
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    print(f"Chi-squared test: chi2={chi2:.3f}, p={p_chi2:.3f}, dof={dof}")
    print(f"  (Appropriate caveat: expected counts < 5 in some cells; interpret with caution)")

    # Use boschloo for pairwise 2x2 comparisons
    models = ct.index.tolist()
    print("\nPairwise Fisher's exact tests (2x2):")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            tbl_2x2 = ct.loc[[m1, m2]].values
            if tbl_2x2.shape == (2, 2):
                _, p_fisher = stats.fisher_exact(tbl_2x2)
                print(f"  {m1} vs {m2}: p = {p_fisher:.3f}")

    print("\nFor paper: report chi2 p-value with small-sample caveat,")
    print("plus pairwise Fisher tests for the primary comparisons.")
    print()


# ---------------------------------------------------------------------------
# 1b. Model x scenario outcome table
# ---------------------------------------------------------------------------
def model_scenario_table():
    """Cross-tab of model x scenario with outcomes."""
    print("=" * 70)
    print("1b. MODEL x SCENARIO OUTCOME TABLE")
    print("=" * 70)

    df = admissible_df()
    df["won"] = (df["end_condition"] == "victory").astype(int)

    rows = []
    for (model, scenario), grp in df.groupby(["model", "scenario"]):
        rows.append({
            "Model": model,
            "Scenario": scenario,
            "Games": len(grp),
            "Victories": grp["won"].sum(),
            "Defeats": len(grp) - grp["won"].sum(),
        })

    table = pd.DataFrame(rows)
    print("\n| Model | Scenario | Games | Victories | Defeats |")
    print("| ----- | -------- | ----- | --------- | ------- |")
    for _, r in table.iterrows():
        print(f"| {r['Model']} | {r['Scenario']} | {r['Games']} | {r['Victories']} | {r['Defeats']} |")
    print()


# ---------------------------------------------------------------------------
# 1c. SDs on mean calls/turn (requires Azure)
# ---------------------------------------------------------------------------
def calls_per_turn_sds():
    """Compute mean and SD of calls/turn per model."""
    print("=" * 70)
    print("1c. CALLS PER TURN — MEAN ± SD")
    print("=" * 70)

    try:
        from analysis.data_loader import load_tools_df, load_games_df
        tools_df = load_tools_df()
        games_df = load_games_df()
    except Exception as e:
        print(f"  Skipped (Azure unavailable): {e}")
        return {}

    if tools_df.empty:
        print("  No tool call data available.")
        return {}

    # Calls per turn per game
    per_game_turn = tools_df.groupby(["game_id", "turn"]).size().reset_index(name="calls")
    # Merge model info
    model_map = tools_df.groupby("game_id")["model"].first()
    per_game_turn["model"] = per_game_turn["game_id"].map(model_map)

    # Per-game mean calls/turn
    per_game_mean = per_game_turn.groupby(["game_id", "model"])["calls"].mean().reset_index()

    # Per-model mean and SD of (per-game mean calls/turn)
    result = per_game_mean.groupby("model")["calls"].agg(["mean", "std", "count"])
    print("\n  Per-model mean ± SD of calls/turn:")
    sds = {}
    for model, row in result.iterrows():
        sd_str = f"{row['std']:.1f}" if not np.isnan(row['std']) else "n=1"
        print(f"    {model}: {row['mean']:.1f} ± {sd_str}  (n={int(row['count'])} games)")
        sds[model] = row["std"]

    print("\n  Macros to add:")
    for model, row in result.iterrows():
        safe_name = model.replace("-", "").replace(".", "").replace("_", "")
        # Map to paper macro names
        name_map = {
            "claudeopus46": "Claude",
            "gemini31propreview": "Gemini",
            "gpt54": "GPT",
            "KimiK25": "Kimi",
        }
        macro_name = name_map.get(safe_name, safe_name)
        if not np.isnan(row["std"]):
            print(f"    \\newcommand{{\\SDCallsPerTurn{macro_name}}}{{{row['std']:.1f}}}")

    # Also report total tool calls
    total = len(tools_df)
    print(f"\n  Total tool calls across all admissible games: {total:,}")
    print(f"    \\newcommand{{\\NTotalToolCalls}}{{{total}}}")
    print()
    return sds


# ---------------------------------------------------------------------------
# 1d. KW test on between-model PMR
# ---------------------------------------------------------------------------
def kw_test_pmr():
    """Kruskal-Wallis test on per-game PMR across models."""
    print("=" * 70)
    print("1d. KRUSKAL-WALLIS TEST ON BETWEEN-MODEL PMR")
    print("=" * 70)

    try:
        from analysis.data_loader import load_tools_df
        tools_df = load_tools_df()
    except Exception as e:
        print(f"  Skipped (Azure unavailable): {e}")
        return

    if tools_df.empty:
        print("  No tool call data available.")
        return

    # Compute per-game PMR
    pmr_rows = []
    for gid, gdf in tools_df.groupby("game_id"):
        model = gdf["model"].iloc[0]
        denom = gdf["is_pmr_denominator"].sum()
        if denom == 0:
            continue
        strat_count = (gdf["category"] == "strategic_monitoring").sum()
        pmr_rows.append({"game_id": gid, "model": model, "pmr": strat_count / denom})

    pmr_df = pd.DataFrame(pmr_rows)
    print("\n  Per-game PMR by model:")
    for model, grp in pmr_df.groupby("model"):
        vals = grp["pmr"].values * 100  # percent
        print(f"    {model}: {vals.mean():.2f}% ± {vals.std():.2f}%  (n={len(vals)})")

    # KW test
    groups = [grp["pmr"].values for _, grp in pmr_df.groupby("model") if len(grp) >= 2]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis: H={H:.3f}, p={p:.3f}")
        if p < 0.05:
            print("  → Significant difference in PMR across models")
        else:
            print("  → No significant difference (expected at small n)")
    else:
        print("\n  Too few groups with n≥2 for KW test")

    # Updated PMR table values
    print("\n  Updated PMR values for macros/paper (percent):")
    for model, grp in pmr_df.groupby("model"):
        overall = grp["pmr"].mean() * 100
        print(f"    {model} overall PMR: {overall:.2f}%")

    # Victory monitoring sub-rate
    vm_rows = []
    for gid, gdf in tools_df.groupby("game_id"):
        model = gdf["model"].iloc[0]
        denom = gdf["is_pmr_denominator"].sum()
        if denom == 0:
            continue
        vm_count = (gdf["pmr_subcategory"] == "victory_monitoring").sum()
        vm_rows.append({"game_id": gid, "model": model, "vm_rate": vm_count / denom,
                        "vm_calls": vm_count})

    vm_df = pd.DataFrame(vm_rows)
    print("\n  Victory monitoring rates:")
    for model, grp in vm_df.groupby("model"):
        rate = grp["vm_rate"].mean() * 100
        calls = grp["vm_calls"].mean()
        print(f"    {model}: {rate:.2f}%, mean {calls:.1f} calls/game")
    print()


# ---------------------------------------------------------------------------
# 1e. BH-FDR summary from inflection deltas
# ---------------------------------------------------------------------------
def bhfdr_summary():
    """Compute BH-FDR results from inflection delta analysis."""
    print("=" * 70)
    print("1e. BH-FDR CORRECTION — INFLECTION DELTAS")
    print("=" * 70)

    try:
        from analysis.data_loader import load_diary_df, load_games_df
        diary_df = load_diary_df()
        games_df = load_games_df()
    except Exception as e:
        print(f"  Skipped: {e}")
        return

    import warnings
    from scipy.stats import spearmanr, false_discovery_control

    metrics = ["score", "science", "cities", "culture", "gold", "exploration_pct"]
    scenario = "ground_control"
    window, t_min, t_max, t_step = 5, 20, 245, 5

    gc_games = games_df[games_df["scenario"] == scenario][
        ["game_id", "normalised_score"]
    ].set_index("game_id")

    agent_diary = diary_df[
        diary_df["is_agent"] & diary_df["game_id"].isin(gc_games.index)
    ]

    results: dict[str, list[tuple[int, float, float, int]]] = {m: [] for m in metrics}
    for t in range(t_min, t_max + 1, t_step):
        at_t0 = agent_diary[agent_diary["turn"] == t][["game_id"] + metrics]
        at_t1 = agent_diary[agent_diary["turn"] == t + window][["game_id"] + metrics]
        merged = at_t0.merge(at_t1, on="game_id", suffixes=("_t0", "_t1"))
        merged = merged.merge(gc_games, left_on="game_id", right_index=True)
        for m in metrics:
            sub = merged[[f"{m}_t0", f"{m}_t1", "normalised_score"]].dropna()
            n = len(sub)
            if n < 4:
                results[m].append((t, float("nan"), float("nan"), n))
                continue
            delta = sub[f"{m}_t1"] - sub[f"{m}_t0"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = spearmanr(delta, sub["normalised_score"])
            results[m].append((t, rho, p, n))

    all_tests = [
        (m, t, rho, p, n)
        for m in metrics
        for t, rho, p, n in results[m]
        if not np.isnan(rho)
    ]

    print(f"\n  Total (metric, window) tests: {len(all_tests)}")

    if all_tests:
        p_vals = np.array([x[3] for x in all_tests])
        p_adj = false_discovery_control(p_vals, method="bh")
        surviving = [(x[0], x[1], x[2], x[3], pa)
                     for x, pa in zip(all_tests, p_adj) if pa < 0.05]
        print(f"  Tests surviving BH-FDR at α=0.05: {len(surviving)}")
        if surviving:
            print("  Surviving tests:")
            for m, t, rho, p_raw, p_adj_val in surviving:
                print(f"    {m} at T={t}: ρ={rho:+.3f}, p_raw={p_raw:.4f}, p_adj={p_adj_val:.4f}")
        else:
            print("  No tests survive FDR correction — individual correlations are")
            print("  suggestive but do not withstand multiple-comparison correction.")
    print()


# ---------------------------------------------------------------------------
# 1f. Reflection quality correlations
# ---------------------------------------------------------------------------
def reflection_quality_stats():
    """Compute Spearman ρ for commitment density and word count vs score."""
    print("=" * 70)
    print("1f. REFLECTION QUALITY CORRELATIONS")
    print("=" * 70)

    try:
        from analysis.data_loader import load_games_df
        games_df = load_games_df()
    except Exception as e:
        print(f"  Skipped: {e}")
        return

    csv_path = ROOT / "analysis" / "rag_auto_results.csv"
    if not csv_path.exists():
        print("  rag_auto_results.csv not found — skipping.")
        return

    rag_df = pd.read_csv(csv_path)

    # Panel A: commitment density per game
    commit_agg = (
        rag_df.groupby("run_id")
        .agg(n_commitments=("commitment", "size"),
             n_turns=("turn", "nunique"))
        .reset_index()
    )
    commit_agg["density"] = commit_agg["n_commitments"] / commit_agg["n_turns"].clip(lower=1)

    # Merge with normalised score
    games_df_slim = games_df[["run_id", "normalised_score", "model"]].copy()
    panel_a = commit_agg.merge(games_df_slim, on="run_id", how="inner")

    if len(panel_a) >= 3:
        rho_a, p_a = stats.spearmanr(panel_a["density"], panel_a["normalised_score"])
        print(f"\n  Panel A — Commitment density vs normalised score:")
        print(f"    Spearman ρ = {rho_a:+.3f}, p = {p_a:.3f}, n = {len(panel_a)}")
    else:
        print(f"\n  Panel A: too few data points (n={len(panel_a)})")

    # Panel B: planning word count (would need reflections data)
    print("\n  Panel B — Word count: requires load_reflections (Convex).")
    print("  (Will be visible on the generated figure.)")
    print()


# ---------------------------------------------------------------------------
# 1g. Tool error rate summary
# ---------------------------------------------------------------------------
def tool_error_summary():
    """Top failing tools and model differences."""
    print("=" * 70)
    print("1g. TOOL ERROR RATE SUMMARY")
    print("=" * 70)

    try:
        from analysis.data_loader import load_tools_df
        tools_df = load_tools_df()
    except Exception as e:
        print(f"  Skipped (Azure unavailable): {e}")
        return

    if tools_df.empty:
        print("  No tool call data available.")
        return

    t = tools_df.copy()
    t["failed"] = ~t["success"].astype(bool)

    # Overall failure rate
    total_calls = len(t)
    total_failures = t["failed"].sum()
    print(f"\n  Overall: {total_failures:,} failures / {total_calls:,} calls ({100*total_failures/total_calls:.1f}%)")

    # Per-model failure rate
    print("\n  Per-model failure rate:")
    for model, grp in t.groupby("model"):
        n, f = len(grp), grp["failed"].sum()
        print(f"    {model}: {f}/{n} ({100*f/n:.1f}%)")

    # Top 10 failing tools
    failures = t[t["failed"]].groupby("tool").size().sort_values(ascending=False)
    print("\n  Top 10 failing tools:")
    for tool, count in failures.head(10).items():
        total_for_tool = len(t[t["tool"] == tool])
        print(f"    {tool}: {count} failures / {total_for_tool} calls ({100*count/total_for_tool:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# 1h. Cost estimation from logs
# ---------------------------------------------------------------------------
def cost_estimation():
    """Estimate per-model costs from tool call logs."""
    print("=" * 70)
    print("1h. COST ESTIMATION FROM LOGS")
    print("=" * 70)

    try:
        from analysis.data_loader import load_tools_df, load_games_df
        tools_df = load_tools_df()
        games_df = load_games_df()
    except Exception as e:
        print(f"  Skipped (Azure unavailable): {e}")
        return

    if tools_df.empty:
        print("  No tool call data available.")
        return

    # Published pricing (per 1M tokens) as of early 2026 — rough estimates
    PRICING = {
        "claude-opus-4-6": {"input": 15.0, "output": 75.0},
        "gpt-5.4": {"input": 10.0, "output": 30.0},
        "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.0},
        "Kimi-K2.5": {"input": 2.0, "output": 8.0},
    }

    # Estimate tokens per tool call: ~500 input + ~800 output on average
    # (rough heuristic from typical MCP tool invocations)
    EST_INPUT_TOKENS = 500
    EST_OUTPUT_TOKENS = 800

    print("\n  | Model | Games | Mean calls/game | Est. tokens/game (M) | Est. $/game | Duration range |")
    print("  | ----- | ----- | --------------- | -------------------- | ----------- | -------------- |")

    for model in sorted(tools_df["model"].unique()):
        model_tools = tools_df[tools_df["model"] == model]
        model_games = games_df[games_df["model"] == model]
        n_games = len(model_games)
        calls_per_game = model_tools.groupby("game_id").size()
        mean_calls = calls_per_game.mean()

        est_input = mean_calls * EST_INPUT_TOKENS / 1e6
        est_output = mean_calls * EST_OUTPUT_TOKENS / 1e6

        pricing = PRICING.get(model, {"input": 10.0, "output": 30.0})
        est_cost = est_input * pricing["input"] + est_output * pricing["output"]

        # Duration from game turns
        turns = model_games["turns_played"] if "turns_played" in model_games.columns else pd.Series()
        if not turns.empty:
            dur_str = f"T{int(turns.min())}-T{int(turns.max())}"
        else:
            dur_str = "—"

        print(f"  | {model} | {n_games} | {mean_calls:.0f} | {est_input + est_output:.2f} | ${est_cost:.0f} | {dur_str} |")

    print("\n  Note: token estimates are rough (500 input + 800 output per tool call).")
    print("  Actual costs depend heavily on context window size, which grows per turn.")
    print("  These are lower bounds; real costs are likely 3-10x higher due to context.")
    print()


# ---------------------------------------------------------------------------
# Updated macros for 22-game dataset
# ---------------------------------------------------------------------------
def updated_game_counts():
    """Print updated game counts and win rates for macros.tex."""
    print("=" * 70)
    print("UPDATED GAME COUNTS & WIN RATES (22-game dataset)")
    print("=" * 70)

    df = admissible_df()
    df["won"] = (df["end_condition"] == "victory").astype(int)

    print(f"\n  Total admissible games: {len(df)}")
    gc = df[df["scenario"] == "ground_control"]
    sf = df[df["scenario"] == "snowflake"]
    print(f"  Ground Control: {len(gc)}")
    print(f"  Snowflake: {len(sf)}")
    print()

    for model, grp in df.groupby("model"):
        wins = grp["won"].sum()
        total = len(grp)
        rate = 100 * wins / total if total > 0 else 0
        gc_n = len(grp[grp["scenario"] == "ground_control"])
        sf_n = len(grp[grp["scenario"] == "snowflake"])
        print(f"  {model}: {total} games ({gc_n} GC, {sf_n} SF), {wins} victories ({rate:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nCivBench Paper Statistics\n")

    # These don't need Azure
    updated_game_counts()
    fisher_exact_win_rates()
    model_scenario_table()

    # These need Azure for tool call data
    calls_per_turn_sds()
    kw_test_pmr()
    tool_error_summary()
    cost_estimation()

    # These need Convex for diary/trajectory data
    bhfdr_summary()
    reflection_quality_stats()
