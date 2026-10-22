"""Build analysis/civbench_analysis.ipynb from section definitions.

Run this script to (re)generate the notebook:
    .venv/bin/python analysis/build_notebook.py
"""

from __future__ import annotations
import json
from pathlib import Path

import nbformat

NB_PATH = Path(__file__).parent / "civbench_analysis.ipynb"


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

cells = []

# ── Setup ──────────────────────────────────────────────────────────────────
cells.append(md("# CivBench Analysis\n\nMulti-model LLM benchmark using Civilization VI. Three models (Claude Opus 4.6, GPT-5.4, Gemini Flash) across up to three scenarios."))

cells.append(code("""\
import sys, warnings
from pathlib import Path

# Add project root to path
ROOT = Path.cwd().parent if Path.cwd().name == "analysis" else Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Consistent aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "claude-opus-4-6": "#E07B39",
    "gpt-5.4": "#4A90D9",
    "gemini-3-flash-preview": "#5CB85C",
    "gemini-3.1-pro-preview": "#9B59B6",
}
FIGURES = ROOT / "analysis" / "figures"
FIGURES.mkdir(exist_ok=True)

print("Environment OK")
"""))

# ── S1: Load data ──────────────────────────────────────────────────────────
cells.append(md("## Section 1: Dataset Overview & Data Integrity"))

cells.append(code("""\
from analysis.data_loader import load_all
data = load_all(verbose=True)

games    = data["games"]
diary    = data["diary"]       # agent rows only
diary_all = data["diary_all"]  # all players
metrics  = data["metrics"]
founding = data["founding"]

# Short-game flag: < 50 turns may inflate normalised score
games["short_game"] = games["turns_played"] < 50
"""))

cells.append(code("""\
# Games per model × scenario
pivot = games.groupby(["model", "scenario"]).size().unstack(fill_value=0)
print("Games per model × scenario:")
print(pivot.to_string())
print()
print(f"Total games: {len(games)}")
print(f"Short games (< 50 turns): {games['short_game'].sum()} (excluded from headline metrics)")
"""))

cells.append(code("""\
# Outcome distribution
outcome_table = games.groupby(["model", "outcome"]).size().unstack(fill_value=0)
print("Outcome distribution:")
print(outcome_table.to_string())
"""))

# ── S2: Discriminative Power (ICC) ─────────────────────────────────────────
cells.append(md("""\
## Section 2: Discriminative Power (ICC)

For each metric, compute the **Intraclass Correlation Coefficient** (ICC):
ICC = between-model variance / (between-model + within-model variance).

- ICC > 0.5 → benchmark reliably discriminates
- ICC < 0.2 → metric is noise

This determines which metrics to feature in subsequent sections.

Also: cross-scenario consistency (how often Model A beats Model B across scenarios).
"""))

cells.append(code("""\
def compute_icc(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    \"\"\"One-way ICC for whether group_col explains value_col variance.\"\"\"
    groups = df.groupby(group_col)[value_col].apply(list)
    k = len(groups)
    if k < 2:
        return float("nan")
    grand_mean = df[value_col].mean()
    n_total = len(df)

    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    # Within-group sum of squares
    ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)

    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0:
        return float("nan")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Avg group size (harmonic)
    n_hat = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / df_between

    icc = (ms_between - ms_within) / (ms_between + (n_hat - 1) * ms_within)
    return float(np.clip(icc, -1, 1))


# Metrics computable from Convex trajectories
CANDIDATE_METRICS = [
    "normalised_score",
    "turns_played",
    "final_cities",
    "final_exploration_pct",
    "final_science",
    "final_gold",
    "final_military",
    "final_culture",
    "city_t50",
    "city_t100",
    "city_t150",
    "city_t200",
    "exploration_t50",
    "exploration_t100",
]

# Filter to ground_control games (most data) and exclude short games
gc = games[
    (games["scenario"] == "ground_control") & (~games["short_game"])
].copy()
gc_m = metrics[metrics["game_id"].isin(gc["game_id"])].copy()

icc_rows = []
for col in CANDIDATE_METRICS:
    if col not in gc_m.columns:
        continue
    sub = gc_m[["model", col]].dropna()
    if len(sub) < 4:
        continue
    icc_val = compute_icc(sub, "model", col)
    within_sd = sub.groupby("model")[col].std().mean()
    between_sd = sub.groupby("model")[col].mean().std()
    verdict = "discriminative" if icc_val > 0.5 else ("marginal" if icc_val > 0.2 else "noise")
    icc_rows.append({
        "metric": col,
        "ICC": round(icc_val, 3),
        "within_SD": round(within_sd, 3) if not np.isnan(within_sd) else None,
        "between_SD": round(between_sd, 3) if not np.isnan(between_sd) else None,
        "verdict": verdict,
    })

icc_df = pd.DataFrame(icc_rows).sort_values("ICC", ascending=False)
print("Discriminative Power (ICC) — ground_control, full games only:")
print(icc_df.to_string(index=False))

DISCRIMINATIVE = icc_df[icc_df["ICC"] > 0.2]["metric"].tolist()
print(f"\\nMetrics with ICC > 0.2 (will get full figure treatment): {DISCRIMINATIVE}")
"""))

cells.append(code("""\
# Kruskal-Wallis + pairwise Mann-Whitney U for top discriminative metrics
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

print("Statistical tests (ground_control, full games):")
print(f"{'Metric':<25} {'KW p-value':<15} {'Pairwise significant pairs'}")
print("-" * 80)
for col in DISCRIMINATIVE[:6]:
    groups = {m: grp[col].dropna().values for m, grp in gc_m.groupby("model") if col in grp}
    vals = [v for v in groups.values() if len(v) >= 2]
    if len(vals) < 2:
        continue
    try:
        _, kw_p = kruskal(*vals)
    except Exception:
        kw_p = float("nan")
    sig_pairs = []
    for (m1, v1), (m2, v2) in combinations(groups.items(), 2):
        if len(v1) < 2 or len(v2) < 2:
            continue
        try:
            _, p = mannwhitneyu(v1, v2, alternative="two-sided")
            if p < 0.05:
                sig_pairs.append(f"{m1[:8]} vs {m2[:8]} (p={p:.3f})")
        except Exception:
            pass
    print(f"{col:<25} {kw_p:<15.4f} {'; '.join(sig_pairs) or 'none'}")
print()
print("Note: study powered for large effects only (n ≈ 5–9 per model).")
"""))

cells.append(code("""\
# Cross-scenario consistency: Model A > Model B in X/N scenarios
# Only possible across scenarios where both models have data
all_scenarios = games["scenario"].unique()
# games is already filtered to ADMISSIBLE_IDS upstream (load_games_df default)
models = sorted(games["model"].unique())

print("Cross-scenario consistency (normalised_score, full games only):")
full_games = games[~games["short_game"]]
for (m1, m2) in combinations(models, 2):
    wins_m1, wins_m2, total = 0, 0, 0
    for sc in all_scenarios:
        g1 = full_games[(full_games["model"] == m1) & (full_games["scenario"] == sc)]["normalised_score"].dropna()
        g2 = full_games[(full_games["model"] == m2) & (full_games["scenario"] == sc)]["normalised_score"].dropna()
        if len(g1) == 0 or len(g2) == 0:
            continue
        total += 1
        if g1.mean() > g2.mean():
            wins_m1 += 1
        else:
            wins_m2 += 1
    if total > 0:
        print(f"  {m1[:20]:20s} vs {m2[:20]:20s}: {m1[:8]} wins {wins_m1}/{total} scenarios")
"""))

# ── S3: Game Outcomes ──────────────────────────────────────────────────────
cells.append(md("## Section 3: Game Outcomes"))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Outcome heatmap
outcome_counts = games.groupby(["model", "outcome"]).size().unstack(fill_value=0)
sns.heatmap(outcome_counts, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0], cbar=False)
axes[0].set_title("Game Outcomes by Model")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# Victory/defeat breakdown — which victory types ended games
ended = games[games["outcome"].isin(["victory", "defeat"])].copy()
if not ended.empty:
    vtype = ended.groupby(["model", "victory_type"]).size().unstack(fill_value=0)
    sns.heatmap(vtype, annot=True, fmt="d", cmap="Blues", ax=axes[1], cbar=False)
    axes[1].set_title("Victory/Defeat Type by Model")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
else:
    axes[1].text(0.5, 0.5, "No concluded games with outcome data", ha="center", va="center")
    axes[1].set_title("Victory/Defeat Type (no data)")

plt.tight_layout()
plt.savefig(FIGURES / "s3_outcomes.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s3_outcomes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: s3_outcomes.pdf/png")
"""))

cells.append(code("""\
# Normalised score distribution
full = games[~games["short_game"]].copy()
fig, ax = plt.subplots(figsize=(10, 5))
order = full.groupby("model")["normalised_score"].median().sort_values(ascending=False).index
sns.boxplot(data=full, x="model", y="normalised_score", order=order,
            palette=PALETTE, ax=ax, width=0.5)
sns.stripplot(data=full, x="model", y="normalised_score", order=order,
              palette=PALETTE, ax=ax, size=6, jitter=True, alpha=0.7)
ax.set_title("Normalised Score by Model (full games only, ground_control)")
ax.set_xlabel("")
ax.set_ylabel("Normalised Score (agent / winner)")
ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="0.5 reference")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "s3_normalised_score.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s3_normalised_score.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── S3b: Multi-Metric Normalised Performance ──────────────────────────────
cells.append(md("""\
## Section 3b: Multi-Metric Normalised Performance

For each game, compute `agent_value / max(all_players_value)` at the final recorded turn,
for score, science, culture, gold, military, cities, faith, tourism, and exploration.

This shows where the agent is competitive vs where it falls furthest behind —
independent of who won and without conflating overall score with individual yields.
"""))

cells.append(code("""\
NORM_METRICS = [
    ("score",           "Score"),
    ("science",         "Science"),
    ("culture",         "Culture"),
    ("gold",            "Gold"),
    ("military",        "Military"),
    ("cities",          "Cities"),
    ("faith",           "Faith"),
    ("tourism",         "Tourism"),
    ("exploration_pct", "Exploration"),
]

full_ids = games[~games["short_game"]]["game_id"].tolist()
da_full = diary_all[diary_all["game_id"].isin(full_ids)].copy()

norm_rows = []
for game_id, gdf in da_full.groupby("game_id"):
    final_turn = gdf["turn"].max()
    final = gdf[gdf["turn"] == final_turn]
    agent_row = final[final["is_agent"]]
    if agent_row.empty:
        continue
    agent = agent_row.iloc[0]
    meta = games[games["game_id"] == game_id].iloc[0]

    row = {
        "game_id": game_id,
        "run_id": agent["run_id"],
        "model": agent["model"],
        "scenario": agent["scenario"],
        "final_turn": int(final_turn),
    }
    for col, label in NORM_METRICS:
        agent_val = agent[col] if col in agent.index else None
        max_val = final[col].max() if col in final.columns else None
        if agent_val is None or max_val is None or max_val <= 0 or pd.isna(max_val):
            row[label] = float("nan")
        else:
            row[label] = min(1.0, float(agent_val) / float(max_val))
    norm_rows.append(row)

norm_df = pd.DataFrame(norm_rows)
metric_labels = [label for _, label in NORM_METRICS]
print(f"Computed normalised metrics for {len(norm_df)} games")
print()
print(norm_df.groupby("model")[metric_labels].mean().round(3).to_string())
"""))

cells.append(code("""\
# Heatmap: each game (row) × metric (col), grouped by model
gc_norm = norm_df[norm_df["scenario"] == "ground_control"].copy()
gc_norm = gc_norm.sort_values(["model", "run_id"])

fig, ax = plt.subplots(figsize=(13, max(4, len(gc_norm) * 0.55 + 1.5)))

heat_data = gc_norm[metric_labels].values.astype(float)
row_labels = [f"{r['model'].replace('claude-opus-4-6','Claude').replace('gpt-5.4','GPT')} — {r['run_id'][:18]}"
              for _, r in gc_norm.iterrows()]

im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(metric_labels)))
ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=10)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)

# Annotate cells
for i in range(len(gc_norm)):
    for j in range(len(metric_labels)):
        val = heat_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.25 < val < 0.85 else "white")

# Divider between models
models_in_order = gc_norm["model"].tolist()
for i in range(1, len(models_in_order)):
    if models_in_order[i] != models_in_order[i - 1]:
        ax.axhline(i - 0.5, color="white", lw=2)

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Agent / Best Player")
ax.set_title("Multi-Metric Normalised Performance — ground_control (per game)")
plt.tight_layout()
plt.savefig(FIGURES / "s3b_norm_heatmap.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s3b_norm_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# Grouped bar chart: model × metric (mean across games)
model_order = sorted(gc_norm["model"].unique())
x = np.arange(len(metric_labels))
width = 0.8 / len(model_order)

fig, ax = plt.subplots(figsize=(14, 5))
for i, model in enumerate(model_order):
    mdata = gc_norm[gc_norm["model"] == model]
    vals = [mdata[label].mean() for label in metric_labels]
    offset = (i - (len(model_order) - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=model,
                  color=PALETTE.get(model, "gray"), alpha=0.85)

ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, rotation=20, ha="right")
ax.set_ylabel("Agent / Best Player (mean across games)")
ax.set_ylim(0, 1.1)
ax.set_title("Multi-Metric Normalised Performance by Model — ground_control")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "s3b_norm_bars.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s3b_norm_bars.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── S4: Score Trajectories ─────────────────────────────────────────────────
cells.append(md("## Section 4: Score Trajectories\n\nPer-turn score, averaged per model with variance shading."))

cells.append(code("""\
def plot_metric_trajectories(diary_df, games_df, metric: str, ylabel: str,
                             title: str, scenario: str = "ground_control",
                             max_turn: int = 320, min_games: int = 2,
                             figsize=(12, 5)):
    \"\"\"Plot per-model trajectory with variance shading.

    Mean line and SD band are only drawn where at least min_games games
    contribute — prevents artificially narrow bands at late turns when
    most games have already ended.
    \"\"\"
    fig, ax = plt.subplots(figsize=figsize)

    full_ids = games_df[~games_df["short_game"]]["game_id"].tolist()
    sub = diary_df[
        (diary_df["scenario"] == scenario) &
        (diary_df["game_id"].isin(full_ids)) &
        (diary_df["turn"] <= max_turn)
    ].copy()

    for model in sub["model"].unique():
        if "AI_" in model:
            continue
        color = PALETTE.get(model, "gray")
        mdata = sub[sub["model"] == model].copy()

        # Per-game lines (faint)
        for gid, gdf in mdata.groupby("game_id"):
            ax.plot(gdf["turn"], gdf[metric], color=color, alpha=0.15, lw=0.8)

        # Mean + 1SD band — only where >= min_games contribute
        count_s = mdata.groupby("turn")[metric].count()
        mean_s = mdata.groupby("turn")[metric].mean()
        std_s = mdata.groupby("turn")[metric].std().fillna(0)
        pivot = pd.DataFrame({"mean": mean_s, "std": std_s, "n": count_s}).dropna(subset=["mean"])
        pivot = pivot[pivot["n"] >= min_games]
        ax.plot(pivot.index, pivot["mean"], color=color, lw=2, label=model.replace("-", "‑"))
        ax.fill_between(pivot.index,
                        pivot["mean"] - pivot["std"],
                        pivot["mean"] + pivot["std"],
                        color=color, alpha=0.15)

    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, max_turn)
    plt.tight_layout()
    return fig, ax

fig, ax = plot_metric_trajectories(
    diary, games, "score", "Score", "Score Trajectory — ground_control"
)
plt.savefig(FIGURES / "s4_score_trajectory.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s4_score_trajectory.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── S5: Yield Trajectories ─────────────────────────────────────────────────
cells.append(md("## Section 5: Yield Trajectories\n\nGold, science, culture, and faith over time — data no prior LLM benchmark produces."))

cells.append(code("""\
YIELDS = [
    ("science", "Science/turn"),
    ("gold", "Gold (cumulative)"),
    ("culture", "Culture/turn"),
    ("faith", "Faith (cumulative)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

full_ids = games[~games["short_game"]]["game_id"].tolist()
sub = diary[
    (diary["scenario"] == "ground_control") &
    (diary["game_id"].isin(full_ids)) &
    (diary["turn"] <= 320)
].copy()

MIN_GAMES = 2
for ax, (metric, ylabel) in zip(axes, YIELDS):
    for model in sub["model"].unique():
        if "AI_" in model:
            continue
        color = PALETTE.get(model, "gray")
        mdata = sub[sub["model"] == model]
        count_s = mdata.groupby("turn")[metric].count()
        mean_s = mdata.groupby("turn")[metric].mean()
        std_s = mdata.groupby("turn")[metric].std().fillna(0)
        pivot = pd.DataFrame({"mean": mean_s, "std": std_s, "n": count_s}).dropna(subset=["mean"])
        pivot = pivot[pivot["n"] >= MIN_GAMES]
        ax.plot(pivot.index, pivot["mean"], color=color, lw=2,
                label=model.replace("-", "‑"))
        ax.fill_between(pivot.index,
                        (pivot["mean"] - pivot["std"]).clip(lower=0),
                        pivot["mean"] + pivot["std"],
                        color=color, alpha=0.12)
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8)

plt.suptitle("Yield Trajectories — ground_control (full games)", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES / "s5_yield_trajectories.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s5_yield_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── S6: Expansion Timing ───────────────────────────────────────────────────
cells.append(md("## Section 6: Expansion Timing\n\nCity count at milestone turns and turn of founding for each city."))

cells.append(code("""\
# City count at milestone turns
milestones = ["city_t50", "city_t100", "city_t150", "city_t200"]
milestone_turns = [50, 100, 150, 200]
labels = ["T50", "T100", "T150", "T200"]

full_gc = metrics[
    (metrics["scenario"] == "ground_control") &
    (metrics["game_id"].isin(games[~games["short_game"]]["game_id"]))
].copy()

# ── Baseline: human data takes priority; fall back to AI opponent average ──
# To add human data: populate HUMAN_BASELINE with city counts at each
# milestone turn from ground_control human play, keyed by turn.
# Example: HUMAN_BASELINE = {50: 3.1, 100: 4.8, 150: 6.2, 200: 7.5}
HUMAN_BASELINE: dict[int, float] | None = None  # set when human data available

if HUMAN_BASELINE is not None:
    baseline_vals = [HUMAN_BASELINE[t] for t in milestone_turns]
    baseline_label = "Human baseline"
    baseline_color = "green"
else:
    # Compute AI opponent average from diary_all (non-agent players, ground_control)
    full_ids = games[~games["short_game"]]["game_id"].tolist()
    ai_rows = diary_all[
        (~diary_all["is_agent"]) &
        (diary_all["scenario"] == "ground_control") &
        (diary_all["game_id"].isin(full_ids))
    ].copy()
    baseline_vals = []
    for t in milestone_turns:
        # For each game, take the first turn >= t for each AI player, then average
        at_t = (
            ai_rows[ai_rows["turn"] >= t]
            .sort_values("turn")
            .groupby(["game_id", "player_id"])
            .first()["cities"]
            .dropna()
        )
        baseline_vals.append(float(at_t.mean()) if len(at_t) else float("nan"))
    baseline_label = "AI opponent avg"
    baseline_color = "red"

x = np.arange(len(milestones))
width = 0.25
# full_gc is already admissible upstream — no need for Gemini exclusion
model_list = sorted(full_gc["model"].unique())

fig, ax = plt.subplots(figsize=(11, 5))
for i, model in enumerate(sorted(model_list)):
    vals = [full_gc[full_gc["model"] == model][col].mean() for col in milestones]
    bars = ax.bar(x + i * width, vals, width, label=model,
                  color=PALETTE.get(model, "gray"), alpha=0.85)

# Baseline lines
for j, bval in enumerate(baseline_vals):
    if not np.isnan(bval):
        ax.hlines(bval, x[j] - 0.1, x[j] + width * len(model_list) + 0.1,
                  colors=baseline_color, linestyles="dashed", lw=1.5, alpha=0.7,
                  label=baseline_label if j == 0 else "")

ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("City Count")
ax.set_title("City Count at Milestone Turns — ground_control")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "s6_city_milestones.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s6_city_milestones.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Baseline used: {baseline_label}")
print(f"Baseline values: { {l: round(v, 2) for l, v in zip(labels, baseline_vals)} }")
"""))

cells.append(code("""\
# Turn of founding per city (averaged per model)
full_ids = games[~games["short_game"]]["game_id"].tolist()
f_sub = founding[founding["game_id"].isin(full_ids) & (founding["scenario"] == "ground_control")].copy()

founding_pivot = (
    f_sub[f_sub["city_number"] <= 6]
    .groupby(["model", "city_number"])["turn_founded"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 5))
for model in sorted(founding_pivot["model"].unique()):
    if "gemini-3.1" in model:
        continue
    mdata = founding_pivot[founding_pivot["model"] == model]
    color = PALETTE.get(model, "gray")
    ax.plot(mdata["city_number"], mdata["mean"], marker="o", color=color,
            label=model, lw=2)
    ax.fill_between(mdata["city_number"],
                    mdata["mean"] - mdata["std"].fillna(0),
                    mdata["mean"] + mdata["std"].fillna(0),
                    color=color, alpha=0.15)

# Benchmark lines
for city_n, turn_b in [(2, 40), (3, 75), (4, 100)]:
    ax.axhline(turn_b, color="red", ls="--", alpha=0.4, lw=0.8)
    ax.text(city_n + 0.05, turn_b + 3, f"T{turn_b} bench", color="red", fontsize=7)

ax.set_xlabel("City Number")
ax.set_ylabel("Turn Founded (mean ± 1SD)")
ax.set_title("Expansion Timing — ground_control")
ax.legend(fontsize=9)
ax.set_xticks(range(1, 7))
plt.tight_layout()
plt.savefig(FIGURES / "s6_expansion_timing.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s6_expansion_timing.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── S7: PMR (placeholder if no Azure) ────────────────────────────────────
cells.append(md("""\
## Section 7: Proactive Monitoring Rate (PMR)

> **Note:** PMR analysis requires Azure blob log.jsonl files.
> Configure `AZURE_STORAGE_CONNECTION_STRING` in `evals/.env` to enable.
> The cells below will run if Azure is available; otherwise they output a placeholder.
"""))

cells.append(code("""\
# PMR analysis — skip if Azure unavailable
from scripts.analyze import _load_env
env = _load_env()
AZURE_AVAILABLE = bool(
    env.get("AZURE_STORAGE_CONNECTION_STRING") or env.get("AZURE_STORAGE_ACCOUNT_NAME")
)
print(f"Azure available: {AZURE_AVAILABLE}")
if not AZURE_AVAILABLE:
    print("Skipping PMR analysis — configure evals/.env to enable")
"""))

cells.append(code("""\
if AZURE_AVAILABLE:
    from scripts.analyze import cloud_log
    from analysis.tool_taxonomy import categorise, pmr_denominator, TOOL_PMR_SUBCATEGORY

    tool_rows = []
    for _, g in games.iterrows():
        entries = cloud_log(g["run_id"])
        for e in entries:
            if e.get("type") != "tool_call":
                continue
            tool = e.get("tool", "unknown")
            cat, pmr_sub = categorise(tool)
            tool_rows.append({
                "game_id": g["game_id"],
                "model": g["model"],
                "scenario": g["scenario"],
                "turn": e.get("turn", 0),
                "tool": tool,
                "category": cat,
                "pmr_subcategory": pmr_sub,
                "is_pmr_denominator": pmr_denominator(tool),
                "success": e.get("success", True),
                "duration_ms": e.get("duration_ms"),
            })

    tools_df = pd.DataFrame(tool_rows)
    print(f"Loaded {len(tools_df)} tool calls from {tools_df['game_id'].nunique()} games")
else:
    tools_df = pd.DataFrame()
"""))

cells.append(code("""\
if AZURE_AVAILABLE and not tools_df.empty:
    # PMR subcategory bar chart
    full_ids = games[~games["short_game"]]["game_id"].tolist()
    t_full = tools_df[tools_df["game_id"].isin(full_ids)]

    pmr_rows = []
    for gid, gdf in t_full.groupby("game_id"):
        model = gdf["model"].iloc[0]
        denom = gdf["is_pmr_denominator"].sum()
        if denom == 0:
            continue
        for sub in ["victory_monitoring", "strategic_map", "resource_monitoring"]:
            count = (gdf["pmr_subcategory"] == sub).sum()
            pmr_rows.append({"game_id": gid, "model": model, "subcategory": sub,
                             "rate": count / denom})

    pmr_df = pd.DataFrame(pmr_rows)
    pmr_mean = pmr_df.groupby(["model", "subcategory"])["rate"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    sub_order = ["victory_monitoring", "strategic_map", "resource_monitoring"]
    sns.barplot(data=pmr_mean, x="subcategory", y="rate", hue="model",
                order=sub_order, palette=PALETTE, ax=ax)
    ax.set_title("Proactive Monitoring Rate by Subcategory")
    ax.set_xlabel("")
    ax.set_ylabel("Calls / Total Calls")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "s7_pmr_subcategories.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / "s7_pmr_subcategories.png", dpi=150, bbox_inches="tight")
    plt.show()
"""))

cells.append(code("""\
if AZURE_AVAILABLE and not tools_df.empty:
    # Stacked area: tool category composition over turns (10-turn rolling)
    full_ids = games[~games["short_game"]]["game_id"].tolist()
    t_full = tools_df[
        (tools_df["game_id"].isin(full_ids)) &
        (tools_df["scenario"] == "ground_control") &
        (tools_df["turn"] > 0) & (tools_df["turn"] <= 320)
    ].copy()

    CAT_ORDER = ["unit_action", "state_query", "strategic_monitoring", "diplomacy", "other"]
    CAT_COLORS = {
        "unit_action": "#E74C3C",
        "state_query": "#3498DB",
        "strategic_monitoring": "#2ECC71",
        "diplomacy": "#9B59B6",
        "other": "#95A5A6",
    }

    model_list_plot = [m for m in t_full["model"].unique() if "gemini-3.1" not in m]
    fig, axes = plt.subplots(1, len(model_list_plot), figsize=(5 * len(model_list_plot), 5), sharey=True)
    if len(model_list_plot) == 1:
        axes = [axes]

    for ax, model in zip(axes, sorted(model_list_plot)):
        mdata = t_full[t_full["model"] == model].copy()
        # Count calls per category per turn, average across games
        cat_counts = (
            mdata.groupby(["turn", "category"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=CAT_ORDER, fill_value=0)
        )
        # Divide by number of games for that turn to get avg per game
        n_games = mdata.groupby("turn")["game_id"].nunique()
        cat_avg = cat_counts.div(n_games, axis=0)
        # Rolling 10-turn window
        cat_smooth = cat_avg.rolling(10, min_periods=1).mean()
        ax.stackplot(cat_smooth.index, [cat_smooth[c] for c in CAT_ORDER],
                     labels=CAT_ORDER, colors=[CAT_COLORS[c] for c in CAT_ORDER], alpha=0.85)
        ax.set_title(model.replace("claude-opus-4-6", "Claude").replace("gpt-5.4", "GPT-5.4")
                         .replace("gemini-3-flash-preview", "Gemini Flash"))
        ax.set_xlabel("Turn")
        if ax == axes[0]:
            ax.set_ylabel("Avg calls / turn")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    plt.suptitle("Tool Category Composition Over Time (10-turn rolling avg)")
    plt.tight_layout()
    plt.savefig(FIGURES / "s7_tool_stacked_area.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / "s7_tool_stacked_area.png", dpi=150, bbox_inches="tight")
    plt.show()
"""))

# ── S8: RAG placeholder ────────────────────────────────────────────────────
cells.append(md("""\
## Section 8: Reflection-Action Gap (RAG)

Requires:
1. Diary planning field audit (run `load_reflections()` and inspect output)
2. Human validation of Haiku extraction (≥ 0.7 agreement gate)
3. Azure log.jsonl for commitment matching

See `experiments.md` E04–E06 for status.
"""))

cells.append(code("""\
# Quick diary planning field sample (10 entries per model)
# Uncomment to run — fetches from Convex (slow first time, then cached)

# from analysis.data_loader import load_reflections
# sample_refs = load_reflections(every_n_turns=20)
# import random
# for model in set(r["model"] for r in sample_refs):
#     model_refs = [r for r in sample_refs if r["model"] == model]
#     sample = random.sample(model_refs, min(3, len(model_refs)))
#     print(f"\\n=== {model} ===")
#     for r in sample:
#         print(f"  T{r['turn']}: {r['planning'][:200]}")
"""))

# ── S9: Summary ────────────────────────────────────────────────────────────
cells.append(md("## Section 9: Multi-Model Summary"))

cells.append(code("""\
# Radar chart: 6 axes from discriminative metrics
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Build summary scores per model (z-scored within ground_control, full games)
gc_full = metrics[
    (metrics["scenario"] == "ground_control") &
    (metrics["game_id"].isin(games[~games["short_game"]]["game_id"]))
].copy()

RADAR_AXES = {
    "normalised_score": "Score",
    "city_t100": "Expansion",
    "exploration_t100": "Exploration",
    "final_culture": "Culture",
    "final_science": "Science",
    "final_military": "Military",
}

# Z-score each axis
radar_data = {}
for model in sorted(gc_full["model"].unique()):
    if "gemini-3.1" in model:
        continue
    radar_data[model] = {}
    for col, label in RADAR_AXES.items():
        if col not in gc_full.columns:
            radar_data[model][label] = 0
            continue
        global_mean = gc_full[col].mean()
        global_std = gc_full[col].std()
        model_mean = gc_full[gc_full["model"] == model][col].mean()
        z = (model_mean - global_mean) / global_std if global_std > 0 else 0
        radar_data[model][label] = float(np.clip(z, -2, 2))  # clip for readability

# Plot radar
labels = list(RADAR_AXES.values())
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for model, vals in radar_data.items():
    color = PALETTE.get(model, "gray")
    v = [vals.get(l, 0) for l in labels]
    v += v[:1]
    ax.plot(angles, v, color=color, lw=2, label=model)
    ax.fill(angles, v, color=color, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=11)
ax.set_ylim(-2.5, 2.5)
ax.set_title("Multi-Model Performance Radar\\n(z-scored, ground_control)", pad=15)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES / "s9_radar.pdf", bbox_inches="tight")
plt.savefig(FIGURES / "s9_radar.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""\
# z-scored metric table
print("Per-model z-scored metrics (ground_control, full games):")
table_rows = []
for model in sorted(gc_full["model"].unique()):
    if "gemini-3.1" in model:
        continue
    row = {"model": model}
    for col, label in RADAR_AXES.items():
        if col not in gc_full.columns:
            row[label] = float("nan")
            continue
        global_mean = gc_full[col].mean()
        global_std = gc_full[col].std()
        model_mean = gc_full[gc_full["model"] == model][col].mean()
        z = (model_mean - global_mean) / global_std if global_std > 0 else 0
        row[label] = round(z, 2)
    table_rows.append(row)

summary_table = pd.DataFrame(table_rows).set_index("model")
print(summary_table.to_string())
"""))

# ── Footer ─────────────────────────────────────────────────────────────────
cells.append(md("""\
## Next Steps

1. Configure Azure credentials in `evals/.env` to unblock PMR, tool stacked area, and RAG
2. Run `load_reflections()` for diary planning field audit (E04)
3. Complete RAG human validation (E05) — target ≥ 0.7 agreement
4. Run `analysis/plots.py` to generate publication-quality figures
5. Draft `docs/paper/civbench_paper.md`
"""))

cells.append(code("""\
# Summary of figures generated
import os
figs = sorted(FIGURES.glob("*.png"))
print(f"Figures in {FIGURES}:")
for f in figs:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45s} {size_kb:.0f} KB")
"""))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (civ6-mcp)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

nbformat.write(nb, NB_PATH)
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
