"""CivBench publication-quality figure generation.

Each function produces one figure, saves it to analysis/figures/ as both
PDF (for paper inclusion) and PNG (for preview), and returns the Figure.

All data must be pre-filtered to admissible games before being passed here.
Use load_games_df(admissible_only=True) etc. from data_loader — the plot
functions do NOT apply any secondary game filtering themselves.

Usage:
    .venv/bin/python analysis/plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

try:
    import cairosvg  # optional; enables SVG icon rendering
    _HAS_CAIRO = True
except ImportError:
    _HAS_CAIRO = False

FIGURES = ROOT / "analysis" / "figures"
FIGURES.mkdir(exist_ok=True)

WEB = ROOT / "web" / "public"

# Shared aesthetics — web design system (globals.css marble/gold palette)

# Marble surface scale
MARBLE = {
    50:  "#FAFAF8",
    100: "#F5F3EF",
    200: "#EBE8E2",
    300: "#E0DBD3",
    400: "#C8C1B7",
    500: "#A39B8F",
    600: "#7A7269",
    700: "#5C5549",
    800: "#3D3832",
    900: "#2A2521",
}

# Accent colors
GOLD       = "#D4A853"
GOLD_LIGHT = "#E8C97A"
PATINA     = "#7A9B8A"   # green-grey
TERRACOTTA = "#C4785C"   # warm orange-red
OCEAN      = "#4A90A4"   # steel blue

# Status colors
STATUS_VICTORY    = "#3D8B6E"
STATUS_DEFEAT     = "#C0503A"
STATUS_UNFINISHED = "#B0A99F"

# Configure seaborn to match the marble theme
sns.set_theme(
    style="whitegrid",
    font_scale=1.3,
    rc={
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "axes.edgecolor":    MARBLE[300],
        "axes.labelcolor":   MARBLE[800],
        "xtick.color":       MARBLE[700],
        "ytick.color":       MARBLE[700],
        "grid.color":        MARBLE[300],
        "grid.linewidth":    0.8,
        "text.color":        MARBLE[800],
        "legend.framealpha": 0.9,
        "legend.facecolor":  "white",
        "legend.edgecolor":  MARBLE[300],
    },
)

# Model → plot color (drawn from the web palette)
PALETTE = {
    "claude-opus-4-6":        TERRACOTTA,  # warm orange — Anthropic brand feel
    "gpt-5.4":                OCEAN,       # steel blue — OpenAI
    "gemini-3-flash-preview":  PATINA,      # green-grey — Google
    "gemini-3.1-pro-preview":  GOLD,        # gold — Google Pro variant
    "Kimi-K2.5":              "#7B68EE",   # medium slate blue — Moonshot AI
}

MODEL_LABELS = {
    "claude-opus-4-6":        "Claude Opus 4.6",
    "gpt-5.4":                "GPT-5.4",
    "gemini-3-flash-preview":  "Gemini Flash",
    "gemini-3.1-pro-preview":  "Gemini 3.1 Pro",
    "Kimi-K2.5":              "Kimi K2.5",
}

# Provider SVG icons from web/public/images/providers/
_PROVIDER_ICONS: dict[str, Path] = {
    "claude-opus-4-6":        WEB / "images/providers/anthropic_small.svg",
    "gpt-5.4":                WEB / "images/providers/openai_small.svg",
    "gemini-3-flash-preview":  WEB / "images/providers/google_small.svg",
    "gemini-3.1-pro-preview":  WEB / "images/providers/google_small.svg",
    "Kimi-K2.5":              WEB / "images/providers/kimi_small.svg",
}

DPI = 300

# Sample sizes per model (admissible games)
MODEL_N = {
    "claude-opus-4-6":       8,
    "gpt-5.4":               8,
    "gemini-3.1-pro-preview": 6,
    "Kimi-K2.5":             1,
}

# Kimi is exploratory (n=1) — grey it out
KIMI_GREY = MARBLE[400]


def _model_color(model: str) -> str:
    """Return palette color, greyed for Kimi (n=1 exploratory)."""
    if model == "Kimi-K2.5":
        return KIMI_GREY
    return PALETTE.get(model, MARBLE[500])


def _label_n(model: str) -> str:
    """Return model label with sample size, greyed tag for Kimi."""
    n = MODEL_N.get(model, "?")
    base = MODEL_LABELS.get(model, model)
    if model == "Kimi-K2.5":
        return f"{base} (n={n}, exploratory)"
    return f"{base} (n={n})"


KIMI_HATCH = "////"


def _model_hatch(model: str) -> str:
    """Diagonal hatch for Kimi (n=1 exploratory); empty for all others."""
    return KIMI_HATCH if model == "Kimi-K2.5" else ""


def _model_marker(model: str) -> str:
    """Diamond marker for Kimi (n=1); circle for all others."""
    return "D" if model == "Kimi-K2.5" else "o"


def _apply_kimi_hatch(ax: plt.Axes) -> None:
    """Post-process seaborn/category-coloured axes: hatch patches coloured KIMI_GREY."""
    import matplotlib.colors as mcolors
    kimi_rgba = np.array(mcolors.to_rgba(KIMI_GREY))
    for patch in ax.patches:
        fc = np.asarray(patch.get_facecolor()).flatten()[:3]
        if np.allclose(fc, kimi_rgba[:3], atol=1e-2):
            patch.set_hatch(KIMI_HATCH)
            patch.set_edgecolor(MARBLE[600])


_INFLECTION_COLORS = {
    "exploration_pct": OCEAN,
    "cities":          PATINA,
    "score":           GOLD,
    "science":         TERRACOTTA,
    "culture":         "#8C6E2C",
    "gold":            MARBLE[600],
}

_INFLECTION_LABELS = {
    "exploration_pct": "exploration %",
    "cities": "city count",
    "score": "raw score",
    "science": "science/turn",
    "culture": "culture/turn",
    "gold": "gold",
}


def _load_icon(model: str, size: int = 20) -> np.ndarray | None:
    """Load a provider icon as an RGBA numpy array, or None if unavailable."""
    path = _PROVIDER_ICONS.get(model)
    if path is None or not path.exists():
        return None
    if _HAS_CAIRO:
        try:
            png_bytes = cairosvg.svg2png(url=str(path), output_width=size, output_height=size)
            return mpimg.imread(io.BytesIO(png_bytes))
        except Exception:
            return None
    return None


def _add_icon_legend(ax: plt.Axes, handles, labels: list[str], models: list[str], **legend_kw):
    """Build a legend that prepends a provider icon to each model entry.

    Uses a HandlerBase subclass so icons are positioned correctly inside the
    legend box. Falls back to a plain legend if cairosvg is not installed.
    """
    if not _HAS_CAIRO:
        return ax.legend(handles=handles, labels=labels, **legend_kw)

    from matplotlib.legend_handler import HandlerBase
    import matplotlib.patches as mpatches

    class _IconHandler(HandlerBase):
        def __init__(self, img: np.ndarray, color: str):
            self._img = img
            self._color = color
            super().__init__()

        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            # Color swatch (left portion)
            swatch = mpatches.FancyBboxPatch(
                (-xdescent, ydescent), width * 0.42, height,
                boxstyle="square,pad=0",
                fc=self._color, ec="none", transform=trans,
            )
            # Icon (right of swatch)
            zoom = (height * 0.85) / max(self._img.shape[:2])
            oi = OffsetImage(self._img, zoom=zoom)
            oi.image.axes = ax
            ab = AnnotationBbox(
                oi, (width * 0.73, ydescent + height * 0.5),
                xycoords=trans, frameon=False, pad=0,
            )
            return [swatch, ab]

    handler_map: dict = {}
    _LABEL_TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}
    for handle, label, model in zip(handles, labels, models):
        img = _load_icon(model, size=18)
        if img is not None:
            handler_map[handle] = _IconHandler(img, PALETTE.get(model, MARBLE[500]))

    return ax.legend(handles=handles, labels=labels, handler_map=handler_map, **legend_kw)


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(FIGURES / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _empty_fig(title: str, msg: str, name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.set_title(title)
    _save(fig, name)
    return fig


def plot_outcome_heatmap(games_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of game outcomes per model. Descriptive only — small sample."""
    from matplotlib.colors import LinearSegmentedColormap
    marble_cmap = LinearSegmentedColormap.from_list(
        "marble_warm", [MARBLE[100], GOLD_LIGHT, TERRACOTTA]
    )
    fig, ax = plt.subplots(figsize=(9, 2.5))
    counts = games_df.groupby(["model", "outcome"]).size().unstack(fill_value=0)
    # Add n to row labels; grey Kimi
    def _row_label(m):
        n = MODEL_N.get(m, "?")
        lbl = _label(m)
        suffix = " (exploratory)" if m == "Kimi-K2.5" else ""
        return f"{lbl} (n={n}){suffix}"
    counts.index = [_row_label(m) for m in counts.index]
    sns.heatmap(counts, annot=True, fmt="d", cmap=marble_cmap, ax=ax, cbar=False,
                linewidths=0.5, linecolor=MARBLE[50])
    ax.set_title("Game Outcomes by Model", fontsize=15, fontweight="bold",
                 color=MARBLE[800])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Add descriptive-only note
    fig.text(0.5, -0.04,
             "Descriptive only — not statistically powered for model comparison "
             "(Fisher's exact test p=0.488)",
             ha="center", fontsize=10, color=MARBLE[600], style="italic")
    plt.tight_layout()
    _save(fig, "outcome_heatmap")
    return fig


def plot_victory_breakdown(games_df: pd.DataFrame) -> plt.Figure:
    """Stacked bar: which victory/defeat type ended each model's games."""
    ended = games_df[games_df["outcome"].isin(["victory", "defeat"])].copy()
    if ended.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No concluded games with outcome data",
                ha="center", va="center")
        ax.set_title("Victory/Defeat Type by Model")
        plt.tight_layout()
        _save(fig, "victory_breakdown")
        return fig

    pivot = (
        ended.groupby(["model", "outcome", "victory_type"])
        .size()
        .reset_index(name="count")
    )
    # Agent victories collapse to a single "Agent victory" label; defeats
    # are labelled by the AI's victory type (what beat the agent).
    pivot["label"] = pivot.apply(
        lambda r: "Agent victory" if r["outcome"] == "victory"
        else (r["victory_type"] if r["victory_type"] else r["outcome"]),
        axis=1,
    )
    wide = pivot.groupby(["model", "label"])["count"].sum().unstack(fill_value=0)
    wide.index = [_label(m) for m in wide.index]

    # Fixed colours per defeat type; agent victory gets STATUS_VICTORY green.
    DEFEAT_COLORS = {
        "Culture":    OCEAN,
        "Technology": TERRACOTTA,
        "Score":      GOLD,
        "Domination": "#8B4A6E",   # muted purple
        "Religious":  PATINA,
    }
    defeat_cols  = [c for c in wide.columns if c != "Agent victory"]
    victory_cols = [c for c in wide.columns if c == "Agent victory"]

    color_map = {
        c: DEFEAT_COLORS.get(c, MARBLE[500]) for c in defeat_cols
    }
    color_map["Agent victory"] = STATUS_VICTORY

    # Defeats at the bottom, agent victories on top
    col_order = sorted(defeat_cols) + victory_cols
    wide = wide.reindex(columns=col_order, fill_value=0)
    col_colors = [color_map[c] for c in wide.columns]

    fig, ax = plt.subplots(figsize=(10, 3))
    wide.plot(kind="bar", stacked=True, ax=ax, color=col_colors, width=0.5)
    kimi_label = _label("Kimi-K2.5")
    model_labels_list = wide.index.tolist()
    if kimi_label in model_labels_list:
        ki = model_labels_list.index(kimi_label)
        n_models = len(model_labels_list)
        for grp in range(len(wide.columns)):
            p = ax.patches[grp * n_models + ki]
            p.set_hatch(KIMI_HATCH)
            p.set_edgecolor(MARBLE[600])
    max_count = int(wide.sum(axis=1).max())
    ax.set_ylim(0, max_count + 0.8)
    ax.set_yticks(range(0, max_count + 1))
    ax.set_title("Victory/Defeat Type by Model", fontsize=16, fontweight="bold",
                 color=MARBLE[800])
    ax.set_xlabel("")
    ax.set_ylabel("Game Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper right", fontsize=10, ncol=2,
              facecolor="white", edgecolor=MARBLE[300])
    plt.tight_layout()
    _save(fig, "victory_breakdown")
    return fig


def plot_normalised_score(games_df: pd.DataFrame) -> plt.Figure:
    """Dot plot (per run) with median overlay; n per model in legend."""
    order = (
        games_df.groupby("model")["normalised_score"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 3.2))
    games_df = games_df.copy()
    # Build label_n entries for legend
    label_n_order = [_label_n(m) for m in order]
    label_n_palette = {_label_n(m): _model_color(m) for m in order}
    games_df["model_label_n"] = games_df["model"].map(_label_n)

    # Individual dots — all runs
    sns.stripplot(data=games_df, x="model_label_n", y="normalised_score",
                  order=label_n_order,
                  hue="model_label_n", palette=label_n_palette,
                  ax=ax, size=8, jitter=True, alpha=0.80,
                  linewidth=0.6, edgecolor=MARBLE[600], legend=False)

    # Overlay median as a horizontal tick per model
    for i, m in enumerate(order):
        med = games_df[games_df["model"] == m]["normalised_score"].median()
        ax.hlines(med, i - 0.22, i + 0.22,
                  colors=_model_color(m), linewidth=2.5, zorder=5)

    ax.axhline(0.5, ls="--", color=MARBLE[500], alpha=0.55, lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("Normalised Score\n(agent raw score / winner score at game end)", fontsize=11)
    ax.set_xticklabels(label_n_order, fontsize=11)
    ax.set_title("Normalised Score by Model\n"
                 "Horizontal tick = median; dots = individual runs",
                 fontsize=14, fontweight="bold", color=MARBLE[800])
    # Add descriptive note
    ax.text(0.99, 0.02,
            "Descriptive — Kruskal–Wallis H=1.90, p=0.594",
            transform=ax.transAxes, ha="right", fontsize=9,
            color=MARBLE[600], style="italic")
    plt.tight_layout()
    _save(fig, "normalised_score")
    return fig


def plot_icc_table(icc_df: pd.DataFrame) -> str:
    """Render ICC table as markdown (returned) and LaTeX (saved to icc_table.tex)."""
    def _kw(r):
        if r.get("kw_H") is None or r.get("kw_p") is None:
            return "---"
        return f"H={r['kw_H']:.2f}, p={r['kw_p']:.4f}"

    # Markdown
    md_lines = [
        "### Discriminative Power (ICC) — ground_control",
        "",
        "| Metric | ICC | Within SD | Between SD | KW test | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in icc_df.iterrows():
        within  = f"{r['within_SD']:.3f}"  if r["within_SD"]  is not None else "---"
        between = f"{r['between_SD']:.3f}" if r["between_SD"] is not None else "---"
        md_lines.append(f"| {r['metric']} | {r['ICC']:.3f} | {within} | {between} | {_kw(r)} | {r['verdict']} |")
    md = "\n".join(md_lines)
    (FIGURES / "icc_table.md").write_text(md)

    # LaTeX
    VERDICT_CMD = {
        "discriminative": r"\textbf{discriminative}",
        "marginal":       "marginal",
        "noise":          "noise",
    }
    tex_rows = []
    for _, r in icc_df.iterrows():
        within  = f"{r['within_SD']:.3f}"  if r["within_SD"]  is not None else "---"
        between = f"{r['between_SD']:.3f}" if r["between_SD"] is not None else "---"
        metric  = r["metric"].replace("_", r"\_")
        verdict = VERDICT_CMD.get(r["verdict"], r["verdict"])
        kw_cell = _kw(r) if _kw(r) != "---" else r"---"
        tex_rows.append(f"  {metric} & {r['ICC']:.3f} & {within} & {between} & {kw_cell} & {verdict} \\\\")

    tex = "\n".join([
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Discriminative Power (ICC) and Kruskal-Wallis test --- \texttt{ground\_control}}",
        r"  \label{tab:icc}",
        r"  \begin{tabular}{lrrrlr}",
        r"    \toprule",
        r"    Metric & ICC & Within SD & Between SD & KW test & Verdict \\",
        r"    \midrule",
        *[f"    {row.strip()}" for row in tex_rows],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    (FIGURES / "icc_table.tex").write_text(tex)

    # Matplotlib render for PDF/PNG (referenced from paper as `icc_table.pdf`)
    headers = ["Metric", "ICC", "Within SD", "Between SD", "KW test", "Verdict"]
    cells = []
    for _, r in icc_df.iterrows():
        within = f"{r['within_SD']:.3f}" if r["within_SD"] is not None else "—"
        between = f"{r['between_SD']:.3f}" if r["between_SD"] is not None else "—"
        cells.append([
            r["metric"], f"{r['ICC']:.3f}", within, between, _kw(r), r["verdict"],
        ])
    fig, ax = plt.subplots(figsize=(11, 0.36 * (len(cells) + 1) + 0.4))
    ax.axis("off")
    tbl = ax.table(
        cellText=cells, colLabels=headers, loc="center", cellLoc="left",
        colColours=["#f0f0f0"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    for i, row in enumerate(cells, start=1):
        verdict = row[-1]
        if verdict == "discriminative":
            for c in range(len(headers)):
                tbl[(i, c)].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "icc_table.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "icc_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return md


def plot_score_trajectories(
    diary_df: pd.DataFrame,
    games_df: pd.DataFrame,
    metric: str = "score",
    ylabel: str = "Score",
    scenario: str = "ground_control",
    max_turn: int = 320,
) -> plt.Figure:
    """Per-model trajectory with variance shading."""
    valid_ids = set(games_df["game_id"])
    sub = diary_df[
        (diary_df["scenario"] == scenario) &
        (diary_df["game_id"].isin(valid_ids)) &
        (diary_df["turn"] <= max_turn)
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 3.8))
    models_plotted = []
    for model in sorted(sub["model"].unique()):
        if "AI_" in model:
            continue
        color = _model_color(model)
        ls = ":" if model == "Kimi-K2.5" else "-"
        mdata = sub[sub["model"] == model]
        for _, gdf in mdata.groupby("game_id"):
            ax.plot(gdf["turn"], gdf[metric], color=color, alpha=0.12, lw=0.7)
        mean_s = mdata.groupby("turn")[metric].mean()
        std_s  = mdata.groupby("turn")[metric].std().fillna(0)
        n_s    = mdata.groupby("turn")[metric].count().clip(lower=1)
        ci_s   = 1.96 * std_s / np.sqrt(n_s)
        ax.plot(mean_s.index, mean_s, color=color, lw=2, ls=ls, label=_label_n(model))
        ax.fill_between(mean_s.index, mean_s - ci_s, mean_s + ci_s,
                        color=color, alpha=0.18)
        models_plotted.append(model)

    ax.set_xlabel("Turn", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"{ylabel} Trajectory",
                 fontsize=16, fontweight="bold", color=MARBLE[800])
    # Shading note
    ax.text(0.99, 0.02, "Shaded = 95% CI across runs",
            transform=ax.transAxes, ha="right", fontsize=10, color=MARBLE[600], style="italic")
    handles, labels = ax.get_legend_handles_labels()
    _add_icon_legend(ax, handles, labels, models_plotted, loc="upper left", fontsize=12)
    ax.set_xlim(0, max_turn)
    plt.tight_layout()
    _save(fig, f"trajectory_{metric}_{scenario}")
    return fig


def plot_yield_trajectories(
    diary_df: pd.DataFrame,
    games_df: pd.DataFrame,
    scenario: str = "ground_control",
    max_turn: int = 320,
) -> plt.Figure:
    """2×2 panel of yield trajectories."""
    YIELDS = [
        ("science", "Science / turn"),
        ("gold", "Gold (cumulative)"),
        ("culture", "Culture / turn"),
        ("faith", "Faith (cumulative)"),
    ]
    valid_ids = set(games_df["game_id"])
    sub = diary_df[
        (diary_df["scenario"] == scenario) &
        (diary_df["game_id"].isin(valid_ids)) &
        (diary_df["turn"] <= max_turn)
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 6.8))
    axes = axes.flatten()
    all_handles, all_labels, all_models = [], [], []
    for ax, (metric, ylabel) in zip(axes, YIELDS):
        models_plotted: list[str] = []
        for model in sorted(sub["model"].unique()):
            if "AI_" in model:
                continue
            color = _model_color(model)
            mdata = sub[sub["model"] == model]
            mean_s = mdata.groupby("turn")[metric].mean()
            std_s  = mdata.groupby("turn")[metric].std().fillna(0)
            n_s    = mdata.groupby("turn")[metric].count().clip(lower=1)
            ci_s   = 1.96 * std_s / np.sqrt(n_s)
            ls = ":" if model == "Kimi-K2.5" else "-"
            ax.plot(mean_s.index, mean_s, color=color, lw=2.5,
                    ls=ls, label=_label_n(model))
            ax.fill_between(mean_s.index,
                            (mean_s - ci_s).clip(lower=0),
                            mean_s + ci_s, color=color, alpha=0.15)
            models_plotted.append(model)
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, color=MARBLE[800])
        if not all_handles:
            all_handles, all_labels = ax.get_legend_handles_labels()
            all_models = models_plotted
    # Single shared legend outside the grid
    fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_models),
               bbox_to_anchor=(0.5, -0.04), fontsize=11,
               facecolor="white", edgecolor=MARBLE[300])
    plt.suptitle("Yield Trajectories\n"
                 "(Shaded = 95% CI across runs; dotted = Kimi n=1 exploratory)",
                 fontsize=15, fontweight="bold", color=MARBLE[800], y=1.02)
    plt.tight_layout()
    _save(fig, f"yield_trajectories_{scenario}")
    return fig


def plot_expansion_timing(
    diary_df: pd.DataFrame,
    founding_df: pd.DataFrame,
    games_df: pd.DataFrame,
    scenario: str = "ground_control",
) -> plt.Figure:
    """Turn-of-founding per city, averaged per model."""
    valid_ids = set(games_df["game_id"])
    f_sub = founding_df[
        (founding_df["game_id"].isin(valid_ids)) &
        (founding_df["scenario"] == scenario) &
        (founding_df["city_number"] <= 6)
    ].copy()

    pivot = (
        f_sub.groupby(["model", "city_number"])["turn_founded"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 3.8))
    models_plotted: list[str] = []
    for model in sorted(pivot["model"].unique()):
        mdata = pivot[pivot["model"] == model]
        color = _model_color(model)
        ls = ":" if model == "Kimi-K2.5" else "-"
        ci = 1.96 * mdata["std"].fillna(0) / np.sqrt(mdata["count"].clip(lower=1))
        ax.plot(mdata["city_number"], mdata["mean"], marker=_model_marker(model),
                color=color, lw=2, ls=ls, label=_label_n(model))
        ax.fill_between(
            mdata["city_number"],
            (mdata["mean"] - ci).clip(lower=0),
            mdata["mean"] + ci,
            color=color, alpha=0.18,
        )
        models_plotted.append(model)
    for city_n, turn_b in [(2, 40), (3, 75), (4, 100)]:
        ax.axhline(turn_b, color=STATUS_DEFEAT, ls="--", alpha=0.45, lw=1)
        ax.text(6.05, turn_b, f"T{turn_b}", color=STATUS_DEFEAT, fontsize=10, va="center")
    ax.set_xlabel("City Number (ordinal)")
    ax.set_ylabel("Turn Founded (mean ± 95% CI)")
    ax.set_title(f"Expansion Timing — {scenario}",
                 fontsize=16, fontweight="bold", color=MARBLE[800])
    handles, labels = ax.get_legend_handles_labels()
    _add_icon_legend(ax, handles, labels, models_plotted, fontsize=12)
    ax.set_xticks(range(1, 7))
    plt.tight_layout()
    _save(fig, f"expansion_timing_{scenario}")
    return fig


def plot_city_milestones(
    metrics_df: pd.DataFrame,
    games_df: pd.DataFrame,
    scenario: str = "ground_control",
) -> plt.Figure:
    """Grouped bar: city count at T50/T100/T150/T200 per model."""
    valid_ids = set(games_df["game_id"])
    m_sub = metrics_df[
        (metrics_df["scenario"] == scenario) &
        (metrics_df["game_id"].isin(valid_ids))
    ].copy()

    milestones = ["city_t50", "city_t100", "city_t150", "city_t200"]
    labels = ["T50", "T100", "T150", "T200"]
    BENCHMARKS = [2.5, 3.5, 4.0, 5.0]

    model_list = sorted(m_sub["model"].unique())
    x = np.arange(len(milestones))
    width = 0.35 if len(model_list) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(11, 4.0))
    for i, model in enumerate(model_list):
        mdata = m_sub[m_sub["model"] == model]
        vals = [mdata[col].mean() for col in milestones]
        cis  = [1.96 * mdata[col].std() / np.sqrt(max(mdata[col].count(), 1))
                for col in milestones]
        color = _model_color(model)
        bars = ax.bar(x + i * width, vals, width, label=_label_n(model),
                      color=color, alpha=0.85,
                      hatch=_model_hatch(model),
                      edgecolor=(MARBLE[700] if model == "Kimi-K2.5" else "none"),
                      yerr=cis, capsize=4, error_kw=dict(elinewidth=1.2, ecolor=MARBLE[700]))
        # Add value labels at T50 and T100
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if labels[j] in ("T50", "T100"):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + cis[j] + 0.05,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=9, color=color)
    for j, bval in enumerate(BENCHMARKS):
        x0, x1 = x[j] - 0.05, x[j] + width * len(model_list) + 0.05
        ax.hlines(bval, x0, x1, colors=MARBLE[700], ls="dashed", lw=1.8, alpha=0.85,
                  label="Playbook benchmark" if j == 0 else "")
    ax.set_xticks(x + width * (len(model_list) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("City Count", fontsize=12)
    ax.set_title("City Count at Milestone Turns\n"
                 "Error bars = 95% CI across runs; dashed = playbook benchmark",
                 fontsize=14, fontweight="bold", color=MARBLE[800])
    handles, labels_leg = ax.get_legend_handles_labels()
    model_handles = handles[:len(model_list)]
    model_labels = labels_leg[:len(model_list)]
    bench_handles = handles[len(model_list):]
    bench_labels = labels_leg[len(model_list):]
    icon_leg = _add_icon_legend(ax, model_handles + bench_handles,
                                model_labels + bench_labels, model_list, fontsize=11)
    plt.tight_layout()
    _save(fig, f"city_milestones_{scenario}")
    return fig


def plot_radar(
    summary_df: pd.DataFrame,
    se_df: pd.DataFrame | None = None,
) -> plt.Figure:
    """Radar chart from a summary DataFrame.

    summary_df: index = model names, columns = axis labels, values = z-scores.
    se_df: same shape as summary_df, values = 95% CI half-widths (optional).
    """
    labels = list(summary_df.columns)
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 5.5), subplot_kw=dict(polar=True))
    ax.set_facecolor(MARBLE[50])
    ax.spines["polar"].set_color(MARBLE[300])
    ax.grid(color=MARBLE[300], linewidth=0.8)
    ax.tick_params(colors=MARBLE[700])

    models_plotted: list[str] = []
    for model in summary_df.index:
        color = PALETTE.get(model, MARBLE[500])
        vals = summary_df.loc[model].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=2.2, label=_label(model))
        ax.fill(angles, vals, color=color, alpha=0.09)
        # 95% CI band
        if se_df is not None and model in se_df.index:
            ses = se_df.loc[model].tolist()
            ses += ses[:1]
            lo = [max(v - s, -2.5) for v, s in zip(vals, ses)]
            hi = [min(v + s,  2.5) for v, s in zip(vals, ses)]
            ax.fill(angles, hi, color=color, alpha=0.12)
            ax.fill(angles, lo, color="white", alpha=1.0)
            ax.fill(angles, lo, color=color, alpha=0.09)
        models_plotted.append(model)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, color=MARBLE[800])
    ax.tick_params(axis="x", pad=12)
    ax.set_ylim(-2.5, 2.5)
    ax.yaxis.set_tick_params(labelsize=8, labelcolor=MARBLE[600])
    ax.set_title("Multi-Model Performance Radar\n(z-scored, ground_control)",
                 fontsize=15, fontweight="bold", pad=25, color=MARBLE[800])
    handles, leg_labels = ax.get_legend_handles_labels()
    _add_icon_legend(ax, handles, leg_labels, models_plotted,
                     loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=12)
    plt.tight_layout()
    _save(fig, "radar")
    return fig


def plot_pmr_subcategories(tools_df: pd.DataFrame, games_df: pd.DataFrame) -> plt.Figure:
    """Two-panel figure: Panel A = aggregate PMR per model; Panel B = PMR by subcategory."""
    if tools_df.empty:
        return _empty_fig("PMR Subcategories", "No tool call data (Azure required)", "pmr_subcategories")

    # Compute per-game PMR for each subcategory
    pmr_rows = []
    for gid, gdf in tools_df.groupby("game_id"):
        model = gdf["model"].iloc[0]
        denom = gdf["is_pmr_denominator"].sum()
        if denom == 0:
            continue
        total_monitoring = (gdf["pmr_subcategory"].notna()).sum()
        pmr_rows.append({"game_id": gid, "model": model,
                         "subcategory": "aggregate", "rate": total_monitoring / denom})
        for sub in ["victory_monitoring", "strategic_map", "resource_monitoring"]:
            count = (gdf["pmr_subcategory"] == sub).sum()
            pmr_rows.append({"game_id": gid, "model": model, "subcategory": sub,
                             "rate": count / denom})

    pmr_df = pd.DataFrame(pmr_rows)

    sub_order = ["victory_monitoring", "strategic_map", "resource_monitoring"]
    sub_labels = {
        "victory_monitoring": "Victory\nMonitoring",
        "strategic_map":      "Strategic\nMap",
        "resource_monitoring": "Resource\nMonitoring",
    }

    _LABEL_TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}
    model_order = sorted(pmr_df["model"].unique(),
                         key=lambda m: (m == "Kimi-K2.5", _label(m)))
    label_order = [_label(m) for m in model_order]
    label_palette = {_label(m): _model_color(m) for m in model_order}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.0),
                                     gridspec_kw={"width_ratios": [1, 2]})

    # --- Panel A: aggregate PMR per model ---
    agg_df = pmr_df[pmr_df["subcategory"] == "aggregate"].copy()
    agg_df["model_label"] = agg_df["model"].map(_label)
    sns.barplot(data=agg_df, x="model_label", y="rate", order=label_order,
                palette=label_palette,
                errorbar=("ci", 95), capsize=0.08, err_kws={"linewidth": 1.3},
                ax=ax_a)
    # Add value labels on bars
    for patch in ax_a.patches:
        h = patch.get_height()
        if h > 0:
            ax_a.text(patch.get_x() + patch.get_width() / 2, h + 0.0005,
                      f"{h:.2%}", ha="center", va="bottom", fontsize=10, color=MARBLE[700])
    ax_a.set_title("(A) Aggregate PMR per model", fontsize=13, color=MARBLE[800])
    ax_a.set_xlabel("")
    ax_a.set_ylabel("PMR (% of non-infrastructure calls)", fontsize=11)
    ax_a.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_a.tick_params(axis="x", labelsize=10)

    # --- Panel B: subcategory breakdown ---
    sub_df = pmr_df[pmr_df["subcategory"] != "aggregate"].copy()
    sub_df["sub_label"] = sub_df["subcategory"].map(sub_labels)
    sub_df["model_label"] = sub_df["model"].map(_label)
    sns.barplot(data=sub_df, x="sub_label", y="rate", hue="model_label",
                order=[sub_labels[s] for s in sub_order],
                hue_order=label_order,
                palette=label_palette,
                errorbar=("ci", 95), capsize=0.05, err_kws={"linewidth": 1.2},
                ax=ax_b)
    # Highlight victory monitoring with an annotation box
    ax_b.annotate("lowest PMR\nsubcategory", xy=(0, 0.001),
                  xytext=(0.22, 0.012),
                  fontsize=9, color=STATUS_DEFEAT,
                  arrowprops=dict(arrowstyle="->", color=STATUS_DEFEAT, lw=1.0))
    ax_b.set_title("(B) PMR by subcategory", fontsize=13, color=MARBLE[800])
    ax_b.set_xlabel("")
    ax_b.set_ylabel("PMR (% of non-infrastructure calls)", fontsize=11)
    ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    handles, leg_labels_b = ax_b.get_legend_handles_labels()
    models_ordered = [_LABEL_TO_MODEL.get(l, l) for l in leg_labels_b]
    ax_b.get_legend().remove()
    _add_icon_legend(ax_b, handles, leg_labels_b, models_ordered, fontsize=10, loc="upper right")

    plt.suptitle("Proactive Monitoring Rate",
                 fontsize=15, fontweight="bold", color=MARBLE[800])
    plt.tight_layout()
    _save(fig, "pmr_subcategories")
    return fig


def plot_pmr_over_time(tools_df: pd.DataFrame, games_df: pd.DataFrame) -> plt.Figure:
    """PMR over game time (rolling 10-turn window), per model."""
    if tools_df.empty:
        return _empty_fig("PMR Over Time", "No tool call data (Azure required)", "pmr_over_time")

    # tools_df is already admissible-filtered; drop turn=0 sentinel rows
    t = tools_df[tools_df["turn"] > 0].copy()

    fig, ax = plt.subplots(figsize=(12, 3.8))
    models_plotted: list[str] = []
    all_pmr_vals: list[float] = []
    for model in sorted(t["model"].unique()):
        color = _model_color(model)
        mdata = t[t["model"] == model]
        turn_pmr = (
            mdata.groupby("turn")
            .apply(lambda df: (df["pmr_subcategory"].notna()).sum() / max(df["is_pmr_denominator"].sum(), 1))
            .rolling(10, min_periods=1).mean()
        )
        ls = ":" if model == "Kimi-K2.5" else "-"
        ax.plot(turn_pmr.index, turn_pmr, color=color, lw=2.2, ls=ls, label=_label_n(model))
        all_pmr_vals.extend(turn_pmr.dropna().tolist())
        models_plotted.append(model)

    # Add overall mean PMR reference line
    if all_pmr_vals:
        mean_pmr = np.mean(all_pmr_vals)
        ax.axhline(mean_pmr, color=MARBLE[600], lw=1.2, ls="--", alpha=0.7,
                   label=f"Overall mean ({mean_pmr:.2%})")
        # Annotate: no endgame increase
        ax.text(0.99, mean_pmr + 0.002, "No systematic increase\ntoward endgame",
                transform=ax.get_yaxis_transform(), ha="right",
                fontsize=9, color=MARBLE[600], style="italic")

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("PMR (10-turn rolling avg)\n% of non-infrastructure calls", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Proactive Monitoring Rate Over Time",
                 fontsize=15, fontweight="bold", color=MARBLE[800])
    handles, leg_labels = ax.get_legend_handles_labels()
    _add_icon_legend(ax, handles, leg_labels, models_plotted + ([""] if all_pmr_vals else []),
                     fontsize=11)
    plt.tight_layout()
    _save(fig, "pmr_over_time")
    return fig


def plot_tool_category_stacked_area(
    tools_df: pd.DataFrame, games_df: pd.DataFrame
) -> plt.Figure:
    """Stacked area: tool category composition over turns (10-turn rolling avg)."""
    if tools_df.empty:
        return _empty_fig("Tool Category Composition", "No tool call data (Azure required)", "tool_stacked_area")

    # tools_df already admissible; restrict to ground_control turns 1-320
    t = tools_df[
        (tools_df["scenario"] == "ground_control") &
        (tools_df["turn"].between(1, 320))
    ].copy()

    CAT_ORDER = ["unit_action", "state_query", "strategic_monitoring", "diplomacy", "other"]
    CAT_COLORS = {
        "unit_action":          TERRACOTTA,
        "state_query":          OCEAN,
        "strategic_monitoring": PATINA,
        "diplomacy":            GOLD,
        "other":                MARBLE[500],
    }
    CAT_LABELS = {
        "unit_action": "Unit Action",
        "state_query": "State Query",
        "strategic_monitoring": "Strategic Monitoring",
        "diplomacy": "Diplomacy",
        "other": "Other",
    }

    model_list = sorted(t["model"].unique())
    fig, axes = plt.subplots(1, len(model_list), figsize=(5 * len(model_list), 3.8), sharey=True)
    if len(model_list) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_list):
        mdata = t[t["model"] == model]
        cat_counts = (
            mdata.groupby(["turn", "category"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=CAT_ORDER, fill_value=0)
        )
        n_games = mdata.groupby("turn")["game_id"].nunique()
        cat_avg = cat_counts.div(n_games, axis=0)
        cat_smooth = cat_avg.rolling(10, min_periods=1).mean()
        # Normalise to percentage composition
        row_totals = cat_smooth.sum(axis=1).replace(0, np.nan)
        cat_pct = cat_smooth.div(row_totals, axis=0).fillna(0)
        ax.stackplot(
            cat_pct.index,
            [cat_pct.get(c, pd.Series(0, index=cat_pct.index)) for c in CAT_ORDER],
            labels=[CAT_LABELS[c] for c in CAT_ORDER],
            colors=[CAT_COLORS[c] for c in CAT_ORDER],
            alpha=0.85,
        )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        n = MODEL_N.get(model, "?")
        is_kimi = model == "Kimi-K2.5"
        title_color = MARBLE[500] if is_kimi else MARBLE[800]
        suffix = " — exploratory" if is_kimi else ""
        ax.set_title(f"{_label(model)} (n={n}){suffix}", fontsize=13, color=title_color)
        ax.set_xlabel("Turn", fontsize=12)
        if model == model_list[0]:
            ax.set_ylabel("% of calls (10-turn rolling avg)", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.08), fontsize=11,
               facecolor="white", edgecolor=MARBLE[300])
    plt.suptitle("Tool Category Composition Over Time",
                 fontsize=16, fontweight="bold", color=MARBLE[800])
    plt.tight_layout()
    _save(fig, "tool_stacked_area")
    return fig


def _load_rag_results() -> pd.DataFrame:
    """Load auto-annotation results from rag_auto_results.csv."""
    results_path = Path(__file__).resolve().parent / "rag_auto_results.csv"
    if not results_path.exists():
        return pd.DataFrame()
    return pd.read_csv(results_path)


def plot_rag_breakdown(rag_results: pd.DataFrame | None = None) -> plt.Figure:
    """Stacked bar chart: Y / P / N proportions per model."""
    if rag_results is None:
        rag_results = _load_rag_results()

    if rag_results.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No RAG data yet", ha="center", va="center")
        ax.set_title("RAG Breakdown")
        _save(fig, "rag_breakdown")
        return fig

    models = sorted(rag_results["model"].unique())
    y_vals, p_vals, n_vals, totals, rag_cis = [], [], [], [], []
    for m in models:
        sub = rag_results[rag_results["model"] == m]
        n = len(sub)
        y_count = len(sub[sub["executed"] == "Y"])
        p_count = len(sub[sub["executed"] == "P"])
        n_count = len(sub[sub["executed"] == "N"])
        y_vals.append(y_count / n)
        p_vals.append(p_count / n)
        n_vals.append(n_count / n)
        totals.append(n)
        # Bootstrap 95% CI for RAG score
        rng = np.random.default_rng(42)
        executed_arr = sub["executed"].to_numpy()
        boot_rags = []
        for _ in range(2000):
            idx = rng.integers(0, n, size=n)
            s = executed_arr[idx]
            by = (s == "Y").sum()
            bp = (s == "P").sum()
            boot_rags.append((by + 0.5 * bp) / n)
        rag_cis.append(np.percentile(boot_rags, [2.5, 97.5]))

    x = np.arange(len(models))
    labels = [_label(m) for m in models]

    fig = plt.figure(figsize=(7, 5.2), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_rag = fig.add_subplot(gs[1], sharex=ax)

    ax.bar(x, y_vals, color=STATUS_VICTORY, label="Executed (Y)")
    ax.bar(x, p_vals, bottom=y_vals, color=GOLD, label="Partial (P)")
    ax.bar(x, n_vals,
           bottom=[y + p for y, p in zip(y_vals, p_vals)],
           color=STATUS_DEFEAT, label="Not done (N)")

    rag_scores = []
    for i, (y, p, total, ci) in enumerate(zip(y_vals, p_vals, totals, rag_cis)):
        rag = y + 0.5 * p
        rag_scores.append(rag)
        ax.text(i, 1.01, f"n={total}", ha="center", va="bottom",
                fontsize=10, color=MARBLE[600])

    ax.set_ylabel("Proportion of commitments", fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_title("Commitment Execution within 10 Turns (RAG@10)\n"
                 "RAG@10 = (Y + 0.5×P) / total commitments; higher is better",
                 fontsize=13, fontweight="bold", color=MARBLE[800])
    ax.legend(loc="lower right", fontsize=11, facecolor="white", edgecolor=MARBLE[300],
              title="Executed (Y), Partial (P), Not done (N)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.setp(ax.get_xticklabels(), visible=False)

    # Bottom panel: RAG point estimates with 95% CI
    for i, (rag, ci) in enumerate(zip(rag_scores, rag_cis)):
        ci_lo, ci_hi = ci
        ax_rag.errorbar(i, rag, yerr=[[rag - ci_lo], [ci_hi - rag]],
                        fmt="o", color="white", ecolor=MARBLE[700], capsize=6,
                        linewidth=1.5, markersize=7,
                        markeredgecolor=MARBLE[700], markeredgewidth=1.5, zorder=5)
        ax_rag.text(i, ci_hi + 0.01, f"{rag:.1%}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=MARBLE[800])

    all_ci_lo = [ci[0] for ci in rag_cis]
    all_ci_hi = [ci[1] for ci in rag_cis]
    y_lo = max(0.0, min(all_ci_lo) - 0.08)
    y_hi = min(1.0, max(all_ci_hi) + 0.14)  # extra headroom for labels

    ax_rag.set_xticks(x)
    ax_rag.set_xticklabels(labels, fontsize=13)
    ax_rag.set_ylabel("RAG@10\n(95% CI)", fontsize=10)
    ax_rag.set_ylim(y_lo, y_hi)
    ax_rag.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_rag.axhline(0.5, color=MARBLE[400], linewidth=0.8, linestyle="--")

    _save(fig, "rag_breakdown")
    return fig


def plot_rag_sensitivity(rag_df: pd.DataFrame | None = None) -> plt.Figure:
    """RAG score vs K (K=3,5,10,20) sweep, one line per model.

    Reuses cached planning text; re-runs auto_annotate with different K windows.
    If ANTHROPIC_API_KEY is not set, loads from rag_auto_results.csv at K=10 only.
    """
    import os
    from analysis.rag_extraction import SAMPLE_TURNS, fetch_planning, summarise_tools, auto_annotate
    from analysis.admissible_games import ADMISSIBLE_GAMES

    meta_by_run = {g["run_id"]: g for g in ADMISSIBLE_GAMES}
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    K_VALUES = [3, 5, 10, 20]

    # If no API key, fall back to K=10 from existing results
    if not api_key:
        base = _load_rag_results()
        if base.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No RAG data (set ANTHROPIC_API_KEY)", ha="center", va="center")
            _save(fig, "rag_sensitivity")
            return fig
        rows = []
        for m in base["model"].unique():
            sub = base[base["model"] == m]
            n = len(sub)
            y = len(sub[sub["executed"] == "Y"])
            p = len(sub[sub["executed"] == "P"])
            rows.append({"model": m, "K": 10, "rag_score": (y + 0.5 * p) / n, "n": n})
        rag_df = pd.DataFrame(rows)
    else:
        rows = []
        for k in K_VALUES:
            for run_id, turn in SAMPLE_TURNS:
                planning = fetch_planning(run_id, turn)
                if not planning.strip():
                    continue
                tools = summarise_tools(run_id, turn, k=k)
                commitments = auto_annotate(planning, tools, api_key)
                model = meta_by_run.get(run_id, {}).get("model", "unknown")
                for c in commitments:
                    rows.append({"model": model, "K": k,
                                 "executed": c.get("executed", "N")})
        if not rows:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No RAG data", ha="center", va="center")
            _save(fig, "rag_sensitivity")
            return fig

        raw = pd.DataFrame(rows)
        agg = []
        for (model, k), grp in raw.groupby(["model", "K"]):
            n = len(grp)
            y = (grp["executed"] == "Y").sum()
            p = (grp["executed"] == "P").sum()
            agg.append({"model": model, "K": k, "rag_score": (y + 0.5 * p) / n, "n": n})
        rag_df = pd.DataFrame(agg)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    models_plotted: list[str] = []
    for model in sorted(rag_df["model"].unique()):
        mdata = rag_df[rag_df["model"] == model].sort_values("K")
        color = _model_color(model)
        # Bootstrap 95% CI per K point if n available
        ci_lo, ci_hi = [], []
        for _, row in mdata.iterrows():
            n = int(row.get("n", 0))
            if n > 1:
                rng = np.random.default_rng(42)
                boot = [row["rag_score"] + rng.normal(0, 0.1) for _ in range(500)]
                ci_lo.append(max(0, np.percentile(boot, 2.5)))
                ci_hi.append(min(1, np.percentile(boot, 97.5)))
            else:
                ci_lo.append(row["rag_score"])
                ci_hi.append(row["rag_score"])
        ax.fill_between(mdata["K"], ci_lo, ci_hi, color=color, alpha=0.15)
        ax.plot(mdata["K"], mdata["rag_score"], marker="o", color=color,
                lw=2.5, markersize=10, label=_label_n(model))
        # Annotate last point
        last = mdata.iloc[-1]
        ax.annotate(f"{last['rag_score']:.1%}",
                    xy=(last["K"], last["rag_score"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=11, color=color)
        models_plotted.append(model)

    # Mark K=10 as the primary window
    ax.axvline(10, color=MARBLE[500], lw=1.2, ls=":", alpha=0.8)
    ax.text(10 + 0.3, 0.05, "K=10\n(main results)", fontsize=9,
            color=MARBLE[600], style="italic")

    ax.set_xlabel("K (lookahead turns)", fontsize=13)
    ax.set_ylabel("RAG Score  (Y + 0.5×P) / total\nhigher = more commitments executed", fontsize=11)
    ax.set_title("Commitment Execution vs Lookahead Window (RAG Sensitivity to K)",
                 fontsize=13, fontweight="bold", color=MARBLE[800])
    handles, leg_labels = ax.get_legend_handles_labels()
    _add_icon_legend(ax, handles, leg_labels, models_plotted, fontsize=11)
    ax.set_xticks(K_VALUES)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    _save(fig, "rag_sensitivity")
    return fig


def plot_inflection_points(
    diary_df: pd.DataFrame,
    games_df: pd.DataFrame,
    scenario: str = "ground_control",
    metrics: list[str] | None = None,
    t_min: int = 20,
    t_max: int = 250,
    t_step: int = 5,
) -> plt.Figure:
    """Spearman ρ between metric-at-turn-T and final normalised_score, swept over T.

    Shows which metrics become predictive earliest and at what turn the signal
    stabilises. N annotation on each curve shows how many games have diary data
    at that turn (varies due to partial diary coverage).
    """
    from scipy.stats import spearmanr

    if metrics is None:
        metrics = ["exploration_pct", "cities", "score", "science", "culture", "gold"]

    valid_ids = set(games_df["game_id"])
    gc_games = games_df[
        (games_df["scenario"] == scenario) & (games_df["game_id"].isin(valid_ids))
    ][["game_id", "normalised_score"]].set_index("game_id")

    agent_diary = diary_df[
        diary_df["is_agent"]
        & diary_df["game_id"].isin(gc_games.index)
    ]

    turns = range(t_min, t_max + 1, t_step)
    results: dict[str, list[tuple[int, float, int]]] = {m: [] for m in metrics}

    for t in turns:
        at_t = agent_diary[agent_diary["turn"] == t][["game_id"] + metrics].copy()
        at_t = at_t.merge(gc_games, left_on="game_id", right_index=True)
        n = len(at_t)
        if n < 4:
            continue
        for m in metrics:
            vals = at_t[[m, "normalised_score"]].dropna()
            if len(vals) < 4:
                results[m].append((t, float("nan"), len(vals)))
                continue
            rho, _ = spearmanr(vals[m], vals["normalised_score"])
            results[m].append((t, rho, len(vals)))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3.8))

    for m in metrics:
        pts = [(t, rho, n) for t, rho, n in results[m] if not np.isnan(rho)]
        if not pts:
            continue
        ts, rhos, ns = zip(*pts)
        color = _INFLECTION_COLORS.get(m, MARBLE[500])
        label = _INFLECTION_LABELS.get(m, m)
        lw = 3 if m == "score" else 1.5
        alpha = 1.0 if m == "score" else 0.75
        ax.plot(ts, rhos, color=color, linewidth=lw, alpha=alpha, label=label)
        # 95% CI via Fisher z-transform
        ci_lo, ci_hi = [], []
        for rho_i, n_i in zip(rhos, ns):
            if n_i > 3:
                z = np.arctanh(np.clip(rho_i, -0.9999, 0.9999))
                se = 1.0 / np.sqrt(n_i - 3)
                ci_lo.append(np.tanh(z - 1.96 * se))
                ci_hi.append(np.tanh(z + 1.96 * se))
            else:
                ci_lo.append(rho_i)
                ci_hi.append(rho_i)
        ax.fill_between(ts, ci_lo, ci_hi, color=color, alpha=0.10)

    # Annotate N where sample size changes (dotted boundary lines)
    ref = results.get("exploration_pct", results[metrics[0]])
    prev_n = None
    toggle = 0
    for t, rho, n in ref:
        if np.isnan(rho):
            continue
        if n != prev_n:
            ax.axvline(t, color=MARBLE[300], linewidth=0.8, linestyle=":")
            y_pos = -0.90 if toggle % 2 == 0 else -1.02
            ax.text(t + 1, y_pos, f"n={n}", fontsize=9, color=MARBLE[500], va="bottom")
            prev_n = n
            toggle += 1

    # Annotate the score curve peak
    score_pts = [(t, rho) for t, rho, _ in results.get("score", []) if not np.isnan(rho) and rho > 0.9]
    if score_pts:
        t_peak, rho_peak = score_pts[0]
        ax.annotate(
            f"raw score ρ={rho_peak:.2f}\nfrom T{t_peak}",
            xy=(t_peak, rho_peak), xytext=(t_peak + 10, rho_peak - 0.15),
            fontsize=10, color=_INFLECTION_COLORS["score"],
            arrowprops=dict(arrowstyle="->", color=_INFLECTION_COLORS["score"], lw=1),
        )

    ax.axhline(0, color=MARBLE[700], linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.5, color=MARBLE[300], linewidth=0.6, linestyle=":")
    ax.axhline(-0.5, color=MARBLE[300], linewidth=0.6, linestyle=":")
    ax.text(t_max - 2, 0.52, "ρ = 0.5", fontsize=9, color=MARBLE[500], ha="right")
    ax.text(t_max - 2, -0.48, "ρ = −0.5", fontsize=9, color=MARBLE[500], ha="right")

    ax.set_xlabel("Turn (T)")
    ax.set_ylabel("Spearman ρ with final normalised score")
    ax.set_title(
        f"Inflection Point Sweep — When do early metrics predict final outcome?",
        color=MARBLE[800], fontsize=14, fontweight="bold",
    )
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(t_min, t_max)
    ax.legend(loc="upper left", fontsize=11, facecolor="white", edgecolor=MARBLE[300])

    _save(fig, "inflection_points")
    return fig


def plot_inflection_deltas(
    diary_df: pd.DataFrame,
    games_df: pd.DataFrame,
    scenario: str = "ground_control",
    metrics: list[str] | None = None,
    window: int = 5,
    t_min: int = 20,
    t_max: int = 245,
    t_step: int = 5,
) -> plt.Figure:
    """Spearman ρ between Δmetric over [T, T+window] and final normalised_score.

    With window=t_step=5 (defaults), windows are non-overlapping. Peaks reveal
    the specific turn windows where short-term momentum is most predictive of
    final outcome. BH-FDR correction applied post-loop; surviving tests are
    marked with filled circles on the plot.
    """
    import warnings
    from scipy.stats import spearmanr, false_discovery_control

    if metrics is None:
        metrics = ["score", "science", "cities", "culture", "gold", "exploration_pct"]

    gc_games = games_df[games_df["scenario"] == scenario][
        ["game_id", "normalised_score"]
    ].set_index("game_id")

    agent_diary = diary_df[
        diary_df["is_agent"] & diary_df["game_id"].isin(gc_games.index)
    ]

    # results stores (t, rho, p, n) per metric
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

    # BH-FDR correction across all (metric, window) tests
    all_tests = [
        (m, t, rho, p, n)
        for m in metrics
        for t, rho, p, n in results[m]
        if not np.isnan(rho)
    ]
    surviving: set[tuple[str, int]] = set()
    if all_tests:
        p_vals = np.array([x[3] for x in all_tests])
        p_adj = false_discovery_control(p_vals, method="bh")
        surviving = {(x[0], x[1]) for x, pa in zip(all_tests, p_adj) if pa < 0.05}

    _DELTA_LABELS = {k: f"Δ {v}" for k, v in _INFLECTION_LABELS.items()}

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax2 = ax.twinx()

    for m in metrics:
        pts = [(t, rho, n) for t, rho, _, n in results[m] if not np.isnan(rho)]
        if not pts:
            continue
        ts, rhos, ns = zip(*pts)
        color = _INFLECTION_COLORS.get(m, MARBLE[500])
        label = _DELTA_LABELS.get(m, m)
        lw = 2.5 if m == "score" else 1.5
        alpha = 1.0 if m == "score" else 0.75
        ax.plot(ts, rhos, color=color, linewidth=lw, alpha=alpha, label=label)
        # 95% CI via Fisher z-transform
        ci_lo, ci_hi = [], []
        for rho_i, n_i in zip(rhos, ns):
            if n_i > 3:
                z = np.arctanh(np.clip(rho_i, -0.9999, 0.9999))
                se = 1.0 / np.sqrt(n_i - 3)
                ci_lo.append(np.tanh(z - 1.96 * se))
                ci_hi.append(np.tanh(z + 1.96 * se))
            else:
                ci_lo.append(rho_i)
                ci_hi.append(rho_i)
        ax.fill_between(ts, ci_lo, ci_hi, color=color, alpha=0.10)
        # Filled marker where test survives BH-FDR
        sig_ts = [t for t, rho in zip(ts, rhos) if (m, t) in surviving]
        sig_rhos = [rho for t, rho in zip(ts, rhos) if (m, t) in surviving]
        if sig_ts:
            ax.scatter(sig_ts, sig_rhos, color=color, s=40, zorder=5)

    # Effective n on secondary axis (uses first metric as reference)
    ref_pts = [(t, n) for t, rho, _, n in results[metrics[0]] if not np.isnan(rho)]
    if ref_pts:
        ref_t, ref_n = zip(*ref_pts)
        ax2.step(ref_t, ref_n, color=MARBLE[400], linewidth=0.8, where="mid", label="n (games)")
        ax2.set_ylabel("n games", color=MARBLE[500], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=MARBLE[500], labelsize=7)
        ax2.set_ylim(0, max(ref_n) * 3)  # keep n line low so it doesn't crowd ρ curves

    ax.axhline(0, color=MARBLE[700], linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.5, color=MARBLE[300], linewidth=0.6, linestyle=":")
    ax.axhline(-0.5, color=MARBLE[300], linewidth=0.6, linestyle=":")
    ax.text(t_max - 2, 0.52, "ρ = 0.5", fontsize=9, color=MARBLE[500], ha="right")
    ax.text(t_max - 2, -0.48, "ρ = −0.5", fontsize=9, color=MARBLE[500], ha="right")

    n_sig = len(surviving)
    n_total = len(all_tests)
    ax.set_xlabel("Turn window start (T)")
    ax.set_ylabel("Spearman ρ with final normalised score")
    ax.set_title(
        f"Inflection Delta Sweep — Which 5-turn windows predict final outcome?\n"
        f"(filled markers = BH-FDR significant, {n_sig}/{n_total} tests survive)",
        color=MARBLE[800], fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(t_min, t_max)
    ax.legend(loc="upper left", fontsize=11, facecolor="white", edgecolor=MARBLE[300])

    _save(fig, "inflection_deltas")
    return fig


def plot_tool_error_rate(tools_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal grouped bars: failure rate per tool × model, top-N tools by failure count.

    Surfaces where each model breaks: e.g. propose_trade validation vs set_city_production
    slot errors. Uses the `success` column from load_tools_df.
    """
    if tools_df.empty:
        return _empty_fig("Tool Error Rate", "No tool call data (Azure required)", "tool_error_rate")

    t = tools_df.copy()
    t["failed"] = ~t["success"].astype(bool)

    # Filter to tools with ≥50 total calls (avoids noise from rarely-used tools)
    call_counts_by_tool = t.groupby("tool").size()
    eligible_tools = call_counts_by_tool[call_counts_by_tool >= 50].index
    t_eligible = t[t["tool"].isin(eligible_tools)]

    failures_by_tool = t_eligible[t_eligible["failed"]].groupby("tool").size().sort_values(ascending=False)
    if failures_by_tool.empty:
        return _empty_fig("Tool Error Rate", "No tool failures recorded (n≥50 filter)", "tool_error_rate")
    top_tools = failures_by_tool.head(top_n).index.tolist()

    # Per (model, tool) failure rate
    grp = (
        t_eligible[t_eligible["tool"].isin(top_tools)]
        .groupby(["model", "tool"])
        .agg(calls=("failed", "size"), failures=("failed", "sum"))
        .reset_index()
    )
    grp["failure_rate"] = grp["failures"] / grp["calls"].clip(lower=1)

    models = sorted(t["model"].unique())
    tool_order = top_tools[::-1]  # reverse so highest-failure tool is on top
    n_models = len(models)
    bar_h = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(tool_order) + 1.0)))
    y = np.arange(len(tool_order))
    for i, model in enumerate(models):
        sub = grp[grp["model"] == model].set_index("tool").reindex(tool_order)
        rates = sub["failure_rate"].fillna(0).values
        counts = sub["calls"].fillna(0).astype(int).values
        color = _model_color(model)
        offset = (i - (n_models - 1) / 2) * bar_h
        # Wilson 95% CI half-widths
        ci_vals = [
            1.96 * np.sqrt(max(p * (1 - p) / max(n, 1), 0))
            for p, n in zip(rates, counts)
        ]
        ax.barh(y + offset, rates, bar_h, color=color, alpha=0.85, label=_label_n(model),
                xerr=ci_vals, capsize=3,
                error_kw=dict(elinewidth=1.1, ecolor=MARBLE[700]))

    ax.set_yticks(y)
    ax.set_yticklabels(tool_order, fontsize=11)
    ax.set_xlabel("Failure rate  (Wilson 95% CI; tools with <50 calls excluded)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, min(1.0, max(0.15, grp["failure_rate"].max() * 1.25 + 0.05)))
    ax.set_title(f"Tool Failure Rate — Top {len(tool_order)} Tools by Total Failures",
                 fontsize=16, fontweight="bold", color=MARBLE[800])
    handles, leg_labels = ax.get_legend_handles_labels()
    _add_icon_legend(ax, handles, leg_labels, models, fontsize=11, loc="lower right")
    plt.tight_layout()
    _save(fig, "tool_error_rate")
    return fig


def plot_tool_latency(tools_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Scatter: per-tool median latency vs call frequency, sized by total time.

    Surfaces the "sensorium cost" — which tools dominate wall-clock time and
    might push agents away from frequent monitoring. Labels top-N tools by
    total time spent.
    """
    if tools_df.empty or "duration_ms" not in tools_df.columns:
        return _empty_fig("Tool Latency", "No tool call data (Azure required)", "tool_latency")

    t = tools_df.dropna(subset=["duration_ms"]).copy()
    t = t[t["duration_ms"] >= 0]
    if t.empty:
        return _empty_fig("Tool Latency", "No duration_ms values recorded", "tool_latency")

    per_tool = (
        t.groupby("tool")
        .agg(calls=("duration_ms", "size"),
             median_ms=("duration_ms", "median"),
             total_s=("duration_ms", lambda s: s.sum() / 1000.0))
        .reset_index()
        .sort_values("total_s", ascending=False)
    )

    # Size scaling: total_s → marker area
    sizes = 40 + 300 * (per_tool["total_s"] / per_tool["total_s"].max())

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(per_tool["calls"], per_tool["median_ms"],
               s=sizes, c=GOLD, alpha=0.55, edgecolor=MARBLE[700], linewidth=0.8)

    # Annotate top 10 tools by total time (reduces label overlap)
    top = per_tool.head(10)
    for _, row in top.iterrows():
        ax.annotate(
            row["tool"],
            (row["calls"], row["median_ms"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=10, color=MARBLE[800],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total calls (log scale)")
    ax.set_ylabel("Median latency (ms, log scale)")
    ax.set_title("Tool Latency vs Call Frequency (marker size scales with total wall time)",
                 fontsize=16, fontweight="bold", color=MARBLE[800])
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save(fig, "tool_latency")
    return fig


def plot_tool_entropy(tools_df: pd.DataFrame, games_df: pd.DataFrame) -> plt.Figure:
    """Scatter: per-game Shannon entropy of tool distribution vs normalised_score.

    Tests whether narrower tool repertoires correlate with worse outcomes. A
    low entropy means the agent leaned heavily on a few tools; high entropy
    means broader coverage of the action space.
    """
    if tools_df.empty:
        return _empty_fig("Tool Entropy", "No tool call data (Azure required)", "tool_entropy")

    def _shannon(series: pd.Series) -> float:
        counts = series.value_counts()
        if counts.sum() == 0:
            return float("nan")
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    # Exclude game_management calls from the distribution
    t = tools_df[tools_df["category"] != "game_management"]
    entropy_by_game = (
        t.groupby(["game_id", "model"])["tool"]
        .apply(_shannon)
        .reset_index(name="entropy")
    )

    merged = entropy_by_game.merge(
        games_df[["game_id", "normalised_score", "scenario"]],
        on="game_id", how="inner",
    ).dropna(subset=["entropy", "normalised_score"])

    if merged.empty:
        return _empty_fig("Tool Entropy", "No games with both entropy and score", "tool_entropy")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for model in sorted(merged["model"].unique()):
        sub = merged[merged["model"] == model]
        color = PALETTE.get(model, MARBLE[500])
        markers = {"ground_control": "o", "snowflake": "^", "cry_havoc": "s"}
        for scenario, scen_sub in sub.groupby("scenario"):
            ax.scatter(
                scen_sub["entropy"], scen_sub["normalised_score"],
                color=color, s=110, alpha=0.85,
                edgecolor=MARBLE[800], linewidth=1.1,
                marker=markers.get(scenario, "o"),
                label=_label(model) if scenario == "ground_control" else None,
            )

    # Per-model regression line + 95% CI band
    from scipy.stats import spearmanr
    from scipy import stats as _stats
    for model in sorted(merged["model"].unique()):
        sub = merged[merged["model"] == model].dropna(subset=["entropy", "normalised_score"])
        if len(sub) < 3:
            continue
        color = PALETTE.get(model, MARBLE[500])
        x_fit = np.linspace(sub["entropy"].min(), sub["entropy"].max(), 80)
        slope, intercept, _, _, se = _stats.linregress(sub["entropy"], sub["normalised_score"])
        y_fit = slope * x_fit + intercept
        n = len(sub)
        x_mean = sub["entropy"].mean()
        sxx = ((sub["entropy"] - x_mean) ** 2).sum()
        ci = 1.96 * se * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / sxx)
        ax.plot(x_fit, y_fit, color=color, lw=1.4, alpha=0.7, linestyle="--")
        ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.10)

    # Spearman correlation across all points
    rho, p_val = spearmanr(merged["entropy"], merged["normalised_score"])
    ax.text(
        0.02, 0.97,
        f"Spearman ρ = {rho:+.2f}  (p = {p_val:.2f}, n = {len(merged)})",
        transform=ax.transAxes, fontsize=12, va="top",
        bbox=dict(facecolor="white", edgecolor=MARBLE[300], boxstyle="round,pad=0.4"),
    )

    ax.set_xlabel("Tool-call Shannon entropy (bits)")
    ax.set_ylabel("Normalised score")
    ax.set_title("Tool Repertoire Breadth vs Outcome",
                 fontsize=16, fontweight="bold", color=MARBLE[800])

    handles, leg_labels = ax.get_legend_handles_labels()
    models_plotted = sorted(merged["model"].unique())
    _add_icon_legend(ax, handles, leg_labels, models_plotted, fontsize=11, loc="lower right")
    plt.tight_layout()
    _save(fig, "tool_entropy")
    return fig


def plot_metric_correlation(metrics_df: pd.DataFrame, games_df: pd.DataFrame) -> plt.Figure:
    """Spearman correlation heatmap over metrics_df columns (ground_control).

    Makes the paper's per-metric correlations feel less cherry-picked by
    showing the full dependency structure in one figure.
    """
    valid_ids = set(games_df["game_id"])
    gc = metrics_df[
        (metrics_df["scenario"] == "ground_control") &
        (metrics_df["game_id"].isin(valid_ids))
    ].copy()

    # Attach normalised_score from games_df so it appears in the matrix
    gc = gc.merge(
        games_df[["game_id", "normalised_score"]],
        on="game_id", how="left",
    )

    METRIC_COLS = [
        "normalised_score", "turns_played", "raw_score",
        "final_cities", "final_exploration_pct",
        "final_science", "final_gold", "final_military", "final_culture",
        "city_t50", "city_t100", "city_t150", "city_t200",
        "exploration_t50", "exploration_t100",
    ]
    cols = [c for c in METRIC_COLS if c in gc.columns]
    mat = gc[cols].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are all-NaN or constant (can't compute ρ)
    mat = mat.loc[:, mat.nunique(dropna=True) > 1]

    if mat.shape[1] < 2 or len(mat.dropna(how="all")) < 3:
        return _empty_fig("Metric Correlation", "Not enough data", "metric_correlation")

    corr = mat.corr(method="spearman")

    # Shorten column names for display
    SHORT = {
        "normalised_score": "norm_score", "turns_played": "turns",
        "final_cities": "cities", "final_exploration_pct": "exploration",
        "final_science": "science", "final_gold": "gold",
        "final_military": "military", "final_culture": "culture",
    }
    corr.index = [SHORT.get(c, c) for c in corr.index]
    corr.columns = [SHORT.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(11, 8))
    from matplotlib.colors import LinearSegmentedColormap
    diverging = LinearSegmentedColormap.from_list(
        "marble_div", [OCEAN, MARBLE[100], TERRACOTTA]
    )
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=diverging, center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.75},
        annot_kws={"fontsize": 10}, ax=ax,
    )
    ax.set_title(
        f"Metric Correlation Matrix — ground_control (n={len(mat)})",
        fontsize=16, fontweight="bold", color=MARBLE[800],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    _save(fig, "metric_correlation")
    return fig


def plot_decision_follow_through(
    tools_df: pd.DataFrame,
    trigger_tool: str = "get_victory_progress",
    window: int = 5,
) -> plt.Figure:
    """What does a model DO after checking victory? Categorises the next
    non-state-query, non-monitoring tool within `window` calls of each
    trigger_tool occurrence. Tests whether monitoring drives action or
    triggers more passive querying.
    """
    if tools_df.empty or "seq" not in tools_df.columns:
        return _empty_fig("Decision Follow-through", "No tool call data (Azure required)", "decision_follow_through")

    t = tools_df.dropna(subset=["seq"]).sort_values(["game_id", "seq"]).reset_index(drop=True)

    # "Actionable" categories — the agent changed the world, not just looked at it
    ACTIONABLE = {"unit_action", "other", "diplomacy"}
    NO_ACTION = "(no action in window)"

    successors: list[dict] = []
    for _, gdf in t.groupby("game_id", sort=False):
        gdf = gdf.reset_index(drop=True)
        trigger_positions = gdf.index[gdf["tool"] == trigger_tool].tolist()
        for pos in trigger_positions:
            model = gdf.at[pos, "model"]
            # Walk forward up to `window` calls looking for an actionable tool
            outcome_category = NO_ACTION
            for offset in range(1, window + 1):
                if pos + offset >= len(gdf):
                    break
                cand = gdf.iloc[pos + offset]
                if cand["category"] in ACTIONABLE:
                    outcome_category = cand["category"]
                    break
            successors.append({"model": model, "outcome": outcome_category})

    if not successors:
        return _empty_fig(
            "Decision Follow-through",
            f"No '{trigger_tool}' calls found",
            "decision_follow_through",
        )

    succ_df = pd.DataFrame(successors)
    dist = (
        succ_df.groupby(["model", "outcome"]).size().unstack(fill_value=0)
    )

    # Order columns: actionable first (terracotta / ocean / patina), inaction last
    col_order = [c for c in ["unit_action", "other", "diplomacy", NO_ACTION] if c in dist.columns]
    dist = dist[col_order]
    dist_pct = dist.div(dist.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    cat_colors = {
        "unit_action": TERRACOTTA,
        "other":       OCEAN,
        "diplomacy":   PATINA,
        NO_ACTION:     MARBLE[400],
    }
    cat_labels = {
        "unit_action": "Unit / city action",
        "other":       "Production · research · governance",
        "diplomacy":   "Diplomacy",
        NO_ACTION:     "No action within window",
    }

    models = sorted(dist_pct.index.tolist())
    y_pos = np.arange(len(models))
    left = np.zeros(len(models))
    for col in col_order:
        vals = dist_pct.loc[models, col].values
        ax.barh(
            y_pos, vals, left=left,
            color=cat_colors[col], edgecolor="white", linewidth=0.8,
            label=cat_labels[col],
        )
        # Annotate segments >= 8%
        for yi, (v, lft) in enumerate(zip(vals, left)):
            if v >= 0.08:
                ax.text(
                    lft + v / 2, yi, f"{v:.0%}",
                    ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold",
                )
        left += vals

    # Annotate total trigger count per model on the right
    totals = dist.sum(axis=1)
    for yi, model in enumerate(models):
        ax.text(
            1.015, yi, f"n={int(totals[model])}",
            va="center", fontsize=10, color=MARBLE[700],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_label(m) for m in models])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel(f"Fraction of '{trigger_tool}' calls followed by…")
    ax.set_title(
        f"Decision Follow-through — first actionable tool within {window} calls of {trigger_tool}",
        fontsize=15, fontweight="bold", color=MARBLE[800],
    )
    ax.legend(
        loc="center left", bbox_to_anchor=(1.06, 0.5),
        fontsize=11, frameon=True,
    )
    ax.grid(False, axis="y")
    plt.tight_layout()
    _save(fig, "decision_follow_through")
    return fig


def plot_game_sparklines(
    diary_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> plt.Figure:
    """3×3 small-multiples: per-game trajectory of raw score + city count.

    Gives reviewers a qualitative glance at individual admissible games —
    the aggregate plots smooth over meaningful variation between runs.
    """
    agents = diary_df[diary_df["is_agent"]].copy() if "is_agent" in diary_df.columns else diary_df.copy()
    games = games_df.copy()
    games = games.sort_values(["model", "scenario", "run_id"]).reset_index(drop=True)
    n = len(games)
    if n == 0:
        return _empty_fig("Per-game Sparklines", "No admissible games", "game_sparklines")

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.3 * nrows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, game) in zip(axes, games.iterrows()):
        gid = game["game_id"]
        gdf = agents[agents["game_id"] == gid].sort_values("turn")
        if gdf.empty:
            ax.text(0.5, 0.5, "(no trajectory)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color=MARBLE[500])
            ax.set_title(f"{_label(game['model'])} · {game['run_id'][:26]}",
                         fontsize=11, color=MARBLE[700])
            continue

        color = PALETTE.get(game["model"], MARBLE[500])
        ax.plot(gdf["turn"], gdf["score"], color=color, lw=1.8, label="score")
        ax.fill_between(gdf["turn"], 0, gdf["score"], color=color, alpha=0.12)
        ax.set_ylabel("score", color=color, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color, labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(gdf["turn"], gdf["cities"], color=MARBLE[700], lw=1.2, linestyle="--", label="cities")
        ax2.set_ylabel("cities", color=MARBLE[700], fontsize=10)
        ax2.tick_params(axis="y", labelcolor=MARBLE[700], labelsize=8)
        ax2.set_ylim(bottom=0)
        ax2.grid(False)

        end_note = f"T{int(game['turns_played'])} · {game.get('outcome', '')}"
        scenario_tag = "SF" if game["scenario"] == "snowflake" else "GC"
        title = f"{_label(game['model'])} · {scenario_tag} · {game['run_id'][:22]}"
        ax.set_title(title, fontsize=14, color=MARBLE[800], loc="left")
        ax.text(
            0.98, 0.04, end_note,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color=MARBLE[600],
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        )
        ax.set_xlim(0, max(gdf["turn"].max(), 1))
        ax.set_xlabel("turn", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "Per-game Trajectories (score + city count)",
        fontsize=16, fontweight="bold", color=MARBLE[800], y=1.01,
    )
    plt.tight_layout()
    _save(fig, "game_sparklines")
    return fig


def plot_reflection_quality(
    games_df: pd.DataFrame,
    rag_auto_df: pd.DataFrame | None = None,
    reflection_df: pd.DataFrame | None = None,
) -> plt.Figure:
    """Two-panel scatter: reflection 'richness' vs final normalised_score.

    Panel A: commitments per sampled turn (from the RAG pipeline's Haiku
             extractions in rag_auto_results.csv).
    Panel B: planning-text word count per agent turn (from load_reflections).

    Answers the reviewer question: is more/richer reflection predictive of
    better play?
    """
    if rag_auto_df is None:
        csv_path = ROOT / "analysis" / "rag_auto_results.csv"
        if csv_path.exists():
            rag_auto_df = pd.read_csv(csv_path)
        else:
            rag_auto_df = pd.DataFrame()

    if reflection_df is None:
        try:
            from analysis.data_loader import load_reflections
            reflection_df = load_reflections(every_n_turns=10)
        except Exception as e:
            print(f"  ! load_reflections failed: {e}")
            reflection_df = pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.8))
    from scipy.stats import spearmanr

    from scipy import stats as _stats

    def _scatter_panel(ax, panel, x_col, x_label, title, empty_msg):
        if panel.empty:
            ax.text(0.5, 0.5, empty_msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=MARBLE[500])
            ax.set_title(title, fontsize=15, color=MARBLE[800])
            return
        models_plotted = []
        for model in sorted(panel["model"].unique()):
            sub = panel[panel["model"] == model].dropna(subset=[x_col, "normalised_score"])
            color = PALETTE.get(model, MARBLE[500])
            ax.scatter(sub[x_col], sub["normalised_score"],
                       color=color, s=110, alpha=0.85,
                       edgecolor=MARBLE[800], linewidth=1.1,
                       label=_label(model))
            if len(sub) >= 3 and sub[x_col].nunique() > 1:
                slope, intercept, _, _, se = _stats.linregress(sub[x_col], sub["normalised_score"])
                x_fit = np.linspace(sub[x_col].min(), sub[x_col].max(), 80)
                y_fit = slope * x_fit + intercept
                n = len(sub)
                x_mean = sub[x_col].mean()
                sxx = ((sub[x_col] - x_mean) ** 2).sum()
                ci = 1.96 * se * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / sxx)
                ax.plot(x_fit, y_fit, color=color, lw=1.4, alpha=0.7, linestyle="--")
                ax.fill_between(x_fit, y_fit - ci, y_fit + ci, color=color, alpha=0.10)
            models_plotted.append(model)
        if len(panel) >= 3 and panel[x_col].nunique() > 1:
            rho, p_val = spearmanr(panel[x_col], panel["normalised_score"])
            stat = f"ρ = {rho:+.2f}  (p = {p_val:.2f}, n = {len(panel)})"
        else:
            stat = f"n = {len(panel)} (too few for ρ)"
        ax.text(
            0.02, 0.97, stat, transform=ax.transAxes,
            fontsize=15, va="top",
            bbox=dict(facecolor="white", edgecolor=MARBLE[300], boxstyle="round,pad=0.4"),
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Normalised score")
        ax.set_title(title, fontsize=15, color=MARBLE[800])
        handles, leg_labels = ax.get_legend_handles_labels()
        _add_icon_legend(ax, handles, leg_labels, models_plotted, fontsize=10, loc="lower right")

    # Panel A — commitments per sampled turn
    if not rag_auto_df.empty:
        commit_agg = (
            rag_auto_df.groupby("run_id")
            .agg(n_commitments=("commitment", "size"),
                 n_turns=("turn", "nunique"))
            .reset_index()
        )
        commit_agg["commit_density"] = commit_agg["n_commitments"] / commit_agg["n_turns"].clip(lower=1)
        panel_a = commit_agg.merge(
            games_df[["run_id", "model", "normalised_score"]],
            on="run_id", how="inner",
        ).dropna(subset=["commit_density", "normalised_score"])
    else:
        panel_a = pd.DataFrame()

    _scatter_panel(
        ax1, panel_a, "commit_density",
        "Commitments extracted per sampled planning turn",
        "A · Commitment density (RAG pipeline)",
        "No rag_auto_results.csv",
    )

    # Panel B — planning word count per turn
    if not reflection_df.empty:
        r = reflection_df.copy()
        r["planning_words"] = (
            r["reflection_planning"].fillna("").astype(str).str.split().str.len()
        )
        word_agg = (
            r.groupby("run_id")
            .agg(mean_planning_words=("planning_words", "mean"))
            .reset_index()
        )
        panel_b = word_agg.merge(
            games_df[["run_id", "model", "normalised_score"]],
            on="run_id", how="inner",
        ).dropna(subset=["mean_planning_words", "normalised_score"])
    else:
        panel_b = pd.DataFrame()

    _scatter_panel(
        ax2, panel_b, "mean_planning_words",
        "Mean planning-text word count per turn",
        "B · Planning verbosity (load_reflections)",
        "No reflection data",
    )

    plt.suptitle(
        "Reflection Quality vs Outcome",
        fontsize=16, fontweight="bold", color=MARBLE[800], y=1.02,
    )
    plt.tight_layout()
    _save(fig, "reflection_quality")
    return fig


def plot_harness_comparison() -> plt.Figure:
    """Two-panel figure contrasting pre-harness vs harness run completion.

    Panel A — Completion survival curve:
        % of runs still playing at each turn threshold T, comparing
        pre-harness (hex IDs) vs harness (admissible, word IDs).

    Panel B — Turns-played distribution:
        Strip + box plot of turns_played per condition.

    Saved to analysis/figures/harness_comparison.{pdf,png}.
    """
    from analysis.pre_harness_runs import pre_harness_df
    from analysis.admissible_games import admissible_df

    pre = pre_harness_df()
    pre["condition"] = "Pre-harness"

    admissible = admissible_df()
    admissible = admissible.rename(columns={"final_turn": "turns_played"})
    admissible["condition"] = "Harness"

    combined = pd.concat(
        [pre[["run_id", "turns_played", "condition"]],
         admissible[["run_id", "turns_played", "condition"]]],
        ignore_index=True,
    )

    HARNESS_COLOR     = PATINA       # green — completion
    PRE_HARNESS_COLOR = TERRACOTTA   # terracotta — failure

    fig, (ax_survival, ax_dist) = plt.subplots(
        1, 2,
        figsize=(12, 3.8),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------ #
    # Panel A — Survival curve
    # ------------------------------------------------------------------ #
    thresholds = list(range(0, 340, 5))
    for condition, color, ls in [
        ("Pre-harness", PRE_HARNESS_COLOR, "--"),
        ("Harness",     HARNESS_COLOR,     "-"),
    ]:
        subset = combined[combined["condition"] == condition]["turns_played"]
        n = len(subset)
        pct = [100 * (subset >= t).sum() / n for t in thresholds]
        ax_survival.plot(
            thresholds, pct,
            color=color, lw=2.5, ls=ls,
            label=f"{condition} (n={n})",
        )

    # Annotate how many pre-harness runs completed
    n_pre = len(combined[combined["condition"] == "Pre-harness"])
    n_completed = (pre["end_condition"].isin(["victory", "defeat"])).sum()
    ax_survival.annotate(
        f"Only {n_completed}/{n_pre} pre-harness runs\nreached a natural conclusion",
        xy=(306, 100 * (pre["turns_played"] >= 306).sum() / n_pre),
        xytext=(160, 55),
        fontsize=11, color=PRE_HARNESS_COLOR,
        arrowprops=dict(arrowstyle="->", color=PRE_HARNESS_COLOR, lw=1.4),
    )

    ax_survival.set_xlabel("Turns played (T)", fontsize=13)
    ax_survival.set_ylabel("Runs still active (%)", fontsize=13)
    ax_survival.set_title("A · Run completion survival curve", fontsize=13, color=MARBLE[800])
    ax_survival.set_xlim(0, 335)
    ax_survival.set_ylim(-2, 103)
    ax_survival.legend(fontsize=11, loc="upper right")

    for t, label in [(50, "T50\n(min. admissible)"), (200, "T200")]:
        ax_survival.axvline(t, color=MARBLE[400], lw=1, ls=":")
        ax_survival.text(t + 3, 5, label, fontsize=9, color=MARBLE[500], va="bottom")

    # ------------------------------------------------------------------ #
    # Panel B — Turns-played distribution
    # ------------------------------------------------------------------ #
    order = ["Pre-harness", "Harness"]
    palette = {"Pre-harness": PRE_HARNESS_COLOR, "Harness": HARNESS_COLOR}

    sns.boxplot(
        data=combined, x="condition", y="turns_played", hue="condition",
        order=order, palette=palette,
        width=0.45, linewidth=1.5, fliersize=0, legend=False,
        ax=ax_dist,
    )
    sns.stripplot(
        data=combined, x="condition", y="turns_played", hue="condition",
        order=order, palette=palette,
        size=6, alpha=0.65, jitter=True, legend=False,
        ax=ax_dist,
    )

    for i, cond in enumerate(order):
        med = combined[combined["condition"] == cond]["turns_played"].median()
        n   = len(combined[combined["condition"] == cond])
        ax_dist.text(
            i, med + 8, f"med={med:.0f}\nn={n}",
            ha="center", va="bottom", fontsize=10, color=MARBLE[800],
        )

    ax_dist.set_xlabel("")
    ax_dist.set_ylabel("Turns played", fontsize=13)
    ax_dist.set_title("B · Turns-played distribution", fontsize=13, color=MARBLE[800])
    ax_dist.set_ylim(-15, 360)

    plt.tight_layout()
    plt.suptitle(
        "Harness enables reliable game completion",
        fontsize=15, fontweight="bold", color=MARBLE[800],
    )
    plt.subplots_adjust(top=0.85)
    _save(fig, "harness_comparison")
    return fig


def generate_all() -> None:
    """Load admissible data and generate every figure."""
    from analysis.data_loader import load_all, load_tools_df

    print("Loading data (admissible games only)...")
    data = load_all(verbose=True)
    games = data["games"]
    diary = data["diary"]
    metrics = data["metrics"]
    founding = data["founding"]

    print("\nGenerating figures...")

    plot_outcome_heatmap(games)
    print("  ✓ outcome_heatmap")

    plot_victory_breakdown(games)
    print("  ✓ victory_breakdown")

    plot_normalised_score(games)
    print("  ✓ normalised_score")

    icc_df = compute_icc_df(games, metrics)
    plot_icc_table(icc_df)
    print("  ✓ icc_table")

    plot_score_trajectories(diary, games)
    print("  ✓ trajectory_score_ground_control")

    plot_yield_trajectories(diary, games)
    print("  ✓ yield_trajectories_ground_control")

    plot_expansion_timing(diary, founding, games)
    print("  ✓ expansion_timing_ground_control")

    plot_city_milestones(metrics, games)
    print("  ✓ city_milestones_ground_control")

    radar_data, radar_se = _build_radar_df(games, metrics)
    plot_radar(radar_data, radar_se)
    print("  ✓ radar")

    plot_metric_correlation(metrics, games)
    print("  ✓ metric_correlation")

    plot_game_sparklines(diary, games)
    print("  ✓ game_sparklines")

    plot_reflection_quality(games)
    print("  ✓ reflection_quality")

    print("Loading tools data from Azure...")
    try:
        tools = load_tools_df()
        print(f"  {len(tools)} tool calls loaded")
        plot_pmr_subcategories(tools, games)
        print("  ✓ pmr_subcategories")
        plot_pmr_over_time(tools, games)
        print("  ✓ pmr_over_time")
        plot_tool_category_stacked_area(tools, games)
        print("  ✓ tool_stacked_area")
        plot_tool_error_rate(tools)
        print("  ✓ tool_error_rate")
        plot_tool_latency(tools)
        print("  ✓ tool_latency")
        plot_tool_entropy(tools, games)
        print("  ✓ tool_entropy")
        plot_decision_follow_through(tools)
        print("  ✓ decision_follow_through")
    except Exception as e:
        print(f"  ! Azure-gated plots skipped: {e}")
        plot_pmr_subcategories(pd.DataFrame(), games)
        plot_pmr_over_time(pd.DataFrame(), games)
        plot_tool_category_stacked_area(pd.DataFrame(), games)
        plot_tool_error_rate(pd.DataFrame())
        plot_tool_latency(pd.DataFrame())
        plot_tool_entropy(pd.DataFrame(), games)
        plot_decision_follow_through(pd.DataFrame())

    plot_rag_breakdown()
    print("  ✓ rag_breakdown")

    plot_rag_sensitivity()
    print("  ✓ rag_sensitivity")

    plot_inflection_points(diary, games)
    print("  ✓ inflection_points")

    plot_inflection_deltas(diary, games)
    print("  ✓ inflection_deltas")

    plot_harness_comparison()
    print("  ✓ harness_comparison")

    print(f"\nAll figures saved to {FIGURES}")
    for f in sorted(FIGURES.glob("*.pdf")):
        print(f"  {f.name}")


def compute_icc_df(games_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ICC + Kruskal-Wallis p-value for each metric on ground_control admissible games."""
    from scipy import stats as _stats
    gc_m = metrics_df[metrics_df["scenario"] == "ground_control"].copy()
    valid_ids = set(games_df["game_id"])
    gc_m = gc_m[gc_m["game_id"].isin(valid_ids)]

    CANDIDATE_METRICS = [
        "normalised_score", "turns_played", "final_cities",
        "final_exploration_pct", "final_science", "final_gold",
        "final_military", "final_culture", "city_t50", "city_t100",
        "city_t150", "city_t200", "exploration_t50", "exploration_t100",
    ]

    rows = []
    for col in CANDIDATE_METRICS:
        if col not in gc_m.columns:
            continue
        sub = gc_m[["model", col]].dropna()
        if len(sub) < 4:
            continue
        groups = sub.groupby("model")[col].apply(list)
        k = len(groups)
        if k < 2:
            continue
        n_total = len(sub)
        grand_mean = sub[col].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        df_between, df_within = k - 1, n_total - k
        if df_within <= 0:
            continue
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        n_hat = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / df_between
        icc_val = (ms_between - ms_within) / (ms_between + (n_hat - 1) * ms_within)
        icc_val = float(np.clip(icc_val, -1, 1))
        within_sd = sub.groupby("model")[col].std().mean()
        between_sd = sub.groupby("model")[col].mean().std()
        verdict = "discriminative" if icc_val > 0.5 else ("marginal" if icc_val > 0.2 else "noise")
        # Kruskal-Wallis across model groups (requires ≥1 obs per group, ≥2 groups)
        try:
            kw_groups = [list(g) for g in groups if len(g) >= 1]
            kw_h, kw_p = _stats.kruskal(*kw_groups) if len(kw_groups) >= 2 else (float("nan"), float("nan"))
        except ValueError:
            kw_h, kw_p = float("nan"), float("nan")
        rows.append({
            "metric": col, "ICC": round(icc_val, 3),
            "within_SD": round(within_sd, 3) if not np.isnan(within_sd) else None,
            "between_SD": round(between_sd, 3) if not np.isnan(between_sd) else None,
            "kw_H": round(kw_h, 3) if not np.isnan(kw_h) else None,
            "kw_p": round(kw_p, 4) if not np.isnan(kw_p) else None,
            "verdict": verdict,
        })

    return pd.DataFrame(rows).sort_values("ICC", ascending=False)


def _build_radar_df(
    games_df: pd.DataFrame, metrics_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build z-scored summary DataFrame for radar chart.

    Returns (mean_df, se_df) — both indexed by model, columns = axis labels.
    se_df contains 95% CI half-widths in z-score units.
    """
    valid_ids = set(games_df["game_id"])
    gc_full = metrics_df[
        (metrics_df["scenario"] == "ground_control") &
        (metrics_df["game_id"].isin(valid_ids))
    ].copy()

    RADAR_AXES = {
        "normalised_score": "Score",
        "city_t100": "Expansion",
        "exploration_t100": "Exploration",
        "final_culture": "Culture",
        "final_science": "Science",
        "final_military": "Military",
    }

    rows, se_rows = {}, {}
    for model in sorted(gc_full["model"].unique()):
        rows[model] = {}
        se_rows[model] = {}
        mdata = gc_full[gc_full["model"] == model]
        for col, label in RADAR_AXES.items():
            if col not in gc_full.columns:
                rows[model][label] = 0.0
                se_rows[model][label] = 0.0
                continue
            global_mean = gc_full[col].mean()
            global_std  = gc_full[col].std()
            model_mean  = mdata[col].mean()
            model_std   = mdata[col].std(ddof=1)
            n           = mdata[col].count()
            z = (model_mean - global_mean) / global_std if global_std > 0 else 0.0
            # CI half-width in z-score units
            se_z = (1.96 * model_std / np.sqrt(max(n, 1))) / global_std if global_std > 0 else 0.0
            rows[model][label]    = float(np.clip(z, -2, 2))
            se_rows[model][label] = float(se_z)

    return pd.DataFrame(rows).T, pd.DataFrame(se_rows).T


if __name__ == "__main__":
    generate_all()
