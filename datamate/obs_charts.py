"""Charts for notebooks/observation.ipynb (matplotlib, flat style; ported from Zalo-Track-2 datamate/obs_charts.py).

PA runs from -1 to +1, so it gets a diverging blue/red pair with grey at 0. Outcomes are states: right = good, wrong = critical,
abstain = warning, always with a text label. Models get one fixed categorical colour each, the same in every chart.
"""
import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
POS, NEG, MID = "#2a78d6", "#e34948", "#b9b8b3"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
MODEL_COLORS = ["#2a78d6", "#d9822b", "#7b3fbf", "#0f9d8a", "#c2185b", "#5f6b7a"]

OUTCOME_STYLE = {  # outcome -> (label, colour)
    "correct": ("Right", STATUS["good"]),
    "wrong": ("Wrong value", STATUS["critical"]),
    "wrong_type": ("Wrong answer type", STATUS["critical"]),
    "missed_unanswerable": ("Should be unanswerable", STATUS["serious"]),
    "false_unanswerable": ("Unanswerable by mistake", STATUS["serious"]),
    "abstain": ("Abstain", STATUS["warning"]),
}


def _axes(height, width=9.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title, subtitle=""):
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), loc="left", color=TEXT, fontsize=11, pad=12)


def stat_tiles(items):
    """A row of number tiles: (value, label, note)."""
    tile = ("display:grid;gap:2px;padding:10px 14px;border:1px solid #d9d8d3;border-radius:8px;min-width:150px;"
            "background:#fcfcfb;color:#0b0b0b")
    cells = "".join(
        f"<div style='{tile}'><div style='font-size:22px;font-weight:700;font-variant-numeric:tabular-nums'>{html.escape(v)}</div>"
        f"<div style='font-size:12px;color:#52514e'>{html.escape(label)}</div><div style='font-size:11px;color:#52514e'>{html.escape(note)}</div></div>"
        for v, label, note in items)
    display(HTML(f"<div style='display:flex;flex-wrap:wrap;gap:10px'>{cells}</div>"))


def pa_bars(df, label_col, title, subtitle=""):
    """Horizontal PA bars per group, sorted; blue above 0, red below."""
    df = df.sort_values("pa")
    fig, ax = _axes(max(2.2, 0.28 * len(df) + 1.2))
    colors = [POS if v > 0 else NEG if v < 0 else MID for v in df["pa"]]
    ax.barh(df[label_col].astype(str), df["pa"], color=colors, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("PA (-1 ... +1)", color=TEXT_2, fontsize=9)
    _title(ax, title, subtitle)
    plt.tight_layout()
    plt.show()


def model_pa_bars(df, title, subtitle=""):
    """PA with its per-shop bootstrap 95% interval, one bar per (split, model). df: split, model, pa, lo, hi."""
    splits, models = list(dict.fromkeys(df["split"])), list(dict.fromkeys(df["model"]))
    fig, ax = _axes(0.55 * len(splits) * len(models) + 1.4)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    y, ticks, labels = 0, [], []
    for s in splits:
        for m_i, m in enumerate(models):
            r = df[(df["split"] == s) & (df["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.barh(y, r["pa"], color=MODEL_COLORS[m_i % len(MODEL_COLORS)], height=0.7, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(r["pa"], y, xerr=[[r["pa"] - r["lo"]], [r["hi"] - r["pa"]]], color=TEXT, capsize=3, linewidth=1)
            ax.text(max(r["hi"], r["pa"]) + 0.02, y, f"{r['pa']:.3f}", va="center", color=TEXT, fontsize=9)
            ticks.append(y); labels.append(f"{s} · {m}")
            y += 1
        y += 0.6
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_xlim(min(-0.05, df["lo"].min() - 0.05), 1.05)
    ax.set_xlabel("PA (-1 ... +1), bars: 95% interval resampling whole shops", color=TEXT_2, fontsize=9)
    _title(ax, title, subtitle)
    plt.tight_layout()
    plt.show()


def grouped_bars(table, title, xlabel="PA", xlim=(-1.05, 1.05)):
    """One row per group (index), one bar per model (columns): e.g. PA by answer type for every model."""
    groups, models = list(table.index), list(table.columns)
    h = 0.8 / len(models)
    fig, ax = _axes(max(2.5, 0.25 * len(groups) * len(models) + 1.2))
    for m_i, m in enumerate(models):
        ys = np.arange(len(groups)) + m_i * h
        ax.barh(ys, table[m].values, height=h, color=MODEL_COLORS[m_i % len(MODEL_COLORS)], label=m, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(np.arange(len(groups)) + h * (len(models) - 1) / 2, [str(g) for g in groups])
    ax.invert_yaxis()
    ax.axvline(0, color=TEXT_2, linewidth=1)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _title(ax, title)
    plt.tight_layout()
    plt.show()


def outcome_stack(counts, title):
    """Outcome mix per run as 100% stacked bars. counts: DataFrame, index = run label, columns = OUTCOME_STYLE keys."""
    cols = [k for k in OUTCOME_STYLE if k in counts.columns and counts[k].sum()]
    share = counts[cols].div(counts[cols].sum(axis=1), axis=0)
    fig, ax = _axes(0.5 * len(share) + 1.6)
    left = np.zeros(len(share))
    for k in cols:
        label, color = OUTCOME_STYLE[k]
        ax.barh(share.index, share[k], left=left, color=color, label=label, edgecolor=SURFACE, linewidth=1.5, height=0.65)
        for i, v in enumerate(share[k]):
            if v >= 0.06:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white", fontsize=8)
        left += share[k].values
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=len(cols), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _title(ax, title)
    plt.tight_layout()
    plt.show()


def latency_box(series_by_run, title, xlabel="seconds per question"):
    """Distribution of a per-question quantity (time, tokens) per run."""
    labels = list(series_by_run)
    fig, ax = _axes(0.45 * len(labels) + 1.4)
    bp = ax.boxplot([pd.Series(series_by_run[k]).dropna() for k in labels], vert=False, tick_labels=labels, showfliers=False, patch_artist=True)
    for i, b in enumerate(bp["boxes"]):
        b.set(facecolor=MODEL_COLORS[i % len(MODEL_COLORS)], alpha=0.75, edgecolor=TEXT_2)
    for m in bp["medians"]:
        m.set(color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    _title(ax, title)
    plt.tight_layout()
    plt.show()


def heatmap(mat, title, fmt="{:.0%}", cmap="Blues"):
    """A small labelled matrix (e.g. pairwise answer agreement between models)."""
    fig, ax = plt.subplots(figsize=(1.2 * len(mat.columns) + 2.5, 0.8 * len(mat) + 1.2), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(mat.values.astype(float), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=9)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", color="white" if v > 0.6 else TEXT, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    _title(ax, title)
    plt.tight_layout()
    plt.show()
