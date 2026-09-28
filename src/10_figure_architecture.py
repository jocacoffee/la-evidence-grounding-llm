"""Figure 1 - pipeline and evidence conditions.

Drawn at its final printed size (JLA text width, about 7 in), so the font sizes below are the sizes in the PDF.
Two-row "snake" layout: prediction pipeline on the first row, generation and evaluation on the second (right to left).
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, SUB, EDGE = "#1f1f1e", "#3b3a36", "#4a4944"
FILL = {"data": "#eef0f2", "la": "#e3effa", "ev": "#fdf0dd", "llm": "#e4f5e7", "eval": "#f3e8f7"}

fig = plt.figure(figsize=(7.0, 3.78))  # height matched to the drawn content (y 4.8-59.5)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(4.8, 59.5)
ax.axis("off")

BW, BH = 17.6, 13.0          # box size (axis units)
XS = [0.8 + i * 20.0 for i in range(5)]
Y1, Y2 = 45.0, 26.0          # bottom of row 1 and row 2


def box(x, y, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=FILL[kind], ec=EDGE, lw=0.8))
    ax.text(x + BW / 2, y + BH - 1.5, title, ha="center", va="top", fontsize=8.6, weight="bold", color=INK)
    ax.text(x + BW / 2, y + BH - 5.2, "\n".join(lines), ha="center", va="top", fontsize=7.0, color=SUB,
            linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.0, color=EDGE,
                                 shrinkA=0, shrinkB=0))


row1 = [("OULAD", ["7 tables, 22 module-", "presentations", "(days from start)"], "data"),
        ("Temporal LA", ["27 features from", "events with day < 28", "(end of week 4)"], "la"),
        ("Risk prediction", ["XGBoost, 4 classes", "train 13B/13J/14B", "test 14J"], "la"),
        ("Explainable ML", ["permutation SHAP on", "P(Fail)+P(Withdrawn)", "(association only)"], "la"),
        ("Evidence store", ["E: observed facts", "M: model inference", "X: SHAP explanation"], "ev")]
for x, (t, l, k) in zip(XS, row1):
    box(x, Y1, t, l, k)
for x in XS[:-1]:
    arrow(x + BW + 0.4, Y1 + BH / 2, x + 20 - 0.4, Y1 + BH / 2)

# row 2, flowing right to left under the evidence store
row2 = [("Retrieval", ["filter: student, cutoff", "rank: cohort deviation", "top-5 E items"], "ev"),
        ("Local LLM", ["Qwen3.5-9B (main)", "Gemma-4-E4B,", "Qwen3-14B; T = 0"], "llm"),
        ("Recommendation", ["facts, inferences,", "actions + rationale,", "evidence limitations"], "llm"),
        ("Evaluation", ["traceability, coverage,", "consistency, individ.,", "robustness to error"], "eval")]
xs2 = XS[::-1][:4]
for x, (t, l, k) in zip(xs2, row2):
    box(x, Y2, t, l, k)
arrow(XS[4] + BW / 2, Y1 - 0.4, XS[4] + BW / 2, Y2 + BH + 0.4)       # evidence store -> retrieval
for a, b in zip(xs2[:-1], xs2[1:]):
    arrow(a - 0.4, Y2 + BH / 2, b + BW + 0.4, Y2 + BH / 2)
ax.text(XS[0] + BW / 2, Y2 + BH / 2, "the context given\nto the LLM varies\nby condition\n(C1–C7, below)",
        ha="center", va="center", fontsize=7.4, color=SUB, style="italic")

# conditions panel
ax.text(0.8, 21.5, "Evidence conditions (only the LLM's context differs)", fontsize=8.4, weight="bold",
        color=INK, va="top")
conds = [("C1", "identifiers only (no evidence)"),
         ("C2", "full record: 27 features as JSON"),
         ("C3", "top-5 retrieved E items (RAG)"),
         ("C4", "C3 + M1 risk + X1–X3 SHAP"),
         ("C5", "C3 + M1 risk (ablation, no SHAP)"),
         ("C6", "all 27 E items with IDs (control)"),
         ("C7", "5 random E items (control)")]
col_x = [0.8, 51.0]
for i, (c, txt) in enumerate(conds):
    col, row = (0, i) if i < 4 else (1, i - 4)
    y = 17.0 - row * 3.5
    ax.text(col_x[col], y, c, fontsize=8.0, weight="bold", color=INK, va="center")
    ax.text(col_x[col] + 4.6, y, txt, fontsize=8.0, color=SUB, va="center")

for ext in ["pdf", "png"]:
    fig.savefig(ROOT / "figures" / f"fig1_architecture.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
shutil.copy(ROOT / "figures" / "fig1_architecture.pdf", ROOT / "paper_jla" / "fig1_architecture.pdf")
