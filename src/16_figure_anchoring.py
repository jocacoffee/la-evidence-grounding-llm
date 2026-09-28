"""Figure: anchoring in the pre-registered confirmatory test (Qwen3.5-9B, 80 new students, C2 vs C4).

Dumbbell chart per group (missed at-risk / false alarms): share of outputs that propose outreach, state low risk,
state high risk. Paired students only (the invalid-JSON pair is excluded, as pre-registered).
Pre-registered hypotheses are annotated with their p-values; the other rows are descriptive.
Output: figures/fig6_anchoring.{pdf,png} (and a copy in paper_jla/)
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_extract import action_cats, stated_band  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
grp = pd.read_csv(ROOT / "data" / "anchoring_sample.csv").set_index("enrolment_id").group
conf = pd.read_csv(R / "anchoring_confirmatory.csv")

rows = []
for o in map(json.loads, open(R / "llm_outputs_anchoring.jsonl")):
    p = o["parsed"] if o["json_ok"] else None
    rows.append({"eid": o["enrolment_id"], "cond": o["condition"], "group": grp[o["enrolment_id"]], "ok": o["json_ok"],
                 "outreach": p is not None and "outreach" in action_cats(p),
                 "low": p is not None and stated_band(p) == "low", "high": p is not None and stated_band(p) == "high"})
D = pd.DataFrame(rows)
bad = set(D[~D.ok].eid)
D = D[~D.eid.isin(bad)]  # paired analysis: drop both outputs of a student with an invalid output

C2_COL, C4_COL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dc"
MEASURES = [("outreach", "Proposes contacting\nthe student"), ("low", "States low risk"), ("high", "States high risk")]
PANELS = [("missed_at_risk", "At-risk students the model missed"), ("false_alarm", "False alarms (not at risk)")]
HYP = {("missed_at_risk", "outreach"): "H1", ("missed_at_risk", "low"): "H2", ("false_alarm", "high"): "H3"}

plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
for ax, (g, title) in zip(axes, PANELS):
    d = D[D.group == g]
    n = d.eid.nunique()
    for y, (m, _) in enumerate(MEASURES):
        c2 = d[d.cond == "C2_student_data"][m].mean() * 100
        c4 = d[d.cond == "C4_shap_rag"][m].mean() * 100
        ax.plot([c2, c4], [y, y], color="#b9b8b0", lw=2, zorder=1, solid_capstyle="round")
        ax.scatter(c2, y, s=46, facecolor="white", edgecolor=C2_COL, linewidth=2, marker="o", zorder=3)
        ax.scatter(c4, y, s=46, color=C4_COL, marker="s", zorder=3, edgecolor="white", linewidth=1)
        left, right = (c2, c4) if c2 <= c4 else (c4, c2)
        pct = lambda v: f"{int(v + 0.5)}%"  # round half up, as in the text and tables
        ax.text(left - 4, y, pct(left), ha="right", va="center", color=INK, fontsize=7)
        ax.text(right + 4, y, pct(right), ha="left", va="center", color=INK, fontsize=7)
        h = HYP.get((g, m))
        if h:
            r = conf[conf.hypothesis.str.startswith(h)].iloc[0]
            ptxt = f"{h}, p = {r.p_holm:.3f}" if h != "H1" else f"{h}, p = {r.p_exact_mcnemar:.3f}"
        else:
            ptxt = "descriptive"
        ax.text(103, y, ptxt, ha="left", va="center", fontsize=6.5, color=INK if h else MUTED,
                style="normal" if h else "italic")
    ax.set_xlim(-14, 100)
    ax.set_ylim(len(MEASURES) - 0.45, -0.55)
    ax.set_yticks(range(len(MEASURES)))
    ax.set_yticklabels([lab for _, lab in MEASURES], color=INK)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_title(f"{title} (n = {n})", fontsize=8, color=INK, loc="left", pad=6)
    ax.set_xlabel("share of outputs", fontsize=7, color=MUTED)
h1 = plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=C2_COL, markeredgewidth=2, ms=6,
                label="C2: full student record (no model output)")
h2 = plt.Line2D([], [], marker="s", ls="", color=C4_COL, ms=6, label="C4: retrieved evidence + model risk + SHAP")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.08, 0.93, 1))
for ext in ["pdf", "png"]:
    fig.savefig(ROOT / "figures" / f"fig6_anchoring.{ext}", dpi=300, bbox_inches="tight")
shutil.copy(ROOT / "figures" / "fig6_anchoring.pdf", ROOT / "paper_jla" / "fig6_anchoring.pdf")
print(D.groupby(["group", "cond"])[["outreach", "low", "high"]].mean().round(3), "\nexcluded:", bad)
