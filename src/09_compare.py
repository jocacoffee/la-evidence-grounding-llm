"""Step 9 - Experiment 8: comparison of the four conditions, figure 5, metrics.json, case studies.

Per-condition values are means of per-output metrics (outputs where the metric is undefined, e.g. no
claims, are excluded and counted). 95% CIs: percentile bootstrap over students (1000 resamples).
Paired differences: Wilcoxon signed-rank on students where both outputs define the metric, Holm-adjusted.
Case studies are selected by fixed, documented rules (no manual cherry-picking).
Usage: python src/09_compare.py [main|pilot]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SET = sys.argv[1] if len(sys.argv) > 1 else "main"
CFG = json.load(open(ROOT / "configs" / "config.json"))
SEED = CFG["seed"]
CONDS = ["C1_llm_only", "C2_student_data", "C3_rag", "C4_shap_rag"]
NAMES = {"C1_llm_only": "LLM Only", "C2_student_data": "LLM + Student Data", "C3_rag": "LLM + RAG",
         "C4_shap_rag": "LLM + SHAP + RAG"}
METRICS = [("unsupported_claim_rate", "Unsupported claim rate", "lower"),
           ("unsupported_fact_rate", "Unsupported rate (observed_facts only)", "lower"),
           ("evidence_coverage", "Evidence coverage", "higher"),
           ("factual_consistency", "Factual consistency", "higher"),
           ("shap_consistency", "SHAP consistency (top-3)", "higher"),
           ("recommendation_specificity", "Recommendation specificity", "higher")]

df = pd.read_csv(ROOT / "results" / f"eval_per_output_{SET}.csv")
audit = {(r["enrolment_id"], r["condition"]): r for r in map(json.loads, open(ROOT / "results" / f"eval_audit_{SET}.jsonl"))}
outs = {(o["enrolment_id"], o["condition"]): o for o in map(json.loads, open(ROOT / "results" / f"llm_outputs_{SET}.jsonl"))}
rng = np.random.default_rng(SEED)
students = df.enrolment_id.unique()
boot_idx = [rng.choice(len(students), len(students)) for _ in range(1000)]


def ci(values_by_student):
    v = values_by_student.reindex(students).values
    bs = [np.nanmean(v[i]) for i in boot_idx]
    return np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


rows = []
for c in CONDS:
    d = df[df.condition == c].set_index("enrolment_id")
    r = {"condition": NAMES[c], "condition_id": c, "n_outputs": len(d), "json_valid": d.json_ok.mean(),
         "mean_claims_per_output": d.n_claims.mean(), "outputs_without_claims": int((d.n_claims == 0).sum()),
         "outputs_without_recommendations": int((d.n_recommendations == 0).sum())}
    for m, _, _ in METRICS:
        lo, hi = ci(d[m])
        r[m] = d[m].mean()
        r[f"{m}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        r[f"{m}_n_defined"] = int(d[m].notna().sum())
    r["micro_unsupported_claim_rate"] = d.n_unsupported.sum() / max(d.n_claims.sum(), 1)
    r["micro_evidence_coverage"] = d.n_linked.sum() / max(d.n_claims.sum(), 1)
    r["micro_factual_consistency"] = d.n_numeric_consistent.sum() / max(d.n_numeric.sum(), 1)
    r["n_numeric_mentions"] = int(d.n_numeric.sum())
    r["rec_problem"], r["rec_evidence"], r["rec_action"] = d.rec_problem.mean(), d.rec_evidence.mean(), d.rec_action.mean()
    r["citation_invalid_share"] = d.n_cited_invalid.sum() / max(d.n_cited_ids.sum(), 1)
    r["insufficiency_declared"] = d.insufficiency_declared.mean()
    r["asserted_non_observable_share"] = d.asserted_non_observable.mean()
    r["causal_language_rate"] = d.causal_language_rate.mean()
    rows.append(r)
comp = pd.DataFrame(rows)
comp.to_csv(ROOT / "results" / ("llm_comparison.csv" if SET == "main" else f"llm_comparison_{SET}.csv"), index=False)

# paired tests
tests = []
pairs = [("C4_shap_rag", "C3_rag"), ("C4_shap_rag", "C2_student_data"), ("C3_rag", "C2_student_data"),
         ("C4_shap_rag", "C1_llm_only"), ("C3_rag", "C1_llm_only"), ("C2_student_data", "C1_llm_only")]
for m, _, _ in METRICS:
    for a, b in pairs:
        x = df[df.condition == a].set_index("enrolment_id")[m]
        y = df[df.condition == b].set_index("enrolment_id")[m]
        j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
        if len(j) < 5 or (j.a - j.b).abs().sum() == 0:
            p = np.nan
        else:
            p = wilcoxon(j.a, j.b, zero_method="zsplit").pvalue
        tests.append({"metric": m, "a": a, "b": b, "n_pairs": len(j), "mean_diff_a_minus_b":
                      (j.a - j.b).mean() if len(j) else np.nan, "p_wilcoxon": p})
tests = pd.DataFrame(tests)
for m in tests.metric.unique():  # Holm within metric
    s = tests[(tests.metric == m) & tests.p_wilcoxon.notna()].sort_values("p_wilcoxon")
    k = len(s)
    adj = np.maximum.accumulate([min(1, p * (k - i)) for i, p in enumerate(s.p_wilcoxon)])
    tests.loc[s.index, "p_holm"] = adj
tests.to_csv(ROOT / "results" / f"llm_paired_tests_{SET}.csv", index=False)

pd.set_option("display.width", 250)
print(comp[["condition", "n_outputs", "json_valid", "mean_claims_per_output", "outputs_without_claims"] +
           [m for m, _, _ in METRICS]].round(3).to_string())
print(comp[["condition", "micro_unsupported_claim_rate", "micro_evidence_coverage", "micro_factual_consistency",
            "n_numeric_mentions", "rec_problem", "rec_evidence", "rec_action", "citation_invalid_share",
            "insufficiency_declared", "asserted_non_observable_share", "causal_language_rate"]].round(3).to_string())
print(tests.round(4).to_string())
if SET != "main":
    sys.exit()

# ---------------------------------------------------------------- Figure 5
fig, axes = plt.subplots(1, 5, figsize=(12, 3.0), sharey=False)
colors = ["#adb5bd", "#74c0fc", "#339af0", "#1864ab"]
for ax, (m, label, better) in zip(axes, [x for x in METRICS if x[0] != "unsupported_fact_rate"]):
    vals = comp[m].values
    cis = [tuple(map(float, s.strip("[]").split(","))) if "nan" not in s else (np.nan, np.nan)
           for s in comp[f"{m}_ci95"]]
    err = np.array([[v - lo for v, (lo, hi) in zip(vals, cis)], [hi - v for v, (lo, hi) in zip(vals, cis)]])
    ax.bar(range(4), np.nan_to_num(vals), color=colors, yerr=np.nan_to_num(err), capsize=3)
    top = 0.15 if m == "unsupported_claim_rate" else 1.05
    for i, v in enumerate(vals):
        y = 0 if np.isnan(v) else v + (err[1][i] if not np.isnan(err[1][i]) else 0)
        ax.text(i, y + top * 0.02, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["C1", "C2", "C3", "C4"], fontsize=8)
    ax.set_ylim(0, top)
    ax.set_title(f"{label}\n({better} is better)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.02, "C1 LLM only · C2 LLM + student data · C3 LLM + RAG · C4 LLM + SHAP + RAG   "
         f"(n = {len(students)} students per condition; error bars: 95% bootstrap CI)", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig5_llm_condition_comparison.png", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- metrics.json
ml = pd.read_csv(ROOT / "results" / "ml_results.csv")
ret = pd.read_csv(ROOT / "results" / "retrieval_metrics.csv")
metrics = {
    "ml": ml.to_dict(orient="records"),
    "shap": json.load(open(ROOT / "results" / "shap_config.json")),
    "shap_global_top10": pd.read_csv(ROOT / "results" / "shap_global_importance.csv").head(10).to_dict(orient="records"),
    "retrieval": ret.to_dict(orient="records"),
    "llm_comparison": comp.to_dict(orient="records"),
    "llm_paired_tests": tests.to_dict(orient="records"),
    "llm_config": {**CFG["llm"], "reasoning_effort": "none", "n_students": int(len(students))},
}
json.dump(metrics, open(ROOT / "results" / "metrics.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- case studies (rule-based selection)
W = df.pivot(index="enrolment_id", columns="condition")


def show(eid, cond):
    o, a = outs[(eid, cond)], audit[(eid, cond)]
    ctx = o["user_prompt"].split("CONTEXT", 1)[1].split("\n\nDo not introduce")[0].split("\n\nReturn")[0]
    lines = [f"**{NAMES[cond]}** (`{cond}`)\n", "<details><summary>Context given to the LLM</summary>\n",
             "```\nCONTEXT" + ctx + "\n```\n</details>\n", "Generated output (JSON):\n",
             "```json\n" + json.dumps(o["parsed"], indent=1) + "\n```\n", "Deterministic audit of claim units:\n",
             "| kind | statement | supported by context | linked to store | reasons |", "|---|---|---|---|---|"]
    for u in a["audit"]:
        if u.get("is_claim"):
            lines.append(f"| {u['kind']} | {u['text'].replace('|', '/')} | {u['supported']} | {u['linked']} | "
                         f"{'; '.join(u['reasons']) or '—'} |")
    lines.append(f"\nPer-output metrics: UCR={a['unsupported_claim_rate']}, coverage={a['evidence_coverage']}, "
                 f"factual consistency={a['factual_consistency']}, SHAP consistency={a['shap_consistency']:.2f} "
                 f"(top-3: {', '.join(a['shap_top3'])}; mentioned: {', '.join(a['shap_hits']) or 'none'}), "
                 f"specificity={a['recommendation_specificity']}\n")
    return "\n".join(lines)


md = ["# Case studies (automatically selected)\n",
      f"Set: {SET}. Selection rules are fixed in `src/09_compare.py`; ties broken by enrolment_id. "
      "Metrics are deterministic (see EXPERIMENT_REPORT.md). The true final_result is shown only for the "
      "reader; it was never given to the LLM.\n"]
ucr = W["unsupported_claim_rate"]
labels = pd.read_csv(ROOT / "data" / "llm_sample.csv").set_index("enrolment_id").true_final_result_NOT_FOR_LLM


def case(title, rule, eid, conds):
    md.append(f"\n## {title}\n\n*Selection rule:* {rule}\n")
    if eid is None:
        md.append("**No output in the evaluation sample satisfies this rule.** Reported as such, not constructed.\n")
        return
    md.append(f"Enrolment `{eid}` (true final_result: {labels.get(eid, '?')})\n")
    for c in conds:
        md.append(show(eid, c))


# 1. RAG reduces an unsupported claim
d = (ucr["C2_student_data"] - ucr[["C3_rag", "C4_shap_rag"]].min(1)).dropna()
d = d[d > 0].sort_values(ascending=False, kind="stable")
case("1. Retrieval reduces unsupported claims",
     "largest positive difference UCR(C2) - min(UCR(C3), UCR(C4)) over students with claims in all three",
     d.index[0] if len(d) else None, ["C2_student_data", "C3_rag", "C4_shap_rag"])
# 2. LLM without retrieval produces an incorrect claim
cand = []
for c in ["C1_llm_only", "C2_student_data"]:
    for eid in students:
        a = audit[(eid, c)]
        bad = [u for u in a["audit"] if u.get("is_claim") and (u.get("numbers_inconsistent_with_store") or
                                                            "non_observable" in u.get("reasons", []))]
        if bad:
            cand.append((c, eid, len(bad)))
cand.sort(key=lambda t: (t[0], -t[2], t[1]))
case("2. LLM without retrieved evidence produces an incorrect / unsupported claim",
     "first C1 output (else C2) containing a number inconsistent with the Evidence Store or an asserted "
     "non-observable attribute; most such claims first",
     cand[0][1] if cand else None, [cand[0][0], "C4_shap_rag"] if cand else [])
# 3. SHAP improves the explanation
s = (W["shap_consistency"]["C4_shap_rag"] - W["shap_consistency"]["C3_rag"]).sort_values(ascending=False, kind="stable")
case("3. SHAP improves the explanation", "largest SHAP-consistency gain C4 - C3 (then lowest UCR in C4)",
     s.index[0] if len(s) and s.iloc[0] > 0 else None, ["C3_rag", "C4_shap_rag"])
# 4. insufficient evidence
ev = pd.read_csv(ROOT / "data" / "student_evidence.csv").set_index("enrolment_id")
ins = [e for e in students if pd.isna(ev.loc[e, "fv_submission_rate"]) or ev.loc[e, "fv_total_clicks"] == 0]
ins.sort()
case("4. Insufficient evidence", "first student (by id) with no assessment due before the cutoff "
     "(submission_rate not applicable) or no VLE activity; outputs of C3 and C4 shown", ins[0] if ins else None,
     ["C3_rag", "C4_shap_rag"])
open(ROOT / "results" / "case_studies.md", "w").write("\n".join(md))

# verification sheet: every claim flagged unsupported / not linked, for manual audit of the evaluator
sheet = []
for (eid, c), a in audit.items():
    for u in a["audit"]:
        if u.get("is_claim") and (not u["supported"] or not u["linked"]):
            sheet.append({"enrolment_id": eid, "condition": c, "kind": u["kind"], "statement": u["text"],
                          "supported_by_context": u["supported"], "linked_to_store": u["linked"],
                          "reasons": "; ".join(u["reasons"]),
                          "numbers_inconsistent_with_store": " ".join(u["numbers_inconsistent_with_store"]),
                          "manual_verdict (correct flag / evaluator error)": "", "manual_note": ""})
pd.DataFrame(sheet).sort_values(["condition", "enrolment_id"]).to_csv(
    ROOT / "results" / "evaluator_verification_sheet.csv", index=False)
print("case studies written")
