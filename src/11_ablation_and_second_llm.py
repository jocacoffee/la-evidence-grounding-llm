"""Step 11 - analyses added after the main run (frozen v1 evaluator throughout).

(a) Ablation: C5 = retrieved evidence + predicted risk (M1), no SHAP (Qwen3.5-9B, same 100 students).
    Paired comparisons C5 vs C3 (effect of adding the risk estimate) and C4 vs C5 (effect of adding SHAP).
(b) Second LLM: Gemma-4-E4B on C1-C4, same students, prompts and evaluator.
Outputs: results/ablation_comparison.csv, results/ablation_paired_tests.csv, results/ablation_claims_by_kind.csv,
         results/second_llm_comparison.csv, results/second_llm_paired_tests.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SEED = 42
M = ["unsupported_claim_rate", "unsupported_fact_rate", "evidence_coverage", "factual_consistency",
     "shap_consistency", "recommendation_specificity"]


def summarise(df, conds, label):
    students = df.enrolment_id.unique()
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(len(students), len(students)) for _ in range(1000)]
    rows = []
    for c in conds:
        d = df[df.condition == c].set_index("enrolment_id").reindex(students)
        r = {"llm": label, "condition": c, "n_outputs": int(d.json_ok.notna().sum()), "json_valid": d.json_ok.mean(),
             "claims_per_output": d.n_claims.mean(), "outputs_without_claims": int((d.n_claims == 0).sum()),
             "outputs_without_recommendations": int((d.n_recommendations == 0).sum()),
             "insufficiency_declared": d.insufficiency_declared.mean(),
             "asserted_non_observable_share": d.asserted_non_observable.mean(),
             "citation_invalid_share": d.n_cited_invalid.sum() / max(d.n_cited_ids.sum(), 1)}
        for m in M:
            v = d[m].values
            bs = [np.nanmean(v[i]) if np.isfinite(v[i]).any() else np.nan for i in boots]
            r[m] = np.nanmean(v) if np.isfinite(v).any() else np.nan
            r[f"{m}_ci95"] = "n/a" if np.isnan(r[m]) else f"[{np.nanpercentile(bs, 2.5):.3f}, {np.nanpercentile(bs, 97.5):.3f}]"
        rows.append(r)
    return pd.DataFrame(rows)


def paired(df, pairs, label):
    out = []
    for m in M:
        sub = []
        for a, b in pairs:
            x = df[df.condition == a].set_index("enrolment_id")[m]
            y = df[df.condition == b].set_index("enrolment_id")[m]
            j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
            p = np.nan if len(j) < 5 or (j.a - j.b).abs().sum() == 0 else wilcoxon(j.a, j.b, zero_method="zsplit").pvalue
            sub.append({"llm": label, "metric": m, "a": a, "b": b, "n_pairs": len(j),
                        "mean_diff_a_minus_b": (j.a - j.b).mean() if len(j) else np.nan, "p_wilcoxon": p})
        s = pd.DataFrame(sub)
        ok = s.p_wilcoxon.notna()
        order = s[ok].sort_values("p_wilcoxon").index
        k = len(order)
        adj = np.maximum.accumulate([min(1, s.loc[i, "p_wilcoxon"] * (k - n)) for n, i in enumerate(order)])
        s.loc[order, "p_holm"] = adj
        out.append(s)
    return pd.concat(out)


def claims_by_kind(audit_file, conds):
    rows = []
    for r in map(json.loads, open(audit_file)):
        if r["condition"] not in conds:
            continue
        for u in r["audit"]:
            if u.get("is_claim"):
                rows.append({"condition": r["condition"], "kind": u["kind"], "supported": u["supported"],
                             "risk_only": u["reasons"] == ["risk_assertion_without_model_output"]})
    d = pd.DataFrame(rows)
    return d.groupby(["condition", "kind"]).agg(n=("supported", "size"),
                                                unsupported_rate=("supported", lambda s: 1 - s.mean())).reset_index()


pd.set_option("display.width", 250)
# ---------------------------------------------------------------- (a) ablation
main = pd.read_csv(R / "eval_per_output_main.csv")
abl_f = R / "eval_per_output_ablation.csv"
if abl_f.exists():
    # (a1) within-session ablation: C3, C5, C4 all generated in the same (second) server session
    abl = pd.read_csv(abl_f)
    conds = ["C3_rag", "C5_rag_risk", "C4_shap_rag"]
    s_ = summarise(abl, conds, "qwen3.5-9b (session 2)")
    t = paired(abl, [("C5_rag_risk", "C3_rag"), ("C4_shap_rag", "C5_rag_risk"), ("C4_shap_rag", "C3_rag")],
               "qwen3.5-9b (session 2)")
    k = claims_by_kind(R / "eval_audit_ablation.jsonl", conds)
    # (a2) session-to-session variability: same condition, same prompts, main session vs session 2
    var = []
    for c in ["C3_rag", "C4_shap_rag"]:
        x = main[main.condition == c].set_index("enrolment_id")
        y = abl[abl.condition == c].set_index("enrolment_id")
        ids = x.index.intersection(y.index)
        outs1 = {json.loads(l)["enrolment_id"]: json.loads(l)["raw_content"] for l in open(R / "llm_outputs_main.jsonl") if f'"{c}"' in l}
        outs2 = {json.loads(l)["enrolment_id"]: json.loads(l)["raw_content"] for l in open(R / "llm_outputs_ablation.jsonl") if f'"{c}"' in l}
        row = {"condition": c, "n": len(ids), "identical_outputs": np.mean([outs1[i] == outs2[i] for i in ids])}
        for m in M[:4]:
            j = pd.concat([x.loc[ids, m], y.loc[ids, m]], axis=1, keys=["s1", "s2"]).dropna()
            row[f"{m}_session1"] = j.s1.mean()
            row[f"{m}_session2"] = j.s2.mean()
            row[f"{m}_p_wilcoxon"] = np.nan if (j.s1 - j.s2).abs().sum() == 0 else wilcoxon(j.s1, j.s2, zero_method="zsplit").pvalue
        var.append(row)
    var = pd.DataFrame(var)
    s_.to_csv(R / "ablation_comparison.csv", index=False)
    t.to_csv(R / "ablation_paired_tests.csv", index=False)
    k.to_csv(R / "ablation_claims_by_kind.csv", index=False)
    var.to_csv(R / "session_variability.csv", index=False)
    print("=== ABLATION (within session 2)"); print(s_[["condition"] + M + ["claims_per_output"]].round(3).to_string())
    print(s_[["condition"] + [f"{m}_ci95" for m in M[:4]]].to_string())
    print(t[t.metric.isin(M[:4])].round(4).to_string()); print(k.round(3).to_string())
    print("=== SESSION VARIABILITY"); print(var.round(4).T.to_string())

# ---------------------------------------------------------------- (b) second LLM
others = [(lab, st) for lab, st in [("gemma-4-e4b", "gemma"), ("qwen3-14b", "qwen14b")]
          if (R / f"eval_per_output_{st}.csv").exists()]
if others:
    conds = ["C1_llm_only", "C2_student_data", "C3_rag", "C4_shap_rag"]
    pairs = [("C4_shap_rag", "C3_rag"), ("C4_shap_rag", "C2_student_data"), ("C3_rag", "C2_student_data")]
    sums, tsts = [summarise(main, conds, "qwen3.5-9b")], [paired(main, pairs, "qwen3.5-9b")]
    for lab, st in others:
        d = pd.read_csv(R / f"eval_per_output_{st}.csv")
        sums.append(summarise(d, conds, lab))
        tsts.append(paired(d, pairs, lab))
        print(f"--- claims by kind: {lab}"); print(claims_by_kind(R / f"eval_audit_{st}.jsonl", conds).round(3).to_string())
    s_, t = pd.concat(sums), pd.concat(tsts)
    s_.to_csv(R / "second_llm_comparison.csv", index=False)
    t.to_csv(R / "second_llm_paired_tests.csv", index=False)
    print("=== OTHER LLMs"); print(s_[["llm", "condition", "outputs_without_claims", "outputs_without_recommendations"] + M + ["claims_per_output"]].round(3).to_string())
    print(t[t.metric.isin(M[:4])].round(4).to_string())
