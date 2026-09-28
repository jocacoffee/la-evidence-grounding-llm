"""Step 17 - RAG control (added after external review): separates evidence FORMAT from evidence SELECTION.

  C2 full record as JSON (no IDs)            vs  C6 all 27 items as statements with IDs  -> effect of FORMAT
  C6 all 27 items as statements with IDs     vs  C3 top-5 salient items (same format)    -> effect of SELECTION (retrieval)
  C7 5 random items (same format)            vs  C3 top-5 salient items                  -> effect of RANKING (Qwen3.5 only)
C2/C3 come from the main runs; C6/C7 from the control runs (different server session; session effects on
aggregate metrics were < 0.5 percentage points, EXPERIMENT_REPORT §8b-iii). Frozen v1 evaluator throughout.
Outputs: results/rag_control_comparison.csv, results/rag_control_tests.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SETS = {"qwen3.5-9b": ("main", "control_qwen35"), "gemma-4-e4b": ("gemma", "control_gemma"),
        "qwen3-14b": ("qwen14b", "control_qwen14b")}
M = ["unsupported_claim_rate", "unsupported_fact_rate", "evidence_coverage", "factual_consistency", "n_claims"]


def by_kind(audit_file):
    rows = []
    for r in map(json.loads, open(audit_file)):
        for u in r["audit"]:
            if u.get("is_claim"):
                rows.append({"enrolment_id": r["enrolment_id"], "condition": r["condition"], "kind": u["kind"],
                             "supported": u["supported"]})
    return pd.DataFrame(rows)


summ, tests = [], []
for model, (base, ctrl) in SETS.items():
    fb, fc = R / f"eval_per_output_{base}.csv", R / f"eval_per_output_{ctrl}.csv"
    if not (fb.exists() and fc.exists()):
        continue
    d = pd.concat([pd.read_csv(fb), pd.read_csv(fc)])
    k = pd.concat([by_kind(R / f"eval_audit_{base}.jsonl"), by_kind(R / f"eval_audit_{ctrl}.jsonl")])
    for c in ["C2_student_data", "C6_full_evidence", "C7_random5", "C3_rag", "C4_shap_rag"]:
        g = d[d.condition == c]
        if g.empty:
            continue
        kk = k[k.condition == c]
        inf = kk[kk.kind == "inference"]
        summ.append({"llm": model, "condition": c, "n_outputs": len(g), "empty_outputs": int((g.n_claims == 0).sum()),
                     **{m: g[m].mean() for m in M},
                     "ucr_inferences": 1 - inf.supported.mean() if len(inf) else np.nan, "n_inferences": len(inf)})
    pairs = [("C6_full_evidence", "C2_student_data", "format (IDs+statements vs JSON)"),
             ("C3_rag", "C6_full_evidence", "selection (top-5 vs all 27)"),
             ("C3_rag", "C7_random5", "ranking (salient vs random 5)"),
             ("C4_shap_rag", "C6_full_evidence", "model evidence vs all observed evidence")]
    for a, b, label in pairs:
        if not {a, b} <= set(d.condition):
            continue
        for m in M[:4]:
            x = d[d.condition == a].set_index("enrolment_id")[m]
            y = d[d.condition == b].set_index("enrolment_id")[m]
            j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
            p = np.nan if len(j) < 5 or (j.a - j.b).abs().sum() == 0 else wilcoxon(j.a, j.b, zero_method="zsplit").pvalue
            tests.append({"llm": model, "contrast": label, "a": a, "b": b, "metric": m, "n_pairs": len(j),
                          "mean_a": j.a.mean(), "mean_b": j.b.mean(), "diff_a_minus_b": (j.a - j.b).mean(), "p_wilcoxon": p})
S, T = pd.DataFrame(summ), pd.DataFrame(tests)
for (llm, m), g in T.groupby(["llm", "metric"]):  # Holm within LLM x metric
    g = g[g.p_wilcoxon.notna()].sort_values("p_wilcoxon")
    T.loc[g.index, "p_holm"] = np.maximum.accumulate([min(1, p * (len(g) - i)) for i, p in enumerate(g.p_wilcoxon)])
S.to_csv(R / "rag_control_comparison.csv", index=False)
T.to_csv(R / "rag_control_tests.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(3).to_string())
print(T[T.metric.isin(["unsupported_claim_rate", "evidence_coverage"])].round(4).to_string())
