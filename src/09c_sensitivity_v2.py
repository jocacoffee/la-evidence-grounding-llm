"""Step 9c - v2 evaluator sensitivity (post-hoc fixes) and determinism check of the generations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
M = ["unsupported_claim_rate", "unsupported_fact_rate", "evidence_coverage", "factual_consistency",
     "shap_consistency", "recommendation_specificity"]
out = []
for ver, f in [("v1_frozen", "eval_per_output_main.csv"), ("v2_posthoc", "eval_per_output_main_v2.csv")]:
    d = pd.read_csv(ROOT / "results" / f)
    for c, g in d.groupby("condition"):
        out.append({"evaluator": ver, "condition": c, **{m: g[m].mean() for m in M}})
    for m in ["unsupported_claim_rate", "evidence_coverage", "factual_consistency"]:
        for a, b in [("C4_shap_rag", "C3_rag"), ("C4_shap_rag", "C2_student_data"), ("C3_rag", "C2_student_data")]:
            x = d[d.condition == a].set_index("enrolment_id")[m]
            y = d[d.condition == b].set_index("enrolment_id")[m]
            j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
            out.append({"evaluator": ver, "condition": f"{a} vs {b}", "metric": m, "mean_diff": (j.a - j.b).mean(),
                        "p_wilcoxon": wilcoxon(j.a, j.b, zero_method="zsplit").pvalue})
res = pd.DataFrame(out)
res.to_csv(ROOT / "results" / "evaluator_sensitivity_v1_v2.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string())

rows = []
for rep_set, ref_set in [("repro", "main"), ("repro_gemma", "gemma"), ("repro_qwen_reload", "main")]:
    rep = ROOT / "results" / "llm_outputs" / rep_set
    if not rep.exists():
        continue
    for f in sorted(rep.glob("*/*.json")):
        a = json.load(open(f))
        b = json.load(open(ROOT / "results" / "llm_outputs" / ref_set / f.parent.name / f.name))
        rows.append({"check": f"{rep_set} vs {ref_set}", "condition": f.parent.name, "enrolment_id": f.stem,
                     "identical_raw_output": a["raw_content"] == b["raw_content"]})
if rows:
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results" / "determinism_check.csv", index=False)
    print(r.groupby(["check", "condition"]).identical_raw_output.agg(["mean", "size"]))
