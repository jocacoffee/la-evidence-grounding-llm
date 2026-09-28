"""Step 14 - confirmatory anchoring test, exactly as pre-registered in results/PREREGISTRATION_anchoring.md.

Requires: results/llm_outputs_anchoring.jsonl and results/eval_per_output_anchoring.csv (src/08_evaluate.py anchoring).
Outputs: results/anchoring_confirmatory.csv, results/anchoring_descriptive.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_extract import action_cats, stated_band  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
grp = pd.read_csv(ROOT / "data" / "anchoring_sample.csv").set_index("enrolment_id").group
ev = pd.read_csv(R / "eval_per_output_anchoring.csv").set_index(["enrolment_id", "condition"])

rows = []
for o in map(json.loads, open(R / "llm_outputs_anchoring.jsonl")):
    p = o["parsed"] if o["json_ok"] else None
    rows.append({"enrolment_id": o["enrolment_id"], "condition": o["condition"], "group": grp[o["enrolment_id"]],
                 "json_ok": o["json_ok"],
                 "outreach": None if p is None else "outreach" in action_cats(p),
                 "band": None if p is None else stated_band(p),
                 "ucr": ev.loc[(o["enrolment_id"], o["condition"]), "unsupported_claim_rate"],
                 "coverage": ev.loc[(o["enrolment_id"], o["condition"]), "evidence_coverage"]})
D = pd.DataFrame(rows)


def mcnemar(group, measure):
    g = D[D.group == group]
    a = g[g.condition == "C2_student_data"].set_index("enrolment_id")
    b = g[g.condition == "C4_shap_rag"].set_index("enrolment_id")
    ids = [i for i in a.index.intersection(b.index) if a.loc[i, "json_ok"] and b.loc[i, "json_ok"]]
    xa, xb = np.array([measure(a.loc[i]) for i in ids]), np.array([measure(b.loc[i]) for i in ids])
    n10, n01 = int((xa & ~xb).sum()), int((~xa & xb).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"n_pairs": len(ids), "excluded": len(a.index.union(b.index)) - len(ids), "C2_share": xa.mean(),
            "C4_share": xb.mean(), "only_C2": n10, "only_C4": n01, "p_exact_mcnemar": p}


res = [{"hypothesis": "H1 (primary)", "group": "missed_at_risk", "measure": "outreach", "expected": "C4 < C2",
        **mcnemar("missed_at_risk", lambda r: bool(r.outreach))},
       {"hypothesis": "H2", "group": "missed_at_risk", "measure": "states_low_risk", "expected": "C4 > C2",
        **mcnemar("missed_at_risk", lambda r: r.band == "low")},
       {"hypothesis": "H3", "group": "false_alarm", "measure": "states_high_risk", "expected": "C4 > C2",
        **mcnemar("false_alarm", lambda r: r.band == "high")}]
T = pd.DataFrame(res)
sec = T.index[T.hypothesis != "H1 (primary)"]
ps = T.loc[sec, "p_exact_mcnemar"].sort_values()
T.loc[ps.index, "p_holm"] = np.maximum.accumulate([min(1, p * (len(ps) - i)) for i, p in enumerate(ps)])
T.loc[T.hypothesis == "H1 (primary)", "p_holm"] = T.loc[T.hypothesis == "H1 (primary)", "p_exact_mcnemar"]
T["direction_as_expected"] = np.where(T.expected == "C4 < C2", T.C4_share < T.C2_share, T.C4_share > T.C2_share)
T.to_csv(R / "anchoring_confirmatory.csv", index=False)

desc = D.groupby(["group", "condition"]).agg(
    n=("enrolment_id", "size"), json_ok=("json_ok", "mean"), outreach=("outreach", "mean"),
    states_high=("band", lambda s: (s == "high").mean()), states_low=("band", lambda s: (s == "low").mean()),
    states_none=("band", lambda s: s.isna().mean()), ucr=("ucr", "mean"), coverage=("coverage", "mean")).reset_index()
desc.to_csv(R / "anchoring_descriptive.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(4).to_string())
print(desc.round(3).to_string())
