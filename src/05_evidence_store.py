"""Step 5 - Experiment 3: Student Evidence Store.

Turns every test-set enrolment's LA features, model prediction and SHAP explanation into explicit,
ID-addressable evidence items. ID prefixes separate the epistemic status of each item:
  E<n>  observed / observed_derived / static data from OULAD (up to the cutoff)
  M<n>  model inference (predicted risk, predicted class). NOT an observed fact.
  X<n>  model explanation (SHAP contribution of a feature to the predicted risk). NOT causal.
Every item carries its source feature, source table and cutoff, so it can be traced back.
Also stores, for retrieval, a model-independent salience score: two-sided deviation of the value from
the same-module TRAINING cohort (past presentations only), salience = |2*percentile - 1|.
Output: data/evidence_store.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_dictionary import EDU_LABELS, FEATURES, INFERENCE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
WEEK = CFG["cutoff_week"]
FEAT = list(FEATURES)

df = pd.read_csv(ROOT / "data" / "features.csv")
tr = df[df.split == "train"]
recs = [json.loads(l) for l in open(ROOT / "data" / "student_evidence.jsonl")]

NA_TEXT = {
    "submission_rate": "Not applicable: no assessment had a deadline before the end of week 4",
    "mean_score": "No score available: no assessment was submitted (and scored) before the end of week 4",
    "activity_trend": "Not defined: no VLE activity between day 0 and day 27",
    "days_since_last_activity": "Not defined: no VLE activity recorded up to the end of week 4",
    "date_registration": "Registration date not recorded in OULAD",
}


def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if kind == "int" or kind == "regdays":
        return str(int(round(v)))
    if kind == "pct":
        return f"{round(v * 100)}%"
    if kind == "ratio":
        return f"{v:.2f}"
    if kind == "index":
        return f"{v:+.2f}"
    if kind == "score":
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if kind == "bool":
        return "yes" if v >= 0.5 else "no"
    if kind == "edu":
        return EDU_LABELS[int(v)]
    return str(v)


# same-module training-cohort reference distributions (mid-rank percentile handles ties)
ref = {m: {f: np.sort(g[f].dropna().values) for f in FEAT} for m, g in tr.groupby("code_module")}


def percentile(module, f, v):
    a = ref[module][f]
    if v is None or len(a) == 0:
        return None
    lo, hi = np.searchsorted(a, v, "left"), np.searchsorted(a, v, "right")
    return float((lo + 0.5 * (hi - lo)) / len(a))


out = open(ROOT / "data" / "evidence_store.jsonl", "w")
n_items = 0
for r in recs:
    common = {"enrolment_id": r["enrolment_id"], "student_id": r["student_id"], "course": r["course"],
              "presentation": r["presentation"], "time_cutoff_week": WEEK}
    items = []
    for n, f in enumerate(FEAT, 1):
        meta = FEATURES[f]
        v = r["feature_values"][f]
        dv = fmt(v, meta["fmt"])
        p = percentile(r["course"], f, v)
        items.append({"id": f"E{n}", **common, "type": meta["type"], "category": meta["category"],
                      "feature": f, "value": v, "display_value": dv,
                      "statement": meta["label"].format(v=dv) if dv is not None else
                      f"{meta['label'].split(':')[0]}: {NA_TEXT.get(f, 'not available')}",
                      "source": f"OULAD {meta['source']}", "cohort_percentile": p,
                      "salience": 0.0 if p is None else abs(2 * p - 1)})
    for n, (f, meta) in enumerate(INFERENCE.items(), 1):
        v = r[f]
        dv = fmt(v, meta["fmt"])
        items.append({"id": f"M{n}", **common, "type": meta["type"], "category": meta["category"],
                      "feature": f, "value": v, "display_value": dv,
                      "statement": meta["label"].format(v=dv) + " (model inference, not an observed fact)",
                      "source": meta["source"]})
    for n, t in enumerate(r["top_SHAP_features"], 1):
        f = t["feature"]
        e_id = f"E{FEAT.index(f) + 1}"
        dv = fmt(t["feature_value"], FEATURES[f]["fmt"])
        items.append({"id": f"X{n}", **common, "type": "model_explanation", "category": "model_explanation",
                      "feature": f, "value": t["shap_value"], "feature_value": t["feature_value"],
                      "display_value": dv, "linked_evidence": e_id, "shap_rank": n,
                      "statement": (f"SHAP rank {n}: {f} (value {dv if dv is not None else 'not available'}, "
                                    f"see {e_id}) {t['direction']} by {abs(t['shap_value']) * 100:.1f} "
                                    f"percentage points (association in the model, not a cause)"),
                      "source": "SHAP PermutationExplainer on XGBoost (this study)"})
    for it in items:
        it["uid"] = f"{r['enrolment_id']}#{it['id']}"
        out.write(json.dumps(it) + "\n")
    n_items += len(items)
out.close()
print("students:", len(recs), "evidence items:", n_items)
print(json.dumps(items, indent=1)[:3000])
