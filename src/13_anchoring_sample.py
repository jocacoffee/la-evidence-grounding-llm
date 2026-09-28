"""Step 13 - sample for the confirmatory anchoring test (see results/PREREGISTRATION_anchoring.md).

From the 2014J test set, excluding the 100 main-sample and 8 pilot enrolments, draw (seed 42):
  40 'missed at-risk'  : predicted_risk < 0.5 and true final_result in {Fail, Withdrawn}
  40 'false alarm'     : predicted_risk >= 0.5 and true final_result in {Pass, Distinction}
Retrieval is identical to src/06_retrieval.py (same filter, salience ranking, K).
Outputs: data/anchoring_sample.csv, data/retrieved_evidence_anchoring.jsonl
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
SEED, K, CUT_WEEK = CFG["seed"], CFG["retrieval_k"], CFG["cutoff_week"]
RETRIEVABLE = {"observed", "observed_derived", "static"}
N_PER_GROUP = 40

ev = pd.read_csv(ROOT / "data" / "student_evidence.csv")
used = set(pd.read_csv(ROOT / "data" / "llm_sample.csv").enrolment_id) | set(pd.read_csv(ROOT / "data" / "pilot_sample.csv").enrolment_id)
e = ev[~ev.enrolment_id.isin(used)].copy()
true_risk = e.true_final_result_NOT_FOR_LLM.isin(["Fail", "Withdrawn"])
e["group"] = None
e.loc[(e.predicted_risk < 0.5) & true_risk, "group"] = "missed_at_risk"
e.loc[(e.predicted_risk >= 0.5) & ~true_risk, "group"] = "false_alarm"
sample = pd.concat([e[e.group == g].sample(N_PER_GROUP, random_state=SEED) for g in ["missed_at_risk", "false_alarm"]])
sample[["enrolment_id", "course", "presentation", "group", "predicted_risk", "true_final_result_NOT_FOR_LLM"]] \
    .to_csv(ROOT / "data" / "anchoring_sample.csv", index=False)

ids = set(sample.enrolment_id)
items = {i: [] for i in ids}
for line in open(ROOT / "data" / "evidence_store.jsonl"):
    it = json.loads(line)
    if it["enrolment_id"] in ids:
        items[it["enrolment_id"]].append(it)
with open(ROOT / "data" / "retrieved_evidence_anchoring.jsonl", "w") as fh:
    for eid in sample.enrolment_id:
        cand = [i for i in items[eid] if i["time_cutoff_week"] <= CUT_WEEK and i["category"] in RETRIEVABLE]
        cand.sort(key=lambda i: (-i["salience"], int(i["id"][1:])))
        fh.write(json.dumps({"enrolment_id": eid, "query": {"enrolment_id": eid, "cutoff_week": CUT_WEEK,
                                                          "task": "educator recommendation"},
                             "k": K, "retrieved": cand[:K]}) + "\n")
print(sample.groupby("group").agg(n=("enrolment_id", "size"), mean_risk=("predicted_risk", "mean")).round(3))
print(sample.groupby("group").true_final_result_NOT_FOR_LLM.value_counts().to_dict())
print(sample.course.value_counts().to_dict())
