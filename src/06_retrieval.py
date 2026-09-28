"""Step 6 - Experiment 4: retrieval over the structured Evidence Store.

Query  q = (enrolment_id, cutoff_week, task="educator recommendation").
Stage 1 (hard filter): items of that enrolment with time_cutoff_week <= cutoff and category in
         {observed, observed_derived, static}. Model items (M*, X*) are never retrieved here, so
         the retriever is model-independent (Condition 3 does not see the model).
Stage 2 (ranking): salience = deviation of the value from the same-module training cohort
         (|2*percentile-1|), ties broken by evidence ID. Top-K returned (K = config retrieval_k).

Evaluation. There is no human relevance ground truth in OULAD, so none is invented. We report
 (a) integrity: share of retrieved items belonging to the query enrolment and to cutoff <= week 4;
 (b) agreement with a MODEL-DEFINED PROXY relevance set: the enrolment's SHAP top-k features.
     P@K, R@K and MRR against this proxy measure how far a model-agnostic retriever recovers
     the evidence the predictive model relied on; they are not a measure of pedagogical relevance.
 Baselines: random ranking (mean of 20 seeded permutations) and fixed dictionary order.
Also draws the LLM evaluation sample and a disjoint pilot sample.
Outputs: data/retrieved_evidence.jsonl, data/llm_sample.csv, data/pilot_sample.csv,
         results/retrieval_metrics.csv
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
K = CFG["retrieval_k"]
SEED = CFG["seed"]
CUT_WEEK = CFG["cutoff_week"]
RETRIEVABLE = {"observed", "observed_derived", "static"}

store = defaultdict(list)
for line in open(ROOT / "data" / "evidence_store.jsonl"):
    it = json.loads(line)
    store[it["enrolment_id"]].append(it)


def retrieve(enrolment_id, cutoff_week=CUT_WEEK, k=K, all_items=None):
    items = all_items if all_items is not None else store[enrolment_id]
    cand = [i for i in items if i["enrolment_id"] == enrolment_id and i["time_cutoff_week"] <= cutoff_week
            and i["category"] in RETRIEVABLE]
    cand.sort(key=lambda i: (-i["salience"], int(i["id"][1:])))
    return cand[:k]


def prf(ranked, rel, k):
    top = ranked[:k]
    hits = [f in rel for f in top]
    rr = next((1 / (n + 1) for n, f in enumerate(ranked) if f in rel), 0.0)
    return sum(hits) / k, sum(hits) / len(rel), rr


rng = np.random.default_rng(SEED)
rows = []
integrity_ok = 0
integrity_total = 0
retrieved_all = {}
for eid, items in store.items():
    ret = retrieve(eid)
    retrieved_all[eid] = ret
    integrity_total += len(ret)
    integrity_ok += sum(i["enrolment_id"] == eid and i["time_cutoff_week"] <= CUT_WEEK for i in ret)
    full = [i["feature"] for i in retrieve(eid, k=10 ** 6)]
    shap_rank = [i["feature"] for i in sorted([i for i in items if i["category"] == "model_explanation"],
                                              key=lambda i: i["shap_rank"])]
    fixed = [i["feature"] for i in sorted([i for i in items if i["category"] in RETRIEVABLE],
                                          key=lambda i: int(i["id"][1:]))]
    for rel_k in [3, 5]:
        rel = set(shap_rank[:rel_k])
        for kk in [3, 5]:
            p, r, mrr = prf(full, rel, kk)
            rows.append(dict(method="salience", rel_k=rel_k, k=kk, p=p, r=r, mrr=mrr))
            p, r, mrr = prf(fixed, rel, kk)
            rows.append(dict(method="fixed_order", rel_k=rel_k, k=kk, p=p, r=r, mrr=mrr))
            for _ in range(20):
                perm = list(rng.permutation(fixed))
                p, r, mrr = prf(perm, rel, kk)
                rows.append(dict(method="random", rel_k=rel_k, k=kk, p=p, r=r, mrr=mrr))

m = pd.DataFrame(rows).groupby(["method", "rel_k", "k"]).mean().reset_index()
m = m.rename(columns={"p": "precision_at_k", "r": "recall_at_k", "mrr": "mrr"})
m["n_queries"] = len(store)
m["relevance_definition"] = "proxy: SHAP top-rel_k features of the enrolment (model-defined)"
m["integrity_same_enrolment_and_cutoff"] = integrity_ok / integrity_total
m.to_csv(ROOT / "results" / "retrieval_metrics.csv", index=False)
print(m.round(3).to_string())

# retrieved-feature frequency (what the retriever tends to return)
freq = pd.Series([i["feature"] for r in retrieved_all.values() for i in r]).value_counts(normalize=True)
freq.rename("share_of_retrieved_items").to_csv(ROOT / "results" / "retrieval_feature_frequency.csv")

# ---------------------------------------------------------------- LLM sample (stratified by outcome)
ev = pd.read_csv(ROOT / "data" / "student_evidence.csv")
n_per = CFG["llm_sample_size"] // 4
sample = ev.groupby("true_final_result_NOT_FOR_LLM", group_keys=False).apply(
    lambda g: g.sample(n_per, random_state=SEED))
rest = ev[~ev.enrolment_id.isin(sample.enrolment_id)]
pilot = rest.groupby("true_final_result_NOT_FOR_LLM", group_keys=False).apply(
    lambda g: g.sample(2, random_state=SEED))
sample[["enrolment_id", "course", "presentation", "true_final_result_NOT_FOR_LLM"]] \
    .to_csv(ROOT / "data" / "llm_sample.csv", index=False)
pilot[["enrolment_id", "course", "presentation", "true_final_result_NOT_FOR_LLM"]] \
    .to_csv(ROOT / "data" / "pilot_sample.csv", index=False)
with open(ROOT / "data" / "retrieved_evidence.jsonl", "w") as fh:
    for eid in list(sample.enrolment_id) + list(pilot.enrolment_id):
        fh.write(json.dumps({"enrolment_id": eid, "query": {"enrolment_id": eid, "cutoff_week": CUT_WEEK,
                             "task": "educator recommendation"}, "k": K,
                             "retrieved": retrieved_all[eid]}) + "\n")
print("sample:", len(sample), sample.course.value_counts().to_dict(), "pilot:", len(pilot))
print(freq.head(10).round(3).to_dict())
