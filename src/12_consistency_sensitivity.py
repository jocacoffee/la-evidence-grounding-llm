"""Step 12 - post-hoc analyses on existing outputs (no new generations):

(A) Cross-model consistency (RQ2): for the same student and condition, do different LLMs say the same things?
    - feature agreement: Jaccard of the LA features referenced in the claim units (from the frozen evaluator audit)
    - stated-risk agreement: same risk band (high / moderate / low) stated by both LLMs
    - action agreement: Jaccard of action categories (outreach, assessment, engagement, support, monitor, positive)
(B) Sensitivity to LA indicators (individualisation):
    - does the risk band stated by the LLM track the model-predicted risk (Spearman) and the true outcome (AUROC)?
    - genericity: mean pairwise TF-IDF cosine similarity of the recommended actions across students
(C) Mispredictions: grounding and stated risk when the predictive model is right vs wrong
    (at-risk threshold 0.5 on P(Fail)+P(Withdrawn); true at-risk = Fail or Withdrawn).
All extraction is deterministic (regex lexicons below); these are descriptive, post-hoc analyses.
Outputs: results/cross_model_consistency.csv, results/llm_sensitivity.csv, results/misprediction_analysis.csv
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
CONDS = ["C2_student_data", "C3_rag", "C4_shap_rag"]
MODELS = {"qwen3.5-9b": "main", "gemma-4-e4b": "gemma", "qwen3-14b": "qwen14b"}
MODELS = {k: v for k, v in MODELS.items() if (R / f"llm_outputs_{v}.jsonl").exists() and (R / f"eval_audit_{v}.jsonl").exists()}

from llm_extract import ACTIONS, BAND, action_cats, jacc, stated_band  # noqa: E402

rows = []
for model, s in MODELS.items():
    audits = {(r["enrolment_id"], r["condition"]): r for r in map(json.loads, open(R / f"eval_audit_{s}.jsonl"))}
    for o in map(json.loads, open(R / f"llm_outputs_{s}.jsonl")):
        if o["condition"] not in CONDS:
            continue
        a = audits[(o["enrolment_id"], o["condition"])]
        feats = {f for u in a["audit"] if u.get("is_claim") for f in u["spec"]} - {"predicted_risk", "predicted_class"}
        p = o["parsed"] or {}
        rows.append({"model": model, "enrolment_id": o["enrolment_id"], "condition": o["condition"], "features": feats,
                     "band": stated_band(p), "actions": action_cats(p),
                     "action_text": " ".join(x.get("action", "") for x in p.get("recommendations", [])),
                     "ucr": a["unsupported_claim_rate"], "coverage": a["evidence_coverage"]})
D = pd.DataFrame(rows)
ev = pd.read_csv(ROOT / "data" / "student_evidence.csv").set_index("enrolment_id")
D["model_risk"] = D.enrolment_id.map(ev.predicted_risk)
D["true_at_risk"] = D.enrolment_id.map(ev.true_final_result_NOT_FOR_LLM.isin(["Fail", "Withdrawn"]))
D["model_band"] = pd.cut(D.model_risk, [-1, 0.33, 0.66, 2], labels=["low", "moderate", "high"]).astype(str)
pd.set_option("display.width", 250)

# ---------------------------------------------------------------- (A) cross-model consistency
cons = []
for m1, m2 in itertools.combinations(MODELS, 2):
    for c in CONDS:
        x = D[(D.model == m1) & (D.condition == c)].set_index("enrolment_id")
        y = D[(D.model == m2) & (D.condition == c)].set_index("enrolment_id")
        ids = x.index.intersection(y.index)
        fj = [jacc(x.loc[i, "features"], y.loc[i, "features"]) for i in ids]
        aj = [jacc(x.loc[i, "actions"], y.loc[i, "actions"]) for i in ids]
        both = [(x.loc[i, "band"], y.loc[i, "band"]) for i in ids if x.loc[i, "band"] and y.loc[i, "band"]]
        cons.append({"pair": f"{m1} vs {m2}", "condition": c, "n": len(ids),
                     "feature_jaccard": np.nanmean(fj), "action_jaccard": np.nanmean(aj),
                     "risk_band_both_stated": len(both), "risk_band_agreement": np.mean([a == b for a, b in both]) if both else np.nan,
                     "fj_list": fj})
C = pd.DataFrame(cons)
tests = []
for pair, g in C.groupby("pair"):
    f = {r.condition: np.array(r.fj_list, dtype=float) for r in g.itertuples()}
    for a, b in [("C3_rag", "C2_student_data"), ("C4_shap_rag", "C3_rag"), ("C4_shap_rag", "C2_student_data")]:
        ok = ~np.isnan(f[a]) & ~np.isnan(f[b])
        tests.append({"pair": pair, "metric": "feature_jaccard", "a": a, "b": b, "mean_diff": np.mean(f[a][ok] - f[b][ok]),
                      "p_wilcoxon": wilcoxon(f[a][ok], f[b][ok], zero_method="zsplit").pvalue})
C.drop(columns="fj_list").to_csv(R / "cross_model_consistency.csv", index=False)
pd.DataFrame(tests).to_csv(R / "cross_model_consistency_tests.csv", index=False)
print("=== (A) CROSS-MODEL CONSISTENCY"); print(C.drop(columns="fj_list").round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())

# ---------------------------------------------------------------- (B) sensitivity to LA indicators
sens = []
for (model, c), g in D.groupby(["model", "condition"]):
    st = g.dropna(subset=["band"])
    b = st.band.map(BAND)
    txt = g.action_text.fillna("")
    sim = cosine_similarity(TfidfVectorizer().fit_transform(txt))
    iu = np.triu_indices(len(g), 1)
    sens.append({"model": model, "condition": c, "n": len(g), "risk_band_stated_share": len(st) / len(g),
                 "stated_band_dist": st.band.value_counts().to_dict(),
                 "spearman_band_vs_model_risk": spearmanr(b, st.model_risk).correlation if len(st) > 5 else np.nan,
                 "auroc_band_vs_true_outcome": roc_auc_score(st.true_at_risk, b) if st.true_at_risk.nunique() == 2 else np.nan,
                 "band_matches_model_band": (st.band == st.model_band).mean(),
                 "action_genericity_mean_cosine": sim[iu].mean(),
                 "distinct_action_texts": txt.nunique() / len(g)})
S = pd.DataFrame(sens)
ref = D.drop_duplicates("enrolment_id")
S.attrs["model_risk_auroc_same_students"] = roc_auc_score(ref.true_at_risk, ref.model_risk)
S.to_csv(R / "llm_sensitivity.csv", index=False)
print("=== (B) SENSITIVITY  (model-risk AUROC on the same 100 students:", round(S.attrs["model_risk_auroc_same_students"], 3), ")")
print(S.round(3).to_string())

# ---------------------------------------------------------------- (C) mispredictions
D["pred_group"] = np.select([(D.model_risk >= 0.5) & D.true_at_risk, (D.model_risk < 0.5) & ~D.true_at_risk,
                             (D.model_risk >= 0.5) & ~D.true_at_risk],
                            ["correct", "correct", "false_alarm"], "missed_at_risk")
mis = D.groupby(["model", "condition", "pred_group"]).agg(
    n=("enrolment_id", "size"), ucr=("ucr", "mean"), coverage=("coverage", "mean"),
    states_high_risk=("band", lambda s: (s == "high").mean()),
    states_low_risk=("band", lambda s: (s == "low").mean()),
    outreach_action=("actions", lambda s: np.mean(["outreach" in a for a in s])),
    positive_action=("actions", lambda s: np.mean(["positive" in a for a in s]))).reset_index()
mis.to_csv(R / "misprediction_analysis.csv", index=False)
print("=== (C) MISPREDICTIONS"); print(mis.round(3).to_string())

# exact paired test (McNemar, binomial on discordant pairs): C2 vs C4 within each prediction group
from scipy.stats import binomtest  # noqa: E402
ex = []
for model in MODELS:
    for grp in ["missed_at_risk", "false_alarm", "correct"]:
        g = D[(D.model == model) & (D.pred_group == grp)]
        a = g[g.condition == "C2_student_data"].set_index("enrolment_id")
        b = g[g.condition == "C4_shap_rag"].set_index("enrolment_id")
        ids = a.index.intersection(b.index)
        for name, f in [("outreach_action", lambda r: "outreach" in r.actions), ("states_high_risk", lambda r: r.band == "high")]:
            xa = np.array([f(a.loc[i]) for i in ids])
            xb = np.array([f(b.loc[i]) for i in ids])
            n10, n01 = int((xa & ~xb).sum()), int((~xa & xb).sum())
            p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else np.nan
            ex.append({"model": model, "group": grp, "n": len(ids), "measure": name, "C2_share": xa.mean(),
                       "C4_share": xb.mean(), "only_C2": n10, "only_C4": n01, "p_exact_mcnemar": p})
EX = pd.DataFrame(ex)
EX.to_csv(R / "misprediction_exact_tests.csv", index=False)
print("=== (C) EXACT TESTS C2 vs C4"); print(EX.round(4).to_string())
