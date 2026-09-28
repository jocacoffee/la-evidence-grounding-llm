"""Step 4 - Experiment 2: SHAP explanations of the main XGBoost model (test set, 2014J).

Primary explanation target: predicted at-risk probability  P(Fail) + P(Withdrawn)  of the
4-class model, on the probability scale. Computed with the exact-additive Permutation SHAP
(interventional, background = 200 random training enrolments, fixed seed). Values are
contributions to the *model's* predicted risk; they are not causal effects.
Complement: TreeSHAP (log-odds, per class) for global importance.
Outputs: data/shap_values_risk.csv, data/shap_treeshap_global.csv, data/student_evidence.{jsonl,csv},
         figures/fig4_shap_importance.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
F = json.load(open(ROOT / "configs" / "features.json"))["features"]
CLASSES = CFG["classes"]
RISK = [CLASSES.index(c) for c in CFG["risk_classes"]]
K = CFG["shap_top_k"]
SEED = CFG["seed"]

df = pd.read_csv(ROOT / "data" / "features.csv")
tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
pred = pd.read_csv(ROOT / "data" / "predictions.csv")
assert (pred.enrolment_id.values == te.enrolment_id.values).all()
model = xgb.XGBClassifier()
model.load_model(ROOT / "models" / "xgb_main.json")


def f_risk(Z):
    return model.predict_proba(pd.DataFrame(Z, columns=F))[:, RISK].sum(1)


bg = tr[F].sample(200, random_state=SEED)
ex = shap.PermutationExplainer(f_risk, shap.maskers.Independent(bg, max_samples=200), seed=SEED)
sv = ex(te[F], max_evals=2 * len(F) + 1 + 2 * 56, silent=True)
S = sv.values
base = float(np.mean(sv.base_values))
add_err = float(np.abs(S.sum(1) + sv.base_values - f_risk(te[F].values)).max())
print("base value", base, "max additivity error", add_err)

shap_df = pd.DataFrame(S, columns=F)
shap_df.insert(0, "enrolment_id", te.enrolment_id)
shap_df.to_csv(ROOT / "data" / "shap_values_risk.csv", index=False)

# TreeSHAP complement (per class, log-odds)
T = np.array(shap.TreeExplainer(model).shap_values(te[F]))  # (n, features, classes)
tree_imp = pd.DataFrame({c: np.abs(T[:, :, i]).mean(0) for i, c in enumerate(CLASSES)}, index=F)
tree_imp["mean_all_classes"] = tree_imp.mean(1)
tree_imp.sort_values("mean_all_classes", ascending=False).to_csv(ROOT / "data" / "shap_treeshap_global.csv")

glob = pd.DataFrame({"feature": F, "mean_abs_shap_risk": np.abs(S).mean(0),
                     "mean_shap_risk": S.mean(0)}).sort_values("mean_abs_shap_risk", ascending=False)
glob.to_csv(ROOT / "results" / "shap_global_importance.csv", index=False)
print(glob.head(12).round(4).to_string())

# ---------------------------------------------------------------- per-student records
records = []
for i, r in te.iterrows():
    order = np.argsort(-np.abs(S[i]))[:K]
    top = [{"feature": F[j], "shap_value": round(float(S[i, j]), 4),
            "direction": "increases predicted risk" if S[i, j] > 0 else "decreases predicted risk",
            "feature_value": None if pd.isna(r[F[j]]) else float(r[F[j]])} for j in order]
    p = pred.iloc[i]
    records.append({
        "enrolment_id": r.enrolment_id, "student_id": int(r.id_student), "course": r.code_module,
        "presentation": r.code_presentation, "time_cutoff_week": CFG["cutoff_week"],
        "predicted_class": p.predicted_class, "prediction_probability": round(float(p.prediction_probability), 4),
        "class_probabilities": {c: round(float(p[f"p_{c}"]), 4) for c in CLASSES},
        "predicted_risk": round(float(p.predicted_risk), 4),
        "shap_base_value_risk": round(base, 4),
        "top_SHAP_features": top,
        "feature_values": {c: (None if pd.isna(r[c]) else float(r[c])) for c in F},
        "true_final_result_NOT_FOR_LLM": r.final_result,
    })
with open(ROOT / "data" / "student_evidence.jsonl", "w") as fh:
    for rec in records:
        fh.write(json.dumps(rec) + "\n")
flat = []
for rec in records:
    row = {k: rec[k] for k in ["enrolment_id", "student_id", "course", "presentation", "time_cutoff_week",
                               "predicted_class", "prediction_probability", "predicted_risk",
                               "true_final_result_NOT_FOR_LLM"]}
    for k, t in enumerate(rec["top_SHAP_features"], 1):
        row[f"shap{k}_feature"], row[f"shap{k}_value"], row[f"shap{k}_feature_value"] = \
            t["feature"], t["shap_value"], t["feature_value"]
    row.update({f"fv_{c}": v for c, v in rec["feature_values"].items()})
    flat.append(row)
pd.DataFrame(flat).to_csv(ROOT / "data" / "student_evidence.csv", index=False)
json.dump({"base_value_risk": base, "max_additivity_error": add_err, "n_explained": len(te),
           "background_size": 200, "explainer": "PermutationExplainer (interventional)",
           "target": "P(Fail)+P(Withdrawn)"}, open(ROOT / "results" / "shap_config.json", "w"), indent=2)

# ---------------------------------------------------------------- Figure 4
top = glob.head(15).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
axes[0].barh(top.feature, top.mean_abs_shap_risk, color="#1864ab")
axes[0].set_xlabel("mean |SHAP| on predicted at-risk probability")
axes[0].set_title("Global importance (test 2014J)", fontsize=10)
axes[0].tick_params(axis="y", labelsize=7)
axes[0].spines[["top", "right"]].set_visible(False)
plt.sca(axes[1])
idx = [F.index(f) for f in glob.feature.head(15)]
shap.summary_plot(S[:, idx], te[F].iloc[:, idx], show=False, plot_size=None, max_display=15,
                  color_bar=True)
axes[1].set_xlabel("SHAP value (contribution to P(at-risk))", fontsize=8)
axes[1].tick_params(axis="y", labelsize=7)
axes[1].set_title("Per-student SHAP values", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig4_shap_importance.png", dpi=300)
print("records:", len(records))
