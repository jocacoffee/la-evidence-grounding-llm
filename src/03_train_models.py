"""Step 3 - Experiment 1: student outcome prediction at the week-4 cutoff.

Models: majority-class reference, multinomial Logistic Regression (baseline), XGBoost (main),
and XGBoost without score features (sensitivity to the mark-availability assumption, §6.2).
Split: train = 2013B, 2013J, 2014B; test = 2014J. XGBoost's n_estimators is chosen by early stopping
on 2014B (model fit on 2013B+2013J), then refit on all train presentations.
Outputs: results/ml_results.csv, results/confusion_matrices.csv, results/ml_results.png,
         data/predictions.csv, models/*.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
FEATURES = json.load(open(ROOT / "configs" / "features.json"))["features"]
SEED = CFG["seed"]
CLASSES = CFG["classes"]
RISK = [CLASSES.index(c) for c in CFG["risk_classes"]]
SCORE_FEATURES = ["mean_score"]
LOG_COLS = [c for c in FEATURES if c.startswith("clicks") or c in
            ["total_clicks", "active_days", "distinct_materials", "pre_course_clicks",
             "active_days_weeks_3_4", "days_since_last_activity", "studied_credits"]]
(ROOT / "models").mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "features.csv")
df["y"] = df.final_result.map({c: i for i, c in enumerate(CLASSES)})
tr, te = df[df.split == "train"], df[df.split == "test"]


def logt(X, idx):
    X = X.astype(float).copy()
    X[:, idx] = np.log1p(np.clip(X[:, idx], 0, None))
    return X


def lr_model(cols):
    idx = [cols.index(c) for c in LOG_COLS if c in cols]
    return make_pipeline(FunctionTransformer(logt, kw_args={"idx": idx}), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))


XGB_PARAMS = dict(objective="multi:softprob", num_class=len(CLASSES), learning_rate=0.05, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  tree_method="hist", random_state=SEED, n_jobs=4, eval_metric="mlogloss")


def fit_xgb(cols, name):
    val_p = CFG["validation_presentation"]
    a, b = tr[tr.code_presentation != val_p], tr[tr.code_presentation == val_p]
    m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **XGB_PARAMS)
    m.fit(a[cols], a.y, sample_weight=compute_sample_weight("balanced", a.y),
          eval_set=[(b[cols], b.y)], sample_weight_eval_set=[compute_sample_weight("balanced", b.y)],
          verbose=False)
    n = m.best_iteration + 1
    final = xgb.XGBClassifier(n_estimators=n, **XGB_PARAMS)
    final.fit(tr[cols], tr.y, sample_weight=compute_sample_weight("balanced", tr.y))
    final.save_model(ROOT / "models" / f"{name}.json")
    return final, n


def metrics(y, P, rng=None):
    pred = P.argmax(1)
    Y = np.eye(len(CLASSES))[y]
    r = {"macro_f1": f1_score(y, pred, average="macro"),
         "accuracy": accuracy_score(y, pred),
         "balanced_accuracy": balanced_accuracy_score(y, pred)}
    if rng is None:  # full-set only metrics
        r["auroc_macro_ovr"] = roc_auc_score(y, P, multi_class="ovr", average="macro")
        r["brier_multiclass"] = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
        yb = np.isin(y, RISK).astype(int)
        pb = P[:, RISK].sum(1)
        r["auroc_at_risk_binary"] = roc_auc_score(yb, pb)
        r["brier_at_risk_binary"] = brier_score_loss(yb, pb)
        for i, c in enumerate(CLASSES):
            r[f"f1_{c}"] = f1_score(y == i, pred == i)
    return r


def bootstrap(y, P, n=1000):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out.append(metrics(y[i], P[i], rng=True))
    b = pd.DataFrame(out)
    return {f"{k}_ci95": f"[{b[k].quantile(.025):.3f}, {b[k].quantile(.975):.3f}]" for k in b}


rows, cms, probs = [], {}, {}
y_te = te.y.values

# majority reference
maj = np.zeros((len(te), len(CLASSES)))
maj[:, tr.y.value_counts().idxmax()] = 1
probs["Majority class"] = maj

# logistic regression
lr = lr_model(FEATURES).fit(tr[FEATURES].values, tr.y)
joblib.dump(lr, ROOT / "models" / "logreg.joblib")
probs["Logistic Regression"] = lr.predict_proba(te[FEATURES].values)

# xgboost main
xm, n_main = fit_xgb(FEATURES, "xgb_main")
probs["XGBoost"] = xm.predict_proba(te[FEATURES])

# sensitivity: no score features
cols_ns = [c for c in FEATURES if c not in SCORE_FEATURES]
xs, n_ns = fit_xgb(cols_ns, "xgb_no_scores")
probs["XGBoost (no scores)"] = xs.predict_proba(te[cols_ns])

for name, P in probs.items():
    r = {"model": name, "n_train": len(tr), "n_test": len(te)}
    r.update(metrics(y_te, P))
    r.update(bootstrap(y_te, P))
    if name.startswith("XGBoost"):
        r["n_estimators"] = n_main if name == "XGBoost" else n_ns
    rows.append(r)
    cms[name] = confusion_matrix(y_te, P.argmax(1), labels=range(len(CLASSES)))

res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "ml_results.csv", index=False)
cm_rows = []
for name, cm in cms.items():
    for i, c in enumerate(CLASSES):
        cm_rows.append({"model": name, "true": c, **{f"pred_{p}": int(cm[i, j]) for j, p in enumerate(CLASSES)}})
pd.DataFrame(cm_rows).to_csv(ROOT / "results" / "confusion_matrices.csv", index=False)

# predictions of main model (test set)
P = probs["XGBoost"]
pred = te[["enrolment_id", "id_student", "code_module", "code_presentation", "final_result"]].copy()
for i, c in enumerate(CLASSES):
    pred[f"p_{c}"] = P[:, i]
pred["predicted_class"] = [CLASSES[i] for i in P.argmax(1)]
pred["prediction_probability"] = P.max(1)
pred["predicted_risk"] = P[:, RISK].sum(1)
pred.to_csv(ROOT / "data" / "predictions.csv", index=False)

# ---------------------------------------------------------------- Figure 3
show = ["Majority class", "Logistic Regression", "XGBoost", "XGBoost (no scores)"]
mets = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced acc."), ("accuracy", "Accuracy"),
        ("auroc_macro_ovr", "AUROC (macro OvR)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
w = 0.2
colors = ["#adb5bd", "#4dabf7", "#1864ab", "#74c0fc"]
for k, mname in enumerate(show):
    rr = res.set_index("model").loc[mname]
    vals = [rr[m] for m, _ in mets]
    ax.bar(np.arange(len(mets)) + (k - 1.5) * w, vals, w, label=mname, color=colors[k])
ax.set_xticks(range(len(mets)))
ax.set_xticklabels([l for _, l in mets], fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Score (test: 2014J)")
ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Outcome prediction at week 4", fontsize=10)
cm = cms["XGBoost"]
cmn = cm / cm.sum(1, keepdims=True)
ax = axes[1]
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if cmn[i, j] > .5 else "black")
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, fontsize=7, rotation=20)
ax.set_yticklabels(CLASSES, fontsize=7)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("XGBoost confusion matrix (row-normalised)", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "results" / "ml_results.png", dpi=300)
fig.savefig(ROOT / "figures" / "fig3_la_model_performance.png", dpi=300)

pd.set_option("display.width", 250)
print(res[["model", "macro_f1", "macro_f1_ci95", "balanced_accuracy", "accuracy", "auroc_macro_ovr",
           "brier_multiclass", "auroc_at_risk_binary", "f1_Distinction", "f1_Pass", "f1_Fail",
           "f1_Withdrawn"]].round(3).to_string())
print("n_estimators:", n_main, n_ns)
for k in ["Logistic Regression", "XGBoost"]:
    print(k, "\n", cms[k])
