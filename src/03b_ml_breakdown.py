"""Step 3b - per-module breakdown of the main model on the test set (reads data/predictions.csv only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
p = pd.read_csv(ROOT / "data" / "predictions.csv")
rows = []
for m, g in [("ALL", p)] + list(p.groupby("code_module")):
    y = np.isin(g.final_result, CFG["risk_classes"]).astype(int)
    rows.append({"module": m, "n": len(g), "at_risk_share": y.mean(),
                 "macro_f1_4class": f1_score(g.final_result, g.predicted_class, average="macro"),
                 "balanced_acc_4class": balanced_accuracy_score(g.final_result, g.predicted_class),
                 "auroc_at_risk": roc_auc_score(y, g.predicted_risk)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "ml_results_by_module.csv", index=False)
print(df.round(3).to_string())
