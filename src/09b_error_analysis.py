"""Step 9b - breakdown of unsupported / unlinked claims by reason, unit kind and condition (main set)."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for r in map(json.loads, open(ROOT / "results" / "eval_audit_main.jsonl")):
    for u in r["audit"]:
        if not u.get("is_claim"):
            continue
        cats = sorted({x.split(":")[0] for x in u["reasons"]}) or ["supported"]
        rows.append({"condition": r["condition"], "kind": u["kind"], "supported": u["supported"],
                     "linked": u["linked"], "generic_only": not u["spec"], "reason": "+".join(cats),
                     "risk_only": u["reasons"] == ["risk_assertion_without_model_output"]})
d = pd.DataFrame(rows)
by_kind = d.groupby(["condition", "kind"]).agg(n=("supported", "size"), unsupported_rate=("supported", lambda s: 1 - s.mean()),
                                               coverage=("linked", "mean")).reset_index()
reasons = d[~d.supported].groupby(["condition", "reason"]).size().rename("n").reset_index()
reasons["share_of_all_claims"] = reasons.apply(lambda x: x.n / (d.condition == x.condition).sum(), axis=1)
unl = d[~d.linked].groupby(["condition"]).agg(n_unlinked=("linked", "size"), generic_only=("generic_only", "sum")).reset_index()
# UCR excluding risk assertions made without a model estimate
alt = d[~d.risk_only].groupby("condition").supported.apply(lambda s: 1 - s.mean()).rename("ucr_excluding_risk_only")
by_kind.to_csv(ROOT / "results" / "claims_by_kind.csv", index=False)
reasons.to_csv(ROOT / "results" / "unsupported_reasons.csv", index=False)
pd.set_option("display.width", 200)
print(by_kind.round(3).to_string()); print(reasons.round(3).to_string()); print(unl.to_string()); print(alt.round(4))
alt.to_csv(ROOT / "results" / "ucr_excluding_risk_only.csv")
