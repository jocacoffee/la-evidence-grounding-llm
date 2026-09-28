"""Step 15 - human validation of the frozen deterministic evaluator (unsupported-claim decisions).

  python src/15_evaluator_validation.py prepare   -> validation/ sheets (blind: evaluator verdict hidden)
  python src/15_evaluator_validation.py score     -> results/evaluator_validation.csv (+ printed summary)

Sampling (seed 42), main set (Qwen3.5-9B, C2-C4):
  * ALL claims the evaluator flagged as unsupported (precision)
  * 50 random claims per condition that the evaluator judged supported (false negatives -> estimated recall)
Items are shuffled; 60 random items are also given to a second annotator (inter-rater agreement).
Human labels: supported / unsupported / not_a_claim / unclear (see validation/GUIDELINES.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402
V = Path(os.environ.get("VALIDATION_DIR", ROOT / "validation"))  # override only for testing
R = ROOT / "results"
SEED = 42
N_UNFLAGGED_PER_COND = 50
N_SECOND = 60
CONDS = ["C2_student_data", "C3_rag", "C4_shap_rag"]
LABELS = ["supported", "unsupported", "not_a_claim", "unclear"]


def context_block(prompt):
    return "CONTEXT" + prompt.split("CONTEXT", 1)[1].split("\n\nDo not introduce")[0].split("\n\nReturn")[0]


def prepare():
    V.mkdir(exist_ok=True)
    outs = {(o["enrolment_id"], o["condition"]): o for o in map(json.loads, open(R / "llm_outputs_main.jsonl"))}
    units = []
    for r in map(json.loads, open(R / "eval_audit_main.jsonl")):
        if r["condition"] not in CONDS:
            continue
        for k, u in enumerate(r["audit"]):
            if u.get("is_claim"):
                units.append({"enrolment_id": r["enrolment_id"], "condition": r["condition"], "unit_index": k,
                              "kind": u["kind"], "statement": u["text"], "evaluator_supported": u["supported"],
                              "evaluator_reasons": "; ".join(u["reasons"])})
    U = pd.DataFrame(units)
    flagged = U[~U.evaluator_supported].assign(stratum="flagged")
    unfl = pd.concat([U[(U.evaluator_supported) & (U.condition == c)].sample(N_UNFLAGGED_PER_COND, random_state=SEED)
                      for c in CONDS]).assign(stratum="unflagged_sample")
    S = pd.concat([flagged, unfl]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    S["item_id"] = [f"V{i + 1:03d}" for i in range(len(S))]
    S["context_given_to_LLM"] = [context_block(outs[(e, c)]["user_prompt"]) for e, c in zip(S.enrolment_id, S.condition)]
    S["kind_label"] = S.kind.map({"fact": "observed fact", "inference": "model inference", "rationale": "recommendation rationale"})
    # stratum sizes for recall weighting
    sizes = U.groupby(["condition", "evaluator_supported"]).size().rename("population").reset_index()
    sizes.to_csv(V / "_population_sizes.csv", index=False)
    rng = np.random.default_rng(SEED)
    S["second_annotator"] = False
    S.loc[rng.choice(len(S), N_SECOND, replace=False), "second_annotator"] = True
    S[["item_id", "enrolment_id", "condition", "unit_index", "kind", "stratum", "evaluator_supported",
       "evaluator_reasons", "second_annotator"]].to_csv(V / "_key_DO_NOT_OPEN.csv", index=False)
    blind = S[["item_id", "kind_label", "statement", "context_given_to_LLM"]].rename(
        columns={"kind_label": "statement_type"}).assign(label="", note="")
    for name, df in [("annotation_sheet_A", blind), ("annotation_sheet_B", blind[S.second_annotator.values])]:
        df.to_csv(V / f"{name}.csv", index=False)
        write_xlsx(df, V / f"{name}.xlsx")
    print(S.groupby(["stratum", "condition"]).size(), "\ntotal items:", len(S), "second annotator:", S.second_annotator.sum())


def write_xlsx(df, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "annotate"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    widths = {"item_id": 8, "statement_type": 16, "statement": 60, "context_given_to_LLM": 90, "label": 14, "note": 30}
    for i, col in enumerate(df.columns, 1):
        letter = ws.cell(1, i).column_letter
        ws.column_dimensions[letter].width = widths.get(col, 15)
        ws.cell(1, i).font = Font(bold=True)
        for r in range(2, len(df) + 2):
            ws.cell(r, i).alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    lab = list(df.columns).index("label") + 1
    dv.add(f"{ws.cell(2, lab).column_letter}2:{ws.cell(len(df) + 1, lab).column_letter}{len(df) + 1}")
    ws.freeze_panes = "C2"
    wb.save(path)


def read_labels(name):
    for ext in ["ods", "xlsx", "csv"]:  # the annotator's saved file wins (LibreOffice .ods first)
        f = V / f"{name}.{ext}"
        if f.exists():
            df = pd.read_excel(f, engine="odf") if ext == "ods" else pd.read_excel(f) if ext == "xlsx" else pd.read_csv(f)
            df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"nan": ""})
            return df[["item_id", "label"]]
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((np.mean([x == c for x in a]) * np.mean([y == c for y in b])) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def score():
    key = pd.read_csv(V / "_key_DO_NOT_OPEN.csv")
    pop = pd.read_csv(V / "_population_sizes.csv")
    A = read_labels("annotation_sheet_A")
    K = key.merge(A, on="item_id", how="left")
    done = K[K.label.isin(LABELS[:3])]
    print(f"labelled (excluding unclear/blank): {len(done)} of {len(K)}")
    out = []
    for cond in CONDS + ["ALL"]:
        d = done if cond == "ALL" else done[done.condition == cond]
        fl, un = d[d.stratum == "flagged"], d[d.stratum == "unflagged_sample"]
        tp = (fl.label == "unsupported").sum()
        prec = tp / len(fl) if len(fl) else np.nan
        fn_rate = (un.label == "unsupported").sum() / len(un) if len(un) else np.nan
        conds = CONDS if cond == "ALL" else [cond]
        n_flag = pop[(pop.condition.isin(conds)) & (~pop.evaluator_supported)].population.sum()
        n_unfl = pop[(pop.condition.isin(conds)) & (pop.evaluator_supported)].population.sum()
        if cond == "ALL":  # combine per-condition estimates (strata weights)
            tp_est = fn_est = 0.0
            for c in CONDS:
                dc = done[done.condition == c]
                f, u = dc[dc.stratum == "flagged"], dc[dc.stratum == "unflagged_sample"]
                nf = pop[(pop.condition == c) & (~pop.evaluator_supported)].population.sum()
                nu = pop[(pop.condition == c) & (pop.evaluator_supported)].population.sum()
                tp_est += (f.label == "unsupported").mean() * nf if len(f) else 0
                fn_est += (u.label == "unsupported").mean() * nu if len(u) else 0
        else:
            tp_est, fn_est = prec * n_flag, fn_rate * n_unfl
        lo, hi = wilson(tp, len(fl))
        rec = tp_est / (tp_est + fn_est) if (tp_est + fn_est) else np.nan
        out.append({"condition": cond, "flagged_labelled": len(fl), "precision": prec,
                    "est_f1": 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec + rec) else np.nan, "precision_ci95_low": lo,
                    "precision_ci95_high": hi, "false_negative_rate_in_unflagged": fn_rate,
                    "est_recall": tp_est / (tp_est + fn_est) if (tp_est + fn_est) else np.nan,
                    "evaluator_ucr_micro": n_flag / (n_flag + n_unfl),
                    "human_corrected_ucr_micro": (tp_est + fn_est) / (n_flag + n_unfl),
                    "flagged_not_a_claim_share": (fl.label == "not_a_claim").mean() if len(fl) else np.nan})
    O = pd.DataFrame(out)
    B = read_labels("annotation_sheet_B")
    if B is not None and (B.label != "").any():
        j = K[["item_id", "label"]].merge(B, on="item_id", suffixes=("_A", "_B"))
        j = j[j.label_A.isin(LABELS[:3]) & j.label_B.isin(LABELS[:3])]
        O.attrs["kappa"] = kappa(list(j.label_A), list(j.label_B))
        O.attrs["kappa_binary"] = kappa(list(j.label_A == "unsupported"), list(j.label_B == "unsupported"))
        O["kappa_3class"], O["kappa_binary_unsupported"], O["kappa_n"] = O.attrs["kappa"], O.attrs["kappa_binary"], len(j)
    O.to_csv((V / "_TEST_evaluator_validation.csv") if os.environ.get("VALIDATION_DIR") else (R / "evaluator_validation.csv"), index=False)
    pd.set_option("display.width", 250)
    print(O.round(3).to_string())
    errs = done[(done.stratum == "flagged") & (done.label != "unsupported")]
    print("\nflagged but judged NOT unsupported, by evaluator reason:")
    print(errs.evaluator_reasons.str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
