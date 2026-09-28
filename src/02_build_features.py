"""Step 2 - Temporal Learning Analytics features at the cutoff.

Only events with day < cutoff are used. Population: enrolments registered before the
cutoff and not unregistered before the cutoff (DATASET_ANALYSIS.md §6.1).
Undefined features are left as NaN (e.g. submission_rate when no assessment was due),
so that "not applicable" is never confused with 0.
Output: data/features.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
RAW = (ROOT / CFG["raw_dir"]).resolve()
CUT = CFG["cutoff_day"]
KEY = ["code_module", "code_presentation"]
SKEY = KEY + ["id_student"]
EDU = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
       "HE Qualification": 3, "Post Graduate Qualification": 4}


def load(n):
    return pd.read_csv(RAW / f"{n}.csv", na_values=["?"])


si, sr = load("studentInfo"), load("studentRegistration")
a, sa, v, sv = load("assessments"), load("studentAssessment"), load("vle"), load("studentVle")

# ---------------------------------------------------------------- population
pop = si.merge(sr, on=SKEY)
pop = pop[~(pop.date_unregistration < CUT) & ~(pop.date_registration >= CUT)].copy()
base = pop[SKEY + ["final_result", "num_of_prev_attempts", "studied_credits",
                   "highest_education", "date_registration"]].copy()
base["highest_education"] = base.highest_education.map(EDU)

# ---------------------------------------------------------------- VLE features (day < CUT)
sv = sv[sv.date < CUT].merge(v[["id_site", "activity_type"]], on="id_site")
sv = sv.merge(base[SKEY], on=SKEY)  # restrict to population
g = sv.groupby(SKEY)
vle = pd.DataFrame({
    "total_clicks": g.sum_click.sum(),
    "active_days": g.date.nunique(),
    "distinct_materials": g.id_site.nunique(),
    "last_active_day": g.date.max(),
})
def clicks_in(mask):
    return sv[mask].groupby(SKEY).sum_click.sum()
vle["pre_course_clicks"] = clicks_in(sv.date < 0)
vle["clicks_weeks_1_2"] = clicks_in((sv.date >= 0) & (sv.date < 14))
vle["clicks_weeks_3_4"] = clicks_in((sv.date >= 14) & (sv.date < 28))
vle["clicks_week_4"] = clicks_in((sv.date >= 21) & (sv.date < 28))
vle["active_days_weeks_3_4"] = sv[(sv.date >= 14) & (sv.date < 28)].groupby(SKEY).date.nunique()
for col, types in {"clicks_content": ["oucontent"], "clicks_forum": ["forumng"],
                   "clicks_quiz": ["quiz"], "clicks_homepage": ["homepage"],
                   "clicks_resource": ["resource", "url", "subpage", "page"]}.items():
    vle[col] = clicks_in(sv.activity_type.isin(types))
vle = vle.reset_index()

df = base.merge(vle, on=SKEY, how="left")
count_cols = ["total_clicks", "active_days", "distinct_materials", "pre_course_clicks",
              "clicks_weeks_1_2", "clicks_weeks_3_4", "clicks_week_4", "active_days_weeks_3_4",
              "clicks_content", "clicks_forum", "clicks_quiz", "clicks_homepage", "clicks_resource"]
df[count_cols] = df[count_cols].fillna(0)  # no logged event = 0 clicks (observed absence)
# trend in [-1, 1]: (w3-4 - w1-2) / (w1-2 + w3-4); undefined (NaN) when no activity in weeks 1-4
den = df.clicks_weeks_1_2 + df.clicks_weeks_3_4
df["activity_trend"] = np.where(den > 0, (df.clicks_weeks_3_4 - df.clicks_weeks_1_2) / den.replace(0, 1), np.nan)
# days between last logged activity and the last observable day (CUT-1); NaN if never active
df["days_since_last_activity"] = (CUT - 1) - df.last_active_day
df = df.drop(columns="last_active_day")
# cohort-relative engagement: clicks / median clicks of the same module-presentation at the cutoff
# (uses only same-cutoff information of the cohort, no future information)
med = df.groupby(KEY).total_clicks.transform("median")
df["clicks_vs_cohort_median"] = df.total_clicks / med.replace(0, np.nan)

# ---------------------------------------------------------------- assessment features
sa = sa.merge(a[["id_assessment"] + KEY + ["assessment_type", "date", "weight"]], on="id_assessment")
sa = sa.merge(base[SKEY], on=SKEY)
due = a[(a.date < CUT) & (a.assessment_type != "Exam")].groupby(KEY).size().rename("n_assessments_due")
df = df.merge(due.reset_index(), on=KEY, how="left")
df["n_assessments_due"] = df.n_assessments_due.fillna(0).astype(int)

own = sa[(sa.is_banked == 0) & (sa.date_submitted < CUT)]
own_due = own[own.date < CUT]
agg = pd.DataFrame({
    "n_submissions": own.groupby(SKEY).size(),
    "n_submitted_due": own_due.groupby(SKEY).size(),
    "mean_score": own.groupby(SKEY).score.mean(),       # NaN scores (173 TMA) ignored
    "n_late_submissions": own[own.date_submitted > own.date].groupby(SKEY).size(),
}).reset_index()
df = df.merge(agg, on=SKEY, how="left")
for c in ["n_submissions", "n_submitted_due", "n_late_submissions"]:
    df[c] = df[c].fillna(0).astype(int)
df["submission_rate"] = np.where(df.n_assessments_due > 0,
                                 df.n_submitted_due / df.n_assessments_due.replace(0, 1), np.nan)
banked = sa[sa.is_banked == 1][SKEY].drop_duplicates().assign(has_banked_assessment=1)
df = df.merge(banked, on=SKEY, how="left")
df["has_banked_assessment"] = df.has_banked_assessment.fillna(0).astype(int)

# ---------------------------------------------------------------- split
df["split"] = np.where(df.code_presentation.isin(CFG["test_presentations"]), "test", "train")
df["enrolment_id"] = df.code_module + "_" + df.code_presentation + "_" + df.id_student.astype(str)
df.to_csv(ROOT / "data" / "features.csv", index=False)

FEATURES = [c for c in df.columns if c not in SKEY + ["final_result", "split", "enrolment_id"]]
json.dump({"features": FEATURES}, open(ROOT / "configs" / "features.json", "w"), indent=2)
print(df.shape, df.split.value_counts().to_dict())
print(df[FEATURES].describe().T.round(2).to_string())
print("NaN counts:", df[FEATURES].isna().sum()[lambda s: s > 0].to_dict())
