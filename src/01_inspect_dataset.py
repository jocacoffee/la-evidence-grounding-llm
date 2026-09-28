"""Step 1 - Inspection of the OULAD dataset (read-only).

Produces results/dataset_inspection.json with all statistics used in
DATASET_ANALYSIS.md, and figures/fig2_final_result_distribution.png.
The original CSV files are never modified.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "open+university+learning+analytics+dataset"
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

KEY = ["code_module", "code_presentation"]
SKEY = KEY + ["id_student"]
CUTOFF_DAY = 28  # candidate cutoff: end of week 4 (days 0..27 = weeks 1-4)

out = {}


def load(name):
    # OULAD encodes missing values as "?"
    return pd.read_csv(RAW / f"{name}.csv", na_values=["?"])


tables = {n: load(n) for n in ["courses", "assessments", "vle", "studentInfo",
                               "studentRegistration", "studentAssessment", "studentVle"]}

# ---------------------------------------------------------------- tables
out["tables"] = {}
for n, df in tables.items():
    out["tables"][n] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicated_rows": int(df.duplicated().sum()),
    }

c, a, v = tables["courses"], tables["assessments"], tables["vle"]
si, sr = tables["studentInfo"], tables["studentRegistration"]
sa, sv = tables["studentAssessment"], tables["studentVle"]

# ---------------------------------------------------------------- keys
out["keys"] = {
    "courses_unique_(module,presentation)": bool(not c.duplicated(KEY).any()),
    "studentInfo_unique_(module,presentation,student)": bool(not si.duplicated(SKEY).any()),
    "studentRegistration_unique_(module,presentation,student)": bool(not sr.duplicated(SKEY).any()),
    "assessments_unique_id_assessment": bool(a.id_assessment.is_unique),
    "vle_unique_id_site": bool(v.id_site.is_unique),
    "studentAssessment_unique_(assessment,student)": bool(not sa.duplicated(["id_assessment", "id_student"]).any()),
    "studentAssessment_ids_in_assessments": bool(sa.id_assessment.isin(a.id_assessment).all()),
    "studentVle_sites_in_vle": bool(sv.id_site.isin(v.id_site).all()),
    "info_vs_registration_same_rows": bool(
        si[SKEY].merge(sr[SKEY], how="outer", indicator=True)["_merge"].eq("both").all()),
}
# studentAssessment has no module/presentation -> recover via assessments
sa = sa.merge(a[["id_assessment"] + KEY + ["assessment_type", "date", "weight"]], on="id_assessment")
sa_enr = sa.merge(si[SKEY], on=SKEY, how="left", indicator=True)
out["keys"]["studentAssessment_rows_matching_an_enrolment"] = int((sa_enr._merge == "both").sum())
sv_enr = sv[SKEY].drop_duplicates().merge(si[SKEY], on=SKEY, how="left", indicator=True)
out["keys"]["studentVle_enrolments_matching_studentInfo"] = int((sv_enr._merge == "both").sum())
out["keys"]["studentVle_enrolments_total"] = int(len(sv_enr))

# ---------------------------------------------------------------- counts
out["counts"] = {
    "enrolments_(student x module-presentation)": int(len(si)),
    "unique_students": int(si.id_student.nunique()),
    "modules": int(c.code_module.nunique()),
    "module_presentations": int(len(c)),
    "presentations": sorted(c.code_presentation.unique().tolist()),
    "students_with_>1_enrolment": int((si.groupby("id_student").size() > 1).sum()),
    "students_repeating_same_module": int((si.groupby(["id_student", "code_module"]).size() > 1).sum()),
    "vle_materials": int(len(v)),
    "vle_log_rows": int(len(sv)),
    "total_clicks": int(sv.sum_click.sum()),
    "assessments": int(len(a)),
    "assessment_submissions": int(len(sa)),
}
out["courses"] = c.merge(si.groupby(KEY).size().rename("n_students").reset_index(), on=KEY) \
    .to_dict(orient="records")
out["assessment_types"] = a.assessment_type.value_counts().to_dict()
out["assessments_missing_date_by_type"] = a[a.date.isna()].assessment_type.value_counts().to_dict()

# ---------------------------------------------------------------- target
fr = si.final_result.value_counts()
out["final_result"] = {"counts": fr.to_dict(), "proportions": (fr / fr.sum()).round(4).to_dict()}
out["final_result_by_module_presentation"] = (
    pd.crosstab([si.code_module, si.code_presentation], si.final_result, normalize="index")
    .round(3).reset_index().to_dict(orient="records"))

# ---------------------------------------------------------------- demographics missing
out["demographic_missing"] = {col: int(si[col].isna().sum()) for col in si.columns}

# ---------------------------------------------------------------- temporal information
out["temporal"] = {
    "studentVle.date_range": [int(sv.date.min()), int(sv.date.max())],
    "studentVle_rows_before_day0": int((sv.date < 0).sum()),
    "assessments.date_range": [float(a.date.min()), float(a.date.max())],
    "studentAssessment.date_submitted_range": [int(sa.date_submitted.min()), int(sa.date_submitted.max())],
    "registration.date_registration_range": [float(sr.date_registration.min()), float(sr.date_registration.max())],
    "registration.date_unregistration_range": [float(sr.date_unregistration.min()), float(sr.date_unregistration.max())],
    "vle_week_from_missing_pct": round(float(v.week_from.isna().mean()), 4),
    "course_length_range_days": [int(c.module_presentation_length.min()), int(c.module_presentation_length.max())],
}

# ---------------------------------------------------------------- is_banked
out["banked"] = {
    "n_banked": int(sa.is_banked.sum()),
    "banked_date_submitted_describe": sa.loc[sa.is_banked == 1, "date_submitted"].describe().round(1).to_dict(),
}

# ---------------------------------------------------------------- withdrawal vs unregistration
m = si.merge(sr, on=SKEY)
out["withdrawal_consistency"] = pd.crosstab(m.final_result, m.date_unregistration.notna()).to_dict()
w = m[m.date_unregistration.notna()]
out["unregistration_date_describe"] = w.date_unregistration.describe().round(1).to_dict()

# ---------------------------------------------------------------- cutoff analysis (day 28)
cut = {}
cut["cutoff_day_exclusive"] = CUTOFF_DAY
cut["enrolments_unregistered_before_cutoff"] = int((m.date_unregistration < CUTOFF_DAY).sum())
cut["enrolments_unregistered_on_or_before_day0"] = int((m.date_unregistration <= 0).sum())
cut["enrolments_registered_after_cutoff"] = int((m.date_registration >= CUTOFF_DAY).sum())
cut["enrolments_missing_registration_date"] = int(m.date_registration.isna().sum())
# assessments with deadline before cutoff
a_pre = a[a.date < CUTOFF_DAY]
cut["assessments_due_before_cutoff_by_mp"] = (
    a_pre.groupby(KEY).assessment_type.apply(lambda s: s.value_counts().to_dict()).unstack(fill_value=0)
    .reset_index().to_dict(orient="records") if len(a_pre) else [])
cut["module_presentations_with_assessment_due_before_cutoff"] = int(a_pre[KEY].drop_duplicates().shape[0])
first_due = a.dropna(subset=["date"]).groupby(KEY).date.min()
cut["first_assessment_due_day_by_mp"] = {f"{k[0]}-{k[1]}": float(d) for k, d in first_due.items()}
# submissions made (non-banked) before cutoff
sa_pre = sa[(sa.date_submitted < CUTOFF_DAY) & (sa.is_banked == 0)]
cut["enrolments_with_>=1_submission_before_cutoff"] = int(sa_pre[SKEY].drop_duplicates().shape[0])
# vle activity before cutoff
sv_pre = sv[sv.date < CUTOFF_DAY]
act = sv_pre.groupby(SKEY).size()
cut["enrolments_with_vle_activity_before_cutoff"] = int(len(act))
# among those still registered at cutoff
still = m[~(m.date_unregistration < CUTOFF_DAY)]
cut["enrolments_still_registered_at_cutoff"] = int(len(still))
cut["final_result_still_registered_at_cutoff"] = still.final_result.value_counts().to_dict()
still_act = still.merge(act.rename("n").reset_index(), on=SKEY, how="left")
cut["still_registered_without_vle_activity_before_cutoff"] = int(still_act.n.isna().sum())
cut["final_result_of_still_registered_without_activity"] = \
    still_act[still_act.n.isna()].final_result.value_counts().to_dict()
out["cutoff_analysis"] = cut

# Sensitivity: how many assessments are due by weeks 2..8
out["assessments_due_by_week"] = {
    wk: int((a.date < 7 * wk).sum()) for wk in [2, 3, 4, 5, 6, 8]}
out["mp_with_assessment_due_by_week"] = {
    wk: int(a[a.date < 7 * wk][KEY].drop_duplicates().shape[0]) for wk in [2, 3, 4, 5, 6, 8]}

# ---------------------------------------------------------------- split feasibility
out["presentations_per_module"] = c.groupby("code_module").code_presentation.apply(sorted).to_dict()
yr = si.code_presentation.str[:4]
out["final_result_by_year"] = pd.crosstab(yr, si.final_result).to_dict()
out["enrolments_by_presentation"] = si.code_presentation.value_counts().sort_index().to_dict()

# ---------------------------------------------------------------- activity types
out["vle_activity_types"] = v.activity_type.value_counts().to_dict()
sv_types = sv.merge(v[["id_site", "activity_type"]], on="id_site")
out["clicks_by_activity_type"] = sv_types.groupby("activity_type").sum_click.sum() \
    .sort_values(ascending=False).to_dict()

with open(RES / "dataset_inspection.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# ---------------------------------------------------------------- Figure 2
order = ["Distinction", "Pass", "Fail", "Withdrawn"]
counts = fr.reindex(order)
fig, ax = plt.subplots(figsize=(5, 3.2))
bars = ax.bar(order, counts.values, color=["#2b8a3e", "#74b816", "#e8590c", "#868e96"])
for b, n in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, n, f"{n:,}\n({n / counts.sum():.1%})",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Enrolments")
ax.set_title(f"OULAD final_result (N = {counts.sum():,} enrolments)", fontsize=10)
ax.set_ylim(0, counts.max() * 1.22)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig2_final_result_distribution.png", dpi=300)

print(json.dumps({k: out[k] for k in ["counts", "keys", "final_result", "temporal", "banked",
                                      "withdrawal_consistency", "unregistration_date_describe",
                                      "assessment_types", "assessments_missing_date_by_type",
                                      "assessments_due_by_week", "mp_with_assessment_due_by_week",
                                      "enrolments_by_presentation", "final_result_by_year",
                                      "presentations_per_module", "demographic_missing"]},
                 indent=1, default=str))
print(json.dumps({k: v for k, v in cut.items() if k != "assessments_due_before_cutoff_by_mp"}, indent=1, default=str))
print(json.dumps(cut["assessments_due_before_cutoff_by_mp"], default=str))
for n, t in out["tables"].items():
    print(n, t["rows"], t["missing"], "dups:", t["duplicated_rows"])
print(pd.DataFrame(out["courses"]).to_string())
print(out["clicks_by_activity_type"])

# ================================================================ extra checks (appended)
extra = {}
# registrations with missing date
extra["missing_registration_date_final_result"] = m[m.date_registration.isna()].final_result.value_counts().to_dict()
extra["missing_registration_date_unregistered"] = int(m[m.date_registration.isna()].date_unregistration.notna().sum())
# studentVle duplicated rows: same student/site/day/clicks -> no intra-day timestamp, cannot distinguish
extra["studentVle_duplicate_rows"] = int(sv.duplicated().sum())
extra["studentVle_duplicate_key_rows_(student,site,date)"] = int(sv.duplicated(SKEY + ["id_site", "date"]).sum())
# score missing
extra["score_missing_by_type"] = sa[sa.score.isna()].assessment_type.value_counts().to_dict()
# submissions before cutoff for assessments whose deadline is after the cutoff (early submissions)
extra["pre_cutoff_submissions_with_deadline_after_cutoff"] = int(
    ((sa.date_submitted < CUTOFF_DAY) & (sa.date >= CUTOFF_DAY) & (sa.is_banked == 0)).sum())
extra["pre_cutoff_submissions_total_nonbanked"] = int(len(sa_pre))
extra["pre_cutoff_submissions_by_type"] = sa_pre.assessment_type.value_counts().to_dict()
# late submissions overall
extra["late_submission_rate_nonbanked"] = round(float(
    (sa.loc[sa.is_banked == 0, "date_submitted"] > sa.loc[sa.is_banked == 0, "date"]).mean()), 4)
# assessments due before cutoff: weights
extra["assessments_due_before_cutoff"] = a_pre[KEY + ["id_assessment", "assessment_type", "date", "weight"]] \
    .to_dict(orient="records")
# exams: submissions exist?
extra["exam_submissions"] = int((sa.assessment_type == "Exam").sum())
# weights per mp
extra["weight_sum_non_exam_by_mp"] = a[a.assessment_type != "Exam"].groupby(KEY).weight.sum() \
    .reset_index().to_dict(orient="records")
# split feasibility: train = 2013B, 2013J, 2014B ; test = 2014J
pop = m[~(m.date_unregistration < CUTOFF_DAY) & ~(m.date_registration >= CUTOFF_DAY)]
extra["population_after_exclusions"] = int(len(pop))
extra["population_final_result"] = pop.final_result.value_counts().to_dict()
tr, te = pop[pop.code_presentation != "2014J"], pop[pop.code_presentation == "2014J"]
extra["split_train_n"] = int(len(tr))
extra["split_test_n"] = int(len(te))
extra["split_train_final_result"] = tr.final_result.value_counts().to_dict()
extra["split_test_final_result"] = te.final_result.value_counts().to_dict()
extra["students_in_both_train_and_test"] = int(len(set(tr.id_student) & set(te.id_student)))
extra["test_modules_seen_in_train"] = sorted(set(te.code_module) & set(tr.code_module))
# alternative split 2013 -> 2014
extra["modules_2014_not_in_2013"] = sorted(set(pop[pop.code_presentation.str.startswith("2014")].code_module)
                                           - set(pop[pop.code_presentation.str.startswith("2013")].code_module))
# withdrawals happening after cutoff: timing distribution within population
wpop = pop[pop.final_result == "Withdrawn"]
extra["withdrawn_in_population_unreg_date_quantiles"] = wpop.date_unregistration.quantile(
    [0, .1, .25, .5, .75, .9, 1]).to_dict()
extra["withdrawn_in_population_missing_unreg_date"] = int(wpop.date_unregistration.isna().sum())
# pre-course VLE activity
extra["enrolments_with_pre_start_vle_activity"] = int(sv[sv.date < 0][SKEY].drop_duplicates().shape[0])
# vle week_from missing by module
extra["vle_week_from_available_by_module"] = v.groupby("code_module").week_from.apply(
    lambda s: round(float(s.notna().mean()), 3)).to_dict()
out["extra"] = extra
with open(RES / "dataset_inspection.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(extra, indent=1, default=str))
