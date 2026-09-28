"""Feature dictionary shared by the evidence store (05) and the deterministic evaluator (08).

For each feature:
  type      evidence type
  category  observed           - count/value read directly from OULAD logs up to the cutoff
            observed_derived   - deterministic statistic computed from observed values
            static             - recorded at enrolment
  source    OULAD table(s)
  label     human-readable statement template ({v} = formatted value)
  fmt       how the value is formatted / which numeric forms count as a match in text
            int -> integer; pct -> value*100 (%); ratio -> 2 decimals; index -> signed 2 decimals
            (activity_trend; changed from pct after the pilot, see report); score -> 0-100 score
  lex       regex lexicon used by the evaluator to detect a reference to this feature
Frozen after the pilot (see EXPERIMENT_REPORT.md, "Evaluator development").
"""

FEATURES = {
    "total_clicks": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="Total VLE clicks from registration up to the end of week 4: {v}",
        lex=r"\b(total|overall)\b[^.;]{0,30}\b(clicks?|interactions?)\b|\bclicks? (in|on) (the )?vle\b|\bvle (clicks?|interactions?)\b"),
    "active_days": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="Number of distinct days with VLE activity up to the end of week 4: {v}",
        lex=r"\bactive days?\b|\bdays? (of|with) (vle )?activity\b|\bdays? active\b|\bactive on\b[^.;]{0,25}\bdays\b|\bdistinct days\b"),
    "distinct_materials": dict(
        type="vle_engagement", category="observed", source="studentVle+vle", fmt="int",
        label="Number of distinct VLE materials accessed up to the end of week 4: {v}",
        lex=r"\b(distinct|different|unique)\b[^.;]{0,20}\b(materials?|resources?|items?|pages?)\b|\bmaterials? accessed\b"),
    "pre_course_clicks": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="VLE clicks before the module start (day < 0): {v}",
        lex=r"\bpre-?course\b|\bbefore the (module|course) (start|began|started)|\bprior to the (module|course) start"),
    "clicks_weeks_1_2": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="VLE clicks in weeks 1-2 (days 0-13): {v}",
        lex=r"\bweeks? 1\s*(-|–|to|and)\s*2\b|\bfirst two weeks\b"),
    "clicks_weeks_3_4": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="VLE clicks in weeks 3-4 (days 14-27): {v}",
        lex=r"\bweeks? 3\s*(-|–|to|and)\s*4\b|\blast two weeks\b|\bmost recent two weeks\b"),
    "clicks_week_4": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="VLE clicks in week 4 (days 21-27): {v}",
        lex=r"\b(in|during) week 4\b(?! of)|\bweek 4 (clicks|activity)\b|\brecent (week|activity)\b|\blast week\b|\bfinal week\b"),
    "active_days_weeks_3_4": dict(
        type="vle_engagement", category="observed", source="studentVle", fmt="int",
        label="Days with VLE activity in weeks 3-4 (out of 14): {v}",
        lex=r"\bactive days? in weeks? 3\b|\bdays? (of|with) activity in weeks? 3|\bof the (last )?14 days\b|\bweeks? 3\s*(-|–|and|to)\s*4\b[^.;]{0,40}\bdays\b|\bdays\b[^.;]{0,40}\bweeks? 3\s*(-|–|and|to)\s*4\b"),
    "clicks_content": dict(
        type="vle_engagement", category="observed", source="studentVle+vle", fmt="int",
        label="Clicks on course content pages (oucontent) up to the end of week 4: {v}",
        lex=r"\b(course |learning )?content\b[^.;]{0,15}\bclicks?\b|\bclicks? on (course )?content\b|\boucontent\b"),
    "clicks_forum": dict(
        type="vle_engagement", category="observed", source="studentVle+vle", fmt="int",
        label="Clicks on discussion forums (forumng) up to the end of week 4: {v}",
        lex=r"\bforums?\b|\bdiscussions?\b|\bforumng\b"),
    "clicks_quiz": dict(
        type="vle_engagement", category="observed", source="studentVle+vle", fmt="int",
        label="Clicks on quizzes up to the end of week 4: {v}",
        lex=r"\bquiz(zes)?\b"),
    "clicks_homepage": dict(
        type="vle_engagement", category="observed", source="studentVle+vle", fmt="int",
        label="Clicks on the module homepage up to the end of week 4: {v}",
        lex=r"\bhome ?page\b"),
    "clicks_resource": dict(
        type="vle_engagement", category="observed", source="studentVle+vle", fmt="int",
        label="Clicks on resources/URLs/subpages up to the end of week 4: {v}",
        lex=r"\bresources?\b[^.;]{0,15}\bclicks?\b|\bclicks? on resources\b|\bsubpages?\b|\burls?\b"),
    "activity_trend": dict(
        type="vle_trend", category="observed_derived", source="studentVle", fmt="index",
        label=("Activity trend index = (clicks weeks 3-4 - clicks weeks 1-2) / (clicks weeks 1-2 + clicks weeks 3-4); "
               "-1 = all activity in weeks 1-2, 0 = equal, +1 = all activity in weeks 3-4 (not a percent change): {v}"),
        lex=r"\b(activity|engagement|click)s? trend\b|\btrend\b|\b(declin|decreas|drop|increas|ris)\w*\b[^.;]{0,30}\b(activity|engagement|clicks?)\b|\b(activity|engagement|clicks?)\b[^.;]{0,30}\b(declin|decreas|drop|increas|fell|rose)\w*"),
    "days_since_last_activity": dict(
        type="vle_recency", category="observed_derived", source="studentVle", fmt="int",
        label="Days between the last VLE activity and the end of week 4: {v}",
        lex=r"\bdays? since\b|\blast (vle )?(activity|login|access|interaction)\b|\binactiv\w*|\bno activity (for|in the last)\b|\bdays? ago\b"),
    "clicks_vs_cohort_median": dict(
        type="vle_engagement_relative", category="observed_derived", source="studentVle", fmt="ratio",
        label="Total clicks relative to the median of the same module-presentation cohort (1.0 = median): {v}",
        lex=r"\bcohort\b|\bmedian\b|\bpeers?\b|\bclassmates\b|\bcompared (to|with) (other|the) (students|class)\b|\brelative to (other|the) (students|class)\b|\btimes the\b|\bbelow average\b|\babove average\b"),
    "n_assessments_due": dict(
        type="assessment_schedule", category="observed", source="assessments", fmt="int",
        label="Number of assessments (TMA/CMA) with a deadline before the end of week 4: {v}",
        lex=r"\bassessments? (were |was )?(due|scheduled)\b|\bdue(?! to)\b|\bdeadlines?\b|\bno assessments?\b"),
    "n_submitted_due": dict(
        type="assessment_submission", category="observed", source="studentAssessment", fmt="int",
        label="Number of those assessments submitted before the end of week 4: {v}",
        lex=r"\bsubmitted\b|\bsubmissions? of (the )?(due|required)\b|\bnot submitted\b|\bmissed\b|\bmissing (assessment|submission)"),
    "submission_rate": dict(
        type="assessment_submission", category="observed_derived", source="studentAssessment+assessments", fmt="pct",
        label="Submission rate of assessments due before the end of week 4: {v}",
        lex=r"\bsubmission rate\b|\bsubmission\b|\bsubmitted\b|\bcompleted\b[^.;]{0,20}\bassessments?\b"),
    "n_submissions": dict(
        type="assessment_submission", category="observed", source="studentAssessment", fmt="int",
        label="Total assessment submissions (any deadline) before the end of week 4: {v}",
        lex=r"\bsubmissions?\b|\bsubmitted\b"),
    "mean_score": dict(
        type="assessment_performance", category="observed", source="studentAssessment", fmt="score",
        label="Mean score (0-100) of assessments submitted before the end of week 4: {v}",
        lex=r"\bscores?\b|\bgrades?\b|\bmarks?\b|\bassessment (average|performance|result)\b|\bperformed\b|\bpass(ing)? mark\b"),
    "n_late_submissions": dict(
        type="assessment_submission", category="observed", source="studentAssessment+assessments", fmt="int",
        label="Assessments submitted after their deadline (before the end of week 4): {v}",
        lex=r"\blate\b|\bafter (the|its) deadline\b|\bpast the deadline\b|\boverdue\b"),
    "has_banked_assessment": dict(
        type="assessment_history", category="observed", source="studentAssessment", fmt="bool",
        label="Has an assessment result transferred (banked) from a previous presentation: {v}",
        lex=r"\bbanked\b|\btransferred\b|\bcarried over\b"),
    "num_of_prev_attempts": dict(
        type="academic_history", category="static", source="studentInfo", fmt="int",
        label="Number of previous attempts at this module: {v}",
        lex=r"\bprevious(ly)? attempt|\bprior attempt|\bfirst attempt\b|\brepeat(ing|ed)?\b|\bretak|\battempted (this|the) module"),
    "studied_credits": dict(
        type="academic_load", category="static", source="studentInfo", fmt="int",
        label="Total credits of modules being studied in this presentation: {v}",
        lex=r"\bcredits?\b|\bstudy load\b|\bworkload\b"),
    "highest_education": dict(
        type="academic_background", category="static", source="studentInfo", fmt="edu",
        label="Highest education level on entry: {v}",
        lex=r"\beducation(al)? (level|background)\b|\bqualifications?\b|\ba levels?\b|\bhe qualification\b|\bpost ?graduate\b|\bno formal\b"),
    "date_registration": dict(
        type="registration", category="static", source="studentRegistration", fmt="regdays",
        label="Registration day relative to module start (negative = days before start): {v}",
        lex=r"\bregist(ered|ration)\b|\benrol(l)?ed\b[^.;]{0,20}\bdays?\b"),
}

# model inference items (not observed facts)
INFERENCE = {
    "predicted_risk": dict(
        type="model_prediction", category="model_inference", source="XGBoost (this study)", fmt="pct",
        label="Model-estimated probability of an unsuccessful outcome (Fail or Withdrawn): {v}",
        lex=r"\brisk\b|\bprobability\b|\blikelihood\b|\bpredicted\b|\bmodel (estimates|predicts|suggests)\b|\bchance\b"),
    "predicted_class": dict(
        type="model_prediction", category="model_inference", source="XGBoost (this study)", fmt="text",
        label="Most likely outcome class according to the model: {v}",
        lex=r"\bpredicted (outcome|class|result)\b|\bmost likely (outcome|result)\b"),
}

EDU_LABELS = {0: "No Formal quals", 1: "Lower Than A Level", 2: "A Level or Equivalent",
              3: "HE Qualification", 4: "Post Graduate Qualification"}

# Claims about attributes that OULAD does not contain (never supportable by the evidence store)
NON_OBSERVABLE = (
    r"\bstruggl\w*\b|\bdifficult(y|ies) (with|in|understanding)\b|\bconfus\w*\b|\bmotivat\w*\b|\bstress\w*\b|"
    r"\banxi\w*\b|\boverwhelm\w*\b|\bpersonal (issues|problems|circumstances)\b|\bfamily\b|\bjob\b|"
    r"\bemployment\b|\bhealth\b|\billness\b|\bmental\b|\bburn-?out\b|\bdisengaged from (the )?(topic|subject)\b|"
    r"\blacks? (understanding|knowledge|skills?)\b|\bunderstanding of\b|\bprocrastinat\w*\b|"
    r"\btime management\b|\blearning (style|disabilit\w*)\b|\bdyslex\w*\b|\bcompeting (commitments|priorities)\b|"
    r"\bwork commitments\b|\bcaring responsibilit\w*\b|\b(is|are|seems?|appears?) (lazy|uninterested|bored|frustrated)\b|"
    r"\btechnical (issues|problems|difficulties)\b|\binternet (access|connection)\b|\bconfidence\b"
)
