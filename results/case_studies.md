# Case studies (automatically selected)

Set: main. Selection rules are fixed in `src/09_compare.py`; ties broken by enrolment_id. Metrics are deterministic (see EXPERIMENT_REPORT.md). The true final_result is shown only for the reader; it was never given to the LLM.


## 1. Retrieval reduces unsupported claims

*Selection rule:* largest positive difference UCR(C2) - min(UCR(C3), UCR(C4)) over students with claims in all three

Enrolment `EEE_2014J_2592973` (true final_result: Pass)

**LLM + Student Data** (`C2_student_data`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - STUDENT DATA (structured record from the learning analytics database; null = not available / not applicable):
{
 "total_clicks": 925.0,
 "active_days": 25.0,
 "distinct_materials": 25.0,
 "pre_course_clicks": 86.0,
 "clicks_weeks_1_2": 624.0,
 "clicks_weeks_3_4": 215.0,
 "clicks_week_4": 64.0,
 "active_days_weeks_3_4": 10.0,
 "clicks_content": 525.0,
 "clicks_forum": 50.0,
 "clicks_quiz": 180.0,
 "clicks_homepage": 131.0,
 "clicks_resource": 34.0,
 "activity_trend": -0.4875,
 "days_since_last_activity": 2.0,
 "clicks_vs_cohort_median": 2.8418,
 "n_assessments_due": 0.0,
 "n_submitted_due": 0.0,
 "submission_rate": null,
 "n_submissions": 0.0,
 "mean_score": null,
 "n_late_submissions": 0.0,
 "has_banked_assessment": 0.0,
 "num_of_prev_attempts": 0.0,
 "studied_credits": 30.0,
 "highest_education": 1.0,
 "date_registration": -115.0
}
DATA DICTIONARY:
{
 "total_clicks": "Total VLE clicks from registration up to the end of week 4",
 "active_days": "Number of distinct days with VLE activity up to the end of week 4",
 "distinct_materials": "Number of distinct VLE materials accessed up to the end of week 4",
 "pre_course_clicks": "VLE clicks before the module start (day < 0)",
 "clicks_weeks_1_2": "VLE clicks in weeks 1-2 (days 0-13)",
 "clicks_weeks_3_4": "VLE clicks in weeks 3-4 (days 14-27)",
 "clicks_week_4": "VLE clicks in week 4 (days 21-27)",
 "active_days_weeks_3_4": "Days with VLE activity in weeks 3-4 (out of 14)",
 "clicks_content": "Clicks on course content pages (oucontent) up to the end of week 4",
 "clicks_forum": "Clicks on discussion forums (forumng) up to the end of week 4",
 "clicks_quiz": "Clicks on quizzes up to the end of week 4",
 "clicks_homepage": "Clicks on the module homepage up to the end of week 4",
 "clicks_resource": "Clicks on resources/URLs/subpages up to the end of week 4",
 "activity_trend": "Activity trend index = (clicks weeks 3-4 - clicks weeks 1-2) / (clicks weeks 1-2 + clicks weeks 3-4); -1 = all activity in weeks 1-2, 0 = equal, +1 = all activity in weeks 3-4 (not a percent change)",
 "days_since_last_activity": "Days between the last VLE activity and the end of week 4",
 "clicks_vs_cohort_median": "Total clicks relative to the median of the same module-presentation cohort (1.0 = median)",
 "n_assessments_due": "Number of assessments (TMA/CMA) with a deadline before the end of week 4",
 "n_submitted_due": "Number of those assessments submitted before the end of week 4",
 "submission_rate": "Submission rate of assessments due before the end of week 4",
 "n_submissions": "Total assessment submissions (any deadline) before the end of week 4",
 "mean_score": "Mean score (0-100) of assessments submitted before the end of week 4",
 "n_late_submissions": "Assessments submitted after their deadline (before the end of week 4)",
 "has_banked_assessment": "Has an assessment result transferred (banked) from a previous presentation",
 "num_of_prev_attempts": "Number of previous attempts at this module",
 "studied_credits": "Total credits of modules being studied in this presentation",
 "highest_education": "Highest education level on entry",
 "date_registration": "Registration day relative to module start (negative = days before start)"
}
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has high total engagement relative to the cohort median (2.84x).",
   "evidence_ids": [
    "clicks_vs_cohort_median"
   ]
  },
  {
   "statement": "The student has accessed a wide variety of materials (25 distinct materials).",
   "evidence_ids": [
    "distinct_materials"
   ]
  },
  {
   "statement": "The student has high total click volume (925 clicks).",
   "evidence_ids": [
    "total_clicks"
   ]
  },
  {
   "statement": "Activity has decreased in the most recent period (weeks 3-4) compared to weeks 1-2.",
   "evidence_ids": [
    "activity_trend"
   ]
  },
  {
   "statement": "The student has not yet submitted any assessments due by the end of week 4.",
   "evidence_ids": [
    "n_submitted_due",
    "n_assessments_due"
   ]
  },
  {
   "statement": "The student has not completed any assessments (submitted or late) by the end of week 4.",
   "evidence_ids": [
    "n_submissions",
    "n_late_submissions"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The student is at risk of missing upcoming assessment deadlines due to a lack of submissions.",
   "evidence_ids": [
    "n_submitted_due",
    "n_assessments_due",
    "activity_trend"
   ]
  },
  {
   "statement": "The student's recent activity rate is lower than their initial activity rate.",
   "evidence_ids": [
    "activity_trend"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Monitor the student's assessment submission status closely.",
   "rationale": "The student has zero submissions for assessments due by the end of week 4, which may indicate a risk of missing deadlines.",
   "evidence_ids": [
    "n_submitted_due",
    "n_assessments_due"
   ]
  },
  {
   "action": "Encourage the student to maintain their high engagement levels to ensure they complete upcoming assessments on time.",
   "rationale": "While the student's total activity is high, the activity trend is negative, suggesting a drop in engagement that could impact timely completion of assessments.",
   "evidence_ids": [
    "activity_trend",
    "clicks_vs_cohort_median"
   ]
  }
 ],
 "evidence_limitations": [
  "The data does not specify the exact deadlines for the assessments due, making it impossible to determine if the current lack of submissions is critical or if the deadlines are further out.",
  "The data does not provide information on the student's specific learning difficulties or content understanding, only on engagement metrics."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has high total engagement relative to the cohort median (2.84x). | True | True | — |
| fact | The student has accessed a wide variety of materials (25 distinct materials). | True | True | — |
| fact | The student has high total click volume (925 clicks). | True | True | — |
| fact | Activity has decreased in the most recent period (weeks 3-4) compared to weeks 1-2. | True | True | — |
| fact | The student has not yet submitted any assessments due by the end of week 4. | True | True | — |
| fact | The student has not completed any assessments (submitted or late) by the end of week 4. | False | True | risk_assertion_without_model_output |
| inference | The student is at risk of missing upcoming assessment deadlines due to a lack of submissions. | False | True | risk_assertion_without_model_output |
| inference | The student's recent activity rate is lower than their initial activity rate. | True | True | — |
| rationale | The student has zero submissions for assessments due by the end of week 4, which may indicate a risk of missing deadlines. | True | True | — |
| rationale | While the student's total activity is high, the activity trend is negative, suggesting a drop in engagement that could impact timely completion of assessments. | True | True | — |

Per-output metrics: UCR=0.2, coverage=1.0, factual consistency=1.0, SHAP consistency=0.67 (top-3: clicks_vs_cohort_median, n_assessments_due, highest_education; mentioned: clicks_vs_cohort_median, n_assessments_due), specificity=1.0

**LLM + RAG** (`C3_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E9] Clicks on course content pages (oucontent) up to the end of week 4: 525
[E5] VLE clicks in weeks 1-2 (days 0-13): 624
[E11] Clicks on quizzes up to the end of week 4: 180
[E1] Total VLE clicks from registration up to the end of week 4: 925
[E16] Total clicks relative to the median of the same module-presentation cohort (1.0 = median): 2.84
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has clicked on 525 course content pages by the end of week 4.",
   "evidence_ids": [
    "E9"
   ]
  },
  {
   "statement": "The student has clicked on 180 quizzes by the end of week 4.",
   "evidence_ids": [
    "E11"
   ]
  },
  {
   "statement": "The student's total VLE clicks (925) are 2.84 times the median of the cohort.",
   "evidence_ids": [
    "E1",
    "E16"
   ]
  },
  {
   "statement": "The student's activity in weeks 1-2 (624 clicks) represents a significant portion of their total activity.",
   "evidence_ids": [
    "E5",
    "E1"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The student demonstrates high engagement relative to peers, as their click volume is well above the cohort median.",
   "evidence_ids": [
    "E16"
   ]
  },
  {
   "statement": "The student's engagement rate appears to have decreased in week 3 and 4 compared to weeks 1-2, as the increase in total clicks from week 2 to week 4 (925 - 624 = 301) is lower than the activity in the first two weeks.",
   "evidence_ids": [
    "E1",
    "E5"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Monitor the trend in content page clicks (E9) and quiz attempts (E11) for weeks 5 and 6 to determine if the engagement rate stabilizes or continues to decline.",
   "rationale": "The data suggests a potential drop in activity rate in the later weeks of the module compared to the initial weeks.",
   "evidence_ids": [
    "E1",
    "E5",
    "E9",
    "E11"
   ]
  },
  {
   "action": "Consider sending a general check-in message to the student to acknowledge their high overall engagement and ask if they are finding the later weeks of the module challenging.",
   "rationale": "While the student is above the median, the deceleration in click volume might indicate a need for support or a shift in focus.",
   "evidence_ids": [
    "E16"
   ]
  }
 ],
 "evidence_limitations": [
  "The data only covers the first four weeks of the module; long-term engagement trends cannot be determined.",
  "The data does not distinguish between clicks on specific topics or resources, making it impossible to identify specific content areas where the student may be struggling or disengaging.",
  "There is no data on assessment scores or completion rates, so the quality of the student's learning cannot be inferred from click counts alone."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has clicked on 525 course content pages by the end of week 4. | True | False | — |
| fact | The student has clicked on 180 quizzes by the end of week 4. | True | True | — |
| fact | The student's total VLE clicks (925) are 2.84 times the median of the cohort. | True | True | — |
| fact | The student's activity in weeks 1-2 (624 clicks) represents a significant portion of their total activity. | True | True | — |
| inference | The student demonstrates high engagement relative to peers, as their click volume is well above the cohort median. | True | True | — |
| inference | The student's engagement rate appears to have decreased in week 3 and 4 compared to weeks 1-2, as the increase in total clicks from week 2 to week 4 (925 - 624 = 301) is lower than the activity in the first two weeks. | False | False | number_not_in_context:['301'] |
| rationale | The data suggests a potential drop in activity rate in the later weeks of the module compared to the initial weeks. | False | True | no_referenced_feature_in_context:['activity_trend'] |
| rationale | While the student is above the median, the deceleration in click volume might indicate a need for support or a shift in focus. | True | True | — |

Per-output metrics: UCR=0.25, coverage=0.75, factual consistency=0.8571428571428571, SHAP consistency=0.33 (top-3: clicks_vs_cohort_median, n_assessments_due, highest_education; mentioned: clicks_vs_cohort_median), specificity=1.0

**LLM + SHAP + RAG** (`C4_shap_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - PREDICTIVE MODEL OUTPUT (model inference, not an observed fact):
[M1] Model-estimated probability of an unsuccessful outcome (Fail or Withdrawn): 29% (model inference, not an observed fact)

CONTEXT - MODEL EXPLANATION (SHAP: features that most changed the model's predicted risk for this student; associations in the model, not causes):
[X1] SHAP rank 1: clicks_vs_cohort_median (value 2.84, see E16) decreases predicted risk by 11.2 percentage points (association in the model, not a cause)
[X2] SHAP rank 2: n_assessments_due (value 0, see E17) decreases predicted risk by 6.9 percentage points (association in the model, not a cause)
[X3] SHAP rank 3: highest_education (value Lower Than A Level, see E26) increases predicted risk by 5.6 percentage points (association in the model, not a cause)

CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E1] Total VLE clicks from registration up to the end of week 4: 925
[E5] VLE clicks in weeks 1-2 (days 0-13): 624
[E9] Clicks on course content pages (oucontent) up to the end of week 4: 525
[E11] Clicks on quizzes up to the end of week 4: 180
[E16] Total clicks relative to the median of the same module-presentation cohort (1.0 = median): 2.84
[E17] Number of assessments (TMA/CMA) with a deadline before the end of week 4: 0
[E26] Highest education level on entry: Lower Than A Level
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has completed 925 total VLE clicks from registration to the end of week 4.",
   "evidence_ids": [
    "E1"
   ]
  },
  {
   "statement": "The student has made 525 clicks on course content pages (oucontent) up to the end of week 4.",
   "evidence_ids": [
    "E9"
   ]
  },
  {
   "statement": "The student has made 180 clicks on quizzes up to the end of week 4.",
   "evidence_ids": [
    "E11"
   ]
  },
  {
   "statement": "The student's total click activity is 2.84 times the cohort median.",
   "evidence_ids": [
    "E16"
   ]
  },
  {
   "statement": "The student has not submitted any assessments (TMA/CMA) with a deadline before the end of week 4.",
   "evidence_ids": [
    "E17"
   ]
  },
  {
   "statement": "The student's highest education level on entry was Lower Than A Level.",
   "evidence_ids": [
    "E26"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The model estimates a 29% probability of an unsuccessful outcome (Fail or Withdrawn).",
   "evidence_ids": [
    "M1"
   ]
  },
  {
   "statement": "High click activity relative to the cohort (2.84x median) is associated with a lower predicted risk of failure.",
   "evidence_ids": [
    "X1",
    "E16"
   ]
  },
  {
   "statement": "A lack of submitted assessments by week 4 is associated with a lower predicted risk of failure.",
   "evidence_ids": [
    "X2",
    "E17"
   ]
  },
  {
   "statement": "Lower entry education levels (Lower Than A Level) are associated with a higher predicted risk of failure.",
   "evidence_ids": [
    "X3",
    "E26"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Monitor assessment submission timelines closely.",
   "rationale": "The data shows zero assessments submitted by the end of week 4 (E17). While the model associates this with lower risk (X2), this pattern is unusual for a distance learning module and may indicate a misunderstanding of deadlines or a lack of engagement with assessment tasks specifically, despite high general activity. Proactive reminders are warranted.",
   "evidence_ids": [
    "E17",
    "X2"
   ]
  },
  {
   "action": "Provide targeted support regarding assessment requirements and deadlines.",
   "rationale": "The student has high general engagement (925 clicks, 2.84x cohort median) but no assessment submissions. This suggests a potential disconnect between consuming content and completing required tasks. Support should focus on bridging this gap.",
   "evidence_ids": [
    "E1",
    "E9",
    "E16",
    "E17"
   ]
  },
  {
   "action": "Leverage the student's high engagement to encourage assessment completion.",
   "rationale": "The student is highly active (E1, E9, E16). Interventions should frame assessment completion as a natural next step in their current learning journey rather than a new demand, capitalizing on their existing momentum.",
   "evidence_ids": [
    "E1",
    "E9",
    "E16"
   ]
  }
 ],
 "evidence_limitations": [
  "The model's prediction of 29% risk (M1) is based on associations (X1, X2, X3) and not proven causality; the specific reasons for the student's current lack of assessment submission are unknown.",
  "The data is limited to the end of week 4; long-term trends and final outcomes are not yet available.",
  "The model indicates that lower education levels (E26) increase risk, but the student's high engagement (E16) appears to be a stronger protective factor in the current model iteration, as it has the largest impact on reducing risk (X1)."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has completed 925 total VLE clicks from registration to the end of week 4. | True | True | — |
| fact | The student has made 525 clicks on course content pages (oucontent) up to the end of week 4. | True | True | — |
| fact | The student has made 180 clicks on quizzes up to the end of week 4. | True | True | — |
| fact | The student's total click activity is 2.84 times the cohort median. | True | True | — |
| fact | The student has not submitted any assessments (TMA/CMA) with a deadline before the end of week 4. | True | True | — |
| fact | The student's highest education level on entry was Lower Than A Level. | True | True | — |
| inference | The model estimates a 29% probability of an unsuccessful outcome (Fail or Withdrawn). | True | True | — |
| inference | High click activity relative to the cohort (2.84x median) is associated with a lower predicted risk of failure. | True | True | — |
| inference | A lack of submitted assessments by week 4 is associated with a lower predicted risk of failure. | True | True | — |
| inference | Lower entry education levels (Lower Than A Level) are associated with a higher predicted risk of failure. | True | True | — |
| rationale | The data shows zero assessments submitted by the end of week 4 (E17). While the model associates this with lower risk (X2), this pattern is unusual for a distance learning module and may indicate a misunderstanding of deadlines or a lack of engagement with assessment tasks specifically, despite high general activity. Proactive reminders are warranted. | True | True | — |
| rationale | The student has high general engagement (925 clicks, 2.84x cohort median) but no assessment submissions. This suggests a potential disconnect between consuming content and completing required tasks. Support should focus on bridging this gap. | True | True | — |
| rationale | The student is highly active (E1, E9, E16). Interventions should frame assessment completion as a natural next step in their current learning journey rather than a new demand, capitalizing on their existing momentum. | True | False | — |

Per-output metrics: UCR=0.0, coverage=0.9230769230769231, factual consistency=1.0, SHAP consistency=1.00 (top-3: clicks_vs_cohort_median, n_assessments_due, highest_education; mentioned: clicks_vs_cohort_median, n_assessments_due, highest_education), specificity=1.0


## 2. LLM without retrieved evidence produces an incorrect / unsupported claim

*Selection rule:* first C1 output (else C2) containing a number inconsistent with the Evidence Store or an asserted non-observable attribute; most such claims first

Enrolment `FFF_2014J_636469` (true final_result: Distinction)

**LLM + Student Data** (`C2_student_data`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - STUDENT DATA (structured record from the learning analytics database; null = not available / not applicable):
{
 "total_clicks": 833.0,
 "active_days": 27.0,
 "distinct_materials": 62.0,
 "pre_course_clicks": 64.0,
 "clicks_weeks_1_2": 352.0,
 "clicks_weeks_3_4": 417.0,
 "clicks_week_4": 107.0,
 "active_days_weeks_3_4": 10.0,
 "clicks_content": 265.0,
 "clicks_forum": 189.0,
 "clicks_quiz": 21.0,
 "clicks_homepage": 141.0,
 "clicks_resource": 193.0,
 "activity_trend": 0.0845,
 "days_since_last_activity": 1.0,
 "clicks_vs_cohort_median": 1.7354,
 "n_assessments_due": 1.0,
 "n_submitted_due": 1.0,
 "submission_rate": 1.0,
 "n_submissions": 1.0,
 "mean_score": 96.0,
 "n_late_submissions": 0.0,
 "has_banked_assessment": 0.0,
 "num_of_prev_attempts": 0.0,
 "studied_credits": 60.0,
 "highest_education": 2.0,
 "date_registration": -102.0
}
DATA DICTIONARY:
{
 "total_clicks": "Total VLE clicks from registration up to the end of week 4",
 "active_days": "Number of distinct days with VLE activity up to the end of week 4",
 "distinct_materials": "Number of distinct VLE materials accessed up to the end of week 4",
 "pre_course_clicks": "VLE clicks before the module start (day < 0)",
 "clicks_weeks_1_2": "VLE clicks in weeks 1-2 (days 0-13)",
 "clicks_weeks_3_4": "VLE clicks in weeks 3-4 (days 14-27)",
 "clicks_week_4": "VLE clicks in week 4 (days 21-27)",
 "active_days_weeks_3_4": "Days with VLE activity in weeks 3-4 (out of 14)",
 "clicks_content": "Clicks on course content pages (oucontent) up to the end of week 4",
 "clicks_forum": "Clicks on discussion forums (forumng) up to the end of week 4",
 "clicks_quiz": "Clicks on quizzes up to the end of week 4",
 "clicks_homepage": "Clicks on the module homepage up to the end of week 4",
 "clicks_resource": "Clicks on resources/URLs/subpages up to the end of week 4",
 "activity_trend": "Activity trend index = (clicks weeks 3-4 - clicks weeks 1-2) / (clicks weeks 1-2 + clicks weeks 3-4); -1 = all activity in weeks 1-2, 0 = equal, +1 = all activity in weeks 3-4 (not a percent change)",
 "days_since_last_activity": "Days between the last VLE activity and the end of week 4",
 "clicks_vs_cohort_median": "Total clicks relative to the median of the same module-presentation cohort (1.0 = median)",
 "n_assessments_due": "Number of assessments (TMA/CMA) with a deadline before the end of week 4",
 "n_submitted_due": "Number of those assessments submitted before the end of week 4",
 "submission_rate": "Submission rate of assessments due before the end of week 4",
 "n_submissions": "Total assessment submissions (any deadline) before the end of week 4",
 "mean_score": "Mean score (0-100) of assessments submitted before the end of week 4",
 "n_late_submissions": "Assessments submitted after their deadline (before the end of week 4)",
 "has_banked_assessment": "Has an assessment result transferred (banked) from a previous presentation",
 "num_of_prev_attempts": "Number of previous attempts at this module",
 "studied_credits": "Total credits of modules being studied in this presentation",
 "highest_education": "Highest education level on entry",
 "date_registration": "Registration day relative to module start (negative = days before start)"
}
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has high engagement relative to the cohort, with total clicks 1.7354 times the cohort median.",
   "evidence_ids": [
    "clicks_vs_cohort_median"
   ]
  },
  {
   "statement": "The student has maintained consistent activity across the first four weeks, with 27 active days out of a possible 28.",
   "evidence_ids": [
    "active_days"
   ]
  },
  {
   "statement": "Activity levels are stable or slightly increasing in weeks 3-4 compared to weeks 1-2 (activity trend index: 0.0845).",
   "evidence_ids": [
    "activity_trend"
   ]
  },
  {
   "statement": "The student has submitted 100% of assessments due by the end of week 4 (submission rate: 1.0).",
   "evidence_ids": [
    "n_assessments_due",
    "n_submitted_due",
    "submission_rate"
   ]
  },
  {
   "statement": "The student achieved a perfect mean score of 96.0 on submitted assessments.",
   "evidence_ids": [
    "mean_score"
   ]
  },
  {
   "statement": "The student has not submitted any assessments late.",
   "evidence_ids": [
    "n_late_submissions"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The student demonstrates strong time management and adherence to deadlines.",
   "evidence_ids": [
    "n_assessments_due",
    "n_submitted_due",
    "n_late_submissions"
   ]
  },
  {
   "statement": "The student is likely well-prepared for upcoming assessments, given the high mean score and consistent engagement.",
   "evidence_ids": [
    "mean_score",
    "active_days",
    "clicks_vs_cohort_median"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Continue current engagement strategies.",
   "rationale": "The student's activity trend is positive (0.0845) and they are maintaining high activity levels (27 active days). No intervention is required to boost engagement at this stage.",
   "evidence_ids": [
    "activity_trend",
    "active_days"
   ]
  },
  {
   "action": "Monitor for potential overconfidence or complacency.",
   "rationale": "While performance is excellent (96.0 mean score), the student has only attempted one assessment. Ensure they maintain focus as the module progresses and more assessments become due.",
   "evidence_ids": [
    "mean_score",
    "n_submissions"
   ]
  }
 ],
 "evidence_limitations": [
  "Insufficient data on future assessment performance to predict long-term success or identify potential drops in quality.",
  "No data available on peer interaction depth (only click counts on forums are recorded)."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has high engagement relative to the cohort, with total clicks 1.7354 times the cohort median. | True | True | — |
| fact | The student has maintained consistent activity across the first four weeks, with 27 active days out of a possible 28. | False | False | number_not_in_context:['28'] |
| fact | Activity levels are stable or slightly increasing in weeks 3-4 compared to weeks 1-2 (activity trend index: 0.0845). | True | True | — |
| fact | The student has submitted 100% of assessments due by the end of week 4 (submission rate: 1.0). | True | True | — |
| fact | The student achieved a perfect mean score of 96.0 on submitted assessments. | True | True | — |
| fact | The student has not submitted any assessments late. | True | True | — |
| inference | The student demonstrates strong time management and adherence to deadlines. | False | False | non_observable |
| inference | The student is likely well-prepared for upcoming assessments, given the high mean score and consistent engagement. | True | True | — |
| rationale | The student's activity trend is positive (0.0845) and they are maintaining high activity levels (27 active days). No intervention is required to boost engagement at this stage. | True | True | — |
| rationale | While performance is excellent (96.0 mean score), the student has only attempted one assessment. Ensure they maintain focus as the module progresses and more assessments become due. | True | True | — |

Per-output metrics: UCR=0.2, coverage=0.8, factual consistency=0.9, SHAP consistency=0.67 (top-3: mean_score, clicks_vs_cohort_median, highest_education; mentioned: mean_score, clicks_vs_cohort_median), specificity=1.0

**LLM + SHAP + RAG** (`C4_shap_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - PREDICTIVE MODEL OUTPUT (model inference, not an observed fact):
[M1] Model-estimated probability of an unsuccessful outcome (Fail or Withdrawn): 15% (model inference, not an observed fact)

CONTEXT - MODEL EXPLANATION (SHAP: features that most changed the model's predicted risk for this student; associations in the model, not causes):
[X1] SHAP rank 1: mean_score (value 96, see E21) decreases predicted risk by 22.6 percentage points (association in the model, not a cause)
[X2] SHAP rank 2: clicks_vs_cohort_median (value 1.74, see E16) decreases predicted risk by 6.9 percentage points (association in the model, not a cause)
[X3] SHAP rank 3: highest_education (value A Level or Equivalent, see E26) decreases predicted risk by 4.6 percentage points (association in the model, not a cause)

CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E5] VLE clicks in weeks 1-2 (days 0-13): 352
[E6] VLE clicks in weeks 3-4 (days 14-27): 417
[E9] Clicks on course content pages (oucontent) up to the end of week 4: 265
[E13] Clicks on resources/URLs/subpages up to the end of week 4: 193
[E16] Total clicks relative to the median of the same module-presentation cohort (1.0 = median): 1.74
[E21] Mean score (0-100) of assessments submitted before the end of week 4: 96
[E26] Highest education level on entry: A Level or Equivalent
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has a mean assessment score of 96 out of 100 based on submissions prior to the end of week 4.",
   "evidence_ids": [
    "E21"
   ]
  },
  {
   "statement": "The student's total VLE click activity is 1.74 times the median of the cohort for this module presentation.",
   "evidence_ids": [
    "E16"
   ]
  },
  {
   "statement": "The student's highest education level on entry was A Level or Equivalent.",
   "evidence_ids": [
    "E26"
   ]
  },
  {
   "statement": "The student has completed 352 clicks in weeks 1-2 and 417 clicks in weeks 3-4.",
   "evidence_ids": [
    "E5",
    "E6"
   ]
  },
  {
   "statement": "The student has completed 265 clicks on course content pages and 193 clicks on resources/URLs/subpages up to the end of week 4.",
   "evidence_ids": [
    "E9",
    "E13"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The predictive model estimates a 15% probability of an unsuccessful outcome (Fail or Withdrawn) for this student.",
   "evidence_ids": [
    "M1"
   ]
  },
  {
   "statement": "The model associates a high mean score with a decreased predicted risk of failure.",
   "evidence_ids": [
    "X1"
   ]
  },
  {
   "statement": "The model associates a high volume of clicks relative to the cohort median with a decreased predicted risk of failure.",
   "evidence_ids": [
    "X2"
   ]
  },
  {
   "statement": "The model associates an A Level or Equivalent education level with a decreased predicted risk of failure.",
   "evidence_ids": [
    "X3"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Continue monitoring engagement patterns.",
   "rationale": "While the student currently shows high engagement (1.74x cohort median) and excellent early assessment performance (96/100), the predictive model indicates a non-negligible 15% risk of an unsuccessful outcome. Sustaining the current level of interaction with course content and resources is associated with lower risk in the model.",
   "evidence_ids": [
    "M1",
    "E16",
    "E21"
   ]
  },
  {
   "action": "Maintain current study habits.",
   "rationale": "The student's high mean score and above-median click activity are positively associated with a lower predicted risk of failure according to the model. Encouraging the student to maintain these behaviors is consistent with the model's risk reduction factors.",
   "evidence_ids": [
    "X1",
    "X2",
    "E21",
    "E16"
   ]
  }
 ],
 "evidence_limitations": [
  "The predictive model output is an inference based on historical data up to week 4; it does not account for future performance or changes in student circumstances after this point.",
  "The model explanation describes associations between features and predicted risk, not causal mechanisms.",
  "There is insufficient evidence to determine if the student's high early scores will be maintained in subsequent assessments or if their engagement levels will fluctuate."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has a mean assessment score of 96 out of 100 based on submissions prior to the end of week 4. | True | True | — |
| fact | The student's total VLE click activity is 1.74 times the median of the cohort for this module presentation. | True | True | — |
| fact | The student's highest education level on entry was A Level or Equivalent. | True | True | — |
| fact | The student has completed 352 clicks in weeks 1-2 and 417 clicks in weeks 3-4. | True | True | — |
| fact | The student has completed 265 clicks on course content pages and 193 clicks on resources/URLs/subpages up to the end of week 4. | True | True | — |
| inference | The predictive model estimates a 15% probability of an unsuccessful outcome (Fail or Withdrawn) for this student. | True | True | — |
| inference | The model associates a high mean score with a decreased predicted risk of failure. | True | True | — |
| inference | The model associates a high volume of clicks relative to the cohort median with a decreased predicted risk of failure. | True | True | — |
| inference | The model associates an A Level or Equivalent education level with a decreased predicted risk of failure. | True | True | — |
| rationale | While the student currently shows high engagement (1.74x cohort median) and excellent early assessment performance (96/100), the predictive model indicates a non-negligible 15% risk of an unsuccessful outcome. Sustaining the current level of interaction with course content and resources is associated with lower risk in the model. | True | True | — |
| rationale | The student's high mean score and above-median click activity are positively associated with a lower predicted risk of failure according to the model. Encouraging the student to maintain these behaviors is consistent with the model's risk reduction factors. | True | True | — |

Per-output metrics: UCR=0.0, coverage=1.0, factual consistency=1.0, SHAP consistency=1.00 (top-3: mean_score, clicks_vs_cohort_median, highest_education; mentioned: mean_score, clicks_vs_cohort_median, highest_education), specificity=1.0


## 3. SHAP improves the explanation

*Selection rule:* largest SHAP-consistency gain C4 - C3 (then lowest UCR in C4)

Enrolment `CCC_2014J_2316201` (true final_result: Pass)

**LLM + RAG** (`C3_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E9] Clicks on course content pages (oucontent) up to the end of week 4: 100
[E2] Number of distinct days with VLE activity up to the end of week 4: 27
[E8] Days with VLE activity in weeks 3-4 (out of 14): 10
[E12] Clicks on the module homepage up to the end of week 4: 106
[E3] Number of distinct VLE materials accessed up to the end of week 4: 36
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has accessed 36 distinct VLE materials by the end of week 4.",
   "evidence_ids": [
    "E3"
   ]
  },
  {
   "statement": "The student has visited the course content pages 100 times by the end of week 4.",
   "evidence_ids": [
    "E9"
   ]
  },
  {
   "statement": "The student has visited the module homepage 106 times by the end of week 4.",
   "evidence_ids": [
    "E12"
   ]
  },
  {
   "statement": "The student has been active on the VLE on 27 distinct days by the end of week 4.",
   "evidence_ids": [
    "E2"
   ]
  },
  {
   "statement": "The student was active on the VLE on 10 days during weeks 3 and 4 combined.",
   "evidence_ids": [
    "E8"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The student's activity frequency decreased in the most recent weeks (weeks 3-4) compared to the overall trend, as 10 days of activity in weeks 3-4 represents a lower proportion of the total 27 distinct days than would be expected if activity were evenly distributed across the 4 weeks.",
   "evidence_ids": [
    "E2",
    "E8"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Monitor the student's engagement trend for weeks 5 and 6.",
   "rationale": "Evidence suggests a potential decline in daily activity frequency in the later weeks of the module. Continued monitoring is necessary to determine if this is a temporary fluctuation or a developing disengagement pattern.",
   "evidence_ids": [
    "E2",
    "E8"
   ]
  }
 ],
 "evidence_limitations": [
  "Insufficient data to determine if the decrease in activity frequency in weeks 3-4 is statistically significant or anomalous without data from weeks 1 and 2.",
  "No data is provided regarding the student's assessment performance, assignment submission status, or specific content areas accessed, which limits the ability to provide targeted academic support recommendations."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has accessed 36 distinct VLE materials by the end of week 4. | True | True | — |
| fact | The student has visited the course content pages 100 times by the end of week 4. | True | False | — |
| fact | The student has visited the module homepage 106 times by the end of week 4. | True | True | — |
| fact | The student has been active on the VLE on 27 distinct days by the end of week 4. | True | True | — |
| fact | The student was active on the VLE on 10 days during weeks 3 and 4 combined. | True | True | — |
| inference | The student's activity frequency decreased in the most recent weeks (weeks 3-4) compared to the overall trend, as 10 days of activity in weeks 3-4 represents a lower proportion of the total 27 distinct days than would be expected if activity were evenly distributed across the 4 weeks. | True | True | — |
| rationale | Evidence suggests a potential decline in daily activity frequency in the later weeks of the module. Continued monitoring is necessary to determine if this is a temporary fluctuation or a developing disengagement pattern. | False | True | no_referenced_feature_in_context:['activity_trend'] |

Per-output metrics: UCR=0.14285714285714285, coverage=0.8571428571428571, factual consistency=1.0, SHAP consistency=0.00 (top-3: highest_education, clicks_vs_cohort_median, n_late_submissions; mentioned: none), specificity=1.0

**LLM + SHAP + RAG** (`C4_shap_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - PREDICTIVE MODEL OUTPUT (model inference, not an observed fact):
[M1] Model-estimated probability of an unsuccessful outcome (Fail or Withdrawn): 53% (model inference, not an observed fact)

CONTEXT - MODEL EXPLANATION (SHAP: features that most changed the model's predicted risk for this student; associations in the model, not causes):
[X1] SHAP rank 1: highest_education (value Lower Than A Level, see E26) increases predicted risk by 7.0 percentage points (association in the model, not a cause)
[X2] SHAP rank 2: clicks_vs_cohort_median (value 1.86, see E16) decreases predicted risk by 4.9 percentage points (association in the model, not a cause)
[X3] SHAP rank 3: n_late_submissions (value 1, see E22) increases predicted risk by 4.7 percentage points (association in the model, not a cause)

CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E2] Number of distinct days with VLE activity up to the end of week 4: 27
[E3] Number of distinct VLE materials accessed up to the end of week 4: 36
[E8] Days with VLE activity in weeks 3-4 (out of 14): 10
[E9] Clicks on course content pages (oucontent) up to the end of week 4: 100
[E12] Clicks on the module homepage up to the end of week 4: 106
[E16] Total clicks relative to the median of the same module-presentation cohort (1.0 = median): 1.86
[E22] Assessments submitted after their deadline (before the end of week 4): 1
[E26] Highest education level on entry: Lower Than A Level
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has accessed 36 distinct VLE materials and 100 clicks on course content pages by the end of week 4.",
   "evidence_ids": [
    "E3",
    "E9"
   ]
  },
  {
   "statement": "The student has been active on 27 distinct days within the first 4 weeks.",
   "evidence_ids": [
    "E2"
   ]
  },
  {
   "statement": "The student has 10 days of activity recorded in weeks 3-4 out of a possible 14.",
   "evidence_ids": [
    "E8"
   ]
  },
  {
   "statement": "The student's total click volume is 1.86 times the cohort median.",
   "evidence_ids": [
    "E16"
   ]
  },
  {
   "statement": "The student has 1 late submission recorded before the end of week 4.",
   "evidence_ids": [
    "E22"
   ]
  },
  {
   "statement": "The student's highest education level on entry was Lower Than A Level.",
   "evidence_ids": [
    "E26"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The model estimates a 53% probability of an unsuccessful outcome (Fail or Withdrawn).",
   "evidence_ids": [
    "M1"
   ]
  },
  {
   "statement": "The model associates 'Lower Than A Level' education with an increased risk of 7.0 percentage points.",
   "evidence_ids": [
    "X1",
    "E26"
   ]
  },
  {
   "statement": "The model associates having 1 late submission with an increased risk of 4.7 percentage points.",
   "evidence_ids": [
    "X3",
    "E22"
   ]
  },
  {
   "statement": "The model associates a click volume of 1.86 (relative to cohort median) with a decreased risk of 4.9 percentage points.",
   "evidence_ids": [
    "X2",
    "E16"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Monitor submission timeliness closely.",
   "rationale": "The student has already recorded 1 late submission, which the model associates with increased risk. Ensuring future submissions are on time is critical.",
   "evidence_ids": [
    "E22",
    "X3"
   ]
  },
  {
   "action": "Maintain current engagement levels.",
   "rationale": "The student's high click volume (1.86x cohort median) is associated with reduced risk in the model. Continue to encourage this level of interaction with course materials.",
   "evidence_ids": [
    "E16",
    "X2"
   ]
  },
  {
   "action": "Provide targeted support regarding academic background.",
   "rationale": "The student's entry education level ('Lower Than A Level') is a top driver of predicted risk in the model. Consider offering additional scaffolding or resources tailored to students with this background.",
   "evidence_ids": [
    "E26",
    "X1"
   ]
  }
 ],
 "evidence_limitations": [
  "The data is limited to the end of week 4; long-term trends and final assessment performance are unknown.",
  "The model output represents an association, not a causal diagnosis of the student's future performance.",
  "There is no observed data on assessment marks or specific content mastery, only engagement metrics and submission timing."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has accessed 36 distinct VLE materials and 100 clicks on course content pages by the end of week 4. | True | True | — |
| fact | The student has been active on 27 distinct days within the first 4 weeks. | True | True | — |
| fact | The student has 10 days of activity recorded in weeks 3-4 out of a possible 14. | False | False | number_not_in_context:['14'] |
| fact | The student's total click volume is 1.86 times the cohort median. | True | True | — |
| fact | The student has 1 late submission recorded before the end of week 4. | True | True | — |
| fact | The student's highest education level on entry was Lower Than A Level. | True | True | — |
| inference | The model estimates a 53% probability of an unsuccessful outcome (Fail or Withdrawn). | True | True | — |
| inference | The model associates 'Lower Than A Level' education with an increased risk of 7.0 percentage points. | True | True | — |
| inference | The model associates having 1 late submission with an increased risk of 4.7 percentage points. | True | True | — |
| inference | The model associates a click volume of 1.86 (relative to cohort median) with a decreased risk of 4.9 percentage points. | True | True | — |
| rationale | The student has already recorded 1 late submission, which the model associates with increased risk. Ensuring future submissions are on time is critical. | True | True | — |
| rationale | The student's high click volume (1.86x cohort median) is associated with reduced risk in the model. Continue to encourage this level of interaction with course materials. | True | True | — |
| rationale | The student's entry education level ('Lower Than A Level') is a top driver of predicted risk in the model. Consider offering additional scaffolding or resources tailored to students with this background. | True | True | — |

Per-output metrics: UCR=0.07692307692307693, coverage=0.9230769230769231, factual consistency=0.9333333333333333, SHAP consistency=1.00 (top-3: highest_education, clicks_vs_cohort_median, n_late_submissions; mentioned: highest_education, clicks_vs_cohort_median, n_late_submissions), specificity=1.0


## 4. Insufficient evidence

*Selection rule:* first student (by id) with no assessment due before the cutoff (submission_rate not applicable) or no VLE activity; outputs of C3 and C4 shown

Enrolment `CCC_2014J_556450` (true final_result: Fail)

**LLM + RAG** (`C3_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E1] Total VLE clicks from registration up to the end of week 4: 0
[E2] Number of distinct days with VLE activity up to the end of week 4: 0
[E3] Number of distinct VLE materials accessed up to the end of week 4: 0
[E16] Total clicks relative to the median of the same module-presentation cohort (1.0 = median): 0.00
[E12] Clicks on the module homepage up to the end of week 4: 0
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has recorded zero VLE clicks, zero distinct days of activity, and zero distinct materials accessed from registration through the end of week 4.",
   "evidence_ids": [
    "E1",
    "E2",
    "E3"
   ]
  },
  {
   "statement": "The student's total click count is 0.00 times the median of the cohort for this module-presentation.",
   "evidence_ids": [
    "E16"
   ]
  },
  {
   "statement": "The student has recorded zero clicks on the module homepage through the end of week 4.",
   "evidence_ids": [
    "E12"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The student has not engaged with the Virtual Learning Environment (VLE) at all during the first four weeks of the module.",
   "evidence_ids": [
    "E1",
    "E2",
    "E3",
    "E16",
    "E12"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Initiate immediate contact with the student to verify their enrollment status and access to the VLE.",
   "rationale": "The complete absence of activity suggests the student may not be aware of the platform, does not have access, or has not yet started the module.",
   "evidence_ids": [
    "E1",
    "E2",
    "E3",
    "E12"
   ]
  },
  {
   "action": "Check if the student has been dropped from the module or if there are technical barriers preventing access.",
   "rationale": "Zero activity across all metrics, including the homepage, indicates a potential systemic issue or non-enrollment rather than simple disengagement.",
   "evidence_ids": [
    "E1",
    "E2",
    "E3",
    "E12"
   ]
  }
 ],
 "evidence_limitations": [
  "The data only covers the first four weeks; it is unknown if the student will engage later in the module.",
  "It is not possible to determine the specific reason for the lack of activity (e.g., technical issues, personal circumstances, or non-enrollment) from the provided data."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has recorded zero VLE clicks, zero distinct days of activity, and zero distinct materials accessed from registration through the end of week 4. | True | True | — |
| fact | The student's total click count is 0.00 times the median of the cohort for this module-presentation. | True | True | — |
| fact | The student has recorded zero clicks on the module homepage through the end of week 4. | True | True | — |
| inference | The student has not engaged with the Virtual Learning Environment (VLE) at all during the first four weeks of the module. | True | False | — |
| rationale | The complete absence of activity suggests the student may not be aware of the platform, does not have access, or has not yet started the module. | True | False | — |
| rationale | Zero activity across all metrics, including the homepage, indicates a potential systemic issue or non-enrollment rather than simple disengagement. | True | True | — |

Per-output metrics: UCR=0.0, coverage=0.6666666666666666, factual consistency=1.0, SHAP consistency=0.33 (top-3: submission_rate, clicks_vs_cohort_median, mean_score; mentioned: clicks_vs_cohort_median), specificity=0.8333333333333333

**LLM + SHAP + RAG** (`C4_shap_rag`)

<details><summary>Context given to the LLM</summary>

```
CONTEXT - PREDICTIVE MODEL OUTPUT (model inference, not an observed fact):
[M1] Model-estimated probability of an unsuccessful outcome (Fail or Withdrawn): 95% (model inference, not an observed fact)

CONTEXT - MODEL EXPLANATION (SHAP: features that most changed the model's predicted risk for this student; associations in the model, not causes):
[X1] SHAP rank 1: submission_rate (value 0%, see E19) increases predicted risk by 18.0 percentage points (association in the model, not a cause)
[X2] SHAP rank 2: clicks_vs_cohort_median (value 0.00, see E16) increases predicted risk by 15.6 percentage points (association in the model, not a cause)
[X3] SHAP rank 3: mean_score (value not available, see E21) increases predicted risk by 6.3 percentage points (association in the model, not a cause)

CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):
[E1] Total VLE clicks from registration up to the end of week 4: 0
[E2] Number of distinct days with VLE activity up to the end of week 4: 0
[E3] Number of distinct VLE materials accessed up to the end of week 4: 0
[E12] Clicks on the module homepage up to the end of week 4: 0
[E16] Total clicks relative to the median of the same module-presentation cohort (1.0 = median): 0.00
[E19] Submission rate of assessments due before the end of week 4: 0%
[E21] Mean score (0-100) of assessments submitted before the end of week 4: No score available: no assessment was submitted (and scored) before the end of week 4
```
</details>

Generated output (JSON):

```json
{
 "observed_facts": [
  {
   "statement": "The student has shown zero engagement with the Virtual Learning Environment (VLE) up to the end of week 4.",
   "evidence_ids": [
    "E1",
    "E2",
    "E3",
    "E12"
   ]
  },
  {
   "statement": "The student has not submitted any assessments due before the end of week 4.",
   "evidence_ids": [
    "E19"
   ]
  },
  {
   "statement": "The student's activity level (clicks) is at 0.00 relative to the cohort median.",
   "evidence_ids": [
    "E16"
   ]
  },
  {
   "statement": "No assessment scores are available for this student as no assessments have been submitted.",
   "evidence_ids": [
    "E21"
   ]
  }
 ],
 "model_inferences": [
  {
   "statement": "The predictive model estimates a 95% probability of an unsuccessful outcome (Fail or Withdrawn).",
   "evidence_ids": [
    "M1"
   ]
  },
  {
   "statement": "The model attributes the high risk primarily to the lack of submission rate (contributing +18.0 percentage points) and low click activity relative to the cohort (contributing +15.6 percentage points).",
   "evidence_ids": [
    "X1",
    "X2"
   ]
  }
 ],
 "recommendations": [
  {
   "action": "Initiate immediate, low-barrier contact with the student to understand their current status and barriers to access.",
   "rationale": "The student has zero recorded activity and zero submissions by week 4, which the model associates with a 95% risk of failure or withdrawal.",
   "evidence_ids": [
    "M1",
    "E1",
    "E19"
   ]
  },
  {
   "action": "Provide clear, direct guidance on the specific assessment requirements and submission deadlines for the current week.",
   "rationale": "The model indicates that the submission rate is the single strongest driver of the predicted risk.",
   "evidence_ids": [
    "X1",
    "E19"
   ]
  },
  {
   "action": "Check the student's technical access to the VLE and provide support if they are unable to log in or view materials.",
   "rationale": "The student has zero clicks and zero material accesses, suggesting a potential access or engagement barrier.",
   "evidence_ids": [
    "E1",
    "E2",
    "E3",
    "E16"
   ]
  }
 ],
 "evidence_limitations": [
  "The data only covers the first four weeks of the module; long-term trends cannot be determined.",
  "The predictive model output is an association based on historical data patterns, not a diagnosis of the student's specific reasons for non-engagement.",
  "No information is available regarding the student's prior knowledge, personal circumstances, or reasons for non-participation."
 ]
}
```

Deterministic audit of claim units:

| kind | statement | supported by context | linked to store | reasons |
|---|---|---|---|---|
| fact | The student has shown zero engagement with the Virtual Learning Environment (VLE) up to the end of week 4. | True | False | — |
| fact | The student has not submitted any assessments due before the end of week 4. | True | True | — |
| fact | The student's activity level (clicks) is at 0.00 relative to the cohort median. | True | True | — |
| fact | No assessment scores are available for this student as no assessments have been submitted. | True | True | — |
| inference | The predictive model estimates a 95% probability of an unsuccessful outcome (Fail or Withdrawn). | True | True | — |
| inference | The model attributes the high risk primarily to the lack of submission rate (contributing +18.0 percentage points) and low click activity relative to the cohort (contributing +15.6 percentage points). | True | True | — |
| rationale | The student has zero recorded activity and zero submissions by week 4, which the model associates with a 95% risk of failure or withdrawal. | True | True | — |
| rationale | The model indicates that the submission rate is the single strongest driver of the predicted risk. | True | True | — |
| rationale | The student has zero clicks and zero material accesses, suggesting a potential access or engagement barrier. | True | False | — |

Per-output metrics: UCR=0.0, coverage=0.7777777777777778, factual consistency=1.0, SHAP consistency=1.00 (top-3: submission_rate, clicks_vs_cohort_median, mean_score; mentioned: submission_rate, clicks_vs_cohort_median, mean_score), specificity=1.0
