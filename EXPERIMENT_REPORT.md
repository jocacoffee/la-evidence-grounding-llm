# EXPERIMENT_REPORT

**From Learning Analytics to Evidence-Grounded Educational Decision Support: Integrating Explainable
Machine Learning and Retrieval-Augmented Generation**

**Research question.** Does retrieval of student-specific learning evidence improve the grounding and
factual consistency of LLM-generated educational recommendations derived from Learning Analytics?

All numbers below were produced by the scripts in `src/` (see README). Nothing was edited by hand.
The study does not evaluate learning impact, pedagogical correctness or causal effects. There was no intervention.

---

## 1. Data
- **Dataset:** OULAD, 7 CSV tables, unmodified (see `DATASET_ANALYSIS.md`). No external data.
- **Unit:** enrolment (student × module-presentation). There are 32,593 enrolments of 28,785 students, in 7 modules and
  22 module-presentations.
- **Target:** `final_result` with 4 classes (Distinction, Pass, Fail, Withdrawn).
- **Temporal cutoff:** end of week 4. Only events with day < 28 are used (days relative to module start),
  including pre-course activity (day < 0).
- **Population at the cutoff:** 27,535 enrolments. Excluded: 5,040 enrolments that unregistered before day 28
  (their label is already known at prediction time) and 18 that registered on or after day 28.
- **Split (temporal, by presentation):** train 2013B + 2013J + 2014B (n = 18,321); test 2014J (n = 9,214).
  All 7 modules appear in both. Caveat: 2014B ends at about the time 2014J starts.
- **Table 1** (dataset statistics) is in `DATASET_ANALYSIS.md` §2. **Figure 2:** `figures/fig2_final_result_distribution.png`.

## 2. Experiment 1: Learning Analytics prediction

**Features (27, all computed from data before the cutoff, `src/02_build_features.py`, `src/feature_dictionary.py`):**
- VLE: total clicks, active days, distinct materials, pre-course clicks, clicks in weeks 1–2, 3–4 and 4,
  active days in weeks 3–4, clicks by activity type (content, forum, quiz, homepage, resource), activity-trend
  index, days since last activity, clicks relative to the same-presentation cohort median.
- Assessment: assessments due before the cutoff, submitted due, submission rate, total submissions, mean score,
  late submissions, banked flag.
- Academic/registration: previous attempts, studied credits, highest education, registration day.
- Undefined values are kept as NaN rather than 0. For example, `submission_rate` is NaN when nothing was due
  (EEE and GGG have no assessment due before day 28).
- **Excluded by design:** `final_result`, `date_unregistration`, events on day ≥ 28, exam results, and the
  protected demographics (gender, disability, age band, IMD band, region; the authors' decision, `DATASET_ANALYSIS.md` §6.5).

**Models:**
- Majority-class reference.
- Multinomial logistic regression: log1p on counts, median imputation with missing-value indicators,
  standardisation, balanced class weights.
- XGBoost (main model): hist trees, depth 5, learning rate 0.05, subsample and colsample 0.8, balanced sample weights.
  n_estimators = 121 was chosen by early stopping on 2014B (model fit on 2013B+2013J), then the model was refit on all train data.
- XGBoost without score features, as a sensitivity check. OULAD has no mark-release date, so the assumption that
  "a score is available at submission" cannot be verified (`DATASET_ANALYSIS.md` §6.2).
- Seed 42 throughout. 95% CIs come from 1,000 bootstrap resamples of the test set.

**Table 2: prediction performance on 2014J (n = 9,214)** (`results/ml_results.csv`)

| Model | Macro-F1 [95% CI] | Balanced acc. [95% CI] | Accuracy | AUROC macro-OvR | Brier (4-class) | AUROC at-risk* | Brier at-risk* |
|---|---|---|---|---|---|---|---|
| Majority class | 0.159 | 0.250 | 0.466 | 0.500 | 1.068 | 0.500 | 0.408 |
| Logistic Regression | 0.357 [0.347, 0.367] | 0.426 [0.415, 0.436] | 0.358 | 0.686 | 0.746 | 0.719 | 0.227 |
| **XGBoost (main)** | **0.382 [0.371, 0.392]** | **0.439 [0.428, 0.450]** | 0.386 | 0.697 | 0.721 | 0.733 | 0.219 |
| XGBoost (no scores) | 0.375 [0.365, 0.386] | 0.422 [0.411, 0.433] | 0.378 | 0.694 | 0.695 | 0.743 | 0.208 |

\*at-risk = P(Fail) + P(Withdrawn) from the 4-class model versus the true Fail/Withdrawn label.
Per-class F1 for XGBoost: Distinction 0.394, Pass 0.412, Fail 0.367, Withdrawn 0.353.
Confusion matrices are in `results/confusion_matrices.csv`. **Figure 3:** `figures/fig3_la_model_performance.png`.

Interpretation:
- Early 4-class prediction at week 4 is hard. XGBoost is better than logistic regression (non-overlapping CIs
  for Macro-F1), but it remains modest.
- Accuracy falls below the majority class because of the class balancing: 34% of true Pass students are predicted
  Distinction. An unweighted diagnostic model reached accuracy 0.49 at the same Macro-F1 (0.38).
- To rule out a pipeline error, a random within-train split gave Macro-F1 = 0.385 and at-risk AUROC = 0.79. The temporal
  shift costs about 0.05 AUROC.
- By module, at-risk AUROC ranges from 0.63 (BBB) to 0.81 (CCC) (`results/ml_results_by_module.csv`).
- Removing the score features changes little: at-risk AUROC is 0.743 without them versus 0.733 with them. The results
  therefore do not hinge on the mark-availability assumption.

## 3. Experiment 2: explainable ML (SHAP)
- **Explained quantity:** the main model's at-risk probability, P(Fail) + P(Withdrawn), on the probability scale.
- **Explainer:** `shap.PermutationExplainer` (interventional, exact additivity; max additivity error 2.2e-16),
  with a background of 200 random training enrolments (seed 42). All 9,214 test enrolments are explained.
  Base value = 0.485.
- TreeSHAP per class (log-odds) is reported as a complement (`data/shap_treeshap_global.csv`).
- **Global importance** (mean |SHAP|, percentage points of risk): mean_score 13.2, clicks_vs_cohort_median 7.5,
  highest_education 5.0, submission_rate 3.5, n_assessments_due 2.2, active_days 1.8, clicks_content 1.7
  (`results/shap_global_importance.csv`, **Figure 4:** `figures/fig4_shap_importance.png`).
- Per-student records (`data/student_evidence.{jsonl,csv}`) contain student_id, course, presentation,
  predicted_class, prediction_probability, class probabilities, predicted_risk, top-5 SHAP features with their
  values and direction, and all feature values.
- SHAP values describe associations **inside the model**. They are not causal effects.
- `n_assessments_due` is a course-design feature: it identifies EEE/GGG, which have no early assessment, so its SHAP value mostly
  reflects module differences, not student behaviour.

## 4. Experiment 3: Student Evidence Store
`data/evidence_store.jsonl` holds 313,276 items for 9,214 enrolments (34 per enrolment). The ID prefix encodes the
epistemic status of each item:

| Prefix | Content | Category |
|---|---|---|
| E1–E27 | one item per feature, with a deterministic natural-language statement, raw value, source table and cutoff | `observed` / `observed_derived` / `static` |
| M1, M2 | predicted risk and predicted class | `model_inference` (the text says "not an observed fact") |
| X1–X5 | SHAP rank, direction and magnitude, linked to the E-item holding the feature value | `model_explanation` (the text says "not a cause") |

Every item has `uid = enrolment_id#id`, `feature`, `source`, `time_cutoff_week = 4` and `value`, so it can be
traced back to the feature table. Inapplicable values get explicit statements, for example "Not applicable: no
assessment had a deadline before the end of week 4". Example:

```json
{"id": "E19", "enrolment_id": "FFF_2014J_651255", "type": "assessment_submission", "category": "observed_derived",
 "feature": "submission_rate", "value": 0.0, "display_value": "0%",
 "statement": "Submission rate of assessments due before the end of week 4: 0%",
 "source": "OULAD studentAssessment+assessments", "time_cutoff_week": 4}
```

## 5. Experiment 4: retrieval
- **Query:** (enrolment_id, cutoff_week = 4, task = "educator recommendation").
- **Stage 1 (hard filter):** items of that enrolment with cutoff ≤ week 4 and category observed, derived or static.
  Model items (M, X) are never retrieved, so C3 stays independent of the model.
- **Stage 2 (ranking):** salience = |2·percentile − 1|, the two-sided deviation of the value from the same-module
  **training** cohort (past presentations only), with ties broken by ID. Top-K = 5 are returned.
  This is simple, deterministic and reproducible. No embeddings and no LangChain are involved.
- **Evaluation:** OULAD contains no relevance judgements, so **no ground truth was invented**. We report:
  - (a) Integrity: **1.000** of retrieved items belong to the queried enrolment and to the week-4 cutoff
    (no cross-student or future leakage).
  - (b) Agreement with a **model-defined proxy**: the enrolment's SHAP top-k features. This measures how well a
    model-agnostic retriever recovers what the predictive model relied on. It is **not** pedagogical relevance.

| Retriever | P@5 (rel = SHAP top-5) | MRR (rel = top-5) | P@3 (rel = top-3) | R@5 (rel = top-3) | MRR (rel = top-3) |
|---|---|---|---|---|---|
| Salience (ours) | **0.279** | **0.604** | **0.200** | **0.294** | **0.464** |
| Random (20 perms) | 0.186 | 0.394 | 0.111 | 0.185 | 0.287 |
| Fixed order | 0.085 | 0.247 | 0.015 | 0.015 | 0.083 |

(`results/retrieval_metrics.csv`, n = 9,214 queries.) Salience retrieval beats random, but it recovers only a
minority of the model's top features. Deviation from the cohort and model importance are different notions.

## 6. Experiments 5–6: LLM and the four conditions
- **LLM:** Qwen3.5-9B (MLX 4-bit) served locally by LM Studio (OpenAI-compatible API), chosen by the authors.
  Qwen3 8B was not installed and nothing was downloaded.
- **Decoding:** temperature = 0, seed = 42, `reasoning_effort = "none"` (thinking disabled), max_tokens = 1500,
  and a JSON-schema constrained output. The same model and parameters were used in all conditions.
- **Sample:** 100 test-set (2014J) enrolments, stratified by true outcome (25 per class; the label was used only for
  sampling). Modules: CCC 27, BBB 22, FFF 22, DDD 13, EEE 10, GGG 5, AAA 1. In this sample, 15 have no assessment
  due before the cutoff, 28 have no score and 1 has no VLE activity.
- **Pilot:** 8 other enrolments (2 per class), used only to develop the evaluator.
- 400 generations: all valid JSON, all `finish_reason = stop`.
- Mean prompt/completion tokens: C1 342/87, C2 1409/635, C3 492/510, C4 764/656.

**Conditions** (only the context block differs):

| | Context |
|---|---|
| C1 LLM only | student ID, module, presentation, time point. "No student-specific learning evidence is available." |
| C2 LLM + student data | full 27-feature record (JSON) plus a data dictionary. No IDs, no model output |
| C3 LLM + RAG | the 5 retrieved E-items (with IDs) |
| C4 LLM + SHAP + RAG | M1 (predicted risk) + X1–X3 (SHAP top-3) + retrieved E-items + the E-items holding the SHAP features' values |

**System prompt** (identical in all conditions):
```
You are a decision-support assistant for educators in an online distance-learning university.
You write for the educator (tutor), not for the student.
Rules:
1. Use only the information provided in the context.
2. Clearly separate observed facts (recorded data) from inferences (model predictions or your own interpretation).
3. Do not invent characteristics, circumstances or data about the student.
4. Do not diagnose the student (no psychological, medical or personal diagnoses).
5. Do not claim causality; describe associations only.
6. Do not invent specific difficulties (e.g., topics, skills, personal problems) that are not present in the data.
7. Give general, actionable recommendations that are compatible with the evidence.
8. When evidence items have IDs (e.g., E3, M1, X2), cite the IDs you use in "evidence_ids"; otherwise leave "evidence_ids" empty.
9. If the information is insufficient to support a conclusion or a specific recommendation, say so in "evidence_limitations".
Answer only with a JSON object that follows the required schema.
```
**User prompt:**
```
TASK: Based only on the information provided, generate a concise educational decision-support recommendation for an educator.
STUDENT: student ID <id>, module <m>, presentation <p>.
TIME POINT: end of week 4 of the module presentation (only information recorded up to this point exists).
<CONTEXT block of the condition>
Do not introduce factual claims that are not supported by the provided evidence.   [C2–C4 only, as specified]
Return the JSON object now.
```
**Output schema:** `observed_facts[{statement, evidence_ids}]` (≤ 6), `model_inferences[...]` (≤ 4),
`recommendations[{action, rationale, evidence_ids}]` (≤ 3), `evidence_limitations[str]`.
Every output file under `results/llm_outputs/main/` stores the full prompts, parameters and raw response.

## 7. Experiment 7: evaluation (deterministic, no human rating, no LLM judge)
`src/08_evaluate.py` and `src/feature_dictionary.py`.
- **Claim units:** the statements in observed_facts, model_inferences and recommendation rationales. A unit is a
  student-specific factual claim if it references a feature (regex lexicon), a generic LA concept, a number, or asserts
  an attribute OULAD does not record (lexicon of non-observable concepts: motivation, stress, struggles, time management…).
  Hedged units ("may", "whether"…) are not counted as asserting non-observables or risk.
- **Unsupported claim rate (UCR):** share of claims *not supported by the context given to the LLM*. A claim is
  unsupported if it asserts a non-observable, if none of its referenced features is in the context, if it asserts
  risk without a model estimate in the context, or if one of its numbers matches no context value (allowed forms:
  %, ×100, 1 − ratio, and the change between the two activity windows).
- **Evidence coverage:** share of claims linkable to a specific Evidence-Store item of that student: at least one
  specific feature, no non-observable content, and numbers consistent with the true store.
- **Factual consistency:** share of numeric mentions attributable to a referenced feature that match the student's
  **true** store value (the whole store, independent of the context).
- **SHAP consistency:** |SHAP top-3 features mentioned anywhere in the response| / 3.
- **Recommendation specificity:** per recommendation, (identifies a problem + uses evidence + proposes an action) / 3,
  averaged over recommendations.
- **Also reported:** invalid citations, insufficiency declared, asserted non-observable, causal-language rate.
- **Statistics:** per-output metrics averaged per condition (outputs where the metric is undefined are excluded and
  counted), 95% bootstrap CIs over students, and paired Wilcoxon tests with Holm correction within each metric.
- **Evaluator development:** the evaluator was developed on the 8-student pilot and then **frozen before the main
  evaluation** (SHA1 in `results/evaluator_frozen_sha1.txt`, copies in `results/FROZEN_*`). Pilot-driven changes:
  lexicon coverage, number stripping for time windows and IDs, "≥ 1 referenced feature in context" instead of
  "all features", SHAP values in the true pool, action-verb stems, and the activity-trend representation (see §10).

## 8. Experiment 8: results

**Table 3: comparison of the LLM conditions** (n = 100 students each; mean [95% bootstrap CI]; `results/llm_comparison.csv`)

| Condition | Unsupported claim rate ↓ | Evidence coverage ↑ | Factual consistency ↑ | SHAP consistency ↑ | Rec. specificity ↑ | Claims/output |
|---|---|---|---|---|---|---|
| C1 LLM only | n/a (0 claims in 100/100) | n/a | n/a (0 numbers) | 0.000 | n/a (0 recs) | 0.0 |
| C2 LLM + student data | 0.053 [0.041, 0.064] | 0.959 [0.945, 0.971] | 0.987 [0.978, 0.994] | 0.583 [0.537, 0.627] | 0.956 [0.941, 0.969] | 10.7 |
| C3 LLM + RAG | 0.061 [0.042, 0.082] | 0.875 [0.848, 0.898] | 0.981 [0.968, 0.992] | 0.317 [0.270, 0.367] | 0.983 [0.974, 0.991] | 8.9 |
| C4 LLM + SHAP + RAG | **0.013 [0.006, 0.020]** | **0.979 [0.970, 0.987]** | 0.999 [0.997, 1.000] | **0.980 [0.963, 0.993]** | 0.984 [0.977, 0.992] | 11.9 |

Supplementary figures: UCR restricted to *observed_facts* is C2 0.017 [0.008, 0.027], C3 0.012 [0.004, 0.023] and
C4 0.015 [0.007, 0.027]. Invalid ID citations: C3 0%, C4 0%. C2 has no IDs, and the LLM nevertheless filled
`evidence_ids` with feature names. Outputs asserting a non-observable attribute: C2 9%, C3 5%, C4 3%.
Causal-language rate per claim: C2 3.0%, C3 0.1%, C4 0.4%. Insufficiency declared: C1 100%, C2 94%, C3 100%, C4 99%.
**Figure 5:** `figures/fig5_llm_condition_comparison.png`.

**Paired tests** (Wilcoxon, Holm-adjusted; `results/llm_paired_tests_main.csv`):

| Metric | C4 vs C3 | C4 vs C2 | C3 vs C2 |
|---|---|---|---|
| UCR | −0.049, p < 0.001 | −0.040, p < 0.001 | +0.009, p = 0.73 |
| UCR, facts only | +0.003, p = 1.0 | −0.001, p = 1.0 | −0.004, p = 1.0 |
| Evidence coverage | +0.105, p < 0.001 | +0.021, p = 0.006 | −0.084, p < 0.001 |
| Factual consistency | +0.018, p = 0.21 | +0.012, p = 0.21 | −0.006, p = 0.59 |
| SHAP consistency | +0.663, p < 0.001 | +0.397, p < 0.001 | −0.267, p < 0.001 |
| Specificity | +0.001, p = 0.94 | +0.028, p = 0.011 | +0.026, p = 0.022 |

**Where the unsupported claims come from** (`src/09b_error_analysis.py`, `results/claims_by_kind.csv`, `results/unsupported_reasons.csv`):

| | UCR, facts | UCR, inferences | UCR, rationales | Main reason for unsupported claims |
|---|---|---|---|---|
| C2 | 0.017 (n = 598) | 0.158 (n = 221) | 0.044 (n = 251) | risk asserted without a model estimate (32/56), e.g. "The student is at high risk of dropping out" |
| C3 | 0.013 (n = 468) | 0.124 (n = 202) | 0.110 (n = 218) | claims about features that were not retrieved (33/55) |
| C4 | 0.017 (n = 538) | 0.000 (n = 371) | 0.025 (n = 281) | a few claims on non-retrieved features (10/16) |

Excluding claims flagged only for "risk without model output", micro-UCR is C2 0.023, C3 0.057, C4 0.013.

**Answer to the research question, as far as these data support it** (revised after the ablation and the
second LLM, §8b):
1. **Retrieval alone (C3) did not improve grounding** over giving the LLM the full structured record (C2): UCR 0.061
   versus 0.053, not significant. Evidence coverage was lower (0.875 versus 0.959), because with only 5 items the LLM makes
   more generic or out-of-context claims. This replicates with Gemma (coverage 0.917 versus 0.974). Retrieval with explicit
   IDs did make every citation valid.
2. **On observed facts, all evidence conditions are already well grounded** (UCR on facts ≈ 1.5%, no differences),
   and numeric factual consistency is ≥ 0.98 in all of them, for both LLMs. When evidence is present, the LLMs rarely
   state wrong numbers. The grounding risk lies in **interpretive claims**.
3. **With Qwen, adding model evidence (C4) gave the lowest UCR** (0.013, significant against C2 and C3). **The ablation
   attributes most of this effect to the risk estimate, not to SHAP.** Adding only M1 to the retrieved evidence (C5)
   lowered UCR from 0.066 to 0.029 (p = 0.004); adding SHAP on top (C4) lowered it further to 0.011, which is not
   significant (p = 0.077). SHAP did significantly raise evidence coverage (0.927 → 0.979). With a model estimate
   available, the LLM reports it instead of producing its own risk judgements.
4. **The UCR effect did not generalise across LLMs.** It appeared only for the LLM that tends to make *unsupported*
   interpretive claims (Qwen3.5-9B: 12–16% of its inference statements unsupported in C2/C3).
   - Gemma-4-E4B rarely interpreted (32 inference statements in C2 versus 221 for Qwen3.5). Its UCR was low in all
     evidence conditions, with no significant differences.
   - Qwen3-14B interpreted a lot, but mostly grounded (3.4% of its 236 C3 inferences unsupported). C4 versus C3 was not
     significant (0.029 versus 0.018, p = 0.21).
   - What replicated across all three LLMs: observed facts well grounded (≤ 2% unsupported) and factual consistency ≥ 0.98
     in every evidence condition; higher coverage for C4 than C3 (p < 0.001 in all three); no fabricated student facts
     without evidence; and model evidence anchoring the stated risk (§8c).
5. **Without evidence (C1), no LLM invented student facts.**
   - Qwen3.5-9B abstained in 100/100 cases (no claims, no recommendations).
   - Gemma gave one generic recommendation in every case ("Monitor progress") plus statements of absence.
   - Qwen3-14B gave 2–3 generic recommendations ("Monitor the student's progress closely…") with general pedagogical rationales.
   - The evaluator flags the Gemma and Qwen3-14B statements (C1 UCR = 1.0), but that is an **evaluator artefact**. We
     inspected all of them: they are statements of absence or general pedagogy, not claims about the student. A subtle issue
     remains: phrases such as "No performance data has been recorded for this student" conflate "no data was given to me"
     with "no data exists".
   - C1 is therefore reported as a **behaviour check**, not as a comparison condition. The abstention is caused by the strict
     system prompt combined with missing data, not by temperature 0.
6. **SHAP consistency** is near-trivially high in C4 (Qwen3.5 0.98, Gemma 0.92, Qwen3-14B 0.99), because the SHAP factors are in its context.
   It shows that the model's explanation is carried into the recommendation. It does **not** show that the explanation is
   better.
7. **Recommendation specificity** is at the ceiling (0.96–1.00) and does not discriminate between conditions.

**Evaluator sensitivity (post-hoc, v2).** After the main run, we inspected the flagged claims and found systematic
errors in the frozen evaluator:
- false positives: constants such as "out of 14/28 possible days" and "vs 1.0 (median)", the singular "1 day in weeks 3–4"
  missed by the lexicon, and the raw education code `2` present in the C2 record;
- false negatives: semantic misreadings the evaluator cannot detect, such as "clicked on 525 course content pages", which
  confuses clicks with pages.

A v2 with only these listed fixes (`EVAL_VERSION=v2`, `results/evaluator_sensitivity_v1_v2.csv`) gives UCR C2 0.049,
C3 0.056, C4 0.008, the same ordering and the same significance pattern (C4 < C2 and C3, p < 0.001; C3 vs C2 p = 0.89).
**The primary results reported are the frozen v1.** All 262 flagged claims are listed for manual audit in
`results/evaluator_verification_sheet.csv`. **We recommend that the authors verify this sheet before submission.** The
evaluator's precision has not been formally validated.

**Reproducibility of the generations:** see §11.

### 8b. Post-hoc analyses (added after the main run; frozen v1 evaluator)

**(i) Ablation: which part of C4 matters?** A fifth condition, **C5 = retrieved evidence + M1 (predicted risk), no SHAP**,
was added (`src/07_generate.py ablation`, `src/11_ablation_and_second_llm.py`). Loading the model again is not
bit-identical (see (iii)), so C3, C5 and C4 were **regenerated together**, with all three conditions of each student
generated in the same server session. A session interruption split the run: students 1–53 ran in one session and
students 54–100 in another (`results/ablation_session2_students.txt`). Pairing stays within student and within session.
Qwen3.5-9B, n = 100 (`results/ablation_comparison.csv`, `results/ablation_paired_tests.csv`,
`results/ablation_claims_by_kind.csv`):

| Condition | UCR [95% CI] | UCR facts | UCR inferences | UCR rationales | Coverage [95% CI] | Factual cons. | Claims/output |
|---|---|---|---|---|---|---|---|
| C3 RAG | 0.066 [0.047, 0.087] | 0.013 | 0.135 (n = 207) | 0.119 | 0.876 [0.852, 0.899] | 0.984 | 9.0 |
| C5 RAG + risk | 0.029 [0.017, 0.042] | 0.017 | 0.009 (n = 109) | 0.070 | 0.927 [0.907, 0.945] | 0.994 | 8.1 |
| C4 RAG + risk + SHAP | 0.011 [0.005, 0.018] | 0.013 | 0.000 (n = 372) | 0.025 | 0.979 [0.969, 0.987] | 0.999 | 11.9 |

Paired tests (Holm): UCR C5 vs C3 −0.037, p = 0.004; C4 vs C5 −0.018, p = 0.077; C4 vs C3 −0.055, p < 0.001.
Coverage C5 vs C3 +0.052, p < 0.001; C4 vs C5 +0.051, p < 0.001. Factual consistency: all differences are n.s.
**Interpretation:** the risk estimate removes almost all unsupported *inferences* (0.135 → 0.009). With the estimate
available, the LLM also produces fewer inference statements (207 → 109). SHAP adds coverage and more (grounded)
model-explanation statements. Its further reduction of UCR, mostly in rationales, is not significant.

**(ii) Other LLMs: Gemma-4-E4B and Qwen3-14B** (MLX 4-bit, LM Studio, same prompts, parameters, students and frozen
evaluator; `results/second_llm_comparison.csv`, `results/second_llm_paired_tests.csv`):

| LLM | Condition | UCR [95% CI] | UCR facts | UCR inferences (n) | Coverage [95% CI] | Factual cons. | Claims/output |
|---|---|---|---|---|---|---|---|
| Gemma-4-E4B | C1 | (1.000, artefact) | n/a | n/a | — | n/a | 0.5 |
| | C2 | 0.003 [0.000, 0.007] | 0.000 | 0.031 (32) | 0.974 [0.948, 0.992] | 0.999 | 7.0 |
| | C3 | 0.018 [0.008, 0.030] | 0.014 | 0.111 (9) | 0.917 [0.895, 0.937] | 0.995 | 6.0 |
| | C4 | 0.005 [0.001, 0.010] | 0.008 | 0.000 (211) | 0.960 [0.946, 0.973] | 0.998 | 8.7 |
| Qwen3-14B | C1 | (1.000, artefact) | n/a | n/a | — | n/a | 1.4 |
| | C2 | **degenerate: 99/100 empty outputs** | — | — | — | — | 0.1 |
| | C3 | 0.029 [0.017, 0.043] | 0.020 | 0.034 (236) | 0.896 [0.873, 0.917] | 0.989 | 8.9 (10 empty outputs) |
| | C4 | 0.018 [0.011, 0.026] | 0.021 | 0.000 (400) | 0.964 [0.953, 0.974] | 0.995 | 12.6 |

- **Gemma:** no UCR difference is significant (Holm p ≥ 0.31). Coverage: C3 < C2 (−0.057, p < 0.001), C4 > C3
  (+0.043, p < 0.001) and C4 < C2 (−0.014, p = 0.008). With Qwen3.5, C4 > C2.
- **Qwen3-14B:** C4 versus C3 gives UCR −0.012 (p = 0.21) and coverage +0.068 (p < 0.001).
- **Qwen3-14B C2 failure.** In 99/100 cases the output was `{"observed_facts": [], "model_inferences": [],
  "recommendations": [], "evidence_limitations": []}`. Diagnosis (one student, outside the reported results): without the JSON
  schema the model copied the input record's JSON format (`observed_facts` as a dictionary of raw values). Under
  schema-constrained decoding, which requires lists, it collapsed to empty lists. This is an interaction between
  JSON-formatted context and constrained decoding. It is a generation failure, not abstention (no limitation is stated), and
  C2 is excluded from Qwen3-14B's comparisons. The prompt was not changed, to keep the conditions identical across LLMs.
- **What replicates across the three LLMs:**
  - facts grounded in all evidence conditions;
  - retrieval alone lowers coverage relative to the full record (Qwen3.5, Gemma);
  - model evidence (C4) raises coverage relative to C3 (all three);
  - no student facts fabricated without evidence (all three).
- **What does not replicate:** the UCR advantage of model evidence. It is specific to Qwen3.5-9B.

**(iii) Session-to-session variability (Qwen3.5-9B).** Main-run C3 and C4 outputs (session 1) were compared with the ablation's
C3 and C4 (sessions 2–3), with the same prompts and parameters (`results/session_variability.csv`).
- Only 53% (C3) and 54% (C4) of the outputs were byte-identical. One different token early in the output changes the rest.
- The metrics nevertheless did not change: UCR 0.061 versus 0.066 (C3, p = 0.34) and 0.013 versus 0.011 (C4, p = 0.71);
  coverage 0.875 versus 0.876 and 0.980 versus 0.979.
- Session-level variability of the aggregate metrics (< 0.5 percentage points) is therefore much smaller than the condition
  effects reported (3.7–5.5 points).

### 8c. Consistency, sensitivity and mispredictions (post-hoc, existing outputs; `src/12_consistency_sensitivity.py`)
All extraction is deterministic and lexicon-based:
- stated risk band: high / moderate / low, taken from a quoted percentage or from risk keywords;
- action categories: outreach, assessment, engagement, support, monitor, positive;
- referenced features: taken from the frozen evaluator audit.

These analyses are descriptive and post-hoc. Qwen3-14B's C2 is excluded (degenerate).

**(A) Cross-model consistency (RQ2)** (`results/cross_model_consistency*.csv`):

| Pair | Feature Jaccard C2 → C3 → C4 | Action Jaccard C2 / C3 / C4 | Same risk band in C4 (n both stated) |
|---|---|---|---|
| Qwen3.5 vs Gemma | 0.52 → 0.75 → 0.66 | 0.45 / 0.51 / 0.46 | 100% (96) |
| Qwen3.5 vs Qwen3-14B | — → 0.71 → 0.72 | — / 0.51 / 0.67 | 93% (83) |
| Gemma vs Qwen3-14B | — → 0.73 → 0.70 | — / 0.41 / 0.46 | 92% (79) |

- Retrieval makes different LLMs talk about the same evidence (C2 → C3: +0.23, p < 0.001, Qwen3.5 vs Gemma).
- A consistent **risk judgement** across LLMs exists only when the model's estimate is supplied. Without it, the LLMs almost
  never state the same risk band, and Gemma and Qwen3-14B state none.
- Recommended **actions** converge little: only within the Qwen family in C4 (0.67).

**(B) Sensitivity to LA indicators** (`results/llm_sensitivity.csv`):
- *Unsupported is not the same as wrong.* Qwen3.5's own risk bands in C2 (stated in 61/100) correlate with the model's
  risk (Spearman 0.55) and discriminate the true outcome (AUROC 0.75, versus 0.80 for the model on the same students). UCR
  measures traceability, not correctness.
- *Genericity.* Gemma's recommendations are largely templated: 17–23% distinct action texts, and a mean pairwise cosine of
  0.46–0.53. In C4, 46/101 of its actions are "Monitor engagement and performance trends", including for high-risk students.
  Qwen3.5 has 98–100% distinct texts and a cosine of 0.15–0.17; Qwen3-14B has 91–100% and 0.21–0.26.
  **Gemma's low UCR partly reflects saying little that is student-specific: grounding and individualisation trade off.**

**(C) When the predictive model is wrong** (threshold 0.5; in the sample, 75 correct, 14 false alarms, 11 missed at-risk
students; `results/misprediction_analysis.csv`, `results/misprediction_exact_tests.csv`):
- Grounding does not depend on prediction correctness: C4 UCR is 0–2% in all groups for all LLMs. Recommendations are
  grounded *and* wrong when the model is wrong.
- **For the 11 missed at-risk students, no LLM stated high risk in C4, 27–45% stated low risk, and none proposed outreach.**
  For false alarms, 43–57% stated high risk in C4.
- **Anchoring (Qwen3.5-9B, C2 versus C4, exact McNemar):** outreach for missed at-risk students fell from 6/11 to 0/11
  (p = 0.031). Outreach also fell among correctly predicted students (43% → 27%, p = 0.008). Supplying the model's risk makes
  recommendations follow the model: less unsupported interpretation, but the model's errors carry over into the advice.
- Qwen3-14B's C2 and Gemma's C2 (Gemma never proposes outreach) do not allow the same C2 versus C4 comparison.

### 8d. Confirmatory anchoring test (pre-registered; `results/PREREGISTRATION_anchoring.md`)
The exploratory anchoring result (§8c-C, 11 students) was tested on **80 new students** from the 2014J test set:
40 missed at-risk students and 40 false alarms, disjoint from all previous samples. The model was Qwen3.5-9B, comparing C2
with C4. Hypotheses, test, exclusion rules, sample and frozen code hashes were registered before any output was generated
(`src/13_anchoring_sample.py`, `src/14_anchoring_confirmatory.py`, frozen extraction rules in `src/llm_extract.py`).

| Hypothesis | Group | C2 | C4 | Discordant (C2 only / C4 only) | Exact McNemar p | Holm p |
|---|---|---|---|---|---|---|
| **H1 (primary):** outreach lower in C4 | missed at-risk (n = 39; 1 invalid JSON excluded) | 28.2% | 10.3% | 8 / 1 | **0.039** | — |
| H2: low risk stated more in C4 | missed at-risk (n = 39) | 23.1% | 56.4% | 5 / 18 | 0.011 | **0.021** |
| H3: high risk stated more in C4 | false alarms (n = 40) | 32.5% | 55.0% | 3 / 12 | 0.035 | **0.035** |

- **All three hypotheses were confirmed.** Supplying the model's risk shifts the stated risk and the recommended
  actions towards the model's (incorrect) prediction in both directions.
  - For missed at-risk students, no C4 output stated high risk (C2: 27.5%).
  - Effect sizes were smaller than in the exploratory sample (outreach 28% → 10% here, versus 55% → 0%).
  - Asymmetry (descriptive): for false alarms, stated high risk rose, but outreach did not (42.5% → 45.0%).
- UCR was again lower in C4 (0.007–0.025 versus 0.048–0.055), replicating the main result for this LLM.

### 8e. RAG control: evidence format versus evidence selection (added after an external review; `src/17_rag_control.py`)
An external review pointed out that C2 (full record, JSON, no IDs) and C3 (top-5 retrieved items, statements with IDs)
differ in two ways at once: format and selection. Two control conditions were therefore added, with the same prompt,
schema and frozen evaluator:
- **C6:** all 27 observed evidence items, in the same statement + ID format as C3, with no selection. Run for all three LLMs.
- **C7:** 5 randomly drawn observed items (seeded per student), in the same format. Run for Qwen3.5-9B only.

C6 and C7 were generated in a later server session than C2 and C3. Session effects on aggregate metrics were < 0.5 points (§8b-iii).
Outputs: 400, all from the intended model. One Gemma output hit the token limit, and Qwen3-14B produced 17 empty C6
outputs. Files: `results/rag_control_comparison.csv`, `results/rag_control_tests.csv`.

| LLM | C2 JSON (UCR / cov.) | **C6 all items + IDs** | C3 top-5 salient | C7 random 5 | C4 top-5 + risk + SHAP |
|---|---|---|---|---|---|
| Qwen3.5-9B | 0.053 / 0.959 | **0.043 / 0.954** | 0.061 / 0.875 | 0.077 / 0.926 | 0.013 / 0.979 |
| Gemma-4-E4B | 0.003 / 0.974 | **0.000 / 0.973** | 0.018 / 0.917 | — | 0.005 / 0.960 |
| Qwen3-14B | (99/100 empty) | **0.007 / 0.967** (17 empty) | 0.029 / 0.896 (10 empty) | — | 0.018 / 0.964 |

Paired Wilcoxon tests (Holm within LLM × metric):
- **Format (C6 vs C2): no significant effect.** UCR −0.010 (p = 0.27) for Qwen3.5 and −0.003 (p = 0.69) for Gemma;
  coverage n.s. Replacing the JSON record with statements largely removed Qwen3-14B's C2 collapse (99 → 17 empty outputs),
  which confirms the JSON-format diagnosis.
- **Selection (C3 top-5 vs C6 all items): coverage fell in all three LLMs** (−0.079, −0.055 and −0.071; all p < 0.001).
  UCR never improved: +0.018 for Qwen3.5 (n.s.), +0.018 for Gemma (p = 0.21) and **+0.025 for Qwen3-14B (p = 0.027)**.
- **Ranking (C3 salient vs C7 random, Qwen3.5):** UCR was n.s. (0.061 versus 0.077), and coverage was *lower* with salience
  ranking (0.875 versus 0.926, p = 0.001).
- **Model evidence versus all observed evidence (C4 vs C6): the effect depends on the LLM.**
  - Qwen3.5: UCR −0.030 (p < 0.001), coverage +0.026 (p < 0.001).
  - Gemma: n.s.
  - Qwen3-14B: UCR **+0.014** (p = 0.035), coverage n.s.

**Revised reading of RQ1.**
- The lower coverage of RAG comes from **selection**, not from format.
- When a student's structured record is small enough to fit the context, as here (27 items), giving the LLM *all* the
  evidence with IDs grounded it at least as well as retrieving a subset, and for two of three LLMs almost perfectly
  (UCR 0.000–0.007).
- Salience-based retrieval did not beat random selection.
- Model evidence (risk + SHAP) helped only the LLM prone to unsupported interpretation. It did not help the others, and it
  slightly hurt Qwen3-14B. The anchoring cost (§8d) applies regardless.

### 8f. Human validation of the evaluator (`src/15_evaluator_validation.py`, `validation/`, `results/evaluator_validation.csv`)
**Design.** One author labelled 277 claims from the main Qwen3.5-9B runs **blind** to the evaluator's decisions, using
`validation/GUIDELINES.md`:
- all 127 claims the evaluator flagged as unsupported;
- 150 random claims the evaluator judged supported (50 per condition, C2–C4).

The items were shuffled, and each was shown with the exact context given to the LLM. Sixty items were re-labelled by the
**same annotator** about 40 minutes later (intra-rater agreement). After the scoring was first run, 9 blank items were
completed; no other label was changed.

**Results.**
- Precision of the evaluator's "unsupported" flags: **16.5%** (95% CI 11–24%).
- Estimated recall: **~11%**.
- Intra-rater κ = **0.71** (n = 60).
- Human-corrected micro-UCR: C2 0.066, C3 0.086, C4 0.040 (evaluator: 0.052, 0.062, 0.013). **The ordering of the
  conditions is preserved.**

**Interpretation.** The annotator's labels followed a "reasonable reading of the data" criterion rather than the
guidelines' strict criterion. Risk judgements without a model estimate and attributions such as "good time management"
were mostly labelled supported, and no item was labelled not_a_claim. Remaining disagreements were evaluator errors,
such as constants like "out of 14 days". Re-annotating with the strict criterion after seeing the evaluator's reasons
would not be blind, so it was **not** done. Consequences for the paper:
- the UCR is reported as **strict traceability** to the given evidence;
- claim-level validity is declared as low;
- conclusions rest on aggregate comparisons, which kept the same ordering, and on the anchoring test, which does not use
  the UCR.

## 9. Qualitative analysis (`results/case_studies.md`, rule-based selection)

The four cases requested were selected by fixed rules (`src/09_compare.py`):
1. **Retrieval reduces unsupported claims** (EEE_2014J_2592973): UCR C2 0.20 → C3 0.25 → C4 0.00.
   - The reduction comes from C4, not from C3.
   - One of the two C2 flags is an evaluator false positive: the regex `not complet*` counted "has not completed any
     assessments" as a risk assertion.
   - The other flag is a genuine LLM-own risk judgement.
2. **LLM without retrieval produces an incorrect or unsupported claim** (FFF_2014J_636469, C2):
   - "The student demonstrates strong time management and adherence to deadlines" asserts a non-observable attribute.
   - "27 active days out of a possible 28" was flagged only because of the constant 28, an evaluator artefact fixed in v2.
   - Since C1 never made claims, this case necessarily comes from C2.
3. **SHAP improves the explanation** (CCC_2014J_2316201): SHAP consistency goes from 0.00 in C3 to 1.00 in C4. However,
   C4's top-3 includes `highest_education`, which illustrates the concern below.
4. **Insufficient evidence** (CCC_2014J_556450, no VLE activity): both C3 and C4 describe the complete absence of
   activity and list what cannot be concluded.

**Table 4: examples of evidence-grounded versus unsupported statements** (verbatim, main set; the verdicts are the frozen
evaluator's, and they were checked by reading)

| Condition | Statement | Verdict |
|---|---|---|
| C4 | "The model estimates a 29% probability of an unsuccessful outcome (Fail or Withdrawn)." [M1] | grounded, labelled as inference |
| C4 | "The model associates a lower predicted risk with having assessments due before the end of week 4." [X1] | grounded, but it reflects a course-design feature |
| C2 | "The student is at high risk of dropping out." | unsupported: no risk estimate in the context |
| C2 | "The student demonstrates strong engagement and time management skills…" | unsupported: non-observable attribute |
| C2 | "…decreased by approximately 58% from weeks 1-2 to weeks 3-4 (activity_trend: -0.5833)." | incorrect: the trend index was read as a percent change |
| C3 | "…below the typical passing threshold often associated with a score of 50…" | unsupported: external claim, and wrong (the OU pass mark is 40) |
| C3 | "…activity on only 1 day out of the 7 days comprising weeks 3 and 4." | incorrect arithmetic (the window is 14 days) |

**Grounded does not mean appropriate.** `highest_education` is in the SHAP top-3 for 76/100 sampled students, and
31/281 C4 recommendations build on it, for example "Provide foundational support tailored to entry qualifications…
bridge potential gaps in prior knowledge". Such statements are fully grounded in the evidence, but they are
deficit-framed, and they target a characteristic the student cannot change. SHAP also leads to "driver of risk" wording,
which is quasi-causal. Consistency with the evidence does not make a recommendation pedagogically sound (rule 10).

## 10. Methodological decisions and problems encountered
1. **Withdrawals before the cutoff were excluded** (5,040), because their label is known at prediction time.
2. **Mark availability is unknown in OULAD.** Scores are used if submitted before day 28 (an assumption), with a
   no-score sensitivity model; the results are almost unchanged.
3. **Structural missingness:** EEE and GGG have no assessment due before week 4. The features use NaN, and the evidence says "not applicable".
4. **Temporal split by presentation** (train 2013B/13J/14B, test 2014J); 2014B and 2014J approximately overlap in time.
5. **Protected demographics were excluded** from the model (authors' decision). `highest_education` was kept, and it
   became a top SHAP factor, with the consequences described in §9.
6. **SHAP explains the at-risk probability,** not a single class, because decision support is about risk.
7. **Activity-trend representation changed after the pilot.** It was first shown as a percentage ("−65%"), which
   the LLM read as a percent change ("65% drop"), but the true drop was 79%. It is now shown as a signed index (−0.65,
   "not a percent change"), and the evaluator no longer accepts the percentage reading. Even with the raw value (C2),
   the LLM sometimes still produced the misreading. *How LA evidence is represented affects faithfulness.*
8. **Pilot bugs fixed before the main run:** C2's `context_ids` wrongly included M/X items (metadata only;
   the prompt was unaffected), and max_tokens was raised from 700 to 1500 after one truncated output.
9. **The LLM needed `reasoning_effort="none"`,** otherwise Qwen3.5 spent the whole token budget on hidden reasoning.
10. **C1 abstained completely (Qwen) or gave only a generic recommendation (Gemma).** The prompt was **not** altered to elicit hallucinations, because that would bias the comparison. C1 is reported as a behaviour check.
11. **The retrieval ground truth is a proxy** (SHAP top-k), because no human relevance labels exist. Integrity is verified.
12. **The evaluator was frozen after the pilot.** Post-hoc v2 fixes are reported only as sensitivity.
13. **LM Studio changed the loaded model during the runs** (twice). A model guard was added to the generator, and every saved output was verified to come from the intended model.

## 11. Reproducibility
- Seeds: 42 (sampling, models, SHAP background and permutations, bootstrap, LLM).
- **Determinism check:** 10 main-sample students × 4 conditions were regenerated and compared byte for byte with the
  original outputs. Result in `results/determinism_check.csv`.
- Saved artefacts:
  - models: `models/`
  - features: `data/features.csv`
  - predictions: `data/predictions.csv`
  - SHAP: `data/shap_values_risk.csv`
  - evidence: `data/evidence_store.jsonl`, `data/student_evidence.*`
  - retrieved evidence: `data/retrieved_evidence.jsonl`
  - LLM outputs: `results/llm_outputs*`
  - metrics: `results/metrics.json`, `results/*.csv`

**Determinism results** (`results/determinism_check.csv`):
- **Within one server session**, 40/40 regenerated outputs were byte-identical for Qwen, and 40/40 for Gemma
  (10 students × 4 conditions each; temperature 0, seed 42).
- **After reloading the model**, outputs are not bit-identical: 9/10 in a 5-student check, and 53–54% over 100 students
  (§8b-iii). Aggregate metrics are stable across sessions.
- Gemma additionally returned a different output for the *first* call of a new prompt than for immediate repeats (a KV-cache
  effect). Each pipeline prompt is called once, so this does not affect the reported runs.
- Every output file records the model that actually answered. All 2,182 LLM outputs used in this report were verified
  to come from the intended model; the generator now aborts if the server answers with a different model.

## 12. Limitations
- **Three small local LLMs (4–14B), a single prompt, a single dataset, a single cutoff (week 4).** The main UCR effect was found with
  Qwen3.5-9B only; it did not replicate with Gemma-4-E4B or Qwen3-14B. The results may not transfer to other models, to prompts without strict grounding
  rules, or to other institutions.
- **Post-hoc analyses:** the ablation (C5), the additional LLMs and the analyses of §8c were added (the anchoring effect was then confirmed in a pre-registered test, §8d) after inspecting the main results. They are
  confirmatory checks of an interpretation, not pre-registered hypotheses.
- **The deterministic evaluator is lexicon-based.**
  - It detects feature references, numbers and a fixed list of non-observable concepts.
  - It cannot judge semantic correctness beyond numbers. Examples: clicks versus pages, the direction of a qualitative
    trend claim, invented topics outside the lexicon.
  - It was tuned on 8 pilot students, and its precision and recall were not validated against human annotation.
  - The number-matching fallback (any context value) is lenient for small integers.
- **UCR depends on what counts as a factual claim.** In particular, risk judgements that the LLM makes itself are
  counted as unsupported when no model estimate is given. Other operationalisations could reduce C2's disadvantage (see the
  sensitivity analysis above: excluding them, C2 has 0.023 versus C4 0.013).
- **SHAP consistency is partly circular for C4.** The specificity metric reached its ceiling.
- **The predictive model is modest** (Macro-F1 0.38). The evidence given to the LLM includes an imperfect risk estimate, and
  a well-grounded recommendation can still rest on a wrong prediction. We did not evaluate how the correctness of the
  prediction interacts with the recommendation.
- **No human evaluation,** no educator judgement of usefulness, and no intervention. We make **no claims** about
  pedagogical quality, learning impact or causality.
- **Sample:** n = 100 students (stratified by outcome, not by module; AAA is represented by 1 student).
- **Retrieval relevance is measured against a model-defined proxy only.**

## 13. Files produced
`DATASET_ANALYSIS.md`, `EXPERIMENT_REPORT.md`, `README.md`, `results/ml_results.{csv,png}`,
`results/llm_comparison.csv`, `results/metrics.json`, `results/case_studies.md`, `figures/fig1–fig5`,
`data/student_evidence.{csv,jsonl}`, `data/evidence_store.jsonl`,
`results/llm_outputs/{pilot,main,repro,ablation,gemma,repro_gemma,repro_qwen_reload}/`,
`results/llm_outputs_{pilot,main,ablation,gemma}.jsonl`, `results/ablation_*.csv`, `results/second_llm_*.csv`,
`results/session_variability.csv`, `results/determinism_check.csv`, `results/cross_model_consistency*.csv`,
`results/llm_sensitivity.csv`, `results/misprediction_*.csv`, `results/PREREGISTRATION_anchoring.md`, `results/anchoring_*.csv`, `results/rag_control_*.csv`, `results/llm_outputs/qwen14b/`, `results/eval_*`, `results/retrieval_metrics.csv`,
`results/evaluator_verification_sheet.csv`.
