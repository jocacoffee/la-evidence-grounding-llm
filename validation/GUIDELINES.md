# Annotation guidelines: validation of the automatic grounding evaluator

**Goal.** Check whether the automatic evaluator correctly decides if a statement written by the LLM is **supported
by the context the LLM received**. You are rating *the metric*, not the quality of the recommendation.

**Files.**
- `annotation_sheet_A.xlsx`: main annotator, 277 items.
- `annotation_sheet_B.xlsx`: second annotator, 60 items (a subset of A). It must be labelled **independently**, without
  looking at the A labels.
- Do **not** open `_key_DO_NOT_OPEN.csv`. It holds the evaluator's decisions; the sheets are blind on purpose.

**Each row has:**
- `statement`: one sentence written by the LLM;
- `statement_type`: observed fact, model inference, or recommendation rationale;
- `context_given_to_LLM`: exactly what the LLM saw about the student. It can be:
  - a JSON record with a data dictionary;
  - a list of evidence items with IDs (E = observed data, M = model prediction, X = SHAP explanation);
  - both.

Fill the `label` column (dropdown) with one of the four labels below. `note` is optional; use it for doubtful cases.

| label | Use when |
|---|---|
| `supported` | Everything the statement asserts **about this student (or about the model's output for this student)** can be verified in the context. Correct simple derivations count as supported: rounding, 0.31 → "31%", "69% below the median" from a ratio of 0.31, differences or percent changes between two given numbers, "no submissions" from a value of 0. |
| `unsupported` | The statement asserts something about the student that the context does not support. |
| `not_a_claim` | The statement asserts nothing specific about this student. Examples: generic pedagogy ("Regular check-ins help students stay engaged"), statements that data are unavailable, or a possibility that is clearly hedged ("it is unclear whether…", "the student *may* be facing…"). |
| `unclear` | You cannot decide. Please add a note. |

**Mark `unsupported`, for example, when the statement:**
1. **Gives a value that is not in the context or is derived incorrectly.** Examples: "24 active days" when the context says 14;
   "7 days in weeks 3–4" (the window is 14 days); "a 65% drop in clicks" when the context only gives an index of −0.65.
2. **Attributes something OULAD does not record**, stated as fact: motivation, stress, confidence, time management,
   personal circumstances, difficulty with specific topics or skills.
3. **States a risk level or a likely outcome** ("the student is at high risk of dropping out") when the context has
   **no** model risk estimate (no M1 item). A risk level that repeats a given M1 value is supported.
4. **Uses knowledge from outside the context about the student's situation** (e.g., "the passing threshold is 50").
5. **Talks about data that exist in OULAD but were not given in this context.** For example, it mentions forum activity when only
   5 items without forum activity were provided. This is unsupported *by the context*, even if it happens to be true.

**Borderline cases:**
- A qualitative description that a given value supports ("low engagement" when clicks are 0.31 × the cohort median) → `supported`.
- The same description with nothing in the context to support it → `unsupported`.
- A clearly signalled interpretation that follows from given values ("this *suggests* a decline in engagement" when the trend
  index is negative) → `supported`. Only mark it `unsupported` if it adds new content (point 2 or 3).
- Mixed statements: if **any** part is unsupported, the statement is `unsupported`.
- Model explanations (X items) are associations in the model. Calling them "drivers" is a wording issue, not a support
  issue: judge only whether the factor and its value are in the context.

**Time:** about 30–40 seconds per item, roughly 2.5 h for sheet A and 30 min for sheet B.
When finished, run: `.venv/bin/python src/15_evaluator_validation.py score`

The output (`results/evaluator_validation.csv`) gives:
- precision of the "unsupported" flags (with 95% CI) and estimated recall, per condition;
- a human-corrected UCR;
- Cohen's κ between annotators A and B.
