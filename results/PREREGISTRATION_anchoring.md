# Pre-registration: confirmatory anchoring test

Written **before** any output of this test was generated (see the timestamp and hashes below). The hypotheses came from
the exploratory result in the main sample (Qwen3.5-9B, 11 missed at-risk students: outreach C2 6/11 versus C4 0/11,
exact McNemar p = 0.031). This test uses **new students** and fixed, frozen code.

## Design
- **Students:** 2014J test set, excluding the 108 main-sample and pilot enrolments. Seed 42 (`src/13_anchoring_sample.py`).
  - 40 **missed at-risk**: predicted risk < 0.5 and true outcome Fail or Withdrawn.
  - 40 **false alarms**: predicted risk ≥ 0.5 and true outcome Pass or Distinction.
- **LLM:** Qwen3.5-9B (LM Studio, MLX 4-bit), temperature 0, seed 42, thinking disabled. Same prompts and schema as the main
  study (`src/07_generate.py anchoring`).
- **Conditions:** C2 (full student record, no model output) and C4 (retrieved evidence + predicted risk + SHAP top-3).
  Both conditions of a student are generated consecutively in the same server session.
- **Outcome extraction:** frozen `src/llm_extract.py` (the rules used in the exploratory analysis).
  - *outreach* = at least one recommendation whose action matches the outreach lexicon;
  - *stated risk band* = `stated_band()`.
  - UCR comes from the frozen evaluator `src/08_evaluate.py` (v1, EVAL_VERSION unset).

## Hypotheses
- **H1 (primary):** for missed at-risk students, the proportion of outputs proposing outreach is **lower in C4 than in C2**.
- **H2 (secondary):** for missed at-risk students, the proportion stating **low** risk is **higher in C4 than in C2**.
- **H3 (secondary):** for false alarms, the proportion stating **high** risk is **higher in C4 than in C2**.

## Analysis
- Paired exact McNemar test (two-sided binomial test on discordant pairs), α = 0.05.
- H1 is tested alone. H2 and H3 are Holm-corrected together.
- Effect sizes are reported as the proportions in C2 and C4, with discordant counts.
- **Exclusions:** a student is excluded from a paired test if either output is not valid JSON; exclusions are reported.
  An output with valid JSON and no recommendations counts as "no outreach".
- **Descriptive only (not confirmatory):** UCR and coverage per group and condition; outreach for false alarms.
- **Results are reported whatever their direction,** including a null or reversed result for H1.

## Frozen artefacts (SHA1)
```
99f4ad39f8bd46bd9b8ba1d4380460b87ab0d295  src/llm_extract.py
f4398c5d190a69435597e3683c4ed2f58c138640  src/08_evaluate.py
531f87c0a7cd15fa9dabc76a1dc302d02983df19  src/feature_dictionary.py
c8ccacec6d37fba7e0671855168e060667eabc92  src/07_generate.py
270582d9f15eca6ddbf43a7cfddd6c65fbd6e4d4  src/13_anchoring_sample.py
0e929f2a4e902f873bd176268b25d7ed5f53e71e  data/anchoring_sample.csv
1177359649f01d04069d540de89d185ca46d9a4f  data/retrieved_evidence_anchoring.jsonl
```

Timestamp (local): 2026-09-26 14:44:23 -04

Outputs existing for this test at registration time: 0 (must be 0)

---
## Results (appended after the run; the registered sections above were not modified)
Run completed 2026-09-26 16:28 -04. There were 160 outputs, all from qwen/qwen3.5-9b. The hashes of the frozen artefacts were re-checked
and are identical. One output (DDD_2014J_359657, C2) hit the token limit and was invalid JSON, so it was excluded from the missed-at-risk pairs
as pre-specified (n = 39).

| Hypothesis | Group | C2 | C4 | Discordant (only C2 / only C4) | Exact McNemar p | Holm p | As expected |
|---|---|---|---|---|---|---|---|
| H1 (primary): outreach lower in C4 | missed at-risk (n = 39) | 28.2% | 10.3% | 8 / 1 | 0.039 | 0.039 (alone) | yes |
| H2: low risk stated more in C4 | missed at-risk (n = 39) | 23.1% | 56.4% | 5 / 18 | 0.011 | 0.021 | yes |
| H3: high risk stated more in C4 | false alarms (n = 40) | 32.5% | 55.0% | 3 / 12 | 0.035 | 0.035 | yes |

**Descriptive (not confirmatory):**
- Missed at-risk students: high risk was stated in 27.5% of C2 outputs versus 0% of C4 outputs.
- False alarms: outreach was 42.5% in C2 versus 45.0% in C4.
- UCR: C2 0.048–0.055; C4 0.007–0.025.
- Coverage: 0.956–0.982.

Files: `results/anchoring_confirmatory.csv`, `results/anchoring_descriptive.csv`.
