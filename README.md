# Grounded, Consistent, Anchored — code, prompts, outputs and audits

Anonymised research artefact accompanying a submission under double-blind review. It contains everything needed to
inspect or reproduce the study: scripts, configuration, prompts, all LLM outputs, evaluation audits, the pre-registration
of the confirmatory test, and the human validation of the evaluator.

The study asks whether student-specific learning analytics (LA) evidence (observed traces, a predicted risk and SHAP
explanations) grounds LLM-generated recommendations for educators. It evaluates three properties: grounding,
individualisation and robustness to predictive-model error. Full protocol and results: `EXPERIMENT_REPORT.md`. Dataset
inspection and methodological decisions: `DATASET_ANALYSIS.md`.

## Data
The raw data are **not** included. Download the public Open University Learning Analytics Dataset (OULAD;
Kuzilek, Hlosta & Zdrahal, 2017, *Scientific Data* 4:170171, https://analyse.kmi.open.ac.uk/open_dataset) and place
the 7 CSV files in a folder named `open+university+learning+analytics+dataset/` **next to** this repository. You can
also edit `raw_dir` in `configs/config.json`.

Derived data are included in `data/`:
- features, predictions and SHAP values;
- the student evidence records;
- the samples;
- the retrieved evidence.

The full evidence store is compressed (`data/evidence_store.jsonl.gz`); decompress it before running the LLM steps:
```bash
gunzip -k data/evidence_store.jsonl.gz
```

## Environment
- Python 3.9: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Also install `odfpy` and
  `openpyxl` to read the annotation sheets.
- LLM steps: a local OpenAI-compatible server (we used LM Studio) serving `qwen/qwen3.5-9b`, `google/gemma-4-e4b` and
  `qwen/qwen3-14b` (MLX 4-bit), at temperature 0, seed 42, with thinking disabled (`reasoning_effort: "none"`). The
  generator aborts if the server answers with a different model than requested.

## Reproduce
`run_all.sh` lists every step in order; `run_control.sh` runs the format and selection controls (C6, C7) for the three
LLMs. The main steps are:

| Step | Script | Main outputs |
|---|---|---|
| Dataset inspection | `src/01_inspect_dataset.py` | `results/dataset_inspection.json` |
| Features (day < 28) | `src/02_build_features.py` | `data/features.csv` |
| Prediction | `src/03_train_models.py`, `src/03b_ml_breakdown.py` | `results/ml_results.csv`, `models/` |
| SHAP | `src/04_explain_shap.py` | `data/shap_values_risk.csv`, `data/student_evidence.*` |
| Evidence store | `src/05_evidence_store.py` | `data/evidence_store.jsonl` |
| Retrieval and samples | `src/06_retrieval.py` | `results/retrieval_metrics.csv`, `data/llm_sample.csv` |
| Generation | `src/07_generate.py <set>` | `results/llm_outputs/<set>/`, `results/llm_outputs_<set>.jsonl` |
| Deterministic evaluation (frozen) | `src/08_evaluate.py <set>` | `results/eval_per_output_<set>.csv`, `results/eval_audit_<set>.jsonl` |
| Comparisons | `src/09_compare.py`, `src/09b_error_analysis.py`, `src/11_ablation_and_second_llm.py`, `src/12_consistency_sensitivity.py`, `src/17_rag_control.py` | `results/*.csv` |
| Confirmatory anchoring test | `src/13_anchoring_sample.py`, `src/14_anchoring_confirmatory.py` | `results/PREREGISTRATION_anchoring.md`, `results/anchoring_*.csv` |
| Human validation of the evaluator | `src/15_evaluator_validation.py score` | `results/evaluator_validation.csv` |
| Figures | `src/10_figure_architecture.py`, `src/16_figure_anchoring.py` | `figures/` |

Every LLM output file stores the exact system prompt, user prompt, request parameters, raw response, token usage,
latency and the model that answered.

## Pre-registration and validation
- `results/PREREGISTRATION_anchoring.md` contains the hypotheses, analysis plan and SHA1 hashes of the frozen code and
  sample, written before any output of the confirmatory test was generated. The results were appended afterwards.
- `validation/` holds the blind annotation sheets, the guidelines and the key. See `EXPERIMENT_REPORT.md` §8f for the
  procedure and results.
- The frozen evaluator is `src/08_evaluate.py` with `EVAL_VERSION` unset. Hashes are in
  `results/evaluator_frozen_sha1.txt`. Later edits only added an opt-in post-hoc variant (`EVAL_VERSION=v2`) and output
  naming; v1 results were verified to be unchanged.

## Licence
Code: MIT. Derived data follow the OULAD licence (CC BY 4.0).
