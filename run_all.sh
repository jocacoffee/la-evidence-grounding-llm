#!/usr/bin/env bash
# Reproduce all experiments. Requires: project/.venv (pip install -r requirements.txt) and, for steps 7-9,
# LM Studio serving qwen/qwen3.5-9b at http://localhost:1234 (see configs/config.json).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/01_inspect_dataset.py        # DATASET_ANALYSIS numbers + Fig. 2
$PY src/02_build_features.py         # temporal features at week 4
$PY src/03_train_models.py           # Exp 1: LR / XGBoost + Fig. 3
$PY src/03b_ml_breakdown.py          # per-module breakdown
$PY src/04_explain_shap.py           # Exp 2: SHAP + student_evidence + Fig. 4 (~10 min)
$PY src/05_evidence_store.py         # Exp 3: evidence store
$PY src/06_retrieval.py              # Exp 4: retrieval + LLM/pilot samples
$PY src/07_generate.py pilot         # pilot generations (evaluator development)
$PY src/08_evaluate.py pilot
$PY src/07_generate.py main          # Exp 5-6: 4 conditions x 100 students (~2 h); resumable
$PY src/08_evaluate.py main          # Exp 7: deterministic evaluation
$PY src/09_compare.py main           # Exp 8: comparison, Fig. 5, metrics.json, case studies
$PY src/09b_error_analysis.py        # breakdown of unsupported claims
EVAL_VERSION=v2 $PY src/08_evaluate.py main   # post-hoc evaluator sensitivity (v2)
$PY src/07_generate.py repro         # determinism check (10 students x 4 conditions)
$PY src/07_generate.py ablation      # C5 ablation (+ C3/C4 regenerated in the same session)
$PY src/08_evaluate.py ablation
# second LLM: load google/gemma-4-e4b in LM Studio first
$PY src/07_generate.py gemma
$PY src/07_generate.py repro_gemma
$PY src/08_evaluate.py gemma
# third LLM: load qwen/qwen3-14b in LM Studio first
$PY src/07_generate.py qwen14b
$PY src/08_evaluate.py qwen14b
$PY src/11_ablation_and_second_llm.py  # ablation, other LLMs, session variability
$PY src/12_consistency_sensitivity.py  # cross-model consistency, sensitivity, mispredictions
# confirmatory anchoring test (pre-registered: results/PREREGISTRATION_anchoring.md); load qwen/qwen3.5-9b first
$PY src/13_anchoring_sample.py
$PY src/07_generate.py anchoring
$PY src/08_evaluate.py anchoring
$PY src/14_anchoring_confirmatory.py
$PY src/16_figure_anchoring.py         # Figure: anchoring (confirmatory test)
./run_control.sh                      # RAG control conditions C6/C7 (3 LLMs; swaps LM Studio models)
$PY src/17_rag_control.py             # format vs selection vs ranking
# human validation of the evaluator: prepare blind sheets, annotate (validation/GUIDELINES.md), then score
# $PY src/15_evaluator_validation.py prepare   # already done; re-running reshuffles nothing (seeded)
# $PY src/15_evaluator_validation.py score
$PY src/09c_sensitivity_v2.py        # v1 vs v2 + determinism comparison
$PY src/10_figure_architecture.py    # Fig. 1
