#!/usr/bin/env bash
# RAG control conditions (C6 full evidence + IDs, C7 random 5) for the three LLMs, run sequentially.
# Swaps the LM Studio model between runs. Each set is resumable.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
LMS=~/.lmstudio/bin/lms
run_set () {  # $1 = model id, $2 = set name
  $LMS unload --all || true
  $LMS load "$1" --context-length 32768 -y
  $PY src/07_generate.py "$2"
  $PY src/08_evaluate.py "$2"
}
run_set qwen/qwen3.5-9b   control_qwen35
run_set google/gemma-4-e4b control_gemma
run_set qwen/qwen3-14b     control_qwen14b
echo "ALL CONTROL RUNS DONE"
