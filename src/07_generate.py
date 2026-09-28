"""Step 7 - Experiments 5-6: LLM generation under four evidence conditions.

C1 llm_only        : task only (student identifiers + time point, no learning evidence)
C2 student_data    : full structured feature record + data dictionary (no evidence IDs, no model output)
C3 rag             : top-K retrieved evidence items (model-independent retriever, with IDs)
C4 shap_rag        : predicted risk (M1) + SHAP top-3 (X1-X3) + retrieved evidence + the evidence
                     items holding the values of the SHAP features
C5 rag_risk        : ABLATION (added after the main run): predicted risk (M1) + retrieved evidence, no SHAP
Same system prompt, same task text and output schema in all conditions; the only difference is the
context block (and the grounding sentence, only added when evidence is provided, as specified).
Model: local LM Studio server (OpenAI-compatible), temperature 0, fixed seed, thinking disabled.
Usage: python src/07_generate.py [pilot|main|repro|ablation|gemma|repro_gemma|repro_qwen_reload|qwen14b|anchoring|control_qwen35|control_gemma|control_qwen14b]
  ablation : C5 + C3 + C4 in a single server session, same LLM as main (Qwen3.5-9B), same 100 students
  gemma    : C1-C4 with google/gemma-4-e4b (second-LLM robustness), same 100 students
Outputs: results/llm_outputs/<set>/<condition>/<enrolment_id>.json and results/llm_outputs_<set>.jsonl
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_dictionary import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "configs" / "config.json"))
LLM = CFG["llm"]
SHAP_K_CONTEXT = 3
SET = sys.argv[1] if len(sys.argv) > 1 else "main"
CONDITIONS = ["C1_llm_only", "C2_student_data", "C3_rag", "C4_shap_rag"]
if SET == "ablation":  # C3, C5, C4 regenerated in ONE server session (Qwen reloads are not bit-identical)
    CONDITIONS = ["C5_rag_risk", "C3_rag", "C4_shap_rag"]
if SET in ("gemma", "repro_gemma"):
    LLM = {**LLM, "model": "google/gemma-4-e4b"}
if SET == "anchoring":  # confirmatory anchoring test: C2 vs C4, Qwen3.5-9B, 80 new students
    CONDITIONS = ["C2_student_data", "C4_shap_rag"]
if SET == "qwen14b":  # third LLM (larger model, earlier Qwen generation), C1-C4 as for Gemma
    LLM = {**LLM, "model": "qwen/qwen3-14b"}
# RAG control (added after external review): separate evidence SELECTION from evidence FORMAT
#   C6_full_evidence : all 27 observed evidence items, same statement+ID format as C3, no selection
#   C7_random5       : 5 randomly drawn observed items (seeded per student), same format as C3
if SET == "control_qwen35":
    CONDITIONS = ["C6_full_evidence", "C7_random5"]
if SET == "control_gemma":
    CONDITIONS = ["C6_full_evidence"]
    LLM = {**LLM, "model": "google/gemma-4-e4b"}
if SET == "control_qwen14b":
    CONDITIONS = ["C6_full_evidence"]
    LLM = {**LLM, "model": "qwen/qwen3-14b"}
if SET == "repro_qwen_reload":  # after reloading Qwen: check that it reproduces the main-run outputs
    CONDITIONS = ["C3_rag", "C4_shap_rag"]

SYSTEM = """You are a decision-support assistant for educators in an online distance-learning university.
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
Answer only with a JSON object that follows the required schema."""

TASK = ("TASK: Based only on the information provided, generate a concise educational decision-support "
        "recommendation for an educator.")
GROUNDING = "Do not introduce factual claims that are not supported by the provided evidence."

SCHEMA = {
    "type": "object",
    "properties": {
        "observed_facts": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "properties": {"statement": {"type": "string"},
                                             "evidence_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement", "evidence_ids"]}},
        "model_inferences": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "properties": {"statement": {"type": "string"},
                                             "evidence_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["statement", "evidence_ids"]}},
        "recommendations": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "properties": {"action": {"type": "string"}, "rationale": {"type": "string"},
                                             "evidence_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["action", "rationale", "evidence_ids"]}},
        "evidence_limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["observed_facts", "model_inferences", "recommendations", "evidence_limitations"],
}

store = defaultdict(dict)
for line in open(ROOT / "data" / "evidence_store.jsonl"):
    it = json.loads(line)
    store[it["enrolment_id"]][it["id"]] = it
retrieved = {json.loads(l)["enrolment_id"]: json.loads(l)["retrieved"]
             for l in open(ROOT / "data" / "retrieved_evidence.jsonl")}
if (ROOT / "data" / "retrieved_evidence_anchoring.jsonl").exists():
    retrieved.update({json.loads(l)["enrolment_id"]: json.loads(l)["retrieved"]
                      for l in open(ROOT / "data" / "retrieved_evidence_anchoring.jsonl")})


def header(eid):
    e = store[eid]["E1"]
    return (f"STUDENT: student ID {e['student_id']}, module {e['course']}, presentation {e['presentation']}.\n"
            f"TIME POINT: end of week {e['time_cutoff_week']} of the module presentation "
            f"(only information recorded up to this point exists).")


def item_line(it):
    return f"[{it['id']}] {it['statement']}"


def context(cond, eid):
    items = store[eid]
    if cond == "C1_llm_only":
        return "CONTEXT: No student-specific learning evidence is available for this student.", []
    if cond == "C2_student_data":
        rec = {it["feature"]: (None if it["value"] is None else round(it["value"], 4))
               for k, it in items.items() if k.startswith("E")}
        dic = {f: FEATURES[f]["label"].split(":")[0] for f in rec}
        return ("CONTEXT - STUDENT DATA (structured record from the learning analytics database; "
                "null = not available / not applicable):\n" + json.dumps(rec, indent=1) +
                "\nDATA DICTIONARY:\n" + json.dumps(dic, indent=1)), [k for k in items if k.startswith("E")]
    if cond == "C6_full_evidence":
        ev = sorted([it for k, it in items.items() if k.startswith("E")], key=lambda i: int(i["id"][1:]))
        return ("CONTEXT - STUDENT EVIDENCE (all observed learning analytics data for this student):\n" +
                "\n".join(item_line(i) for i in ev)), [i["id"] for i in ev]
    if cond == "C7_random5":
        import numpy as np
        pool = sorted([it for k, it in items.items() if k.startswith("E")], key=lambda i: int(i["id"][1:]))
        rng = np.random.default_rng(CFG["seed"] + int(items["E1"]["student_id"]))
        ret = [pool[i] for i in rng.choice(len(pool), CFG["retrieval_k"], replace=False)]
        return ("CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):\n" +
                "\n".join(item_line(i) for i in ret)), [i["id"] for i in ret]
    ret = [items[r["id"]] for r in retrieved[eid]]
    if cond == "C3_rag":
        return ("CONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):\n" +
                "\n".join(item_line(i) for i in ret)), [i["id"] for i in ret]
    if cond == "C5_rag_risk":  # ablation: model risk estimate without SHAP
        txt = ("CONTEXT - PREDICTIVE MODEL OUTPUT (model inference, not an observed fact):\n" + item_line(items["M1"]) +
               "\n\nCONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):\n" +
               "\n".join(item_line(i) for i in ret))
        return txt, ["M1"] + [i["id"] for i in ret]
    # C4
    xs = [items[f"X{n}"] for n in range(1, SHAP_K_CONTEXT + 1)]
    linked = [items[x["linked_evidence"]] for x in xs]
    ev = ret + [l for l in linked if l["id"] not in {i["id"] for i in ret}]
    ev.sort(key=lambda i: int(i["id"][1:]))
    txt = ("CONTEXT - PREDICTIVE MODEL OUTPUT (model inference, not an observed fact):\n" + item_line(items["M1"]) +
           "\n\nCONTEXT - MODEL EXPLANATION (SHAP: features that most changed the model's predicted risk "
           "for this student; associations in the model, not causes):\n" + "\n".join(item_line(x) for x in xs) +
           "\n\nCONTEXT - RETRIEVED EVIDENCE (observed learning analytics data for this student):\n" +
           "\n".join(item_line(i) for i in ev))
    return txt, ["M1"] + [x["id"] for x in xs] + [i["id"] for i in ev]


def build_prompt(cond, eid):
    ctx, ids = context(cond, eid)
    parts = [TASK, header(eid), ctx]
    if cond != "C1_llm_only":
        parts.append(GROUNDING)
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts), ids


def call(user):
    body = {"model": LLM["model"], "temperature": LLM["temperature"], "seed": LLM["seed"],
            "max_tokens": LLM["max_tokens"], "reasoning_effort": "none",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "recommendation", "strict": True, "schema": SCHEMA}}}
    t = time.time()
    r = requests.post(LLM["base_url"] + "/chat/completions", json=body, timeout=600).json()
    if "choices" not in r or r.get("model") != LLM["model"]:
        sys.exit(f"ABORT: server did not answer with {LLM['model']} (response: {str(r)[:300]}). "
                 "Check which model LM Studio has loaded; no output was written for this call.")
    return r, time.time() - t, body


if __name__ == "__main__":
    ids = pd.read_csv(ROOT / "data" / ("pilot_sample.csv" if SET == "pilot" else
                                       "anchoring_sample.csv" if SET == "anchoring" else "llm_sample.csv")).enrolment_id
    if SET in ("repro", "repro_gemma"):  # determinism check: regenerate the first 10 main-sample students
        ids = ids.iloc[:10]
    if SET == "repro_qwen_reload":
        ids = ids.iloc[:5]
    outdir = ROOT / "results" / "llm_outputs" / SET
    for eid in ids:
        for cond in CONDITIONS:
            f = outdir / cond / f"{eid}.json"
            if f.exists():
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            prompt, ctx_ids = build_prompt(cond, eid)
            resp, dt, body = call(prompt)
            content = resp["choices"][0]["message"].get("content", "")
            try:
                parsed, ok = json.loads(content), True
            except Exception:
                parsed, ok = None, False
            json.dump({"enrolment_id": eid, "condition": cond, "context_ids": ctx_ids,
                       "system_prompt": SYSTEM, "user_prompt": prompt, "request_params":
                       {k: v for k, v in body.items() if k != "messages"}, "raw_content": content,
                       "parsed": parsed, "json_ok": ok, "usage": resp.get("usage"),
                       "finish_reason": resp["choices"][0].get("finish_reason"), "seconds": round(dt, 2),
                       "model": resp.get("model")}, open(f, "w"), indent=1)
            print(eid, cond, f"{dt:.1f}s", resp.get("usage", {}).get("completion_tokens"), "ok" if ok else "JSON FAIL",
                  flush=True)
    with open(ROOT / "results" / f"llm_outputs_{SET}.jsonl", "w") as fh:
        for eid in ids:
            for cond in CONDITIONS:
                fh.write(json.dumps(json.load(open(outdir / cond / f"{eid}.json"))) + "\n")
