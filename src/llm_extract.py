"""Deterministic extraction of stated risk band and action categories from LLM outputs.
Shared by src/12_consistency_sensitivity.py and src/14_anchoring_confirmatory.py.
FROZEN before the confirmatory anchoring run (hash in results/PREREGISTRATION_anchoring.md).
"""
import re

import numpy as np

HIGH = re.compile(r"\b(high|elevated|significant|substantial|considerable|serious|critical)\b[^.;]{0,20}\brisk\b|"
                  r"\bat[- ]risk\b|\bhigh (probability|likelihood)\b|\bstrong likelihood\b", re.I)
LOW = re.compile(r"\blow(er)?\b[^.;]{0,20}\b(risk|probability|likelihood)\b|\bminimal risk\b|\bon track\b|"
                 r"\bunlikely to (fail|withdraw)\b", re.I)
MOD = re.compile(r"\b(moderate|medium|intermediate)\b[^.;]{0,20}\b(risk|probability|likelihood)\b", re.I)
PCT = re.compile(r"(\d{1,3}(?:\.\d)?)\s?%[^.;]{0,40}\b(probability|risk|likelihood|unsuccessful)\b|"
                 r"\b(probability|risk|likelihood)\b[^.;]{0,40}?(\d{1,3}(?:\.\d)?)\s?%", re.I)
ACTIONS = {
    "outreach": r"\b(contact|reach(ing)? out|check[- ]in|message|email|call|meet|conversation|one-on-one|1:1)",
    "assessment": r"\b(deadline|submission|submit|assessment|assignment|tma|cma|score)",
    "engagement": r"\b(engag|activity|resource|material|content|forum|vle|log ?in|platform|participat)",
    "support": r"\b(support|tutor|help|foundational|scaffold|guidance|study skill|strateg)",
    "monitor": r"\b(monitor|track|observe|follow[- ]up)",
    "positive": r"\b(acknowledg|prais|commend|recogni|celebrat|maintain|continu|enrichment|advanced|challeng)",
}
ACTIONS = {k: re.compile(v, re.I) for k, v in ACTIONS.items()}
BAND = {"low": 0, "moderate": 1, "high": 2}


def stated_band(parsed):
    """Risk band stated in facts, inferences and rationales. A quoted percentage wins; otherwise keyword votes."""
    txt = " ".join([x.get("statement", "") for x in parsed.get("observed_facts", []) + parsed.get("model_inferences", [])] +
                   [x.get("rationale", "") for x in parsed.get("recommendations", [])])
    m = PCT.search(txt)
    if m:
        v = float(m.group(1) or m.group(4))
        return "high" if v >= 66 else ("moderate" if v >= 33 else "low")
    h, l, mo = len(HIGH.findall(txt)), len(LOW.findall(txt)), len(MOD.findall(txt))
    if h == l == mo == 0:
        return None
    return max([("high", h), ("low", l), ("moderate", mo)], key=lambda t: t[1])[0] if sorted([h, l, mo])[-1] != sorted([h, l, mo])[-2] else None


def action_cats(parsed):
    return {k for x in parsed.get("recommendations", []) for k, rx in ACTIONS.items() if rx.search(x.get("action", ""))}


def jacc(a, b):
    return np.nan if not a and not b else len(a & b) / len(a | b)


