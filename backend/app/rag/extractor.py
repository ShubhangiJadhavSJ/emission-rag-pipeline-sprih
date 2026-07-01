"""LLM extraction step: build context from chunks, call the model, parse JSON.

Robust parsing is important — if the model returns prose or malformed JSON we
degrade to all-NA rather than crash the pipeline (assignment: "Return NA rather
than hallucinate").
"""
import json
import re
from dataclasses import dataclass

from app.llm.client import LLMResult, complete
from app.rag.prompts import build_prompt

_NA_FIELD = {"value": "NA", "unit": "NA"}
_EMPTY = {
    "reporting_year": "NA",
    "scope1": dict(_NA_FIELD),
    "scope2": dict(_NA_FIELD),
    "scope3": dict(_NA_FIELD),
}


@dataclass
class ExtractionResult:
    parsed: dict
    raw_output: str
    prompt: str
    llm: LLMResult


def _build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        blocks.append(f"--- excerpt {i + 1} (score={c.get('score', 0):.3f}) ---\n{c['text']}")
    return "\n\n".join(blocks) if blocks else "(no relevant text found)"


def _coerce_field(obj) -> dict:
    if not isinstance(obj, dict):
        return dict(_NA_FIELD)
    value = obj.get("value", "NA")
    unit = obj.get("unit", "NA")
    return {"value": str(value), "unit": str(unit)}


def _parse_json(text: str) -> dict:
    """Best-effort extraction of the JSON object from the model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return dict(_EMPTY)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(_EMPTY)
    return {
        "reporting_year": str(data.get("reporting_year", "NA")),
        "scope1": _coerce_field(data.get("scope1")),
        "scope2": _coerce_field(data.get("scope2")),
        "scope3": _coerce_field(data.get("scope3")),
    }


def extract(prompt_version: str, chunks: list[dict]) -> ExtractionResult:
    context = _build_context(chunks)
    system, user = build_prompt(prompt_version, context)
    llm = complete(system=system, user=user, max_tokens=512)
    parsed = _parse_json(llm.text)
    full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
    return ExtractionResult(parsed=parsed, raw_output=llm.text, prompt=full_prompt, llm=llm)
