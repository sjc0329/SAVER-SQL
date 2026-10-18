import json
import re
from typing import Any

from evaluation.cost_tracker import Usage
from llm.api_client import LLMClient
from llm.prompt_templates import JSON_REPAIR_SYSTEM, JSON_REPAIR_USER, SAVER_INTENT_SYSTEM, SAVER_INTENT_USER
from llm.response_parser import extract_json


def _confidence(value: str = "high") -> str:
    return value if value in {"high", "medium", "low", "unknown"} else "unknown"


def _formula_hints(question: str, evidence: str = "") -> dict[str, Any]:
    text = f"{question} {evidence}".lower()
    evidence_l = evidence.lower()
    return {
        "explicit_average_via_sum_count": "sum(" in evidence_l and "count(" in evidence_l,
        "ratio_required": any(term in text for term in ["ratio", "percentage", "divide("]),
        "denormalized_email_slots": "email addresses" in text and any(term in text for term in ["email1", "email2", "email3", "admemail"]),
        "denormalized_admin_name_slots": "administrator" in text and any(term in text for term in ["at most 3 administrators", "up to three"]),
        "distinct_list_preferred": any(term in text for term in ["list", "what are", "what kind of", "names of", "elements associated"]) and not any(term in text for term in ["top ", "highest", "lowest", "largest", "smallest"]),
        "preserve_record_multiplicity": _preserve_record_multiplicity(question),
        "set_level_translation_scope": "set of cards" in text or ("translation" in text and "online" in text),
        "fastest_lap_time_required": "fastest lap speed" in text,
        "single_slot_output_preferred": _single_slot_output_preferred(question),
        "rank_output_required": _rank_output_required(question),
        "temporal_label_output": bool(re.match(r"^(which|what)\s+year\b", question.lower().strip())),
    }


def _single_slot_output_preferred(question: str) -> bool:
    text = question.lower().strip()
    explicit_multi_patterns = [
        r"\bshow(?:ing)?\b",
        r"\binclude\b",
        r"\bincluding\b",
        r"\balong with\b",
        r"\bidentify\b[\s\S]{0,80}\band\b",
        r"\band\s+(their|the|its)\s+(link|links|cost|costs|date|dates|type|types|id|ids|number|numbers|name|names|score|scores|event|events)\b",
        r"\bforename\b[\s\S]{0,30}\bsurname\b",
        r"\bfirst name\b[\s\S]{0,30}\blast name\b",
        r"\bfull name\b[\s\S]{0,40}\bcost\b",
        r"\bname\b[\s\S]{0,30}\btype\b",
        r"\blist out\b",
    ]
    if any(re.search(pattern, text, re.I | re.S) for pattern in explicit_multi_patterns):
        return False
    single_slot_starts = [
        "which ",
        "what is ",
        "what are ",
        "name ",
        "list ",
        "list the ",
        "please list ",
        "give ",
        "write ",
    ]
    if any(text.startswith(prefix) for prefix in single_slot_starts):
        return True
    return bool(re.search(r"\bwhich\s+top\s+\d+\b", text))


def _rank_output_required(question: str) -> bool:
    text = question.lower().strip()
    if re.match(r"^(rank|ranking)\b", text):
        return True
    return bool(re.search(r"\bshow(?:ing)?\s+(their|the)\s+.*\brank\b", text))


def _preserve_record_multiplicity(question: str) -> bool:
    text = question.lower().strip()
    if not (
        text.startswith("list ")
        or text.startswith("list the ")
        or text.startswith("please list ")
        or text.startswith("please list all ")
    ):
        return False
    if any(term in text for term in ["top ", "highest", "lowest", "largest", "smallest", "most ", "least "]):
        return False
    return True


def heuristic_intent_frame(question: str, evidence: str = "") -> dict[str, Any]:
    text = f"{question} {evidence}".lower()
    agg_type = "none"
    agg_target = "unknown"
    if any(term in text for term in ["how many", "number of", "count"]):
        agg_type = "count"
    elif any(term in text for term in ["average", "avg", "mean"]):
        agg_type = "avg"
    elif any(term in text for term in ["sum", "total"]):
        agg_type = "sum"
    elif any(term in text for term in ["maximum", "max "]):
        agg_type = "max"
    elif any(term in text for term in ["minimum", "min "]):
        agg_type = "min"

    top_required = any(term in text for term in ["highest", "lowest", "top", "most", "least", "largest", "smallest"])
    direction = "unknown"
    if any(term in text for term in ["highest", "top", "most", "largest", "maximum"]):
        direction = "desc"
    if any(term in text for term in ["lowest", "least", "smallest", "minimum"]):
        direction = "asc"
    top_k = 1 if top_required else None
    match = re.search(r"\btop\s+(\d+)\b", text)
    if match:
        top_k = int(match.group(1))

    grouping_required = any(term in text for term in [" per ", " each ", " for each ", " by "])
    negation_required = any(term in text for term in [" without ", " never ", " no ", " not "])
    filters = []
    for op, pattern in [
        ("after", r"\b(after|since)\s+(\d{4}|\d+(?:\.\d+)?)"),
        ("before", r"\b(before|prior to)\s+(\d{4}|\d+(?:\.\d+)?)"),
        (">", r"\bmore than\s+(\d+(?:\.\d+)?)"),
        ("<", r"\bless than\s+(\d+(?:\.\d+)?)"),
    ]:
        m = re.search(pattern, text)
        if m:
            value = m.group(2) if len(m.groups()) > 1 else m.group(1)
            filters.append({"semantic_field": "unknown", "operator": op, "value": value, "confidence": "medium"})

    expected_rows = "single" if top_k == 1 or agg_type != "none" else "multiple"
    return {
        "target": {"semantic_type": "unknown", "description": "unknown", "confidence": "unknown"},
        "filters": filters,
        "aggregation": {"type": agg_type, "target": agg_target, "confidence": "high" if agg_type != "none" else "unknown"},
        "grouping": {"required": grouping_required, "by": ["unknown"] if grouping_required else [], "confidence": "medium" if grouping_required else "unknown"},
        "ordering": {"required": top_required, "by": "unknown", "direction": direction, "confidence": "high" if top_required else "unknown"},
        "top_k": {"required": top_required, "k": top_k, "confidence": "high" if top_required else "unknown"},
        "answer_shape": {"expected_rows": expected_rows, "expected_columns": "single", "description": "unknown", "confidence": "medium"},
        "negation": {"required": negation_required, "confidence": "medium" if negation_required else "unknown"},
        "formula_hints": _formula_hints(question, evidence),
        "uncertainty": ["heuristic intent frame used; no LLM call"],
    }


def normalize_intent_frame(frame: dict[str, Any], question: str, evidence: str = "") -> dict[str, Any]:
    baseline = heuristic_intent_frame(question, evidence)
    normalized = dict(baseline)
    for key in ("target", "aggregation", "grouping", "ordering", "top_k", "answer_shape", "negation"):
        if isinstance(frame.get(key), dict):
            merged = dict(baseline[key])
            merged.update(frame[key])
            if "confidence" in merged:
                merged["confidence"] = _confidence(str(merged["confidence"]).lower())
            normalized[key] = merged
    if isinstance(frame.get("filters"), list):
        normalized["filters"] = [item for item in frame["filters"] if isinstance(item, dict)]
    if isinstance(frame.get("formula_hints"), dict):
        merged_hints = dict(baseline.get("formula_hints") or {})
        for key, value in frame["formula_hints"].items():
            if key not in merged_hints:
                merged_hints[key] = value
                continue
            if isinstance(merged_hints[key], bool) and isinstance(value, bool):
                merged_hints[key] = merged_hints[key] or value
            elif merged_hints[key] in {None, "", "unknown"}:
                merged_hints[key] = value
        normalized["formula_hints"] = merged_hints
    uncertainty = []
    if isinstance(frame.get("uncertainty"), list):
        uncertainty.extend(frame["uncertainty"])
    missing = [key for key in ("target", "filters", "aggregation", "grouping", "ordering", "top_k", "answer_shape") if key not in frame]
    if missing:
        uncertainty.append(f"intent frame missing fields filled by heuristic defaults: {', '.join(missing)}")
    text = question.lower()
    answer_shape = normalized.get("answer_shape", {})
    aggregation = normalized.get("aggregation", {})
    top_k = normalized.get("top_k", {})
    if (
        answer_shape.get("expected_rows") == "single"
        and aggregation.get("type") in {"none", "unknown"}
        and not top_k.get("required")
        and ("please list" in text or text.startswith("list ") or text.startswith("list the "))
    ):
        answer_shape["expected_rows"] = "multiple"
        uncertainty.append("answer_shape.expected_rows adjusted to multiple for list-style question.")
    formula_hints = normalized.get("formula_hints", {})
    if (
        formula_hints.get("single_slot_output_preferred")
        and answer_shape.get("expected_columns") == "multiple"
    ):
        answer_shape["expected_columns"] = "single"
        uncertainty.append("answer_shape.expected_columns adjusted to single for single-slot output question.")
    normalized["uncertainty"] = uncertainty
    return normalized


class IntentFrameExtractor:
    def __init__(self, client: LLMClient | None = None, temperature: float = 0.0, max_tokens: int | None = None, repair_max_tokens: int | None = None):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repair_max_tokens = repair_max_tokens

    def extract(self, schema: str, question: str, evidence: str, dry_run: bool = False) -> tuple[dict[str, Any], Usage]:
        if dry_run or self.client is None:
            return heuristic_intent_frame(question, evidence), Usage()
        response = self.client.chat(
            [
                {"role": "system", "content": SAVER_INTENT_SYSTEM},
                {"role": "user", "content": SAVER_INTENT_USER.format(schema=schema, question=question, evidence=evidence)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return normalize_intent_frame(extract_json(response.text), question, evidence), response.usage
        except json.JSONDecodeError:
            usage = response.usage
            try:
                repaired = self.client.chat(
                    [
                        {"role": "system", "content": JSON_REPAIR_SYSTEM},
                        {"role": "user", "content": JSON_REPAIR_USER.format(raw_response=response.text[:4000])},
                    ],
                    temperature=0.0,
                    max_tokens=self.repair_max_tokens,
                    response_format={"type": "json_object"},
                )
                usage.add(repaired.usage)
                frame = normalize_intent_frame(extract_json(repaired.text), question, evidence)
                frame.setdefault("uncertainty", []).append("LLM intent frame JSON repair retry used")
                return frame, usage
            except Exception:
                frame = heuristic_intent_frame(question, evidence)
                frame["uncertainty"].append("LLM intent frame JSON parse failed; heuristic fallback used")
                frame["raw_intent_response_preview"] = response.text[:500]
                return frame, usage
