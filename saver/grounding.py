import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundedConstraint:
    semantic_field: str
    grounded_columns: list[dict[str, Any]] = field(default_factory=list)
    grounding_confidence: str = "unknown"
    llm_assisted: bool = False


def _tokens(text: Any) -> set[str]:
    raw = str(text or "").lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_]*|\d+(?:\.\d+)?", raw))
    split = set()
    for token in tokens:
        split.update(part for part in token.split("_") if len(part) > 2)
    return {token for token in tokens | split if len(token) > 2 and token not in {"unknown", "value", "values", "field", "column"}}


def _schema_columns(schema: str) -> list[str]:
    columns: list[str] = []
    current_table = ""
    for line in schema.splitlines():
        table_match = re.search(r"\btable\s+[`\"]?([a-zA-Z_][\w]*)[`\"]?", line, re.I)
        if table_match:
            current_table = table_match.group(1)
        for match in re.finditer(r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)", line):
            columns.append(f"{match.group(1)}.{match.group(2)}")
        if current_table:
            col_match = re.search(r"^\s*[-*]?\s*`?([a-zA-Z_][\w]*)`?\s*[:(]", line)
            if col_match:
                columns.append(f"{current_table}.{col_match.group(1)}")
    return list(dict.fromkeys(columns))


def deterministic_ground_intent(intent_frame: dict[str, Any], schema: str, question: str) -> dict[str, Any]:
    columns = _schema_columns(schema)
    question_tokens = _tokens(question)
    fields: list[str] = []
    target = intent_frame.get("target") if isinstance(intent_frame.get("target"), dict) else {}
    if target.get("description") and target.get("description") != "unknown":
        fields.append(str(target["description"]))
    aggregation = intent_frame.get("aggregation") if isinstance(intent_frame.get("aggregation"), dict) else {}
    if aggregation.get("target") and aggregation.get("target") != "unknown":
        fields.append(str(aggregation["target"]))
    filters = intent_frame.get("filters")
    if not isinstance(filters, list):
        filters = []
    for item in filters:
        if isinstance(item, dict):
            fields.append(str(item.get("semantic_field") or item.get("value") or "unknown"))
    grouping = intent_frame.get("grouping") if isinstance(intent_frame.get("grouping"), dict) else {}
    grouping_by = grouping.get("by")
    if isinstance(grouping_by, list):
        fields.extend(str(item) for item in grouping_by if item != "unknown")
    elif grouping_by not in {None, "", "unknown"}:
        fields.append(str(grouping_by))
    ordering = intent_frame.get("ordering") if isinstance(intent_frame.get("ordering"), dict) else {}
    if ordering.get("by") and ordering.get("by") != "unknown":
        fields.append(str(ordering["by"]))

    grounded = []
    for field in fields:
        field_tokens = _tokens(field)
        scored = []
        for column in columns:
            col_tokens = _tokens(column)
            overlap = len(field_tokens & col_tokens)
            question_overlap = len(question_tokens & col_tokens)
            if overlap or question_overlap:
                confidence = min(0.95, 0.35 + 0.25 * overlap + 0.05 * question_overlap)
                evidence = []
                if overlap:
                    evidence.append("column_name_overlap")
                if question_overlap:
                    evidence.append("question_overlap")
                scored.append({"column": column, "confidence": round(confidence, 3), "evidence": evidence})
        scored = sorted(scored, key=lambda item: item["confidence"], reverse=True)[:5]
        top = scored[0]["confidence"] if scored else 0.0
        confidence_label = "high" if top >= 0.75 else "medium" if top >= 0.5 else "low" if scored else "unknown"
        grounded.append(GroundedConstraint(field, scored, confidence_label).__dict__)

    high_risk_unknown = any(
        [
            intent_frame.get("filters"),
            bool((intent_frame.get("top_k") or {}).get("required")) if isinstance(intent_frame.get("top_k"), dict) else False,
            bool((intent_frame.get("ordering") or {}).get("required")) if isinstance(intent_frame.get("ordering"), dict) else False,
        ]
    ) and any(item.get("grounding_confidence") in {"low", "unknown"} for item in grounded)
    return {
        "constraints": grounded,
        "llm_assisted_grounding_triggered": False,
        "llm_assisted_grounding_recommended": bool(high_risk_unknown),
    }
