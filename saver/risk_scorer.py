import re
from dataclasses import asdict, dataclass, field
from typing import Any

from saver.behavior_signature import BehaviorSignature


RISK_WEIGHTS = {
    "execution_violation": 0.90,
    "target_violation": 0.70,
    "filter_violation": 0.70,
    "aggregation_violation": 0.75,
    "grouping_violation": 0.65,
    "ordering_violation": 0.70,
    "topk_violation": 0.70,
    "answer_shape_violation": 0.60,
    "negation_violation": 0.70,
    "abnormal_result_violation": 0.90,
}


@dataclass
class Violation:
    type: str
    severity: str
    intent_evidence: str
    grounding_evidence: str = ""
    sql_evidence: str = ""


@dataclass
class RiskReport:
    candidate_id: str
    semantic_risk_score: float
    total_score: int
    passed_tests: list[str] = field(default_factory=list)
    failed_tests: list[Any] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    behavior_signature: dict[str, Any] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)

    @property
    def high_severity_failures(self) -> int:
        return sum(1 for item in self.violations if item.severity == "high")


def _confidence_multiplier(confidence: str) -> float:
    return {"high": 1.0, "medium": 0.65, "low": 0.25, "unknown": 0.0}.get(confidence, 0.0)


def _has_aggregation(signature: BehaviorSignature, agg_type: str) -> bool:
    agg = agg_type.upper()
    observed = {item["function"].upper() for item in signature.aggregations}
    if agg == "AVG" and {"SUM", "COUNT"}.issubset(observed) and "/" in signature.raw_sql:
        return True
    return agg in observed


def _has_extrema_ordering(signature: BehaviorSignature, agg_type: str) -> bool:
    expected = {"max": "DESC", "min": "ASC"}.get(agg_type.lower())
    if not expected or not signature.order_by:
        return False
    first = signature.order_by[0]
    if first.get("direction") != expected:
        return False
    return signature.limit == 1 or signature.result_shape.get("rows") == 1


def _count_case_else_zero(sql_l: str) -> bool:
    case_pattern = r"count\s*\(\s*case\s+when[\s\S]*?else\s+0[\s\S]*?end\s*\)"
    iif_pattern = r"count\s*\(\s*iif\s*\([\s\S]*?,[\s\S]*?,\s*0\s*\)\s*\)"
    return bool(re.search(case_pattern, sql_l, re.I) or re.search(iif_pattern, sql_l, re.I))


def _has_rank_projection(signature: BehaviorSignature) -> bool:
    sql_l = signature.raw_sql.lower()
    if re.search(r"\b(rank|dense_rank|row_number)\s*\(", sql_l, re.I):
        return True
    return any("rank" in column.lower() for column in signature.selected_columns)


def _selected_sql_text(signature: BehaviorSignature) -> str:
    return " ".join(signature.selected_columns).lower()


def _has_top_level_distinct(signature: BehaviorSignature) -> bool:
    return bool(re.match(r"^\s*select\s+distinct\b", signature.raw_sql, re.I))


def _split_full_name_allowed(signature: BehaviorSignature, intent_frame: dict[str, Any]) -> bool:
    target = intent_frame.get("target") if isinstance(intent_frame.get("target"), dict) else {}
    answer_shape = intent_frame.get("answer_shape") if isinstance(intent_frame.get("answer_shape"), dict) else {}
    target_text = f"{target.get('description', '')} {answer_shape.get('description', '')}".lower()
    if "full name" not in target_text:
        return False
    if signature.result_shape.get("columns") != 2:
        return False
    selected = _selected_sql_text(signature)
    name_parts = ("forename", "surname", "first_name", "last_name", "fname", "lname", "first name", "last name")
    return any(part in selected for part in name_parts)


def _has_denormalized_slots(signature: BehaviorSignature, base_token: str) -> bool:
    selected = _selected_sql_text(signature)
    return all(f"{base_token}{idx}" in selected for idx in (1, 2, 3))


def _target_specific_violations(intent_frame: dict[str, Any], signature: BehaviorSignature) -> list[Violation]:
    target = intent_frame.get("target") if isinstance(intent_frame.get("target"), dict) else {}
    answer_shape = intent_frame.get("answer_shape") if isinstance(intent_frame.get("answer_shape"), dict) else {}
    top_k = intent_frame.get("top_k") if isinstance(intent_frame.get("top_k"), dict) else {}
    formula_hints = intent_frame.get("formula_hints") if isinstance(intent_frame.get("formula_hints"), dict) else {}
    target_desc = str(target.get("description", "")).lower()
    answer_desc = str(answer_shape.get("description", "")).lower()
    sql_l = signature.raw_sql.lower()
    selected_l = _selected_sql_text(signature)
    tables = set(signature.tables)
    violations: list[Violation] = []

    if "duration" in target_desc and "milliseconds" in sql_l and "duration" not in sql_l:
        violations.append(
            Violation(
                "target_violation",
                "high",
                "Question targets duration.",
                "",
                "SQL projects milliseconds instead of duration.",
            )
        )
    if "category of events" in target_desc and "category" not in selected_l and "type" in selected_l:
        violations.append(
            Violation(
                "target_violation",
                "high",
                "Question targets event category.",
                "",
                "SQL projects type-like field instead of category.",
            )
        )
    if ("email addresses" in target_desc or formula_hints.get("denormalized_email_slots")) and _has_denormalized_slots(signature, "admemail"):
        violations.append(
            Violation(
                "answer_shape_violation",
                "medium",
                "Question asks for a list of email addresses.",
                "",
                "SQL returns denormalized email slot columns instead of flattened addresses.",
            )
        )
    if ("up to three administrator full names" in answer_desc or formula_hints.get("denormalized_admin_name_slots")) and "admfname1" in sql_l:
        has_extra_slots = any(token in sql_l for token in ["admfname2", "admlname2", "admfname3", "admlname3", "union"])
        if not has_extra_slots:
            violations.append(
                Violation(
                    "answer_shape_violation",
                    "medium",
                    "Question expects all administrator full names for the selected school.",
                    "",
                    "SQL only returns the first administrator slot.",
                )
            )
    if (
        formula_hints.get("distinct_list_preferred")
        and not formula_hints.get("preserve_record_multiplicity")
        and answer_shape.get("expected_rows") == "multiple"
        and not top_k.get("required")
    ):
        if not re.search(r"\bselect\s+distinct\b", sql_l, re.I):
            violations.append(
                Violation(
                    "answer_shape_violation",
                    "medium",
                    "Question requests a list-style answer where duplicate rows are likely incorrect.",
                    "",
                    "SQL does not use DISTINCT for a list-style answer.",
                )
            )
    if formula_hints.get("preserve_record_multiplicity") and _has_top_level_distinct(signature):
        violations.append(
            Violation(
                "answer_shape_violation",
                "medium",
                "Question asks for a record-level list where duplicate records may be semantically meaningful.",
                "",
                "SQL applies top-level DISTINCT and may collapse record multiplicity.",
            )
        )
    if formula_hints.get("set_level_translation_scope"):
        if "cards" in tables and "foreign_data" in tables and "set_translations" not in tables:
            violations.append(
                Violation(
                    "target_violation",
                    "high",
                    "Question refers to set-level translation scope.",
                    "",
                    "SQL operates on card-level translation tables instead of set-level translation tables.",
                )
            )
    if formula_hints.get("fastest_lap_time_required") and "fastestlaptime" not in sql_l:
        violations.append(
            Violation(
                "filter_violation",
                "medium",
                "Fastest lap speed should be restricted to rows with a recorded fastest lap time.",
                "",
                "SQL has no fastestLapTime presence constraint.",
            )
        )
    if "bond_type" in target_desc and signature.result_shape.get("unique_rows", signature.result_shape.get("rows", 0)) > 1:
        violations.append(
            Violation(
                "answer_shape_violation",
                "medium",
                "Question expects a single bond type value.",
                "",
                f"SQL returns {signature.result_shape.get('rows')} rows for bond type.",
            )
        )
    if formula_hints.get("temporal_label_output") and "cast(" in selected_l and ("substr(" in selected_l or "strftime(" in selected_l):
        violations.append(
            Violation(
                "target_violation",
                "medium",
                "Question asks for a year label output.",
                "",
                "SQL casts the year extraction to a numeric type instead of returning the year label directly.",
            )
        )
    if formula_hints.get("rank_output_required") and not _has_rank_projection(signature):
        violations.append(
            Violation(
                "answer_shape_violation",
                "medium",
                "Question explicitly asks for a ranking output.",
                "",
                "SQL orders rows but does not project an explicit rank or window-function rank column.",
            )
        )
    return violations


def _semantic_tokens(text: Any) -> set[str]:
    raw = str(text or "").lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_]*|\d+(?:\.\d+)?", raw))
    split_tokens: set[str] = set()
    for token in tokens:
        split_tokens.update(part for part in token.split("_") if part)
    stopwords = {"and", "the", "for", "from", "with", "unknown", "value", "values", "date", "year"}
    return {token for token in tokens | split_tokens if len(token) > 1 and token not in stopwords}


GENERIC_FIELD_TOKENS = {"id", "date", "year", "name", "type", "value", "count", "number", "code"}


def _filter_values(filter_item: dict[str, Any]) -> list[str]:
    value = filter_item.get("value")
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    output: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if not text or text.lower() in {"unknown", "none", "null"}:
            continue
        output.append(text)
    return output


def _normalize_operator(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _requires_not_null(filter_item: dict[str, Any]) -> bool:
    operator = _normalize_operator(filter_item.get("operator"))
    value = filter_item.get("value")
    return operator in {"!=", "<>", "is not", "is not null"} and value is None


def _requires_is_null(filter_item: dict[str, Any]) -> bool:
    operator = _normalize_operator(filter_item.get("operator"))
    value = filter_item.get("value")
    return operator in {"=", "is", "is null"} and value is None


def _sql_contains_filter_value(sql_l: str, value: str) -> bool:
    value_l = value.lower().strip().strip("'\"")
    if not value_l:
        return True
    if value_l in sql_l:
        return True
    tokens = _semantic_tokens(value_l)
    if not tokens:
        return True
    return all(token in sql_l for token in tokens)


def _field_covered(sql_l: str, field: Any) -> bool | None:
    tokens = _semantic_tokens(field)
    if not tokens:
        return None
    specific_tokens = [token for token in tokens if token not in GENERIC_FIELD_TOKENS]
    field_text = str(field or "").lower()
    table_prefix = field_text.split(".", 1)[0] if "." in field_text else ""
    if specific_tokens:
        hits = sum(1 for token in specific_tokens if token in sql_l)
        if table_prefix and table_prefix in sql_l:
            return hits >= 1
        return hits >= max(1, min(2, len(specific_tokens)))
    return any(token in sql_l for token in tokens)


def _field_has_null_check(sql_l: str, field: Any, positive: bool) -> bool:
    tokens = list(_semantic_tokens(field))
    patterns = [r"\bis\s+not\s+null\b", r"!=\s*null", r"<>?\s*null"] if positive else [r"\bis\s+null\b", r"=\s*null"]
    for token in tokens or [""]:
        for pattern in patterns:
            if token:
                if re.search(rf"{re.escape(token)}[\s\S]{{0,80}}{pattern}", sql_l, re.I):
                    return True
            elif re.search(pattern, sql_l, re.I):
                return True
    return False


def _required_filter_violations(filters: list[dict[str, Any]], signature: BehaviorSignature) -> list[Violation]:
    sql_l = signature.raw_sql.lower()
    violations: list[Violation] = []
    for item in filters:
        if not isinstance(item, dict):
            continue
        confidence = str(item.get("confidence", "unknown")).lower()
        if confidence == "low":
            continue
        semantic_field = item.get("semantic_field", "unknown")
        if _requires_not_null(item):
            if not _field_has_null_check(sql_l, semantic_field, positive=True):
                severity = "high" if confidence == "high" else "medium"
                violations.append(
                    Violation(
                        "filter_violation",
                        severity,
                        f"Question requires non-null filter on {semantic_field}.",
                        "",
                        "SQL does not preserve the required IS NOT NULL condition.",
                    )
                )
            continue
        if _requires_is_null(item):
            if not _field_has_null_check(sql_l, semantic_field, positive=False):
                severity = "high" if confidence == "high" else "medium"
                violations.append(
                    Violation(
                        "filter_violation",
                        severity,
                        f"Question requires null filter on {semantic_field}.",
                        "",
                        "SQL does not preserve the required IS NULL condition.",
                    )
                )
            continue
        values = _filter_values(item)
        missing_values = [value for value in values if not _sql_contains_filter_value(sql_l, value)]
        field_ok = _field_covered(sql_l, semantic_field)
        if missing_values:
            severity = "high" if confidence == "high" else "medium"
            violations.append(
                Violation(
                    "filter_violation",
                    severity,
                    f"Question requires filter value(s): {', '.join(values)}.",
                    "",
                    f"Missing value(s) in SQL: {', '.join(missing_values)}.",
                )
            )
            continue
        if field_ok is False and values:
            violations.append(
                Violation(
                    "filter_violation",
                    "medium",
                    f"Question grounds filter to semantic field {semantic_field}.",
                    "",
                    "Filter value appears, but the expected field is not lexically present.",
                )
            )
    if filters and not signature.where_conditions:
        has_all_values = all(
            _sql_contains_filter_value(sql_l, value)
            for item in filters
            if isinstance(item, dict)
            for value in _filter_values(item)
        )
        if not has_all_values:
            violations.append(
                Violation(
                    "filter_violation",
                    "high",
                    "Question includes explicit filtering.",
                    "",
                    "SQL has no WHERE clause and does not preserve all required filter values.",
                )
            )
    return violations


def _has_any_limit_clause(sql_l: str) -> bool:
    return bool(re.search(r"\blimit\s+\d+\b", sql_l))


def _uses_scalar_extrema_subquery_without_limit(sql_l: str, agg_type: str) -> bool:
    agg = agg_type.lower()
    if agg not in {"max", "min"}:
        return False
    pattern = rf"(=|in)\s*\(\s*select\s+{agg}\s*\("
    return bool(re.search(pattern, sql_l, re.I | re.S)) and not _has_any_limit_clause(sql_l)


def _has_order_direction(sql_l: str, expected: str) -> bool:
    return bool(re.search(rf"\border\s+by\b.*\b{expected.lower()}\b", sql_l, re.I | re.S))


def _negation_present(sql_l: str) -> bool:
    return any(
        pattern in sql_l
        for pattern in [" not ", " not in ", " not exists ", " except ", " is null", " is not null"]
    ) or bool(re.search(r"\bleft\s+join\b.*\bis\s+null\b", sql_l, re.I | re.S))


def _add(violations: list[Violation], risk_type: str, severity: str, intent: str, sql: str, grounding: str = "") -> None:
    violations.append(Violation(risk_type, severity, intent, grounding, sql))


def score_candidate_risk(
    candidate_id: str,
    intent_frame: dict[str, Any],
    grounding: dict[str, Any],
    signature: BehaviorSignature,
) -> RiskReport:
    violations: list[Violation] = []
    passed: list[str] = []
    if signature.execution_status != "success":
        _add(violations, "execution_violation", "high", "SQL should execute successfully.", "Execution failed.")
    else:
        passed.append("execution")

    if signature.result_shape.get("empty_result"):
        _add(violations, "abnormal_result_violation", "medium", "Question expects an answer.", "Execution returned an empty result.")
    elif signature.execution_status == "success":
        passed.append("non_empty_result")

    aggregation = intent_frame.get("aggregation") if isinstance(intent_frame.get("aggregation"), dict) else {}
    formula_hints = intent_frame.get("formula_hints") if isinstance(intent_frame.get("formula_hints"), dict) else {}
    agg_type = str(aggregation.get("type", "none")).lower()
    sql_l = signature.raw_sql.lower()
    if agg_type not in {"none", "unknown"}:
        if _has_aggregation(signature, agg_type) or _has_extrema_ordering(signature, agg_type):
            passed.append("aggregation_alignment")
        else:
            _add(violations, "aggregation_violation", "high", f"Question requires {agg_type} aggregation.", "No matching SQL aggregation found.")
    if agg_type == "avg" and formula_hints.get("explicit_average_via_sum_count"):
        observed = {item["function"].upper() for item in signature.aggregations}
        if "AVG" in observed and not ({"SUM", "COUNT"}.issubset(observed) and "/" in signature.raw_sql):
            _add(
                violations,
                "aggregation_violation",
                "medium",
                "Evidence specifies average via SUM/COUNT semantics.",
                "SQL uses AVG() directly instead of the evidenced SUM/COUNT form.",
            )
    if formula_hints.get("ratio_required"):
        observed = {item["function"].upper() for item in signature.aggregations}
        if "/" not in signature.raw_sql:
            _add(
                violations,
                "aggregation_violation",
                "high",
                "Question requires a ratio or percentage computation.",
                "SQL has no explicit division operator for ratio semantics.",
            )
        elif observed == {"COUNT"} and "case when" not in sql_l and "iif(" not in sql_l and "sum(" not in sql_l:
            _add(
                violations,
                "aggregation_violation",
                "medium",
                "Ratio questions often require a conditional numerator and denominator.",
                "SQL uses bare COUNT-based aggregation without explicit conditional ratio construction.",
            )
    if agg_type in {"count", "sum", "avg"} and _count_case_else_zero(signature.raw_sql.lower()):
        _add(
            violations,
            "aggregation_violation",
            "high",
            "Conditional counting/aggregation should preserve only qualifying rows.",
            "COUNT(CASE/IIF ... ELSE 0) likely counts non-qualifying rows as well.",
        )

    grouping = intent_frame.get("grouping") if isinstance(intent_frame.get("grouping"), dict) else {}
    if grouping.get("required"):
        if signature.group_by:
            passed.append("grouping_alignment")
        else:
            _add(violations, "grouping_violation", "high", "Question requires per-group reasoning.", "SQL has no GROUP BY.")

    ordering = intent_frame.get("ordering") if isinstance(intent_frame.get("ordering"), dict) else {}
    inferred_extrema = agg_type in {"max", "min"} and not _has_aggregation(signature, agg_type)
    ordering_required = bool(ordering.get("required")) or inferred_extrema
    if ordering_required:
        expected = str(ordering.get("direction", "unknown")).upper()
        if inferred_extrema and expected not in {"ASC", "DESC"}:
            expected = "DESC" if agg_type == "max" else "ASC"
        if not signature.order_by:
            _add(violations, "ordering_violation", "high", "Question requires ordering.", "SQL has no ORDER BY.")
        elif expected in {"ASC", "DESC"} and not any(item.get("direction") == expected for item in signature.order_by) and not _has_order_direction(sql_l, expected):
            observed = ", ".join(item.get("direction", "unknown") for item in signature.order_by)
            _add(violations, "ordering_violation", "high", f"Question requires {expected} ordering.", f"Observed direction: {observed}.")
        else:
            passed.append("ordering_alignment")

    top_k = intent_frame.get("top_k") if isinstance(intent_frame.get("top_k"), dict) else {}
    topk_required = bool(top_k.get("required")) or inferred_extrema
    topk_value = top_k.get("k")
    if inferred_extrema and topk_value is None:
        topk_value = 1
    if topk_required and topk_value is not None:
        if signature.limit == int(topk_value) or (int(topk_value) == 1 and _has_any_limit_clause(sql_l)):
            passed.append("topk_alignment")
        else:
            _add(violations, "topk_violation", "high", f"Question requires top-{topk_value}.", f"Observed LIMIT: {signature.limit}.")

    answer_shape = intent_frame.get("answer_shape") if isinstance(intent_frame.get("answer_shape"), dict) else {}
    if (
        agg_type in {"max", "min"}
        and answer_shape.get("expected_rows") == "single"
        and signature.result_shape.get("unique_rows", signature.result_shape.get("rows", 0)) > 1
        and signature.limit is None
    ):
        _add(
            violations,
            "topk_violation",
            "high",
            "Question asks for the entity associated with an extrema value.",
            f"SQL returns {signature.result_shape.get('rows')} rows without a top-1 constraint.",
        )
    if (
        agg_type in {"max", "min"}
        and answer_shape.get("expected_rows") == "single"
        and _uses_scalar_extrema_subquery_without_limit(sql_l, agg_type)
    ):
        _add(
            violations,
            "topk_violation",
            "medium",
            "Question asks for a single extrema answer.",
            "SQL relies on a MAX/MIN scalar subquery without an explicit top-1 tie-breaking constraint.",
        )
    if (
        answer_shape.get("expected_columns") == "single"
        and signature.result_shape.get("columns") not in {0, 1}
        and not _split_full_name_allowed(signature, intent_frame)
    ):
        _add(violations, "answer_shape_violation", "high", "Question expects a single answer column.", f"Observed {signature.result_shape.get('columns')} columns.")
    else:
        passed.append("answer_shape_columns")
    if (
        answer_shape.get("expected_rows") == "single"
        and signature.result_shape.get("unique_rows", signature.result_shape.get("rows", 0)) > 1
    ):
        _add(violations, "answer_shape_violation", "medium", "Question expects a single result row.", f"Observed {signature.result_shape.get('rows')} rows.")

    negation = intent_frame.get("negation") if isinstance(intent_frame.get("negation"), dict) else {}
    if negation.get("required"):
        if _negation_present(sql_l):
            passed.append("negation_alignment")
        else:
            _add(violations, "negation_violation", "high", "Question contains negation.", "SQL has no explicit negation operator.")

    filters = intent_frame.get("filters") if isinstance(intent_frame.get("filters"), list) else []
    if filters:
        filter_violations = _required_filter_violations(filters, signature)
        if filter_violations:
            violations.extend(filter_violations)
        else:
            passed.append("filter_presence")

    violations.extend(_target_specific_violations(intent_frame, signature))

    raw_risk = 0.0
    for violation in violations:
        raw_risk += RISK_WEIGHTS.get(violation.type, 0.5)
    risk_score = min(1.0, raw_risk)
    total_score = max(0, int(round(10 * (1.0 - risk_score))) - sum(1 for item in violations if item.severity == "high"))
    return RiskReport(
        candidate_id=candidate_id,
        semantic_risk_score=round(risk_score, 4),
        total_score=total_score,
        passed_tests=passed,
        risk_flags=[item.type for item in violations],
        violations=violations,
        behavior_signature=asdict(signature),
        grounding=grounding,
    )
