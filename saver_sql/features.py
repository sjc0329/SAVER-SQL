import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from saver.risk_scorer import RISK_WEIGHTS


FEATURE_PREFIX = "feature_"

DIAGNOSTIC_REPORT_COUNT_FEATURES = frozenset(
    {
        f"{FEATURE_PREFIX}passed_test_count",
        f"{FEATURE_PREFIX}failed_test_count",
    }
)
RESERVED_COLUMNS = {
    "question_id",
    "db_id",
    "difficulty",
    "source_run",
    "candidate_id",
    "generation_mode",
    "label",
    "oracle_correct",
    "execution_ok",
    "selected_by_majority_vote",
    "selected_by_execution_guided",
    "selected_by_saver",
    "selected_by_llm_judge",
    "selected_by_random",
}
KNOWN_FLAG_NAMES = set(RISK_WEIGHTS) | {"duplicate_list_values"}


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").lower()).strip("_") or "unknown"


def _candidate_order(candidate_id: str) -> int:
    digits = "".join(ch for ch in str(candidate_id) if ch.isdigit())
    return int(digits) if digits else 999


def _bool(value: Any) -> int:
    return 1 if bool(value) else 0


def _get_report(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate.get("saver_report") or candidate.get("crest_report") or {}


def _get_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    report = _get_report(candidate)
    return report.get("behavior_signature") or {}


def _cluster_sizes(item: dict[str, Any]) -> tuple[dict[str, int], int]:
    counter: Counter[str] = Counter()
    for candidate in item.get("candidates", []):
        execution = candidate.get("execution") or {}
        if execution.get("status") != "success":
            continue
        signature = str(execution.get("signature") or "")
        if not signature:
            continue
        counter[signature] += 1
    max_size = max(counter.values(), default=0)
    return dict(counter), max_size


def _selected_candidate_ids(item: dict[str, Any]) -> dict[str, str]:
    results = item.get("selector_results") or {}
    selected: dict[str, str] = {}
    for selector_name, selector_result in results.items():
        if isinstance(selector_result, dict):
            selected[selector_name] = str(selector_result.get("selected_candidate_id") or "")
    return selected


def _grounding_feature_counts(grounding: dict[str, Any]) -> dict[str, int]:
    constraints = grounding.get("constraints") or []
    counts = {
        "constraint_count": len(constraints),
        "grounded_constraint_count": 0,
        "high_conf_count": 0,
        "medium_conf_count": 0,
        "low_conf_count": 0,
        "unknown_conf_count": 0,
        "grounded_column_total": 0,
    }
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        grounded_columns = constraint.get("grounded_columns") or []
        if grounded_columns:
            counts["grounded_constraint_count"] += 1
            counts["grounded_column_total"] += len(grounded_columns)
        confidence = str(constraint.get("grounding_confidence", "unknown")).lower()
        if confidence == "high":
            counts["high_conf_count"] += 1
        elif confidence == "medium":
            counts["medium_conf_count"] += 1
        elif confidence == "low":
            counts["low_conf_count"] += 1
        else:
            counts["unknown_conf_count"] += 1
    return counts


def _sql_text_features(sql: str) -> dict[str, Any]:
    sql_l = str(sql or "").lower()
    return {
        "has_distinct": _bool(re.match(r"^\s*select\s+distinct\b", sql_l)),
        "has_subquery": _bool(re.search(r"\(\s*select\b", sql_l) or re.match(r"^\s*with\b", sql_l)),
        "has_case_when": _bool("case when" in sql_l or "iif(" in sql_l),
        "has_count_distinct": _bool(re.search(r"count\s*\(\s*distinct\b", sql_l)),
        "agg_has_count": _bool(re.search(r"\bcount\s*\(", sql_l)),
        "agg_has_sum": _bool(re.search(r"\bsum\s*\(", sql_l)),
        "agg_has_avg": _bool(re.search(r"\bavg\s*\(", sql_l)),
        "agg_has_max": _bool(re.search(r"\bmax\s*\(", sql_l)),
        "agg_has_min": _bool(re.search(r"\bmin\s*\(", sql_l)),
        "sql_length": len(sql_l),
    }


def _behavior_features(signature: dict[str, Any]) -> dict[str, Any]:
    result_shape = signature.get("result_shape") or {}
    tables = signature.get("tables") or []
    return {
        "parse_ok": _bool(signature.get("parse_status") == "success"),
        "selected_column_count": len(signature.get("selected_columns") or []),
        "table_count": len(tables),
        "join_count": max(0, len(tables) - 1),
        "where_count": len(signature.get("where_conditions") or []),
        "aggregation_count": len(signature.get("aggregations") or []),
        "group_by_count": len(signature.get("group_by") or []),
        "order_by_count": len(signature.get("order_by") or []),
        "has_limit": _bool(signature.get("limit") is not None),
        "limit_value": int(signature.get("limit") or 0),
        "result_rows": int(result_shape.get("rows") or 0),
        "result_columns": int(result_shape.get("columns") or 0),
        "result_unique_rows": int(result_shape.get("unique_rows") or 0),
        "empty_result": _bool(result_shape.get("empty_result")),
    }


def _intent_features(intent_frame: dict[str, Any]) -> dict[str, Any]:
    aggregation = intent_frame.get("aggregation") or {}
    ordering = intent_frame.get("ordering") or {}
    top_k = intent_frame.get("top_k") or {}
    answer_shape = intent_frame.get("answer_shape") or {}
    grouping = intent_frame.get("grouping") or {}
    negation = intent_frame.get("negation") or {}
    formula_hints = intent_frame.get("formula_hints") or {}
    features: dict[str, Any] = {
        "intent_agg_type": str(aggregation.get("type") or "unknown").lower(),
        "intent_agg_confidence": str(aggregation.get("confidence") or "unknown").lower(),
        "intent_topk_required": _bool(top_k.get("required")),
        "intent_topk_k": int(top_k.get("k") or 0),
        "intent_grouping_required": _bool(grouping.get("required")),
        "intent_grouping_confidence": str(grouping.get("confidence") or "unknown").lower(),
        "intent_ordering_required": _bool(ordering.get("required")),
        "intent_ordering_direction": str(ordering.get("direction") or "unknown").lower(),
        "intent_negation_required": _bool(negation.get("required")),
        "intent_expected_rows": str(answer_shape.get("expected_rows") or "unknown").lower(),
        "intent_expected_columns": str(answer_shape.get("expected_columns") or "unknown").lower(),
        "intent_answer_shape_confidence": str(answer_shape.get("confidence") or "unknown").lower(),
        "intent_filter_count": len(intent_frame.get("filters") or []),
    }
    for key, value in formula_hints.items():
        if isinstance(value, bool):
            features[f"hint_{_sanitize_name(key)}"] = _bool(value)
    return features


def _report_features(report: dict[str, Any]) -> dict[str, Any]:
    features = {
        "risk_score": float(report.get("semantic_risk_score") or 0.0),
        "risk_total_score": int(report.get("total_score") or 0),
        "high_severity_failures": int(report.get("high_severity_failures") or 0),
        "risk_flag_count": len(report.get("risk_flags") or []),
        "violation_count": len(report.get("violations") or []),
        "passed_test_count": len(report.get("passed_tests") or []),
        "failed_test_count": len(report.get("failed_tests") or []),
    }
    for flag in KNOWN_FLAG_NAMES | {_sanitize_name(flag) for flag in report.get("risk_flags") or []}:
        flag_name = str(flag)
        features[f"flag__{_sanitize_name(flag_name)}"] = 0
    for flag in report.get("risk_flags") or []:
        features[f"flag__{_sanitize_name(flag)}"] = 1
    violation_counter = Counter(_sanitize_name(item.get("type")) for item in report.get("violations") or [] if isinstance(item, dict))
    for violation_type, count in violation_counter.items():
        features[f"violation_type__{violation_type}"] = count
    return features


def _execution_features(candidate: dict[str, Any]) -> dict[str, Any]:
    execution = candidate.get("execution") or {}
    elapsed_ms = execution.get("elapsed_ms")
    return {
        "execution_ok": _bool(execution.get("status") == "success"),
        "execution_row_count": int(execution.get("row_count") or 0),
        "execution_column_count": int(execution.get("column_count") or 0),
        "execution_elapsed_ms": float(elapsed_ms or 0.0),
        "execution_has_signature": _bool(bool(execution.get("signature"))),
    }


def _normalize_feature_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return 0.0
        return value
    return str(value)


def feature_group_for_name(feature_name: str) -> str:
    if feature_name.startswith(f"{FEATURE_PREFIX}generation_mode") or feature_name.startswith(f"{FEATURE_PREFIX}candidate_order"):
        return "mode"
    if feature_name.startswith(f"{FEATURE_PREFIX}execution_") or feature_name.startswith(f"{FEATURE_PREFIX}cluster_"):
        return "execution"
    if feature_name in DIAGNOSTIC_REPORT_COUNT_FEATURES or feature_name.startswith(f"{FEATURE_PREFIX}risk_") or feature_name.startswith(f"{FEATURE_PREFIX}high_severity_") or feature_name.startswith(f"{FEATURE_PREFIX}flag__") or feature_name.startswith(f"{FEATURE_PREFIX}violation_"):
        return "risk"
    if feature_name.startswith(f"{FEATURE_PREFIX}grounding_"):
        return "grounding"
    if feature_name.startswith(f"{FEATURE_PREFIX}intent_") or feature_name.startswith(f"{FEATURE_PREFIX}hint_"):
        return "intent"
    return "structure"


def extract_candidate_feature_rows(item: dict[str, Any], source_run: str = "") -> list[dict[str, Any]]:
    candidates = item.get("candidates") or []
    cluster_sizes, max_cluster_size = _cluster_sizes(item)
    pool_size = max(1, len(candidates))
    selected_ids = _selected_candidate_ids(item)
    intent_frame = item.get("intent_frame") or item.get("contract") or {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        report = _get_report(candidate)
        signature = _get_signature(candidate)
        grounding = report.get("grounding") or item.get("grounding") or {}
        execution = candidate.get("execution") or {}
        result_signature = str(execution.get("signature") or "")
        cluster_size = cluster_sizes.get(result_signature, 0) if execution.get("status") == "success" else 0

        feature_values: dict[str, Any] = {
            "generation_mode": str(candidate.get("generation_mode") or "unknown").lower(),
            "candidate_order": _candidate_order(str(candidate.get("candidate_id") or "")),
            "cluster_size": cluster_size,
            "cluster_ratio": cluster_size / pool_size,
            "cluster_is_majority": _bool(cluster_size > 0 and cluster_size == max_cluster_size),
            "cluster_max_size": max_cluster_size,
            "cluster_signature_seen": _bool(bool(result_signature)),
        }
        feature_values.update(_execution_features(candidate))
        feature_values.update(_behavior_features(signature))
        feature_values.update(_sql_text_features(candidate.get("sql") or signature.get("raw_sql") or ""))
        feature_values.update(_report_features(report))
        feature_values.update(_intent_features(intent_frame))
        grounding_counts = _grounding_feature_counts(grounding)
        feature_values.update(
            {
                "grounding_constraint_count": grounding_counts["constraint_count"],
                "grounding_grounded_constraint_count": grounding_counts["grounded_constraint_count"],
                "grounding_grounded_column_total": grounding_counts["grounded_column_total"],
                "grounding_high_conf_count": grounding_counts["high_conf_count"],
                "grounding_medium_conf_count": grounding_counts["medium_conf_count"],
                "grounding_low_conf_count": grounding_counts["low_conf_count"],
                "grounding_unknown_conf_count": grounding_counts["unknown_conf_count"],
                "grounding_llm_triggered": _bool(grounding.get("llm_assisted_grounding_triggered")),
                "grounding_llm_recommended": _bool(grounding.get("llm_assisted_grounding_recommended")),
            }
        )

        row: dict[str, Any] = {
            "question_id": int(item["question_id"]),
            "db_id": str(item["db_id"]),
            "difficulty": str(item.get("difficulty") or "unknown"),
            "source_run": source_run,
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "generation_mode": str(candidate.get("generation_mode") or "unknown"),
            "label": int(bool(candidate.get("correct"))),
            "oracle_correct": int(bool(item.get("oracle_correct"))),
            "execution_ok": int(execution.get("status") == "success"),
            "selected_by_majority_vote": _bool(selected_ids.get("majority_vote") == candidate.get("candidate_id")),
            "selected_by_execution_guided": _bool(selected_ids.get("execution_guided") == candidate.get("candidate_id")),
            "selected_by_saver": _bool(selected_ids.get("saver") == candidate.get("candidate_id")),
            "selected_by_llm_judge": _bool(selected_ids.get("llm_judge") == candidate.get("candidate_id")),
            "selected_by_random": _bool(selected_ids.get("random") == candidate.get("candidate_id")),
        }
        for key, value in feature_values.items():
            row[f"{FEATURE_PREFIX}{key}"] = _normalize_feature_value(value)
        rows.append(row)
    return rows


def load_run_items(run_dirs: list[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_question_ids: dict[int, str] = {}
    for run_dir in run_dirs:
        per_item_path = run_dir / "per_item.jsonl"
        if not per_item_path.exists():
            raise FileNotFoundError(f"Missing per_item.jsonl under {run_dir}")
        with open(per_item_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                question_id = int(item["question_id"])
                previous = seen_question_ids.get(question_id)
                if previous is not None:
                    raise RuntimeError(
                        f"Duplicate question_id {question_id} found in {run_dir} and {previous}. "
                        "Use disjoint runs when building the v2 feature table."
                    )
                seen_question_ids[question_id] = str(run_dir)
                item["_source_run"] = str(run_dir)
                items.append(item)
    return items


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows
