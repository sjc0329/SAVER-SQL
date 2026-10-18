from __future__ import annotations

from typing import Any

from saver_sql.features import FEATURE_PREFIX, feature_group_for_name


STATISTICAL_FEATURES = frozenset(
    {
        "feature_candidate_order",
        "feature_execution_ok",
        "feature_execution_row_count",
        "feature_execution_column_count",
        "feature_cluster_size",
        "feature_cluster_ratio",
        "feature_cluster_is_majority",
        "feature_cluster_max_size",
        "feature_selected_column_count",
        "feature_table_count",
        "feature_join_count",
        "feature_where_count",
        "feature_aggregation_count",
        "feature_group_by_count",
        "feature_order_by_count",
        "feature_has_limit",
        "feature_limit_value",
        "feature_result_rows",
        "feature_result_columns",
        "feature_sql_length",
    }
)

PUBLIC_PROFILES = (
    "full",
    "lite",
    "full_minus_semantic_risk",
    "execution_cluster_only",
)


def keep_feature(profile: str, key: str) -> bool:
    if not key.startswith(FEATURE_PREFIX):
        return True
    if profile == "full":
        return True
    if profile == "lite":
        return key in STATISTICAL_FEATURES
    group = feature_group_for_name(key)
    if profile == "full_minus_semantic_risk":
        return group != "risk"
    if profile == "execution_cluster_only":
        return group == "execution"
    raise ValueError(f"Unsupported public feature profile: {profile}")


def apply_feature_profile(rows: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    if profile not in PUBLIC_PROFILES:
        raise ValueError(f"Unsupported public feature profile: {profile}")
    return [{key: value for key, value in row.items() if keep_feature(profile, key)} for row in rows]
