from __future__ import annotations

from typing import Any


def _candidate_order(candidate_id: str) -> int:
    digits = "".join(character for character in str(candidate_id) if character.isdigit())
    return int(digits) if digits else 999


def _generation_mode_priority(mode: str) -> int:
    normalized = (mode or "").lower()
    priorities = {
        "query_plan": 0,
        "decomposition": 1,
        "direct_sample": 2,
        "direct": 3,
    }
    if normalized.startswith("semantic_contrast"):
        return 4
    return priorities.get(normalized, 9)


def select_candidate(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    if not rows or len(rows) != len(scores):
        raise ValueError("Candidate rows and scores must be non-empty and have the same length.")

    def key(pair: tuple[dict[str, Any], float]) -> tuple[float, int, int, float, int, int, int]:
        row, score = pair
        return (
            float(score),
            int(row.get("feature_cluster_size", 0)),
            int(row.get("feature_execution_ok", 0)),
            -float(row.get("feature_risk_score", 0.0)),
            -int(row.get("feature_high_severity_failures", 0)),
            -_generation_mode_priority(str(row.get("generation_mode", ""))),
            -_candidate_order(str(row.get("candidate_id", ""))),
        )

    return max(zip(rows, scores), key=key)[0]
