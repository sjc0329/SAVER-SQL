import ast
import re
from dataclasses import dataclass, field
from typing import Any

from evaluation.sql_executor import ExecutionResult


AGGREGATION_RE = re.compile(r"\b(count|sum|avg|max|min)\s*\((.*?)\)", re.I | re.S)


@dataclass
class BehaviorSignature:
    candidate_id: str
    parse_status: str
    execution_status: str
    selected_columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    where_conditions: list[dict[str, str]] = field(default_factory=list)
    aggregations: list[dict[str, str]] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: list[dict[str, str]] = field(default_factory=list)
    limit: int | None = None
    result_shape: dict[str, Any] = field(default_factory=dict)
    raw_sql: str = ""


def _clause(sql: str, start: str, stops: list[str]) -> str:
    stop_patterns = []
    for stop in stops:
        parts = stop.split()
        stop_patterns.append(r"\s+\b" + r"\s+".join(re.escape(part) for part in parts) + r"\b")
    pattern = rf"\b{start}\b(.*?)(?=" + "|".join(stop_patterns) + r"|$)"
    match = re.search(pattern, sql, re.I | re.S)
    return match.group(1).strip() if match else ""


def _split_csv(text: str) -> list[str]:
    if not text:
        return []
    items: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(char)
    item = "".join(current).strip()
    if item:
        items.append(item)
    return items


def _tables(sql: str) -> list[str]:
    found = re.findall(r"\b(?:from|join)\s+[`\"]?([a-zA-Z_][\w]*)[`\"]?", sql, re.I)
    return list(dict.fromkeys(found))


def _where_conditions(sql: str) -> list[dict[str, str]]:
    where = _clause(sql, "where", ["group by", "order by", "limit"])
    if not where:
        return []
    parts = re.split(r"\s+\bAND\b\s+", where, flags=re.I)
    conditions = []
    for part in parts:
        match = re.search(r"(.+?)\s*(=|!=|<>|>=|<=|>|<|like|in|not\s+in)\s*(.+)", part, re.I | re.S)
        if match:
            conditions.append(
                {
                    "column": match.group(1).strip(),
                    "operator": re.sub(r"\s+", " ", match.group(2).strip()).upper(),
                    "value": match.group(3).strip(),
                }
            )
        else:
            conditions.append({"column": "unknown", "operator": "unknown", "value": part.strip()})
    return conditions


def _order_by(sql: str) -> list[dict[str, str]]:
    clause = _clause(sql, "order by", ["limit"])
    orders = []
    for item in _split_csv(clause):
        direction = "DESC" if re.search(r"\bdesc\b", item, re.I) else "ASC" if re.search(r"\basc\b", item, re.I) else "unknown"
        expression = re.sub(r"\b(asc|desc)\b", "", item, flags=re.I).strip()
        orders.append({"expression": expression, "direction": direction})
    return orders


def _limit(sql: str) -> int | None:
    match = re.search(r"\blimit\s+(\d+)\b", sql, re.I)
    return int(match.group(1)) if match else None


def _unique_row_count(execution: ExecutionResult) -> int:
    if not execution.ok:
        return 0
    rows = getattr(execution, "rows", None)
    if rows:
        return len({repr(row) for row in rows})
    if rows == [] and int(getattr(execution, "row_count", 0) or 0) == 0:
        return 0
    preview = getattr(execution, "preview", None)
    if preview:
        return len({repr(tuple(row)) for row in preview})
    signature = getattr(execution, "signature", "")
    if signature:
        try:
            parsed = ast.literal_eval(signature)
            if isinstance(parsed, list):
                return len({repr(tuple(row) if isinstance(row, (list, tuple)) else row) for row in parsed})
        except Exception:
            pass
    return int(getattr(execution, "row_count", 0) or 0)


def extract_behavior_signature(candidate_id: str, sql: str, execution: ExecutionResult) -> BehaviorSignature:
    stripped = sql.strip().rstrip(";")
    parse_status = "success" if re.match(r"^(select|with)\b", stripped, re.I) else "error"
    selected = _clause(stripped, "select", ["from"])
    group_by = _split_csv(_clause(stripped, "group by", ["order by", "limit"]))
    aggregations = [
        {"function": match.group(1).upper(), "argument": match.group(2).strip()}
        for match in AGGREGATION_RE.finditer(stripped)
    ]
    return BehaviorSignature(
        candidate_id=candidate_id,
        parse_status=parse_status,
        execution_status=execution.status,
        selected_columns=_split_csv(selected),
        tables=_tables(stripped),
        where_conditions=_where_conditions(stripped),
        aggregations=aggregations,
        group_by=group_by,
        order_by=_order_by(stripped),
        limit=_limit(stripped),
        result_shape={
            "rows": execution.row_count,
            "columns": execution.column_count,
            "unique_rows": _unique_row_count(execution),
            "empty_result": execution.ok and execution.row_count == 0,
        },
        raw_sql=stripped,
    )
