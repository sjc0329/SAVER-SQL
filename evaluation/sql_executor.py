import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any


FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|pragma|attach|detach|vacuum)\b", re.I)


@dataclass
class ExecutionResult:
    status: str
    rows: list[tuple[Any, ...]]
    preview: list[list[Any]]
    error: str | None
    elapsed_ms: float
    row_count: int
    column_count: int
    signature: str

    @property
    def ok(self) -> bool:
        return self.status == "success"


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if value is None:
        return None
    return str(value).strip() if not isinstance(value, (int, float)) else value


def canonical_signature(rows: list[tuple[Any, ...]]) -> str:
    normalized = [tuple(_normalize_cell(v) for v in row) for row in rows]
    return repr(sorted(normalized, key=repr))


def _safe_select(sql: str) -> tuple[bool, str | None]:
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False, "empty SQL"
    if FORBIDDEN_SQL.search(stripped):
        return False, "non-read-only SQL is not allowed"
    if not re.match(r"^(select|with)\b", stripped, re.I):
        return False, "only SELECT/WITH queries are allowed"
    return True, None


def execute_sql(
    db_path: str,
    sql: str,
    timeout_seconds: int = 30,
    preview_rows: int = 20,
    max_rows: int = 10000,
) -> ExecutionResult:
    safe, reason = _safe_select(sql)
    if not safe:
        return ExecutionResult("error", [], [], reason, 0.0, 0, 0, "")

    start = time.monotonic()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        deadline = start + timeout_seconds

        def progress_handler() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(progress_handler, 10000)
        cur = conn.execute(sql.strip().rstrip(";"))
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]
        tuples = [tuple(row) for row in rows]
        elapsed = (time.monotonic() - start) * 1000
        col_count = len(cur.description or [])
        error = "result truncated for signature" if truncated else None
        return ExecutionResult(
            "success",
            tuples,
            [list(row) for row in tuples[:preview_rows]],
            error,
            elapsed,
            len(tuples),
            col_count,
            canonical_signature(tuples),
        )
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        return ExecutionResult("error", [], [], str(exc), elapsed, 0, 0, "")
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
