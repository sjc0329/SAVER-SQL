import re
import sqlite3
from dataclasses import dataclass
from typing import Any


STOPWORDS = {
    "about",
    "after",
    "among",
    "before",
    "between",
    "calculate",
    "count",
    "does",
    "each",
    "find",
    "from",
    "give",
    "have",
    "list",
    "many",
    "more",
    "name",
    "number",
    "only",
    "please",
    "show",
    "state",
    "than",
    "that",
    "their",
    "them",
    "there",
    "these",
    "those",
    "what",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class ValueHint:
    table: str
    column: str
    matched_term: str
    value: str


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _date_variants(term: str) -> set[str]:
    variants = {term}
    match = re.fullmatch(r"(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})", term)
    if not match:
        return variants
    a, b, c = match.groups()
    if len(a) == 4:
        year, month, day = a, b, c
    elif len(c) == 4:
        year, month, day = c, a, b
    else:
        return variants
    variants.add(f"{year}-{int(month):02d}-{int(day):02d}")
    variants.add(f"{year}/{int(month):02d}/{int(day):02d}")
    variants.add(f"{year}-{int(month)}-{int(day)}")
    variants.add(f"{year}/{int(month)}/{int(day)}")
    return variants


def _extract_terms(question: str, evidence: str, max_terms: int) -> list[str]:
    text = f"{question}\n{evidence}"
    terms: list[str] = []
    evidence_terms: list[str] = []

    def add(term: str) -> None:
        cleaned = term.strip().strip(".,;:()[]{}")
        if not cleaned:
            return
        lowered = cleaned.lower()
        if len(cleaned) < 2 or lowered in STOPWORDS:
            return
        if lowered not in {item.lower() for item in terms}:
            terms.append(cleaned)

    for pattern in (r"'([^']+)'", r'"([^"]+)"', r"`([^`]+)`"):
        for match in re.finditer(pattern, text):
            add(match.group(1))

    for match in re.finditer(r"[=:]\s*([A-Za-z0-9_.+-]+)", evidence):
        evidence_terms.append(match.group(1))
    for term in evidence_terms:
        add(term)

    for match in re.finditer(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", text):
        for variant in _date_variants(match.group(0)):
            add(variant)

    for match in re.finditer(r"\b[A-Z][A-Z0-9_]{1,}\b", text):
        add(match.group(0))

    for match in re.finditer(r"\bNo\.\s*(\d+)\b", text, re.I):
        add(match.group(1))

    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text):
        add(match.group(0))

    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text):
        add(match.group(0))

    return terms[:max_terms]


def _term_column_preferences(evidence: str) -> dict[str, set[str]]:
    preferences: dict[str, set[str]] = {}
    for match in re.finditer(r"\b([A-Za-z_][\w]*)\b\s*(?:=|!=|>|>=|<|<=)\s*'?([^';,\n]+)'?", evidence):
        column = match.group(1).strip().lower()
        value = match.group(2).strip().strip('"`').lower()
        if column and value:
            preferences.setdefault(value, set()).add(column)
            for variant in _date_variants(value):
                preferences.setdefault(variant.lower(), set()).add(column)
    return preferences


def _tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    return [str(row[0]) for row in rows]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return [str(row[1]) for row in rows]


def collect_value_hints(
    db_path: str,
    question: str,
    evidence: str,
    max_terms: int = 24,
    max_hints: int = 40,
    per_column_limit: int = 2,
) -> list[ValueHint]:
    terms = _extract_terms(question, evidence, max_terms=max_terms)
    if not terms:
        return []
    preferences = _term_column_preferences(evidence)

    hints: list[ValueHint] = []
    seen: set[tuple[str, str, str, str]] = set()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for table in _tables(conn):
            table_sql = _quote_identifier(table)
            for column in _columns(conn, table):
                col_sql = _quote_identifier(column)
                for term in terms:
                    preferred_columns = preferences.get(term.lower())
                    column_l = column.lower()
                    if preferred_columns and not any(pref in column_l or column_l in pref for pref in preferred_columns):
                        continue
                    if len(hints) >= max_hints:
                        return hints
                    rows = _exact_matches(conn, table_sql, col_sql, term, per_column_limit)
                    for row in rows:
                        value = "" if row[0] is None else str(row[0])
                        key = (table, column, term.lower(), value.lower())
                        if value and key not in seen:
                            seen.add(key)
                            hints.append(ValueHint(table, column, term, value))
                            if len(hints) >= max_hints:
                                return hints
    finally:
        conn.close()
    return hints


def _exact_matches(conn: sqlite3.Connection, table_sql: str, col_sql: str, term: str, limit: int) -> list[tuple[Any, ...]]:
    try:
        return conn.execute(
            f"SELECT DISTINCT {col_sql} FROM {table_sql} WHERE {col_sql} = ? COLLATE NOCASE LIMIT ?",
            (term, limit),
        ).fetchall()
    except sqlite3.Error:
        try:
            return conn.execute(
                f"SELECT DISTINCT {col_sql} FROM {table_sql} WHERE CAST({col_sql} AS TEXT) = ? COLLATE NOCASE LIMIT ?",
                (term, limit),
            ).fetchall()
        except sqlite3.Error:
            return []


def format_value_hints(hints: list[ValueHint], max_chars: int = 1800) -> str:
    if not hints:
        return ""
    lines = ["Database value hints matched from the current question/evidence:"]
    for hint in hints:
        lines.append(f"- {hint.table}.{hint.column}: value {hint.value!r} matched term {hint.matched_term!r}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n... [value hints truncated for prompt budget]"


def build_value_hints_text(db_path: str, question: str, evidence: str, config: dict[str, Any] | None = None) -> str:
    cfg = config or {}
    if not bool(cfg.get("enabled", False)):
        return ""
    hints = collect_value_hints(
        db_path,
        question,
        evidence,
        max_terms=int(cfg.get("max_terms", 24)),
        max_hints=int(cfg.get("max_hints", 40)),
        per_column_limit=int(cfg.get("per_column_limit", 2)),
    )
    return format_value_hints(hints, max_chars=int(cfg.get("max_chars", 1800)))
