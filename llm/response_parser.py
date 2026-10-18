import json
import re
from typing import Any


SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.I | re.S)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.I | re.S)


def extract_sql(text: str) -> str:
    text = text.strip()
    block = SQL_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    joined = " ".join(lines)
    match = re.search(r"\b(with|select)\b.*", joined, re.I | re.S)
    return match.group(0).strip().rstrip(";") if match else joined.rstrip(";")


def _balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    block = JSON_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    else:
        balanced = _balanced_json_object(text)
        if balanced is not None:
            text = balanced
    return json.loads(text)
