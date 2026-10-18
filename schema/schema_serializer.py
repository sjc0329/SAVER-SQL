from typing import Any


def _table_columns(schema: dict[str, Any]) -> dict[int, list[tuple[str, str, str]]]:
    tables = schema["table_names_original"]
    table_cols: dict[int, list[tuple[str, str, str]]] = {i: [] for i in range(len(tables))}
    col_names = schema.get("column_names_original", [])
    col_names_nl = schema.get("column_names", col_names)
    col_types = schema.get("column_types", [])
    for idx, col in enumerate(col_names):
        table_idx, raw_name = col
        if table_idx < 0:
            continue
        semantic = col_names_nl[idx][1] if idx < len(col_names_nl) else raw_name
        col_type = col_types[idx] if idx < len(col_types) else "unknown"
        table_cols.setdefault(table_idx, []).append((str(raw_name), str(semantic), str(col_type)))
    return table_cols


def _format_foreign_keys(schema: dict[str, Any]) -> list[str]:
    col_names = schema.get("column_names_original", [])
    table_names = schema.get("table_names_original", [])
    lines: list[str] = []
    for src_idx, dst_idx in schema.get("foreign_keys", []):
        try:
            src_table_idx, src_col = col_names[src_idx]
            dst_table_idx, dst_col = col_names[dst_idx]
            lines.append(f"{table_names[src_table_idx]}.{src_col} -> {table_names[dst_table_idx]}.{dst_col}")
        except Exception:
            continue
    return lines


def serialize_schema(schema: dict[str, Any], max_chars: int = 12000) -> str:
    table_names = schema["table_names_original"]
    table_names_nl = schema.get("table_names", table_names)
    table_cols = _table_columns(schema)
    primary_keys = set()
    for pk in schema.get("primary_keys", []):
        if isinstance(pk, list):
            primary_keys.update(pk)
        else:
            primary_keys.add(pk)

    lines = [f"Database: {schema['db_id']}", "Tables:"]
    col_index = {tuple(col): i for i, col in enumerate(schema.get("column_names_original", []))}
    for idx, table in enumerate(table_names):
        semantic_table = table_names_nl[idx] if idx < len(table_names_nl) else table
        lines.append(f"- {table} ({semantic_table})")
        for raw_name, semantic, col_type in table_cols.get(idx, []):
            marker = " PK" if col_index.get((idx, raw_name)) in primary_keys else ""
            lines.append(f"  - {raw_name}: {col_type}; meaning={semantic}{marker}")

    fks = _format_foreign_keys(schema)
    if fks:
        lines.append("Foreign keys:")
        lines.extend(f"- {fk}" for fk in fks)

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n... [schema truncated for prompt budget]"
