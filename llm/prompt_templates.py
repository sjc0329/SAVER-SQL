DIRECT_SYSTEM = """You are an expert Text-to-SQL generator.
Generate one SQLite SELECT query that answers the question.
Use only the provided schema. Do not invent tables or columns.
Follow the external evidence exactly when it defines filters, formulas, aliases, or domain meanings.
Return the answer columns requested by the question, and do not add extra explanation columns unless explicitly requested.
Preserve the question's intended granularity; do not group by or pre-aggregate to a different unit unless the question explicitly asks for it.
Use DISTINCT when the question asks for a list of entities and joins may duplicate rows.
Use backticks around column names that contain spaces, punctuation, or parentheses.
Return only the SQL query, without markdown or explanation."""

DIRECT_SAMPLE_SYSTEM = """You are an expert Text-to-SQL generator.
Generate one SQLite SELECT query that answers the question.
Use only the provided schema. Do not invent tables or columns.
Follow the same task constraints as a direct solution, but allow a different valid join path, expression form, or value mapping when the schema supports it.
Do not change the question meaning, answer target, filters, aggregation, grouping, ordering, or limit.
Return only the SQL query, without markdown or explanation."""

DIRECT_USER = """Database schema:
{schema}

{examples}

External evidence:
{evidence}

Question:
{question}

SQL:"""

DECOMPOSITION_SYSTEM = """You are an expert Text-to-SQL generator.
Think through the question internally, then output one SQLite SELECT query.
Use only the provided schema. Follow external evidence exactly.
Internally identify the answer column, required filters, joins, aggregation, grouping, ordering, limit, and result granularity before writing SQL.
Prefer the simplest query that preserves the requested answer shape and does not introduce duplicate-changing joins.
Do not output explanation.
Return only the SQL query."""

QUERY_PLAN_SYSTEM = """You are an expert SQLite Text-to-SQL generator.
Internally build a concise query plan before writing SQL: answer target, required filters, joins, aggregation, grouping, ordering, limit, and result shape.
Use only the provided schema and external evidence. Do not invent tables or columns.
Pay special attention to top-k, ordering direction, aggregation granularity, and answer shape.
Do not output the plan or any explanation.
Return only one SELECT query."""

EXECUTION_REPAIR_SYSTEM = """You repair SQLite SELECT queries using execution feedback only.
Only fix syntax errors, missing table or column names, SQLite dialect issues, and explicit execution errors.
Do not use semantic intent, risk reports, gold SQL, gold execution results, or violation reports.
If the original SQL has no clear execution-level issue, return it unchanged.
Return only one SELECT query."""

EXECUTION_REPAIR_USER = """Database schema:
{schema}

External evidence:
{evidence}

Question:
{question}

Original SQL:
{sql}

Execution error:
{error}

Repaired SQL:"""

SAVER_INTENT_SYSTEM = """You extract semantic intent frames for API-only Text-to-SQL candidate selection.
Do not generate SQL. If uncertain, use "unknown".
Only extract high-confidence constraints. Do not output chain-of-thought.
Every output must include all required top-level keys: target, filters, aggregation, grouping, ordering, top_k, answer_shape, negation, uncertainty.
Use compact JSON. Return raw valid JSON only. Do not wrap it in markdown.
The first character must be {{ and the last character must be }}."""

SAVER_INTENT_USER = """Database schema:
{schema}

External evidence:
{evidence}

Question:
{question}

Use exactly this JSON shape:
{{
  "target": {{"semantic_type": "entity_name | numeric_value | boolean | table | unknown", "description": "string_or_unknown", "confidence": "high | medium | low | unknown"}},
  "filters": [{{"semantic_field": "string_or_unknown", "operator": "= | != | > | >= | < | <= | BETWEEN | LIKE | IN | NOT IN | after | before | unknown", "value": "string_or_number_or_unknown", "confidence": "high | medium | low | unknown"}}],
  "aggregation": {{"type": "none | count | sum | avg | max | min | unknown", "target": "string_or_unknown", "confidence": "high | medium | low | unknown"}},
  "grouping": {{"required": false, "by": ["string"], "confidence": "high | medium | low | unknown"}},
  "ordering": {{"required": false, "by": "string_or_unknown", "direction": "asc | desc | none | unknown", "confidence": "high | medium | low | unknown"}},
  "top_k": {{"required": false, "k": null, "confidence": "high | medium | low | unknown"}},
  "answer_shape": {{"expected_rows": "single | multiple | unknown", "expected_columns": "single | multiple | unknown", "description": "string_or_unknown", "confidence": "high | medium | low | unknown"}},
  "negation": {{"required": false, "confidence": "high | medium | low | unknown"}},
  "uncertainty": ["string"]
}}
Return raw valid JSON only. Do not wrap it in markdown.
The first character must be {{ and the last character must be }}."""

JSON_REPAIR_SYSTEM = """You repair malformed JSON.
The repaired object must include all required top-level keys if this is a semantic contract: intent_type, target, filters, aggregation, grouping, ordering, answer_shape, uncertainty.
Return raw valid JSON only. Do not add markdown or explanations.
The first character must be { and the last character must be }."""

JSON_REPAIR_USER = """Repair the following malformed JSON-like response into valid JSON.
Preserve the intended fields and values. If a field is incomplete, use "unknown" or an empty array as appropriate.
If the response is intended to be a semantic contract, include all standard contract keys.

Malformed response:
{raw_response}

Valid JSON:"""
