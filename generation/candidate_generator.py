import re

from evaluation.cost_tracker import Usage
from generation.direct_generator import Candidate, DirectGenerator


def _has_limit(sql: str) -> bool:
    return bool(re.search(r"\blimit\s+\d+\b", sql, re.I))


def _question_text(question: str, evidence: str) -> str:
    return f"{question} {evidence}".lower()


def _needs_single_extrema_answer(question: str, evidence: str) -> bool:
    text = _question_text(question, evidence)
    return any(term in text for term in [" highest ", " lowest ", " maximum ", " minimum ", " max ", " min ", " 1st ", " first "]) or (
        any(term in text for term in ["highest", "lowest", "maximum", "minimum"]) and "list" not in text
    )


def _add_limit_one(sql: str) -> str | None:
    stripped = sql.strip().rstrip(";")
    if _has_limit(stripped):
        return None
    if not re.match(r"^(select|with)\b", stripped, re.I):
        return None
    return f"{stripped} LIMIT 1"


def _needs_distinct_list(question: str, evidence: str) -> bool:
    text = _question_text(question, evidence)
    if any(term in text for term in ["top ", "highest", "lowest", "largest", "smallest"]):
        return False
    return any(term in text for term in ["list", "what are", "what kind of", "names of"])


def _add_select_distinct(sql: str) -> str | None:
    stripped = sql.strip().rstrip(";")
    if re.search(r"\bselect\s+distinct\b", stripped, re.I):
        return None
    return re.sub(r"^\s*select\b", "SELECT DISTINCT", stripped, count=1, flags=re.I)


def _needs_non_distinct_count(question: str, evidence: str) -> bool:
    text = _question_text(question, evidence)
    if not any(term in text for term in ["how many", "number of", "count"]):
        return False
    return not any(term in text for term in ["distinct", "different", "unique"])


def _swap_count_distinct(sql: str) -> str | None:
    stripped = sql.strip().rstrip(";")
    rewritten = re.sub(r"count\s*\(\s*distinct\s+[^)]+\)", "COUNT(*)", stripped, count=1, flags=re.I)
    return rewritten if rewritten != stripped else None


def _needs_unrounded_ratio(question: str, evidence: str) -> bool:
    text = _question_text(question, evidence)
    return any(term in text for term in ["ratio", "percentage"])


def _remove_round_wrapper(sql: str) -> str | None:
    stripped = sql.strip().rstrip(";")
    pattern = re.compile(r"round\s*\(\s*(.+?)\s*,\s*2\s*\)", re.I | re.S)
    rewritten = pattern.sub(r"\1", stripped, count=1)
    return rewritten if rewritten != stripped else None


def semantic_contrast_sql(question: str, evidence: str, sql: str) -> tuple[str | None, str]:
    rules = [
        ("extrema_limit_one", _needs_single_extrema_answer, _add_limit_one),
        ("distinct_list", _needs_distinct_list, _add_select_distinct),
        ("count_without_distinct", _needs_non_distinct_count, _swap_count_distinct),
        ("ratio_unrounded", _needs_unrounded_ratio, _remove_round_wrapper),
    ]
    for name, predicate, rewriter in rules:
        if predicate(question, evidence):
            rewritten = rewriter(sql)
            if rewritten and rewritten.strip().rstrip(";").lower() != sql.strip().rstrip(";").lower():
                return rewritten, name
    return None, "copy_c1"


class CandidateGenerator:
    MODES = ["direct", "direct_sample", "decomposition", "query_plan"]

    def __init__(self, generator: DirectGenerator | None = None):
        self.generator = generator

    def generate(self, schema: str, question: str, evidence: str, examples: str = "") -> list[Candidate]:
        if self.generator is None:
            raise RuntimeError("CandidateGenerator requires a DirectGenerator.")
        return [self.generator.generate(mode, schema, question, evidence, examples=examples) for mode in self.MODES]

    def generate_execution_repair(self, schema: str, question: str, evidence: str, sql: str, error: str) -> Candidate:
        if self.generator is None:
            return Candidate("c5", "execution_repair_unavailable", sql, Usage())
        return self.generator.repair_execution_error(schema, question, evidence, sql, error)

    def generate_semantic_contrast(self, question: str, evidence: str, sql: str) -> Candidate:
        rewritten, variant = semantic_contrast_sql(question, evidence, sql)
        candidate_sql = rewritten if rewritten else sql
        return Candidate("c5", f"semantic_contrast_{variant}", candidate_sql, Usage())
