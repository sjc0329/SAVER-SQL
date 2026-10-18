from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetItem:
    question_id: int
    db_id: str
    question: str
    evidence: str
    difficulty: str
    gold_sql: str
    db_path: str
