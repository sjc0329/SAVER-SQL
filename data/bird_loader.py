import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from data.dataset_item import DatasetItem


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def _read_gold_sql(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    gold: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            sql = line.split("\t", 1)[0].strip()
            gold[idx] = sql
    return gold


class BirdDevLoader:
    def __init__(self, config: dict[str, Any], root_dir: str | Path | None = None):
        self.config = config
        self.root_dir = Path(root_dir or os.getcwd())
        dataset = config["dataset"]
        self.data_dir = self._resolve(dataset["data_dir"])
        self.question_file = self.data_dir / dataset["question_file"]
        self.gold_file = self.data_dir / dataset.get("gold_file", "dev.sql")
        self.tables_file = self.data_dir / dataset["tables_file"]
        self.database_dir = self.data_dir / dataset["database_dir"]

    def _resolve_db_path(self, db_id: str) -> Path:
        primary = self.database_dir / db_id / f"{db_id}.sqlite"
        if primary.exists():
            return primary
        nested_root = self.database_dir / self.database_dir.name
        nested = nested_root / db_id / f"{db_id}.sqlite"
        if nested.exists():
            return nested
        flat = self.database_dir / f"{db_id}.sqlite"
        if flat.exists():
            return flat
        return primary

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root_dir / p

    def load_items(self) -> list[DatasetItem]:
        rows = load_json(self.question_file)
        gold_by_index = _read_gold_sql(self.gold_file)
        items: list[DatasetItem] = []
        for idx, row in enumerate(rows):
            qid = int(row.get("question_id", idx))
            db_id = row["db_id"]
            gold_sql = str(row.get("SQL") or gold_by_index.get(idx, "")).strip()
            db_path = self._resolve_db_path(str(db_id))
            items.append(
                DatasetItem(
                    question_id=qid,
                    db_id=db_id,
                    question=str(row.get("question", "")),
                    evidence=str(row.get("evidence", "")),
                    difficulty=str(row.get("difficulty", "")),
                    gold_sql=gold_sql,
                    db_path=str(db_path),
                )
            )
        return items

    def load_tables(self) -> list[dict[str, Any]]:
        return load_json(self.tables_file)


def stratified_sample(
    items: Sequence[DatasetItem],
    sample_size: int,
    seed: int,
    difficulties: Iterable[str] | None = None,
) -> list[DatasetItem]:
    if sample_size <= 0 or sample_size >= len(items):
        return list(items)

    allowed = {d for d in (difficulties or []) if d}
    pool = [it for it in items if not allowed or it.difficulty in allowed]
    rnd = random.Random(seed)
    by_key: dict[tuple[str, str], list[DatasetItem]] = {}
    for item in pool:
        by_key.setdefault((item.difficulty, item.db_id), []).append(item)

    selected: list[DatasetItem] = []
    buckets = list(by_key.values())
    for bucket in buckets:
        rnd.shuffle(bucket)
    while len(selected) < sample_size and any(buckets):
        rnd.shuffle(buckets)
        for bucket in buckets:
            if bucket and len(selected) < sample_size:
                selected.append(bucket.pop())
    return selected[:sample_size]
