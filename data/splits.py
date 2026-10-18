import json
from pathlib import Path
from typing import Any, Sequence

from data.dataset_item import DatasetItem


def load_split_file(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_question_ids(split_data: dict[str, Any], split_name: str) -> list[int]:
    splits = split_data.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        available = ", ".join(sorted(splits.keys())) if isinstance(splits, dict) else "none"
        raise KeyError(f"Split '{split_name}' not found. Available splits: {available}")
    entry = splits[split_name]
    if not isinstance(entry, dict) or not isinstance(entry.get("question_ids"), list):
        raise ValueError(f"Split '{split_name}' must contain a question_ids list.")
    return [int(qid) for qid in entry["question_ids"]]


def apply_question_id_split(
    items: Sequence[DatasetItem],
    question_ids: Sequence[int],
    sample_size: int | None = None,
) -> list[DatasetItem]:
    by_qid = {item.question_id: item for item in items}
    selected: list[DatasetItem] = []
    missing: list[int] = []
    seen: set[int] = set()
    for qid in question_ids:
        qid_int = int(qid)
        if qid_int in seen:
            raise ValueError(f"Duplicate question_id in split: {qid_int}")
        seen.add(qid_int)
        item = by_qid.get(qid_int)
        if item is None:
            missing.append(qid_int)
        else:
            selected.append(item)
    if missing:
        preview = ", ".join(str(qid) for qid in missing[:10])
        raise ValueError(f"Split contains question_ids not found in dataset: {preview}")
    return selected[:sample_size] if sample_size is not None else selected
