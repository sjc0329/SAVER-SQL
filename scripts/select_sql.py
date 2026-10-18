from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_sql.profiles import apply_feature_profile
from saver_sql.reranker import BaseReranker
from saver_sql.selection import select_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a frozen five-candidate pool and return one original candidate SQL string."
    )
    parser.add_argument("--model", type=Path, required=True, help="Model created by scripts/train_lr.py")
    parser.add_argument("--candidates", type=Path, required=True, help="JSON list, or an object with a candidates list")
    parser.add_argument("--output", type=Path, help="Optional output JSON; stdout is used otherwise")
    return parser.parse_args()


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Expected exactly five candidate feature rows.")
    required = {"candidate_id", "generation_mode", "sql"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Candidate {index} must contain candidate_id, generation_mode, and sql.")
    return rows


def main() -> None:
    args = parse_args()
    original_rows = _load_candidates(args.candidates)
    model = BaseReranker.load(args.model)
    profile = str(getattr(model, "public_profile", "full"))
    model_rows = apply_feature_profile(original_rows, profile)
    scores = model.predict_scores(model_rows)
    selected = select_candidate(model_rows, scores)
    selected_index = model_rows.index(selected)
    result = {
        "candidate_id": original_rows[selected_index]["candidate_id"],
        "score": scores[selected_index],
        "sql": original_rows[selected_index]["sql"],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
