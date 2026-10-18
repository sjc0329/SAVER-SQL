from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_sql.features import extract_candidate_feature_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SAVER-SQL candidate-level features from local per-question JSONL records."
    )
    parser.add_argument("input", type=Path, help="JSONL with one SAVER-SQL per-question record per line")
    parser.add_argument("--output", type=Path, required=True, help="Destination feature JSONL")
    parser.add_argument("--source-name", default="local", help="Non-sensitive provenance label stored in each row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {args.input}:{line_number}: {exc}") from exc
            for row in extract_candidate_feature_rows(item, source_run=args.source_name):
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
    print(f"Wrote {row_count} candidate rows to {args.output}")


if __name__ == "__main__":
    main()
