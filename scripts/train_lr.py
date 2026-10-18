from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saver_sql.features import load_feature_rows
from saver_sql.profiles import PUBLIC_PROFILES, apply_feature_profile
from saver_sql.reranker import build_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fixed SAVER-SQL logistic-regression selector.")
    parser.add_argument("features", type=Path, help="Candidate-level training feature JSONL")
    parser.add_argument("--profile", choices=PUBLIC_PROFILES, default="full")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "full_lr.json")
    parser.add_argument("--output-model", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = apply_feature_profile(load_feature_rows(args.features), args.profile)
    if not rows:
        raise RuntimeError("No feature rows were loaded.")
    if any("label" not in row for row in rows):
        raise RuntimeError("Every training row must contain a binary 'label'.")

    model = build_reranker("logistic", params=config["logistic_regression"], feature_profile="full")
    model.fit(rows)
    model.public_profile = args.profile
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model)
    print(f"Saved {args.profile} LR model trained on {len(rows)} candidate rows to {args.output_model}")


if __name__ == "__main__":
    main()
