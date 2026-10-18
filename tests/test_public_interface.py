import unittest

from saver_sql.profiles import STATISTICAL_FEATURES, apply_feature_profile
from saver_sql.reranker import build_reranker
from saver_sql.selection import select_candidate


class PublicInterfaceTest(unittest.TestCase):
    def test_lite_has_exactly_twenty_features(self) -> None:
        self.assertEqual(len(STATISTICAL_FEATURES), 20)

    def test_lr_selects_an_unchanged_candidate(self) -> None:
        training_rows = []
        for question_id in (1, 2):
            for candidate_number in range(1, 6):
                training_rows.append(
                    {
                        "question_id": question_id,
                        "candidate_id": f"C{candidate_number}",
                        "generation_mode": "direct" if candidate_number == 1 else "sampled",
                        "label": int(candidate_number == 1),
                        "feature_candidate_order": candidate_number,
                        "feature_execution_ok": int(candidate_number == 1),
                        "feature_cluster_size": 2 if candidate_number == 1 else 1,
                    }
                )
        model = build_reranker("logistic", params={"c": 0.25, "class_weight": "balanced", "max_iter": 1000})
        model.fit(training_rows)

        candidates = []
        for candidate_number in range(1, 6):
            candidates.append(
                {
                    "candidate_id": f"C{candidate_number}",
                    "generation_mode": "direct" if candidate_number == 1 else "sampled",
                    "sql": f"SELECT {candidate_number}",
                    "feature_candidate_order": candidate_number,
                    "feature_execution_ok": int(candidate_number == 1),
                    "feature_cluster_size": 2 if candidate_number == 1 else 1,
                }
            )
        scored_rows = apply_feature_profile(candidates, "full")
        selected = select_candidate(scored_rows, model.predict_scores(scored_rows))
        original = next(row for row in candidates if row["candidate_id"] == selected["candidate_id"])
        self.assertEqual(selected["sql"], original["sql"])


if __name__ == "__main__":
    unittest.main()
