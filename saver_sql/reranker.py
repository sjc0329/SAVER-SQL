import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise RuntimeError(
        "SAVER-SQL requires scikit-learn. "
        "Install the pinned dependencies with `python -m pip install -r requirements.txt`."
    ) from exc

from saver_sql.features import FEATURE_PREFIX, feature_group_for_name


FEATURE_PROFILES = {
    "full": {"include_groups": None, "drop_groups": set()},
}


def resolve_feature_profile(
    feature_profile: str | None,
    drop_feature_groups: set[str] | None = None,
) -> tuple[set[str] | None, set[str]]:
    profile = str(feature_profile or "full").lower()
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unsupported feature profile: {profile}")
    profile_spec = FEATURE_PROFILES[profile]
    include_groups = profile_spec["include_groups"]
    merged_drop = set(profile_spec["drop_groups"]) | set(drop_feature_groups or set())
    return (set(include_groups) if include_groups is not None else None, merged_drop)


def _feature_dict(
    row: dict[str, Any],
    drop_feature_groups: set[str] | None = None,
    include_feature_groups: set[str] | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith(FEATURE_PREFIX):
            continue
        group = feature_group_for_name(key)
        if include_feature_groups is not None and group not in include_feature_groups:
            continue
        if drop_feature_groups and group in drop_feature_groups:
            continue
        feature_name = key[len(FEATURE_PREFIX) :]
        features[feature_name] = value
    return features


@dataclass
class PredictionResult:
    candidate_id: str
    score: float


class BaseReranker:
    model_name = "base"

    def fit(self, rows: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    def predict_scores(self, rows: list[dict[str, Any]]) -> list[float]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "BaseReranker":
        with open(path, "rb") as handle:
            return pickle.load(handle)


class ConstantReranker(BaseReranker):
    model_name = "constant"

    def __init__(self, constant_score: float = 0.5, drop_feature_groups: set[str] | None = None, include_feature_groups: set[str] | None = None, feature_profile: str | None = None):
        self.constant_score = float(constant_score)
        self.drop_feature_groups = set(drop_feature_groups or set())
        self.include_feature_groups = set(include_feature_groups or set()) if include_feature_groups is not None else None
        self.feature_profile = str(feature_profile or "full")

    def fit(self, rows: list[dict[str, Any]]) -> None:
        labels = [int(row["label"]) for row in rows]
        if labels:
            self.constant_score = sum(labels) / len(labels)

    def predict_scores(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.constant_score for _ in rows]


class LogisticRegressionReranker(BaseReranker):
    model_name = "logistic_regression"

    def __init__(self, c: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 1000, drop_feature_groups: set[str] | None = None, include_feature_groups: set[str] | None = None, feature_profile: str | None = None):
        self.c = c
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.drop_feature_groups = set(drop_feature_groups or set())
        self.include_feature_groups = set(include_feature_groups or set()) if include_feature_groups is not None else None
        self.feature_profile = str(feature_profile or "full")
        self.pipeline = Pipeline(
            [
                ("vectorizer", DictVectorizer(sparse=True)),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.c,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        self._constant: ConstantReranker | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        labels = [int(row["label"]) for row in rows]
        if len(set(labels)) < 2:
            self._constant = ConstantReranker(drop_feature_groups=self.drop_feature_groups, include_feature_groups=self.include_feature_groups, feature_profile=self.feature_profile)
            self._constant.fit(rows)
            return
        self.pipeline.fit([_feature_dict(row, self.drop_feature_groups, self.include_feature_groups) for row in rows], labels)
        self._constant = None

    def predict_scores(self, rows: list[dict[str, Any]]) -> list[float]:
        if self._constant is not None:
            return self._constant.predict_scores(rows)
        probabilities = self.pipeline.predict_proba([_feature_dict(row, self.drop_feature_groups, self.include_feature_groups) for row in rows])
        return [float(prob[1]) for prob in probabilities]


def build_reranker(
    model_name: str,
    params: dict[str, Any] | None = None,
    drop_feature_groups: set[str] | None = None,
    feature_profile: str | None = None,
) -> BaseReranker:
    params = dict(params or {})
    include_feature_groups, merged_drop = resolve_feature_profile(feature_profile, drop_feature_groups)
    params["drop_feature_groups"] = merged_drop
    params["include_feature_groups"] = include_feature_groups
    params["feature_profile"] = str(feature_profile or "full")
    normalized = str(model_name).lower()
    if normalized in {"logistic", "logreg", "lr"}:
        return LogisticRegressionReranker(**params)
    raise ValueError(f"Unsupported reranker model: {model_name}")
