from __future__ import annotations

"""ML-based violation classifier service."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .violation_review import ViolationReviewDB, ViolationReviewEntry


@dataclass
class ClassifiedSpan:
    """Result of classifying a span of text."""

    violation_type: str
    probability: float
    start: int
    end: int
    text: str


class ViolationClassifier:
    """Trainable text classifier for detecting legal violations."""

    def __init__(self, model_path: str = "./storage/models/violation_clf.joblib") -> None:
        self.model_path = Path(model_path)
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        if self.model_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        data = joblib.load(self.model_path)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]

    def _save(self) -> None:
        if self.model and self.vectorizer:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, self.model_path)

    # ------------------------------------------------------------------
    def train_from_review_db(self, db: ViolationReviewDB, min_conf: float = 0.5) -> None:
        """Train classifier using reviewed violations."""
        records = db.fetch_violations()
        texts: List[str] = []
        labels: List[str] = []
        for rec in records:
            if rec.status.lower() not in {"approved", "modified", "auto_approved"}:
                continue
            if rec.confidence < min_conf:
                continue
            text = rec.description or ""
            if not text.strip():
                continue
            texts.append(text)
            labels.append(rec.violation_type)
        if len(set(labels)) < 2:
            return
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)
        self._save()

    # ------------------------------------------------------------------
    def _predict_proba(self, text: str) -> List[Tuple[str, float]]:
        if not self.model or not self.vectorizer:
            return []
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        return list(zip(self.model.classes_, probs))

    def detect_violations(self, text: str, threshold: float = 0.6) -> List[ClassifiedSpan]:
        """Detect potential violations in text using the classifier."""
        if not self.model or not self.vectorizer:
            return []
        spans: List[ClassifiedSpan] = []
        offset = 0
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            clean = sentence.strip()
            if not clean:
                offset += len(sentence)
                continue
            preds = self._predict_proba(clean)
            for label, prob in preds:
                if prob >= threshold:
                    spans.append(
                        ClassifiedSpan(
                            violation_type=label,
                            probability=prob,
                            start=offset,
                            end=offset + len(sentence),
                            text=clean,
                        )
                    )
            offset += len(sentence) + 1
        return spans
