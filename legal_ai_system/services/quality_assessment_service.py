from __future__ import annotations

"""Service wrapper exposing quality classification utilities."""

from typing import Any, Dict, List

from ..core.detailed_logging import get_detailed_logger, LogCategory
from ..analytics.quality_classifier import (
    QualityClassifier,
    QualityModelMonitor,
    PreprocessingErrorPredictor,
)


class QualityAssessmentService:
    """Manage quality-related models and predictions."""

    def __init__(self) -> None:
        self.logger = get_detailed_logger("QualityAssessmentService", LogCategory.SYSTEM)
        self.classifier = QualityClassifier()
        self.preproc_predictor = PreprocessingErrorPredictor()
        self.monitor = QualityModelMonitor(self.classifier)

    # ------------------------------------------------------------------
    # Wrapper methods for QualityClassifier
    # ------------------------------------------------------------------
    def train_classifier(self, items: List[Dict[str, Any]], labels: List[int]) -> None:
        self.logger.debug("Training quality classifier", parameters={"num_items": len(items)})
        self.classifier.train(items, labels)

    def predict_error_probability(self, item: Dict[str, Any]) -> float:
        return self.classifier.predict_error_probability(item)

    def evaluate_classifier(self, items: List[Dict[str, Any]], labels: List[int]) -> float:
        return self.classifier.evaluate(items, labels)

    # ------------------------------------------------------------------
    # Wrapper methods for QualityModelMonitor
    # ------------------------------------------------------------------
    def check_drift(self, items: List[Dict[str, Any]], labels: List[int]) -> float:
        return self.monitor.check_drift(items, labels)

    # ------------------------------------------------------------------
    # Preprocessing error prediction utilities
    # ------------------------------------------------------------------
    def train_preprocessing_predictor(self, docs: List[Dict[str, Any]], labels: List[int]) -> None:
        self.logger.debug("Training preprocessing predictor", parameters={"num_docs": len(docs)})
        self.preproc_predictor.train(docs, labels)

    def predict_preprocessing_risk(self, doc: Dict[str, Any]) -> float:
        return self.preproc_predictor.predict_risk(doc)
