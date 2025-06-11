from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..core.ml_optimizer import DocumentFeatures


class WorkflowPolicy:
    """Learn routing and concurrency policies based on history."""

    def __init__(self) -> None:
        self.agent_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"success": 0, "fail": 0}
        )
        self.history: List[Dict[str, float]] = []
        self.model = DecisionTreeClassifier(max_depth=3)
        self.trained = False

    # -------------------------- Training Helpers --------------------------
    def _feature_vector(self, features: DocumentFeatures) -> List[float]:
        vec = features.to_vector().tolist()
        # Append global agent success rates
        for stats in self.agent_stats.values():
            total = stats["success"] + stats["fail"] or 1
            vec.append(stats["success"] / total)
        return vec

    def record_step(
        self,
        step_name: str,
        features: DocumentFeatures,
        success: bool,
        duration: float,
    ) -> None:
        """Store training sample and retrain when enough data is available."""
        sample = {
            "step": step_name,
            "duration": duration,
            "success": 1 if success else 0,
            "features": asdict(features),
        }
        self.history.append(sample)
        if len(self.history) >= 10:
            self._train()

    def _train(self) -> None:
        X = [
            self._feature_vector(DocumentFeatures(**h["features"]))
            for h in self.history
        ]
        y = [h["success"] for h in self.history]
        if X and y:
            self.model.fit(X, y)
            self.trained = True

    # -------------------------- Predictions --------------------------
    def should_run_step(self, step_name: str, features: DocumentFeatures) -> bool:
        if not self.trained:
            return True
        x = np.array([self._feature_vector(features)])
        pred = self.model.predict(x)[0]
        return bool(pred)

    def predict_concurrency(self, features: DocumentFeatures) -> int:
        total_success = sum(s["success"] for s in self.agent_stats.values())
        total_attempts = total_success + sum(
            s["fail"] for s in self.agent_stats.values()
        )
        rate = total_success / total_attempts if total_attempts else 1.0
        # Map success rate to concurrency between 1 and 5
        return max(1, min(5, int(round(rate * 5))))

    def update_agent_stats(self, agent: str, success: bool) -> None:
        stats = self.agent_stats[agent]
        if success:
            stats["success"] += 1
        else:
            stats["fail"] += 1
