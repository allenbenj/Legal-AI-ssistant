import pytest

from legal_ai_system.analytics.quality_classifier import QualityClassifier, PreprocessingErrorPredictor


@pytest.mark.unit
def test_quality_classifier_trains_and_predicts():
    items = [
        {"item_type": "ENTITY", "confidence": 0.4, "text_context": "foo"},
        {"item_type": "RELATION", "confidence": 0.9, "text_context": "bar"},
    ]
    labels = [1, 0]
    clf = QualityClassifier()
    clf.train(items, labels)
    prob = clf.predict_error_probability({"item_type": "ENTITY", "confidence": 0.5, "text_context": "foo"})
    assert 0.0 <= prob <= 1.0


@pytest.mark.unit
def test_preprocessing_predictor_trains_and_predicts():
    docs = [
        {"content_preview": "short text", "size": 100},
        {"content_preview": "longer text with more content", "size": 2000},
    ]
    labels = [0, 1]
    pred = PreprocessingErrorPredictor()
    pred.train(docs, labels)
    risk = pred.predict_risk({"content_preview": "short text", "size": 50})
    assert 0.0 <= risk <= 1.0
