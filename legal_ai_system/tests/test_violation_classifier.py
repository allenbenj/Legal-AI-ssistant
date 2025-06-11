import pytest
from datetime import datetime, timezone
from pathlib import Path

from legal_ai_system.services.violation_classifier import ViolationClassifier
from legal_ai_system.services.violation_review import ViolationReviewDB, ViolationReviewEntry


@pytest.mark.unit
def test_violation_classifier_trains_and_detects(tmp_path: Path) -> None:
    db_path = tmp_path / "viol.db"
    db = ViolationReviewDB(db_path=str(db_path))
    now = datetime.now(timezone.utc)
    entry1 = ViolationReviewEntry(
        id="1",
        document_id="doc1",
        violation_type="Brady Violation",
        severity="high",
        status="approved",
        description="Prosecutor withheld exculpatory evidence from defense",
        confidence=0.9,
        detected_time=now,
    )
    entry2 = ViolationReviewEntry(
        id="2",
        document_id="doc2",
        violation_type="4th Amendment Violation",
        severity="high",
        status="approved",
        description="Police conducted an unreasonable search without a warrant",
        confidence=0.85,
        detected_time=now,
    )
    db.insert_violation(entry1)
    db.insert_violation(entry2)

    clf = ViolationClassifier(model_path=str(tmp_path / "model.joblib"))
    clf.train_from_review_db(db)

    text = "The prosecutor withheld exculpatory evidence from the defense team."
    spans = clf.detect_violations(text, threshold=0.5)
    assert any(s.violation_type == "Brady Violation" for s in spans)
