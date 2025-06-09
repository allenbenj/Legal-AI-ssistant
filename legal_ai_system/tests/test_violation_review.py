import uuid
from datetime import datetime
from legal_ai_system.services.violation_review import ViolationReviewDB, ViolationReviewEntry


def test_insert_and_fetch(tmp_path):
    db = ViolationReviewDB(db_path=str(tmp_path / "test.db"))
    entry = ViolationReviewEntry(
        id=str(uuid.uuid4()),
        document_id="doc1",
        violation_type="Data Privacy",
        severity="HIGH",
        status="PENDING",
        description="Sample",
        confidence=0.8,
        detected_time=datetime.now(),
    )
    assert db.insert_violation(entry)
    rows = db.fetch_violations()
    assert len(rows) == 1
    assert rows[0].document_id == "doc1"


def test_update_status(tmp_path):
    db = ViolationReviewDB(db_path=str(tmp_path / "test.db"))
    vid = str(uuid.uuid4())
    entry = ViolationReviewEntry(
        id=vid,
        document_id="doc2",
        violation_type="Compliance",
        severity="MEDIUM",
        status="PENDING",
        description="Another",
        confidence=0.9,
        detected_time=datetime.now(),
    )
    db.insert_violation(entry)
    assert db.update_violation_status(vid, "APPROVED", reviewed_by="tester")
    rec = db.fetch_violations()[0]
    assert rec.status == "APPROVED"
    assert rec.reviewed_by == "tester"
