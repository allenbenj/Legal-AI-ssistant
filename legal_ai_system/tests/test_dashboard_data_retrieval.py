import datetime

from legal_ai_system.services.database_manager import DatabaseManager
from legal_ai_system.services.violation_review import ViolationReviewEntry


def test_dashboard_data_retrieval(tmp_path):
    db_file = tmp_path / "dashboard.db"
    manager = DatabaseManager(str(db_file))

    # Insert documents with entity results
    manager.save_document("d1", "a.txt", "txt", 10, {})
    manager.save_document("d2", "b.txt", "txt", 20, {})
    manager.update_document_status(
        "d1", "COMPLETED", {"entities": [{"type": "PERSON"}, {"type": "ORG"}]}
    )
    manager.update_document_status(
        "d2", "IN_PROGRESS", {"entities": [{"type": "PERSON"}]}
    )

    # Insert violations using the review manager so they are stored correctly
    manager.violation_manager.insert_violation(
        ViolationReviewEntry(
            id="v1",
            document_id="d1",
            violation_type="Data",
            severity="HIGH",
            status="PENDING",
            description="desc",
            confidence=0.9,
            detected_time=datetime.datetime.now(),
        )
    )
    manager.violation_manager.insert_violation(
        ViolationReviewEntry(
            id="v2",
            document_id="d2",
            violation_type="Privacy",
            severity="LOW",
            status="RESOLVED",
            description="desc",
            confidence=0.8,
            detected_time=datetime.datetime.now(),
        )
    )

    # Also record violations in the main database for analytics
    with manager._get_connection() as conn:
        conn.execute(
            "INSERT INTO violations (id, document_id, violation_type, severity, status, description, confidence, detected_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "v1",
                "d1",
                "Data",
                "HIGH",
                "PENDING",
                "desc",
                0.9,
                datetime.datetime.now(),
            ),
        )
        conn.execute(
            "INSERT INTO violations (id, document_id, violation_type, severity, status, description, confidence, detected_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "v2",
                "d2",
                "Privacy",
                "LOW",
                "RESOLVED",
                "desc",
                0.8,
                datetime.datetime.now(),
            ),
        )
        conn.commit()

    docs = manager.get_documents()
    assert len(docs) == 2

    # Aggregate entities from documents
    entities = []
    for doc in docs:
        entities.extend(doc.get("results", {}).get("entities", []))
    assert len(entities) == 3

    violations = manager.get_violations()
    assert len(violations) == 2
    assert len([v for v in violations if v.status == "PENDING"]) == 1

    metrics = manager.get_analytics_data()
    assert metrics["documents"]["total"] == 2
    assert metrics["violations"]["pending"] == 1
