import datetime
import pytest

from legal_ai_system.services.database_manager import DatabaseManager


def test_get_analytics_data(tmp_path):
    db_file = tmp_path / "analytics.db"
    manager = DatabaseManager(str(db_file))

    # Insert documents
    manager.save_document("doc1", "a.txt", "txt", 10, {})
    manager.save_document("doc2", "b.txt", "txt", 20, {})
    manager.update_document_status("doc1", "COMPLETED", {"entities": [{"type": "PERSON"}]})
    manager.update_document_status("doc2", "COMPLETED", {"entities": [{"type": "ORG"}]})

    # Insert violations directly for analytics
    with manager._get_connection() as conn:
        conn.execute(
            "INSERT INTO violations (id, document_id, violation_type, severity, status, description, confidence, detected_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("v1", "doc1", "Data", "HIGH", "PENDING", "desc", 0.9, datetime.datetime.now()),
        )
        conn.execute(
            "INSERT INTO violations (id, document_id, violation_type, severity, status, description, confidence, detected_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("v2", "doc2", "Security", "LOW", "REVIEWED", "desc", 0.8, datetime.datetime.now()),
        )
        conn.commit()

    analytics = manager.get_analytics_data()
    assert analytics["documents"]["total"] == 2
    assert analytics["documents"]["processed"] == 2
    assert analytics["violations"]["total"] == 2
    assert analytics["violations"]["pending"] == 1

