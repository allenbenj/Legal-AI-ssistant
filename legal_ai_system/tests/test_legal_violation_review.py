"""Pytest for LegalViolationReview service."""
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from legal_ai_system.services.legal_violation_review import (
    LegalViolationReview,
    LVRActor,
    LVRRecord,
)


def test_add_and_search_violation(tmp_path: Path) -> None:
    db_path = tmp_path / "lvr.db"
    jsonl_path = tmp_path / "lvr.jsonl"
    index_path = tmp_path / "lvr.index"

    review = LegalViolationReview(
        db_path=str(db_path),
        jsonl_path=str(jsonl_path),
        faiss_index_path=str(index_path),
    )
    review.initialize()

    record = LVRRecord(
        case_id="State_v_Allen_2025",
        violation_type="Suborning Perjury",
        statute="N.C.G.S. § 14-210",
        jurisdiction="North Carolina",
        actor=LVRActor(name="ADA Freeman", role="Prosecutor"),
        related_statements=["stmt_3412"],
        harms=["False timeline admitted at trial"],
        created_by="Test",
    )

    review.add_violation(record)

    loaded = review.get_violation(record.violation_id)
    assert loaded is not None
    assert loaded.violation_id == record.violation_id

    results = review.search_similar("Suborning Perjury", k=1)
    assert results
    assert results[0].violation_id == record.violation_id

    assert review.link_memory_entry(record.violation_id, "mem_1")

