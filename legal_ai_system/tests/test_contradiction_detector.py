from legal_ai_system.tools.contradiction_detector import (
    ContradictionDetector,
    MemoryEntry,
)


def test_contradiction_detected():
    entries = [MemoryEntry(speaker="Bob", statement="It is raining", source="a")]
    detector = ContradictionDetector(entries)
    result = detector.check("Bob", "It is not raining", source="b")
    assert result["count"] == 1
    assert result["contradictions"][0]["conflict"] == "It is not raining"


def test_no_contradiction():
    entries = [MemoryEntry(speaker="Alice", statement="It is sunny", source="a")]
    detector = ContradictionDetector(entries)
    result = detector.check("Alice", "It is sunny", source="b")
    assert result["count"] == 0
    assert result["contradictions"] == []
