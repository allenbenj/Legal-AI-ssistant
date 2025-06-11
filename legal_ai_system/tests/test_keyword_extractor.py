import pytest

from legal_ai_system.analytics.keyword_extractor import extract_keywords


@pytest.mark.unit
def test_extract_keywords_ranks_terms():
    text = (
        "Contract law governs contracts, agreements and obligations. "
        "A contract is a legally binding agreement. "
        "In contract law, the agreement is enforceable."
    )
    keywords = extract_keywords(text, top_k=3)
    assert len(keywords) == 3
    assert keywords[0][0] == "contract"
    assert keywords[0][1] >= keywords[1][1] >= keywords[2][1]
    assert any(k == "agreement" for k, _ in keywords)


@pytest.mark.unit
def test_extract_keywords_empty_text_returns_empty_list():
    assert extract_keywords("", top_k=5) == []
    assert extract_keywords("   ", top_k=2) == []


@pytest.mark.unit
def test_extract_keywords_respects_top_k():
    text = "alpha beta gamma delta epsilon"
    keywords = extract_keywords(text, top_k=2)
    assert len(keywords) == 2
