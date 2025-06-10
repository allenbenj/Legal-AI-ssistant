"""Keyword extraction utilities."""

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Return the top *top_k* keywords ranked by TF-IDF score.

    Parameters
    ----------
    text: str
        Input document text.
    top_k: int
        Number of keywords to return.

    Returns
    -------
    List[Tuple[str, float]]
        Keyword-score pairs sorted in descending order.
    """
    if not text.strip() or top_k <= 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    features = vectorizer.get_feature_names_out()
    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
