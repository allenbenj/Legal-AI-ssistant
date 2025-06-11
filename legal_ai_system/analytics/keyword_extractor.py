"""Keyword extraction utilities.

This module provides a lightweight TF-IDF based keyword extractor.  The
``extract_keywords`` function returns the top scoring terms from a single
document.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Return the top ``top_k`` keywords ranked by TF-IDF score.

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
    # Guard against empty input or invalid ``top_k``
    if not text or not text.strip() or top_k <= 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    features = vectorizer.get_feature_names_out()

    # Pair each term with its TF-IDF score and sort by score descending
    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
