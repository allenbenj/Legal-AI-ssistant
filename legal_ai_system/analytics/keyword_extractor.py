

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text: str, top_k: int = 5) -> List[Tuple[str, float]]:
<<<<<<< codex/fix-function-docstring-and-tf-idf-return
    """Return the highest ranked keywords using a TF-IDF model.

=======
>>>>>>> main
    Parameters
    ----------
    text : str
        Input document text.
<<<<<<< codex/fix-function-docstring-and-tf-idf-return
    top_k : int, optional
        Number of keywords to return, by default 5.
=======
>>>>>>> main

    Returns
    -------
    List[Tuple[str, float]]
        Keyword-score pairs sorted in descending order.
    """

<<<<<<< codex/fix-function-docstring-and-tf-idf-return
    if not text.strip():
        return []

=======
>>>>>>> main
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    features = vectorizer.get_feature_names_out()

    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
