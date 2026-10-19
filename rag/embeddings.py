"""Turn text chunks into numeric vectors (TF-IDF)."""
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """
    Sparse embeddings via TF-IDF. No API key needed, runs fully offline.

    In production you would swap this for dense embeddings
    (e.g. sentence-transformers) - the rest of the pipeline is unchanged.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=20000,
        )
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts):
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        return self.vectorizer.transform(texts)
