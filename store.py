"""In-memory vector store with cosine-similarity search."""
import numpy as np


class VectorStore:
    """Stores chunk vectors + metadata, retrieves top-k by cosine similarity."""

    def __init__(self):
        self.vectors = None      # 2D numpy array, L2-normalized rows
        self.metadatas = []      # parallel list of {"doc": name, "chunk": text}

    def add(self, vectors, metadatas):
        mat = np.asarray(vectors.todense(), dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self.vectors = mat if self.vectors is None else np.vstack([self.vectors, mat])
        self.metadatas.extend(metadatas)

    def search(self, query_vector, top_k=3):
        """Return [(score, metadata), ...] sorted by descending similarity."""
        q = np.asarray(query_vector.todense(), dtype=np.float64).ravel()
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.vectors @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.metadatas[i])
                for i in idx if scores[i] > 0]
