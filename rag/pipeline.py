"""The full RAG pipeline: load -> chunk -> embed -> retrieve -> answer."""
import os
from .loader import load_documents
from .chunker import chunk_text
from .embeddings import TfidfEmbedder
from .store import VectorStore


class RagPipeline:
    def __init__(self, docs_dir, chunk_size=500, overlap=75):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = TfidfEmbedder()
        self.store = VectorStore()
        self._build()

    def _build(self):
        texts, metadatas = [], []
        for doc_name, text in load_documents(self.docs_dir):
            for chunk in chunk_text(text, self.chunk_size, self.overlap):
                texts.append(chunk)
                metadatas.append({"doc": doc_name, "chunk": chunk})
        self.embedder.fit(texts)
        self.store.add(self.embedder.embed(texts), metadatas)
        self.num_chunks = len(texts)

    def retrieve(self, query, top_k=3):
        qv = self.embedder.embed([query])
        return self.store.search(qv, top_k=top_k)

    def answer(self, query, top_k=3):
        """
        Retrieve the most relevant chunks and compose an answer.

        Default mode is extractive: it shows the evidence with scores.
        Set the LLM_API_URL env var to a chat-completions compatible
        endpoint for generative answers (left as an exercise).
        """
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return "I could not find anything relevant in the documents."
        lines = [f"Based on {len(hits)} retrieved chunk(s):", ""]
        for score, meta in hits:
            lines.append(f"[score={score:.3f} | {meta['doc']}]")
            lines.append(meta["chunk"])
            lines.append("")
        if os.environ.get("LLM_API_URL"):
            lines.append("(Generative mode: wire your LLM here to rewrite "
                         "these chunks into a final answer.)")
        return "\n".join(lines).strip()
