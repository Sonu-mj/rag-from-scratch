"""Smoke test: index the sample docs and check retrieval works."""
import os
from rag import RagPipeline

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def test_pipeline_retrieves_relevant_chunk():
    rag = RagPipeline(DOCS)
    assert rag.num_chunks > 0
    hits = rag.retrieve("what is retrieval augmented generation", top_k=2)
    assert hits, "expected at least one hit"
    assert hits[0][0] > 0
    assert "rag" in hits[0][1]["doc"].lower()
