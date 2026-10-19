"""Load raw documents from a folder."""
import os

SUPPORTED = (".txt", ".md")


def load_documents(docs_dir):
    """Return a list of (doc_name, text) for every supported file in docs_dir."""
    documents = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.lower().endswith(SUPPORTED):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            documents.append((fname, text))
    if not documents:
        raise ValueError(f"No .txt/.md documents found in {docs_dir}")
    return documents
