# RAG from Scratch

Retrieval-Augmented Generation for document search, built from scratch in Python. No API keys, no cloud, runs fully offline.

## How it works

```
docs/*.txt ──► loader ──► chunker ──► TF-IDF embeddings ──► vector store
                                                              │
your question ──► embed ──► cosine similarity search (top-k) ──┘
                              │
                              ▼
                     answer from retrieved chunks
```

1. **Load** - read `.txt` / `.md` files from `docs/`
2. **Chunk** - sliding window (~500 chars, 75 overlap), cuts on sentence boundaries
3. **Embed** - TF-IDF vectors via scikit-learn (sparse, offline, no training needed)
4. **Retrieve** - embed the question the same way, rank chunks by cosine similarity, take top-k
5. **Answer** - show the retrieved evidence with scores (extractive mode)

## Quickstart

```bash
pip install -r requirements.txt

# ask one question
python app.py --query "what is retrieval augmented generation"

# interactive mode
python app.py

# smoke test
python app.py --self-test
```

## Project structure

```
rag-from-scratch/
├── app.py               # CLI: one-shot, interactive, and --self-test modes
├── rag/
│   ├── loader.py        # read documents from a folder
│   ├── chunker.py       # overlapping, sentence-aware chunking
│   ├── embeddings.py    # TF-IDF embedder (swap for dense embeddings later)
│   ├── store.py         # in-memory vector store + cosine search
│   └── pipeline.py      # wires it all together
├── docs/                # sample documents (RAG, chunking, vector search)
└── tests/               # smoke test
```

## What I would improve next

- **Dense embeddings** (e.g. sentence-transformers) for better paraphrase matching
- **Reranking** step on the top-k before answering
- **Generative answers**: pass retrieved chunks to an LLM instead of extractive output
- **Chunking eval**: measure retrieval hit-rate across chunk sizes
- **Persistent store**: FAISS or Chroma instead of in-memory numpy

## Why this project

Rebuilding RAG from scratch teaches the parts that matter in interviews:
chunking trade-offs, how embeddings capture meaning, why cosine similarity
works, and where hallucinations come from (hint: skipped retrieval).
