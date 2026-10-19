"""CLI for the RAG document-search pipeline."""
import argparse
from rag import RagPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG document search, built from scratch.")
    parser.add_argument("--docs", default="docs", help="Folder with .txt/.md documents")
    parser.add_argument("--query", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve")
    parser.add_argument("--self-test", action="store_true", help="Run a built-in smoke test")
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.docs)
        return

    rag = RagPipeline(args.docs)
    print(f"Indexed {rag.num_chunks} chunks from '{args.docs}'.")
    if args.query:
        print("\n" + rag.answer(args.query, top_k=args.top_k))
        return

    print("Ask questions (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print("\n" + rag.answer(q, top_k=args.top_k))


def run_self_test(docs_dir):
    rag = RagPipeline(docs_dir)
    assert rag.num_chunks > 0, "no chunks indexed"
    hits = rag.retrieve("what is retrieval augmented generation", top_k=2)
    assert hits, "retrieval returned nothing"
    assert hits[0][0] > 0, "top score should be positive"
    print(f"SELF-TEST OK: {rag.num_chunks} chunks, top score {hits[0][0]:.3f} "
          f"from {hits[0][1]['doc']}")


if __name__ == "__main__":
    main()
