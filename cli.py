"""
cli.py
------
Command-line interface for the OSFI RAG pipeline.

Usage:
    # Index all OSFI documents
    python cli.py index

    # Ask a question
    python cli.py ask "What are the requirements for model risk management?"

    # Index a custom file then ask
    python cli.py index --file path/to/policy.pdf
    python cli.py ask "What does our policy say about outsourcing?"

    # Evaluate retrieval quality
    python cli.py evaluate --method hybrid
    python cli.py evaluate --method semantic
"""

from document_loader import load_all_documents, load_custom_document
from rag_pipeline import VectorStore, ask, ingest_document
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_index(args):
    store = VectorStore()

    if args.file:
        print(f"Loading custom file: {args.file}")
        doc = load_custom_document(args.file)
        if doc:
            ingest_document(doc["text"], doc["name"], store)
            print("Custom document indexed.")
        else:
            print("Failed to load document.")
    else:
        print("Loading all OSFI guidelines...")
        docs = load_all_documents()
        for doc in docs:
            ingest_document(doc["text"], doc["name"], store)
        print(f"\nDone. {len(store.chunks)} total chunks indexed.")


def cmd_ask(args):
    store = VectorStore()

    if store.is_empty:
        print("No documents indexed. Run: python cli.py index")
        return

    query = " ".join(args.query)
    print(f"\nQuestion: {query}\n{'─' * 60}")

    result = ask(query, store)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")

    if args.verbose:
        print(f"\n{'─' * 60}\nRetrieved passages:")
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            print(f"\n[{i}] {chunk['source']} (score: {chunk['score']:.3f})")
            print(chunk["text"][:300] + "...")


def cmd_evaluate(args):
    """Run evaluation on benchmark queries."""
    # Import here to avoid circular dependencies
    from evaluation import EvaluationDataset, RetrieverEvaluator
    from rag_pipeline import HybridVectorStore, hybrid_retrieve, ingest_document

    # Setup vector store
    if args.method == "hybrid":
        store = HybridVectorStore()
    else:
        store = VectorStore()

    if store.is_empty:
        print("Loading all OSFI guidelines...")
        docs = load_all_documents()
        for doc in docs:
            ingest_document(doc["text"], doc["name"], store)
        print(f"Indexed {len(store.chunks)} chunks.\n")

    # Load benchmark
    benchmark_path = Path(args.benchmark) if args.benchmark else Path("data/evaluation_benchmark.json")
    if not benchmark_path.exists():
        print(f"❌ Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    dataset = EvaluationDataset(benchmark_path)
    print(f"Loaded {len(dataset.queries)} evaluation queries.\n")

    # Evaluate
    evaluator = RetrieverEvaluator(dataset)
    use_hybrid = args.method == "hybrid"

    print("Evaluating RAG system...")
    print("─" * 70)

    total = len(dataset.queries)
    for idx, (query_id, query) in enumerate(dataset.queries.items(), 1):
        print(f"[{idx}/{total}] {query_id}: {query.query_text[:50]}...", end=" → ")

        try:
            results = hybrid_retrieve(
                query.query_text,
                store,
                top_k=5,
                use_hybrid=use_hybrid,
            )
            retrieved_sources = [r["source"] for r in results]
            metrics = evaluator.evaluate_query(query_id, retrieved_sources)

            status = "✓" if metrics.mrr > 0 else "✗"
            print(f"{status} P@5: {metrics.precision_at_5:.2f}, NDCG: {metrics.ndcg_at_5:.2f}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("─" * 70 + "\n")

    # Print report
    evaluator.print_report()

    # Save results
    output_dir = Path(args.output or "results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / f"metrics_{args.method}.json"
    evaluator.save_results(metrics_file)
    print(f"Saved detailed results: {metrics_file}")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Export it before running.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="OSFI RAG — Command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # index subcommand
    p_index = subparsers.add_parser("index", help="Index OSFI documents")
    p_index.add_argument(
        "--file", help="Path to a custom PDF or TXT file to index")

    # ask subcommand
    p_ask = subparsers.add_parser("ask", help="Ask a question")
    p_ask.add_argument("query", nargs="+",
                       help="Your question (wrap in quotes)")
    p_ask.add_argument("-v", "--verbose", action="store_true",
                       help="Show retrieved passages")

    # evaluate subcommand
    p_eval = subparsers.add_parser("evaluate", help="Evaluate RAG on benchmark")
    p_eval.add_argument(
        "--method",
        choices=["semantic", "hybrid"],
        default="hybrid",
        help="Retrieval method to evaluate",
    )
    p_eval.add_argument(
        "--benchmark",
        help="Path to benchmark file (default: data/evaluation_benchmark.json)",
    )
    p_eval.add_argument(
        "--output",
        help="Output directory for results",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
