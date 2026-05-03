"""
evaluate_rag.py
---------------
Command-line tool to evaluate the RAG system against benchmark queries.

Usage:
    python evaluate_rag.py [--method semantic|hybrid] [--output results/]
    
Examples:
    # Evaluate using hybrid search (recommended)
    python evaluate_rag.py --method hybrid
    
    # Evaluate using semantic-only search
    python evaluate_rag.py --method semantic
    
    # Save results to custom directory
    python evaluate_rag.py --method hybrid --output ~/eval_results/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_loader import load_all_documents
from rag_pipeline import VectorStore, HybridVectorStore, hybrid_retrieve, ingest_document
from evaluation import EvaluationDataset, RetrieverEvaluator


def setup_vector_store(store_type: str = "hybrid") -> VectorStore:
    """
    Initialize and populate the vector store.
    
    Args:
        store_type: Either 'semantic' or 'hybrid'
        
    Returns:
        Initialized VectorStore
    """
    if store_type == "hybrid":
        store = HybridVectorStore()
    else:
        store = VectorStore()

    if store.is_empty:
        print("Vector store is empty. Indexing documents...")
        docs = load_all_documents()
        for doc in docs:
            ingest_document(doc["text"], doc["name"], store)
        print(f"Indexed {len(store.chunks)} chunks.\n")
    else:
        print(f"Loaded {len(store.chunks)} indexed chunks.\n")

    return store


def evaluate_rag(
    store: VectorStore,
    dataset: EvaluationDataset,
    store_type: str = "semantic",
    use_hybrid: bool = False,
) -> RetrieverEvaluator:
    """
    Run evaluation on all queries in the dataset.
    
    Args:
        store: Initialized vector store
        dataset: Evaluation dataset with queries
        store_type: Type of store ('semantic' or 'hybrid')
        use_hybrid: Whether to use hybrid search
        
    Returns:
        RetrieverEvaluator with results
    """
    evaluator = RetrieverEvaluator(dataset)

    print("Evaluating RAG system...")
    print("─" * 70)

    total = len(dataset.queries)
    for idx, (query_id, query) in enumerate(dataset.queries.items(), 1):
        print(f"[{idx}/{total}] {query_id}: {query.query_text[:50]}...", end=" → ")

        # Retrieve documents
        try:
            results = hybrid_retrieve(
                query.query_text,
                store,
                top_k=5,
                use_hybrid=use_hybrid,
            )
            retrieved_sources = [r["source"] for r in results]

            # Evaluate
            metrics = evaluator.evaluate_query(query_id, retrieved_sources)

            # Print quick feedback
            status = "✓" if metrics.mrr > 0 else "✗"
            print(f"{status} P@5: {metrics.precision_at_5:.2f}, NDCG: {metrics.ndcg_at_5:.2f}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("─" * 70 + "\n")
    return evaluator


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Export it before running.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Evaluate OSFI RAG system on benchmark queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--method",
        choices=["semantic", "hybrid"],
        default="hybrid",
        help="Retrieval method to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation"),
        help="Output directory for results and report",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/evaluation_benchmark.json"),
        help="Benchmark dataset file",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load evaluation dataset
    print(f"Loading benchmark from: {args.benchmark}")
    if not args.benchmark.exists():
        print(f"❌ Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    dataset = EvaluationDataset(args.benchmark)
    print(f"Loaded {len(dataset.queries)} evaluation queries.\n")

    # Setup vector store
    print(f"Setting up {args.method} vector store...")
    store = setup_vector_store(store_type=args.method)

    # Run evaluation
    use_hybrid = args.method == "hybrid"
    evaluator = evaluate_rag(
        store,
        dataset,
        store_type=args.method,
        use_hybrid=use_hybrid,
    )

    # Print report
    evaluator.print_report()

    # Save results
    metrics_file = args.output / f"metrics_{args.method}.json"
    evaluator.save_results(metrics_file)
    print(f"Saved detailed results: {metrics_file}")

    # Save aggregated metrics
    agg_metrics = evaluator.aggregate_metrics()
    agg_file = args.output / f"aggregated_{args.method}.json"
    with open(agg_file, "w") as f:
        json.dump(
            {
                "method": args.method,
                "metrics": {
                    "mean_precision_at_1": agg_metrics.mean_precision_at_1,
                    "mean_precision_at_3": agg_metrics.mean_precision_at_3,
                    "mean_precision_at_5": agg_metrics.mean_precision_at_5,
                    "mean_recall_at_1": agg_metrics.mean_recall_at_1,
                    "mean_recall_at_3": agg_metrics.mean_recall_at_3,
                    "mean_recall_at_5": agg_metrics.mean_recall_at_5,
                    "mean_ndcg_at_5": agg_metrics.mean_ndcg_at_5,
                    "mean_mrr": agg_metrics.mean_mrr,
                    "mean_map_at_5": agg_metrics.mean_map_at_5,
                    "num_queries": agg_metrics.num_queries,
                    "num_with_relevant_results": agg_metrics.num_with_relevant_results,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved aggregated metrics: {agg_file}")

    # Comparison report if both methods exist
    semantic_file = args.output / "aggregated_semantic.json"
    hybrid_file = args.output / "aggregated_hybrid.json"
    if semantic_file.exists() and hybrid_file.exists():
        print("\n" + "="*70)
        print("COMPARISON: SEMANTIC vs HYBRID".center(70))
        print("="*70)
        with open(semantic_file) as f:
            sem = json.load(f)["metrics"]
        with open(hybrid_file) as f:
            hyb = json.load(f)["metrics"]

        print(f"\n{'Metric':<20} {'Semantic':<15} {'Hybrid':<15} {'Improvement':<15}")
        print("─" * 65)
        metrics_keys = [
            ("P@1", "mean_precision_at_1"),
            ("P@3", "mean_precision_at_3"),
            ("P@5", "mean_precision_at_5"),
            ("R@1", "mean_recall_at_1"),
            ("R@3", "mean_recall_at_3"),
            ("R@5", "mean_recall_at_5"),
            ("NDCG@5", "mean_ndcg_at_5"),
            ("MRR", "mean_mrr"),
            ("MAP@5", "mean_map_at_5"),
        ]
        for label, key in metrics_keys:
            s_val = sem[key]
            h_val = hyb[key]
            improvement = ((h_val - s_val) / s_val * 100) if s_val > 0 else 0
            print(
                f"{label:<20} {s_val:<15.3f} {h_val:<15.3f} {improvement:+.1f}%"
            )
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
