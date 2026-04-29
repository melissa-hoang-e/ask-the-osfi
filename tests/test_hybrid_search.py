"""
test_hybrid_search.py
---------------------
Tests for the hybrid search module (BM25 + semantic + RRF).
"""

from hybrid_search import (
    BM25Retriever,
    SemanticRetriever,
    reciprocal_rank_fusion,
    HybridSearcher,
)
import sys
from pathlib import Path
import numpy as np

# Add src to path BEFORE importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_bm25_retriever():
    """Test BM25 keyword-based retrieval."""
    documents = [
        "OSFI manages third-party risk through governance frameworks",
        "Climate risk management requires scenario analysis and stress testing",
        "Cybersecurity controls are mandatory under B-13 guideline",
        "Model risk includes validation and backtesting requirements",
        "Third-party relationships need ongoing monitoring",
    ]

    retriever = BM25Retriever(documents)

    # Query about third-party risk
    results = retriever.search("third-party risk management", top_k=3)

    print("\n=== BM25 Retriever Test ===")
    print(f"Query: 'third-party risk management'")
    print(f"Results: {len(results)} documents")
    for idx, score in results:
        print(f"  [{idx}] Score: {score:.3f} - {documents[idx][:60]}...")

    assert len(results) > 0
    assert results[0][1] > 0  # First result should have positive score
    print("✓ BM25 retriever test passed")


def test_semantic_retriever():
    """Test semantic vector-based retrieval."""
    documents = [
        "OSFI manages third-party risk",
        "Climate risk drives capital requirements",
        "The bank failed cybersecurity audits",
        "Model validation prevents regulatory breaches",
        "Third-party outsourcing needs contracts",
    ]

    # Create mock embeddings (5 docs, 10-dim vectors)
    embeddings = np.random.randn(5, 10).astype(np.float32)
    # Make first two embeddings slightly similar to each other
    embeddings[1] = embeddings[0] + 0.1 * np.random.randn(10)

    retriever = SemanticRetriever(embeddings)

    # Query embedding (similar to first doc)
    query_emb = embeddings[0] + 0.05 * np.random.randn(10)
    results = retriever.search(query_emb, top_k=3)

    print("\n=== Semantic Retriever Test ===")
    print(f"Query embedding similarity search (top-3)")
    print(f"Results: {len(results)} documents")
    for idx, score in results:
        print(f"  [{idx}] Cosine similarity: {score:.3f}")

    assert len(results) > 0
    assert results[0][1] < 1.1  # Cosine similarity should be <= 1.0
    print("✓ Semantic retriever test passed")


def test_reciprocal_rank_fusion():
    """Test RRF fusion algorithm."""
    # Simulate BM25 and semantic results with potentially different rankings
    bm25_results = [
        (0, 10.5),  # Doc 0 ranks 1st in BM25
        (2, 8.3),   # Doc 2 ranks 2nd in BM25
        (4, 6.1),   # Doc 4 ranks 3rd in BM25
    ]

    semantic_results = [
        (1, 0.92),  # Doc 1 ranks 1st in semantic
        (0, 0.88),  # Doc 0 ranks 2nd in semantic
        (3, 0.85),  # Doc 3 ranks 3rd in semantic
    ]

    fused = reciprocal_rank_fusion(bm25_results, semantic_results, k=60.0)

    print("\n=== Reciprocal Rank Fusion Test ===")
    print("BM25 rankings:", [idx for idx, _ in bm25_results])
    print("Semantic rankings:", [idx for idx, _ in semantic_results])
    print(f"\nFused results (top-5):")
    for idx, rrf_score in fused[:5]:
        print(f"  Doc {idx}: RRF score = {rrf_score:.4f}")

    # Doc 0 should rank high since it appears in both
    doc_0_position = next(
        (i for i, (idx, _) in enumerate(fused) if idx == 0), -1)
    assert doc_0_position < 3, "Doc appearing in both lists should rank in top-3"
    print("✓ RRF fusion test passed")


def test_hybrid_searcher():
    """Test end-to-end hybrid search."""
    documents = [
        "Third-party risk management is critical under OSFI Guideline B-10",
        "Climate risk scenarios must include stress testing according to B-15",
        "Cybersecurity requirements in guideline B-13 mandate encryption",
        "Model risk validation requires independent review and backtesting",
        "Third-party service providers must maintain security standards",
    ]

    # Create mock embeddings
    embeddings = np.random.randn(5, 8).astype(np.float32)

    # Initialize searcher
    searcher = HybridSearcher(documents, embeddings)

    # Perform hybrid search
    query = "third-party risk management cybersecurity"
    query_emb = np.random.randn(8).astype(np.float32)

    results = searcher.search(query, query_emb, top_k=3)

    print("\n=== Hybrid Searcher Test ===")
    print(f"Query: '{query}'")
    print(f"Results: {len(results)} documents")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Score: {result['hybrid_score']:.4f}")
        print(f"     BM25 rank: {result['bm25_rank']}")
        print(f"     Semantic rank: {result['semantic_rank']}")
        print(f"     Text: {result['text'][:70]}...")

    assert len(results) <= 3
    assert all('hybrid_score' in r for r in results)
    assert all('bm25_rank' in r or 'semantic_rank' in r for r in results)
    print("\n✓ Hybrid searcher test passed")


if __name__ == "__main__":
    test_bm25_retriever()
    test_semantic_retriever()
    test_reciprocal_rank_fusion()
    test_hybrid_searcher()
    print("\n" + "="*50)
    print("✓ All hybrid search tests passed!")
    print("="*50)
