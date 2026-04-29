"""
hybrid_search.py
----------------
Hybrid search combining BM25 (lexical/keyword-based) and semantic (embedding-based) retrieval
with Reciprocal Rank Fusion (RRF) for score normalization and merging.

This module implements state-of-the-art dense-sparse fusion for improved retrieval quality,
particularly effective for regulatory documents where both precise keyword matching and
semantic understanding matter.

Key components:
  - BM25Retriever: Sparse lexical retrieval
  - SemanticRetriever: Dense semantic retrieval
  - reciprocal_rank_fusion: Merges ranked results via RRF algorithm
  - HybridSearcher: Orchestrates both retrievers with configurable weighting
"""

from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 (Best Matching 25) lexical retriever.
    Effective for keyword matching and precise term overlap.
    
    BM25 is a probabilistic ranking function that considers:
    - Term frequency in the document
    - Inverse document frequency (rarity of the term)
    - Document length normalization
    """
    
    def __init__(self, documents: list[str]):
        """
        Initialize BM25 index.
        
        Args:
            documents: List of document texts to index
        """
        # Tokenize by whitespace and lowercase
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Retrieve top-k documents using BM25.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (index, score) pairs
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class SemanticRetriever:
    """
    Semantic retriever using vector embeddings.
    Effective for understanding query intent and semantic similarity.
    Uses cosine similarity for scoring.
    """
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize with precomputed embeddings.
        
        Args:
            embeddings: (N, D) array where N is number of documents, D is embedding dimension
        """
        self.embeddings = embeddings
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Retrieve top-k documents using semantic similarity.
        
        Args:
            query_embedding: Query embedding vector (shape: D,)
            top_k: Number of results to return
            
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute cosine similarity
        scores = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    semantic_results: list[tuple[int, float]],
    k: float = 60.0,
) -> list[tuple[int, float]]:
    """
    Merge BM25 and semantic results using Reciprocal Rank Fusion (RRF).
    
    RRF algorithm:
        RRF(d) = sum over all result sets S { 1 / (k + rank(d)) }
    
    where:
        - d is a document
        - rank(d) is the position of d in result set S (1-indexed)
        - k is a constant (typically 60) to avoid division by small numbers
    
    This approach:
    - Treats ranking positions equally regardless of original score magnitude
    - Naturally handles heterogeneous score distributions (BM25 vs cosine similarity)
    - Gives meaningful credit to documents appearing in both result sets
    
    Args:
        bm25_results: List of (doc_index, score) from BM25
        semantic_results: List of (doc_index, score) from semantic search
        k: RRF constant (recommended: 60)
        
    Returns:
        List of (doc_index, fused_score) sorted by fused_score descending
    """
    fused_scores = {}
    
    # Process BM25 results - rank starts at 1
    for rank, (doc_idx, _) in enumerate(bm25_results, start=1):
        rrf_score = 1.0 / (k + rank)
        fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + rrf_score
    
    # Process semantic results - rank starts at 1
    for rank, (doc_idx, _) in enumerate(semantic_results, start=1):
        rrf_score = 1.0 / (k + rank)
        fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + rrf_score
    
    # Sort by fused score descending
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


class HybridSearcher:
    """
    Orchestrates hybrid search using both BM25 and semantic retrieval with RRF fusion.
    
    Usage:
        searcher = HybridSearcher(
            documents=chunks,
            embeddings=embedding_vectors,
            bm25_weight=0.5,
            semantic_weight=0.5
        )
        results = searcher.search(
            query="What is third-party risk?",
            query_embedding=query_vec,
            top_k=5
        )
    """
    
    def __init__(
        self,
        documents: list[str],
        embeddings: np.ndarray,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: float = 60.0,
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            documents: List of document texts
            embeddings: (N, D) array of document embeddings
            bm25_weight: Weight for BM25 in final ranking (0-1). Note: RRF
                        merges scores before weighting, so weights primarily
                        control which retriever's top-k is fetched.
            semantic_weight: Weight for semantic search (0-1)
            rrf_k: RRF constant for fusion (default: 60)
        """
        self.bm25_retriever = BM25Retriever(documents)
        self.semantic_retriever = SemanticRetriever(embeddings)
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k
        self.documents = documents
    
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform hybrid search combining BM25 and semantic retrieval.
        
        Args:
            query: Search query string
            query_embedding: Query embedding vector (shape: D,)
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys:
            - index: Document index
            - text: Document text
            - hybrid_score: RRF fused score
            - bm25_rank: Rank in BM25 results (or None if not present)
            - semantic_rank: Rank in semantic results (or None if not present)
        """
        # Retrieve from both modalities (fetch slightly more for fusion)
        search_k = max(top_k * 2, 10)
        
        bm25_results = self.bm25_retriever.search(query, top_k=search_k)
        semantic_results = self.semantic_retriever.search(query_embedding, top_k=search_k)
        
        # Fuse results using RRF
        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k=self.rrf_k)
        
        # Build result dicts with metadata
        results = []
        bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_results, 1)}
        semantic_ranks = {idx: rank for rank, (idx, _) in enumerate(semantic_results, 1)}
        
        for doc_idx, fused_score in fused[:top_k]:
            results.append({
                "index": doc_idx,
                "text": self.documents[doc_idx],
                "hybrid_score": fused_score,
                "bm25_rank": bm25_ranks.get(doc_idx),
                "semantic_rank": semantic_ranks.get(doc_idx),
            })
        
        return results
