"""
evaluation.py
--------------
Evaluation metrics for RAG system: retrieval quality, ranking quality, and answer quality.

Metrics included:
- Precision@K: % of top-K results that are relevant
- Recall: % of all relevant documents that were retrieved
- NDCG: Normalized Discounted Cumulative Gain (ranking quality)
- MRR: Mean Reciprocal Rank (position of first relevant result)
- MAP: Mean Average Precision (overall ranking quality)
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import numpy as np


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class EvaluationQuery:
    """A query with ground truth relevant documents."""
    query_id: str
    query_text: str
    relevant_docs: list[str]  # List of source names that are relevant
    description: Optional[str] = None  # Query context/category


@dataclass
class RetrievalMetrics:
    """Metrics for a single query's retrieval."""
    query_id: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    ndcg_at_5: float
    mrr: float
    map_at_5: float
    num_relevant: int
    num_retrieved: int


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all queries."""
    mean_precision_at_1: float
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_ndcg_at_5: float
    mean_mrr: float
    mean_map_at_5: float
    num_queries: int
    num_with_relevant_results: int = 0


# ── Metrics Computation ───────────────────────────────────────────────────────

def precision_at_k(relevant_set: set, retrieved_set: set, k: int) -> float:
    """
    Precision@K: fraction of top-k results that are relevant.
    
    Args:
        relevant_set: Set of relevant document identifiers
        retrieved_set: Set of retrieved document identifiers (ordered)
        k: Position threshold
        
    Returns:
        Precision@K (0-1)
    """
    if k == 0:
        return 0.0
    # Assume retrieved_set is already limited to top-k
    hits = len(relevant_set & retrieved_set)
    return hits / min(k, len(retrieved_set)) if retrieved_set else 0.0


def recall_at_k(relevant_set: set, retrieved_set: set, k: int) -> float:
    """
    Recall@K: fraction of all relevant documents that appear in top-k.
    
    Args:
        relevant_set: Set of relevant document identifiers
        retrieved_set: Set of retrieved document identifiers (ordered)
        k: Position threshold
        
    Returns:
        Recall@K (0-1)
    """
    if not relevant_set:
        return 0.0
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def ndcg_at_k(
    relevant_set: set,
    retrieved_ranking: list,
    k: int = 5,
) -> float:
    """
    Normalized Discounted Cumulative Gain@K: evaluates ranking quality.
    
    Penalizes relevant items appearing later in the ranking.
    
    Args:
        relevant_set: Set of relevant document identifiers
        retrieved_ranking: Ordered list of retrieved documents (doc_id or source name)
        k: Position threshold
        
    Returns:
        NDCG@K (0-1)
    """
    # DCG: sum of (1 / log2(position+1)) for each relevant item at position
    dcg = 0.0
    for i, doc in enumerate(retrieved_ranking[:k]):
        if doc in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # position i starts at 0, so position 1 = log2(2)

    # Ideal DCG: assume perfect ranking (all relevant items at top)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevant_set: set, retrieved_ranking: list) -> float:
    """
    Mean Reciprocal Rank: 1 / position of first relevant item.
    
    Args:
        relevant_set: Set of relevant document identifiers
        retrieved_ranking: Ordered list of retrieved documents
        
    Returns:
        MRR (0-1, where 1.0 = first result is relevant)
    """
    for i, doc in enumerate(retrieved_ranking):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    relevant_set: set,
    retrieved_ranking: list,
    k: int = 5,
) -> float:
    """
    Average Precision@K: average of precision computed at each relevant position.
    
    Combines precision and recall into one metric.
    
    Args:
        relevant_set: Set of relevant document identifiers
        retrieved_ranking: Ordered list of retrieved documents
        k: Position threshold
        
    Returns:
        AP@K (0-1)
    """
    if not relevant_set:
        return 0.0

    score = 0.0
    num_hits = 0

    for i, doc in enumerate(retrieved_ranking[:k]):
        if doc in relevant_set:
            num_hits += 1
            precision = num_hits / (i + 1)
            score += precision

    return score / len(relevant_set)


# ── Query Dataset Management ──────────────────────────────────────────────────

class EvaluationDataset:
    """Manages evaluation queries with ground truth relevance judgments."""

    def __init__(self, dataset_file: Optional[Path] = None):
        """
        Args:
            dataset_file: Path to JSON file with evaluation queries
        """
        self.queries: dict[str, EvaluationQuery] = {}
        if dataset_file and dataset_file.exists():
            self._load(dataset_file)

    def add_query(
        self,
        query_id: str,
        query_text: str,
        relevant_docs: list[str],
        description: Optional[str] = None,
    ):
        """Add a query with ground truth relevant documents."""
        self.queries[query_id] = EvaluationQuery(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            description=description,
        )

    def _load(self, path: Path):
        """Load queries from JSON file."""
        with open(path) as f:
            data = json.load(f)
        for q in data.get("queries", []):
            self.add_query(
                q["query_id"],
                q["query_text"],
                q["relevant_docs"],
                q.get("description"),
            )

    def save(self, path: Path):
        """Save queries to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        queries_list = [asdict(q) for q in self.queries.values()]
        with open(path, "w") as f:
            json.dump({"queries": queries_list}, f, indent=2)

    def get_query(self, query_id: str) -> Optional[EvaluationQuery]:
        """Retrieve a query by ID."""
        return self.queries.get(query_id)


# ── Evaluation Runner ─────────────────────────────────────────────────────────

class RetrieverEvaluator:
    """Evaluates retrieval quality on a benchmark dataset."""

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset
        self.results: dict[str, RetrievalMetrics] = {}

    def evaluate_query(
        self,
        query_id: str,
        retrieved_sources: list[str],  # Ordered list of source names returned
    ) -> RetrievalMetrics:
        """
        Evaluate a single query's retrieval results.
        
        Args:
            query_id: ID of the query (must exist in dataset)
            retrieved_sources: Ordered list of source names retrieved by the system
            
        Returns:
            RetrievalMetrics object
        """
        query = self.dataset.get_query(query_id)
        if not query:
            raise ValueError(f"Query {query_id} not found in dataset")

        relevant_set = set(query.relevant_docs)
        retrieved_set = set(retrieved_sources)

        # Compute all metrics
        p_at_1 = precision_at_k(relevant_set, retrieved_set, k=1)
        p_at_3 = precision_at_k(relevant_set, retrieved_set, k=3)
        p_at_5 = precision_at_k(relevant_set, retrieved_set, k=5)

        r_at_1 = recall_at_k(relevant_set, retrieved_set, k=1)
        r_at_3 = recall_at_k(relevant_set, retrieved_set, k=3)
        r_at_5 = recall_at_k(relevant_set, retrieved_set, k=5)

        ndcg = ndcg_at_k(relevant_set, retrieved_sources, k=5)
        mrr_score = mrr(relevant_set, retrieved_sources)
        ap = average_precision(relevant_set, retrieved_sources, k=5)

        metrics = RetrievalMetrics(
            query_id=query_id,
            precision_at_1=p_at_1,
            precision_at_3=p_at_3,
            precision_at_5=p_at_5,
            recall_at_1=r_at_1,
            recall_at_3=r_at_3,
            recall_at_5=r_at_5,
            ndcg_at_5=ndcg,
            mrr=mrr_score,
            map_at_5=ap,
            num_relevant=len(relevant_set),
            num_retrieved=len(retrieved_sources),
        )

        self.results[query_id] = metrics
        return metrics

    def aggregate_metrics(self) -> AggregatedMetrics:
        """Aggregate metrics across all evaluated queries."""
        if not self.results:
            raise ValueError("No evaluation results yet. Run evaluate_query() first.")

        n = len(self.results)
        metrics = AggregatedMetrics(
            mean_precision_at_1=np.mean([m.precision_at_1 for m in self.results.values()]),
            mean_precision_at_3=np.mean([m.precision_at_3 for m in self.results.values()]),
            mean_precision_at_5=np.mean([m.precision_at_5 for m in self.results.values()]),
            mean_recall_at_1=np.mean([m.recall_at_1 for m in self.results.values()]),
            mean_recall_at_3=np.mean([m.recall_at_3 for m in self.results.values()]),
            mean_recall_at_5=np.mean([m.recall_at_5 for m in self.results.values()]),
            mean_ndcg_at_5=np.mean([m.ndcg_at_5 for m in self.results.values()]),
            mean_mrr=np.mean([m.mrr for m in self.results.values()]),
            mean_map_at_5=np.mean([m.map_at_5 for m in self.results.values()]),
            num_queries=n,
            num_with_relevant_results=sum(
                1 for m in self.results.values() if m.mrr > 0
            ),
        )
        return metrics

    def print_report(self):
        """Print human-readable evaluation report."""
        agg = self.aggregate_metrics()
        print("\n" + "="*70)
        print("RETRIEVAL EVALUATION REPORT".center(70))
        print("="*70)
        print(f"\nTotal queries evaluated: {agg.num_queries}")
        print(f"Queries with ≥1 relevant result: {agg.num_with_relevant_results}/{agg.num_queries}")

        print("\n" + "─"*70)
        print("PRECISION@K (% of top-k results that are relevant)")
        print("─"*70)
        print(f"  P@1: {agg.mean_precision_at_1:.3f}")
        print(f"  P@3: {agg.mean_precision_at_3:.3f}")
        print(f"  P@5: {agg.mean_precision_at_5:.3f}")

        print("\n" + "─"*70)
        print("RECALL@K (% of all relevant docs found in top-k)")
        print("─"*70)
        print(f"  R@1: {agg.mean_recall_at_1:.3f}")
        print(f"  R@3: {agg.mean_recall_at_3:.3f}")
        print(f"  R@5: {agg.mean_recall_at_5:.3f}")

        print("\n" + "─"*70)
        print("RANKING QUALITY")
        print("─"*70)
        print(f"  NDCG@5: {agg.mean_ndcg_at_5:.3f}  (normalized ranking quality, 0-1)")
        print(f"  MRR:    {agg.mean_mrr:.3f}   (position of first relevant, 0-1)")
        print(f"  MAP@5:  {agg.mean_map_at_5:.3f}   (average precision, 0-1)")

        print("\n" + "="*70 + "\n")

    def save_results(self, output_file: Path):
        """Save per-query results to JSON."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results = {
            query_id: asdict(metrics)
            for query_id, metrics in self.results.items()
        }
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
