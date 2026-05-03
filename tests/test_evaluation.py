"""
tests/test_evaluation.py
------------------------
Unit tests for the evaluation metrics module.

Run with: pytest tests/test_evaluation.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    EvaluationDataset,
    EvaluationQuery,
    RetrieverEvaluator,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
)


# ── Metric Computation Tests ──────────────────────────────────────────────────

class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_perfect_precision(self):
        """All top-k results are relevant."""
        relevant = {"doc_1", "doc_2", "doc_3"}
        retrieved = {"doc_1", "doc_2", "doc_3"}
        assert precision_at_k(relevant, retrieved, k=3) == 1.0

    def test_zero_precision(self):
        """No relevant results in top-k."""
        relevant = {"doc_1", "doc_2"}
        retrieved = {"doc_5", "doc_6"}
        assert precision_at_k(relevant, retrieved, k=2) == 0.0

    def test_partial_precision(self):
        """Some results are relevant."""
        relevant = {"doc_1", "doc_2"}
        retrieved = {"doc_1", "doc_3", "doc_4"}
        assert precision_at_k(relevant, retrieved, k=3) == 1 / 3

    def test_empty_retrieval(self):
        """No documents retrieved."""
        relevant = {"doc_1"}
        retrieved = set()
        assert precision_at_k(relevant, retrieved, k=3) == 0.0

    def test_k_larger_than_retrieved(self):
        """K larger than number of retrieved documents."""
        relevant = {"doc_1", "doc_2"}
        retrieved = {"doc_1"}
        assert precision_at_k(relevant, retrieved, k=5) == 1.0

    def test_k_equals_zero(self):
        """Edge case: k=0."""
        relevant = {"doc_1"}
        retrieved = {"doc_1"}
        assert precision_at_k(relevant, retrieved, k=0) == 0.0


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_recall(self):
        """All relevant documents are retrieved in top-k."""
        relevant = {"doc_1", "doc_2"}
        retrieved = {"doc_1", "doc_2", "doc_5"}
        assert recall_at_k(relevant, retrieved, k=3) == 1.0

    def test_zero_recall(self):
        """No relevant documents retrieved."""
        relevant = {"doc_1", "doc_2"}
        retrieved = {"doc_5", "doc_6"}
        assert recall_at_k(relevant, retrieved, k=2) == 0.0

    def test_partial_recall(self):
        """Some relevant documents retrieved."""
        relevant = {"doc_1", "doc_2", "doc_3"}
        retrieved = {"doc_1", "doc_5"}
        assert recall_at_k(relevant, retrieved, k=2) == 1 / 3

    def test_empty_relevant_set(self):
        """No relevant documents defined."""
        relevant = set()
        retrieved = {"doc_1"}
        assert recall_at_k(relevant, retrieved, k=1) == 0.0

    def test_empty_retrieval(self):
        """No documents retrieved."""
        relevant = {"doc_1", "doc_2"}
        retrieved = set()
        assert recall_at_k(relevant, retrieved, k=3) == 0.0


class TestNDCG:
    """Tests for NDCG (Normalized Discounted Cumulative Gain) metric."""

    def test_perfect_ranking(self):
        """Perfect ranking: all relevant items at top."""
        relevant = {"doc_1", "doc_2"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        ndcg = ndcg_at_k(relevant, ranking, k=3)
        assert ndcg > 0.95  # Should be near perfect

    def test_reversed_ranking(self):
        """Worst ranking: all relevant items at bottom."""
        relevant = {"doc_3"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        ndcg_worst = ndcg_at_k(relevant, ranking, k=3)

        # Compare to best ranking
        relevant = {"doc_1"}
        ranking_best = ["doc_1", "doc_2", "doc_3"]
        ndcg_best = ndcg_at_k(relevant, ranking_best, k=3)

        assert ndcg_worst < ndcg_best  # Worst ranking should score lower

    def test_no_relevant(self):
        """No relevant documents in ranking."""
        relevant = {"doc_10"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        assert ndcg_at_k(relevant, ranking, k=3) == 0.0

    def test_k_constraint(self):
        """NDCG respects k-cutoff."""
        relevant = {"doc_5"}
        ranking = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        ndcg_3 = ndcg_at_k(relevant, ranking, k=3)
        ndcg_5 = ndcg_at_k(relevant, ranking, k=5)
        assert ndcg_5 > ndcg_3  # Relevant item within top-5 should score better


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_result_relevant(self):
        """First result is relevant (best case)."""
        relevant = {"doc_1"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        assert mrr(relevant, ranking) == 1.0

    def test_second_result_relevant(self):
        """Second result is relevant."""
        relevant = {"doc_2"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        assert mrr(relevant, ranking) == 0.5

    def test_fifth_result_relevant(self):
        """Fifth result is relevant."""
        relevant = {"doc_5"}
        ranking = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        assert mrr(relevant, ranking) == 0.2

    def test_no_relevant_result(self):
        """No relevant results (worst case)."""
        relevant = {"doc_10"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        assert mrr(relevant, ranking) == 0.0

    def test_multiple_relevant_first_position(self):
        """Multiple relevant documents, first one determines MRR."""
        relevant = {"doc_1", "doc_3"}
        ranking = ["doc_2", "doc_1", "doc_3"]
        assert mrr(relevant, ranking) == 0.5  # First relevant at position 2


class TestAveragePrecision:
    """Tests for Average Precision metric."""

    def test_perfect_ranking(self):
        """All relevant items at top."""
        relevant = {"doc_1", "doc_2"}
        ranking = ["doc_1", "doc_2", "doc_3"]
        ap = average_precision(relevant, ranking, k=3)
        assert ap == 1.0

    def test_mixed_ranking(self):
        """Relevant items interspersed."""
        relevant = {"doc_1", "doc_3"}
        ranking = ["doc_1", "doc_2", "doc_3", "doc_4"]
        ap = average_precision(relevant, ranking, k=4)
        # Precision at relevant positions: 1/1=1.0 at pos 1, 2/3≈0.67 at pos 3
        # AP = (1.0 + 0.67) / 2 ≈ 0.833
        assert 0.8 < ap < 0.85

    def test_no_relevant(self):
        """No relevant documents."""
        relevant = set()
        ranking = ["doc_1", "doc_2"]
        assert average_precision(relevant, ranking, k=2) == 0.0

    def test_k_constraint(self):
        """AP respects k-cutoff."""
        relevant = {"doc_5"}
        ranking = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        ap_3 = average_precision(relevant, ranking, k=3)  # doc_5 not in top-3
        ap_5 = average_precision(relevant, ranking, k=5)  # doc_5 in top-5
        assert ap_3 == 0.0
        assert ap_5 > 0


# ── Dataset Tests ────────────────────────────────────────────────────────────

class TestEvaluationDataset:
    """Tests for EvaluationDataset class."""

    def test_add_query(self):
        """Add a query to dataset."""
        dataset = EvaluationDataset()
        dataset.add_query(
            query_id="q1",
            query_text="What is OSFI?",
            relevant_docs=["B-10", "B-13"],
            description="Test query",
        )
        assert len(dataset.queries) == 1
        assert dataset.queries["q1"].query_text == "What is OSFI?"

    def test_get_query(self):
        """Retrieve a query by ID."""
        dataset = EvaluationDataset()
        dataset.add_query("q1", "Test query", ["doc1"])
        query = dataset.get_query("q1")
        assert query.query_id == "q1"

    def test_get_nonexistent_query(self):
        """Attempt to retrieve nonexistent query."""
        dataset = EvaluationDataset()
        assert dataset.get_query("q99") is None

    def test_save_and_load(self):
        """Save dataset to file and load it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "queries.json"

            # Save
            dataset1 = EvaluationDataset()
            dataset1.add_query("q1", "Query 1", ["doc1", "doc2"])
            dataset1.add_query("q2", "Query 2", ["doc3"])
            dataset1.save(filepath)

            # Load
            dataset2 = EvaluationDataset(filepath)
            assert len(dataset2.queries) == 2
            assert dataset2.get_query("q1").query_text == "Query 1"
            assert dataset2.get_query("q2").query_text == "Query 2"


# ── Evaluator Tests ──────────────────────────────────────────────────────────

class TestRetrieverEvaluator:
    """Tests for RetrieverEvaluator class."""

    def setup_method(self):
        """Setup for each test."""
        self.dataset = EvaluationDataset()
        self.dataset.add_query("q1", "Query 1", ["doc_1", "doc_2", "doc_3"])
        self.dataset.add_query("q2", "Query 2", ["doc_5", "doc_6"])

    def test_evaluate_query_perfect(self):
        """Evaluate a query with perfect retrieval."""
        evaluator = RetrieverEvaluator(self.dataset)
        metrics = evaluator.evaluate_query("q1", ["doc_1", "doc_2", "doc_3"])

        assert metrics.precision_at_5 == 1.0
        assert metrics.recall_at_5 == 1.0
        assert metrics.ndcg_at_5 > 0.95
        assert metrics.mrr == 1.0

    def test_evaluate_query_no_match(self):
        """Evaluate a query with no matching results."""
        evaluator = RetrieverEvaluator(self.dataset)
        metrics = evaluator.evaluate_query("q1", ["doc_99", "doc_100"])

        assert metrics.precision_at_5 == 0.0
        assert metrics.recall_at_5 == 0.0
        assert metrics.ndcg_at_5 == 0.0
        assert metrics.mrr == 0.0

    def test_evaluate_query_partial_match(self):
        """Evaluate a query with partial relevant results."""
        evaluator = RetrieverEvaluator(self.dataset)
        metrics = evaluator.evaluate_query("q1", ["doc_1", "doc_99", "doc_2"])

        # 2 out of 3 retrieved are relevant
        assert 0.6 < metrics.precision_at_5 < 0.7
        # 2 out of 3 known relevant docs found
        assert 0.6 < metrics.recall_at_5 < 0.7

    def test_evaluate_nonexistent_query(self):
        """Attempt to evaluate nonexistent query."""
        evaluator = RetrieverEvaluator(self.dataset)
        with pytest.raises(ValueError):
            evaluator.evaluate_query("q99", ["doc_1"])

    def test_aggregate_metrics(self):
        """Aggregate metrics across multiple queries."""
        evaluator = RetrieverEvaluator(self.dataset)
        evaluator.evaluate_query("q1", ["doc_1", "doc_2", "doc_3"])
        evaluator.evaluate_query("q2", ["doc_5", "doc_6"])

        agg = evaluator.aggregate_metrics()
        assert agg.num_queries == 2
        assert agg.num_with_relevant_results == 2
        assert agg.mean_precision_at_5 == 1.0
        assert agg.mean_recall_at_5 == 1.0

    def test_aggregate_empty(self):
        """Attempt to aggregate without evaluating any queries."""
        evaluator = RetrieverEvaluator(self.dataset)
        with pytest.raises(ValueError):
            evaluator.aggregate_metrics()

    def test_save_results(self):
        """Save evaluation results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = RetrieverEvaluator(self.dataset)
            evaluator.evaluate_query("q1", ["doc_1", "doc_2"])
            evaluator.evaluate_query("q2", ["doc_5"])

            output_file = Path(tmpdir) / "results.json"
            evaluator.save_results(output_file)

            assert output_file.exists()
            import json
            with open(output_file) as f:
                data = json.load(f)
            assert "q1" in data
            assert "q2" in data
            assert data["q1"]["precision_at_5"] > 0
