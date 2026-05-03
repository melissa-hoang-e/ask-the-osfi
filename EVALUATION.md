# RAG Evaluation Guide

This document describes the evaluation framework for measuring retrieval quality in the OSFI RAG system.

---

## Overview

The evaluation system measures how well the RAG pipeline retrieves relevant documents for a benchmark set of queries. It provides:

- **Retrieval metrics**: Precision, Recall, NDCG, MRR, MAP
- **Comparative analysis**: Compare semantic vs. hybrid search performance
- **Benchmark dataset**: 15 realistic OSFI-focused queries with ground truth relevance judgments
- **Extensible framework**: Add more queries and evaluation campaigns

---

## Metrics

All metrics are computed per-query and then aggregated across the benchmark.

### Precision@K
**Definition**: Fraction of top-k results that are relevant.

$$\text{P@K} = \frac{\text{# relevant in top-k}}{\text{k}}$$

**Interpretation**:
- P@1 = 1.0 means the top result is relevant
- P@5 = 0.8 means 4 out of 5 top results are relevant
- **Range**: 0–1 (higher is better)

### Recall@K
**Definition**: Fraction of all relevant documents that appear in top-k.

$$\text{R@K} = \frac{\text{# relevant in top-k}}{\text{# relevant docs total}}$$

**Interpretation**:
- R@5 = 1.0 means all relevant documents were found in top-5
- R@5 = 0.5 means only half of relevant documents were retrieved
- **Range**: 0–1 (higher is better)

### NDCG@K (Normalized Discounted Cumulative Gain)
**Definition**: Measures ranking quality by penalizing relevant items appearing later.

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Where:
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\mathbb{1}[\text{doc}_i \text{ is relevant}]}{\log_2(i+1)}$$

**Interpretation**:
- Penalizes relevant items that appear lower in the ranking
- NDCG = 1.0 means perfect ranking (all relevant items at top)
- NDCG = 0.9 means nearly perfect ranking quality
- **Range**: 0–1 (higher is better)
- **Use when**: Ranking position matters (e.g., users examine top results first)

### MRR (Mean Reciprocal Rank)
**Definition**: Inverse position of the first relevant document.

$$\text{MRR} = \frac{1}{\text{position of first relevant doc}}$$

**Interpretation**:
- MRR = 1.0 means the first result is relevant
- MRR = 0.5 means the first relevant result is at position 2
- MRR = 0.0 means no relevant results were found
- **Range**: 0–1 (higher is better)
- **Use when**: Finding *any* relevant result quickly is important

### MAP@K (Mean Average Precision)
**Definition**: Average precision computed at each position where a relevant item appears.

$$\text{AP@K} = \frac{1}{\text{# relevant}} \sum_{i=1}^{K} \mathbb{1}[\text{doc}_i \text{ is relevant}] \cdot \text{P@i}$$

**Interpretation**:
- Combines precision and ranking quality
- AP = 1.0 means all relevant items appear at the top in order
- AP = 0.5 means relevant items are scattered, with later items reducing the score
- **Range**: 0–1 (higher is better)
- **Use when**: You want a single metric balancing precision and ranking order

---

## Benchmark Dataset

Located in `data/evaluation_benchmark.json`, contains 15 queries:

| Query ID | Topic | Relevant Docs |
|----------|-------|---------------|
| q_tpm_1 | Third-party risk framework | B-10 |
| q_tpm_2 | Outsourcing governance | B-10 |
| q_tpm_3 | Cloud service controls | B-10, B-13 |
| q_cyber_1 | Cybersecurity guidance | B-13 |
| q_cyber_2 | Incident response | B-13 |
| q_model_1 | Model risk framework | E-23 |
| q_model_2 | Model validation | E-23 |
| q_climate_1 | Climate risk framework | B-15 |
| q_climate_2 | Climate risk assessment | B-15 |
| q_multi_1 | Third-party + cyber overlap | B-10, B-13 |
| q_multi_2 | External dependencies | B-10 |
| q_integrity_1 | Integrity & security | I&S |
| q_compliance_1 | Due diligence | B-10 |
| q_oversight_1 | Service provider monitoring | B-10 |
| q_resilience_1 | Business continuity | B-13 |

**Characteristics**:
- **Single-source queries**: 10 queries require documents from one guideline
- **Multi-source queries**: 5 queries require documents from multiple guidelines
- **Coverage**: Tests retrieval across all indexed documents (B-10, B-13, E-23, B-15, I&S)

---

## Usage

### Quick Start: Run Full Evaluation

```bash
# Evaluate using hybrid search (BM25 + semantic + RRF)
python evaluate_rag.py --method hybrid

# Evaluate using semantic-only search
python evaluate_rag.py --method semantic
```

### CLI Evaluation

```bash
# Via CLI with hybrid search
python cli.py evaluate --method hybrid

# Via CLI with semantic search
python cli.py evaluate --method semantic

# Custom benchmark file
python cli.py evaluate --method hybrid --benchmark data/custom_benchmark.json

# Save results to specific directory
python cli.py evaluate --method hybrid --output ~/my_results/
```

### Programmatic Usage

```python
from pathlib import Path
from evaluation import EvaluationDataset, RetrieverEvaluator
from rag_pipeline import HybridVectorStore, hybrid_retrieve, ingest_document, embed_texts

# Setup
dataset = EvaluationDataset(Path("data/evaluation_benchmark.json"))
store = HybridVectorStore()
evaluator = RetrieverEvaluator(dataset)

# Evaluate each query
for query_id, query in dataset.queries.items():
    results = hybrid_retrieve(query.query_text, store, top_k=5, use_hybrid=True)
    retrieved_sources = [r["source"] for r in results]
    metrics = evaluator.evaluate_query(query_id, retrieved_sources)

# Get aggregated metrics
agg_metrics = evaluator.aggregate_metrics()

# Print report
evaluator.print_report()

# Save results
evaluator.save_results(Path("results/evaluation/metrics.json"))
```

---

## Output Files

### Metrics File: `metrics_{method}.json`
Per-query detailed results:
```json
{
  "q_tpm_1": {
    "query_id": "q_tpm_1",
    "precision_at_1": 1.0,
    "precision_at_3": 0.667,
    "precision_at_5": 0.6,
    "recall_at_1": 0.0,
    "recall_at_3": 0.333,
    "recall_at_5": 0.5,
    "ndcg_at_5": 0.85,
    "mrr": 0.5,
    "map_at_5": 0.65,
    "num_relevant": 2,
    "num_retrieved": 5
  },
  ...
}
```

### Aggregated File: `aggregated_{method}.json`
Summary metrics across all queries:
```json
{
  "method": "hybrid",
  "metrics": {
    "mean_precision_at_1": 0.733,
    "mean_precision_at_3": 0.622,
    "mean_precision_at_5": 0.587,
    "mean_recall_at_1": 0.067,
    "mean_recall_at_3": 0.267,
    "mean_recall_at_5": 0.467,
    "mean_ndcg_at_5": 0.812,
    "mean_mrr": 0.722,
    "mean_map_at_5": 0.634,
    "num_queries": 15,
    "num_with_relevant_results": 14
  }
}
```

### Comparison Report: Semantic vs. Hybrid
Automatically generated if both methods are evaluated:
```
========================================================================
COMPARISON: SEMANTIC vs HYBRID
========================================================================

Metric               Semantic        Hybrid          Improvement
────────────────────────────────────────────────────────────────
P@1                  0.667           0.733           +10.0%
P@3                  0.533           0.622           +16.6%
P@5                  0.480           0.587           +22.3%
R@1                  0.067           0.067           +0.0%
R@3                  0.200           0.267           +33.5%
R@5                  0.400           0.467           +16.8%
NDCG@5               0.720           0.812           +12.8%
MRR                  0.667           0.722           +8.3%
MAP@5                0.560           0.634           +13.2%
```

---

## Adding Custom Queries

### Format

Create a JSON file with query entries:

```json
{
  "queries": [
    {
      "query_id": "q_custom_1",
      "query_text": "Your question here?",
      "relevant_docs": ["B-10", "B-13"],
      "description": "Optional: what topic this tests"
    }
  ]
}
```

### Example: Add a new climate + third-party query

```json
{
  "query_id": "q_multi_climate_tpm",
  "query_text": "How should institutions manage third-party climate risk?",
  "relevant_docs": ["B-10", "B-15"],
  "description": "Cross-cutting question with climate + governance implications"
}
```

### Evaluate Custom Dataset

```bash
python evaluate_rag.py --method hybrid --benchmark data/custom_benchmark.json
```

---

## Interpreting Results

### Strong Performance (What to aim for)
- **P@5 > 0.8**: Most results are relevant
- **R@5 > 0.7**: Most relevant documents found in top-5
- **NDCG@5 > 0.85**: Excellent ranking quality
- **MRR > 0.8**: First relevant result appears early

### Hybrid vs. Semantic Comparison
- **Hybrid typically wins on**:
  - P@K: Better precision through keyword matching
  - NDCG: Better ranking via RRF fusion
  - Regulatory queries with specific terminology
  
- **Semantic can excel on**:
  - Paraphrased queries (e.g., "external dependencies" → "third-party governance")
  - Implicit meaning queries ("How do I manage risk?")

### Troubleshooting

**If metrics are low (< 0.5)**:
1. Check chunk size is appropriate (~800 chars)
2. Verify embeddings are calculated correctly
3. Review benchmark relevance judgments are accurate
4. Check if query terminology matches document language

---

## Integration with Testing

Evaluation results can be integrated into CI/CD pipelines:

```bash
# Run evaluation and fail if metrics drop below threshold
python evaluate_rag.py --method hybrid --output results/

# Check if mean NDCG > 0.80
python -c "
import json
with open('results/evaluation/aggregated_hybrid.json') as f:
    data = json.load(f)
    ndcg = data['metrics']['mean_ndcg_at_5']
    assert ndcg > 0.80, f'NDCG {ndcg} below 0.80 threshold'
    print(f'✓ NDCG check passed: {ndcg:.3f}')
"
```

---

## Performance Baseline

On the included 15-query benchmark:

| Metric | Semantic | Hybrid | Expected Gap |
|--------|----------|--------|--------------|
| P@5 | ~0.55 | ~0.65 | +18% |
| R@5 | ~0.42 | ~0.52 | +24% |
| NDCG@5 | ~0.75 | ~0.85 | +13% |
| MRR | ~0.67 | ~0.73 | +9% |

*(Estimated ranges; actual values depend on index and embeddings)*

---

## See Also

- [HYBRID_SEARCH.md](./HYBRID_SEARCH.md): Detailed hybrid search architecture
- [src/evaluation.py](./src/evaluation.py): Metrics implementation
- [tests/test_evaluation.py](./tests/test_evaluation.py): Unit tests for metrics
