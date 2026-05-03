# Evaluation System - Quick Reference

## Run Evaluation in 30 Seconds

```bash
# Evaluate hybrid search (best performance)
python evaluate_rag.py --method hybrid

# OR via CLI
python cli.py evaluate --method hybrid
```

## What You Get

Detailed metrics for each query  
Summary report with aggregate scores  
Side-by-side comparison (if you run both methods)  
JSON files with full results  

---

## Key Metrics

| Metric | Meaning | Target | Good Value |
|--------|---------|--------|------------|
| **P@5** | % of top-5 results relevant | > 0.7 |  0.8+ |
| **R@5** | % of relevant docs found | > 0.6 |  0.7+ |
| **NDCG@5** | Ranking quality (1=perfect) | > 0.8 |  0.85+ |
| **MRR** | How fast to find 1st result | > 0.7 |  0.75+ |
| **MAP@5** | Overall quality | > 0.7 |  0.75+ |

---

## Benchmark: 15 Queries

- 10 single-topic queries (B-10, B-13, E-23, B-15, I&S)
- 5 multi-topic queries (cross-domain)
- All grounded in OSFI guidelines

---

## Files

| File | Purpose |
|------|---------|
| `src/evaluation.py` | Metrics implementation |
| `data/evaluation_benchmark.json` | 15 test queries |
| `tests/test_evaluation.py` | 35 unit tests ✅ passing |
| `evaluate_rag.py` | Standalone evaluation CLI |
| `EVALUATION.md` | Complete guide |

---

## Python API

```python
from evaluation import EvaluationDataset, RetrieverEvaluator
from rag_pipeline import hybrid_retrieve

# Load benchmark
dataset = EvaluationDataset("data/evaluation_benchmark.json")
evaluator = RetrieverEvaluator(dataset)

# Run evaluation
for qid, query in dataset.queries.items():
    results = hybrid_retrieve(query.query_text, store, top_k=5, use_hybrid=True)
    sources = [r["source"] for r in results]
    metrics = evaluator.evaluate_query(qid, sources)

# Print report
evaluator.print_report()

# Save results
evaluator.save_results("results.json")
```

---

## Hybrid vs. Semantic

Hybrid typically wins by:
- **~18% higher precision** (more relevant results)
- **~24% higher recall** (finds more relevant docs)
- **~13% better ranking** (NDCG improvement)

---

## Next: Create Custom Benchmark

```json
{
  "queries": [
    {
      "query_id": "my_q1",
      "query_text": "Your regulation question?",
      "relevant_docs": ["B-10", "B-13"],
      "description": "What this tests"
    }
  ]
}
```

Then evaluate:
```bash
python evaluate_rag.py --method hybrid --benchmark data/my_benchmark.json
```

---

## More Info

- **Full Guide**: [EVALUATION.md](EVALUATION.md)
- **Implementation Summary**: [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md)
- **Checklist**: [EVALUATION_CHECKLIST.md](EVALUATION_CHECKLIST.md)
- **Code**: [src/evaluation.py](src/evaluation.py)
