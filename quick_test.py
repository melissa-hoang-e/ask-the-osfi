"""
Quick test script to verify hybrid search implementation
"""
import numpy as np
from rag_pipeline import HybridVectorStore, hybrid_retrieve, ask_hybrid
from hybrid_search import BM25Retriever, SemanticRetriever, reciprocal_rank_fusion, HybridSearcher
import sys
sys.path.insert(0, 'src')

print('Testing hybrid search components...')
print()

# Test 1: Import all modules
print('1. Testing imports...')
print('   ✓ All imports successful')

print()

# Test 2: BM25Retriever
print('2. Testing BM25Retriever...')
docs = [
    'Third-party risk management is critical',
    'OSFI requires governance controls',
    'Cybersecurity compliance is mandatory'
]
bm25 = BM25Retriever(docs)
results = bm25.search('third-party risk', top_k=2)
assert len(results) > 0, 'BM25 should return results'
print(f'   ✓ BM25Retriever works (found {len(results)} results)')
print(f'     Results: {results}')

print()

# Test 3: SemanticRetriever
print('3. Testing SemanticRetriever...')
embeddings = np.random.randn(3, 8).astype(np.float32)
semantic = SemanticRetriever(embeddings)
query_emb = np.random.randn(8).astype(np.float32)
results = semantic.search(query_emb, top_k=2)
assert len(results) > 0, 'Semantic should return results'
print(f'   ✓ SemanticRetriever works (found {len(results)} results)')

print()

# Test 4: RRF Fusion
print('4. Testing Reciprocal Rank Fusion...')
bm25_res = [(0, 10.0), (2, 8.0)]
semantic_res = [(1, 0.9), (0, 0.8)]
fused = reciprocal_rank_fusion(bm25_res, semantic_res)
assert len(fused) > 0, 'RRF should return fused results'
assert fused[0][0] == 0, 'Doc 0 should rank first (appears in both)'
print(f'   ✓ RRF fusion works ({len(fused)} fused results)')
print(f'     Ranking: Doc {fused[0][0]} (score={fused[0][1]:.4f})')

print()

# Test 5: HybridSearcher
print('5. Testing HybridSearcher...')
docs = [
    'Third-party risk management controls for vendors',
    'OSFI Guideline B-10 covers outsourcing governance',
    'Cybersecurity requirements under B-13 mandate encryption',
    'Model risk validation requires backtesting',
    'Third-party contracts must include SLAs'
]
embeddings = np.random.randn(5, 10).astype(np.float32)
searcher = HybridSearcher(docs, embeddings)
query = 'third-party risk governance'
query_emb = np.random.randn(10).astype(np.float32)
results = searcher.search(query, query_emb, top_k=3)
assert len(results) <= 3, 'Should return max 3 results'
assert all(
    'hybrid_score' in r for r in results), 'All results should have hybrid_score'
print(f'   ✓ HybridSearcher works ({len(results)} results)')
for i, r in enumerate(results, 1):
    print(f'     [{i}] Score: {r["hybrid_score"]:.4f}, BM25 rank: {r["bm25_rank"]}, Semantic rank: {r["semantic_rank"]}')

print()

# Test 6: HybridVectorStore
print('6. Testing HybridVectorStore initialization...')
store = HybridVectorStore()
assert store.is_empty, 'New store should be empty'
print(f'   ✓ HybridVectorStore initialized (empty={store.is_empty})')

print()
print('='*50)
print('✓ ALL TESTS PASSED')
print('='*50)
