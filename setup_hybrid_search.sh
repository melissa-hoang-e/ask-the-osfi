#!/bin/bash
# setup_hybrid_search.sh
# Completes the hybrid search setup

set -e

echo "Installing hybrid search dependency..."
pip install -q rank-bm25>=0.2.2

echo "✓ Installing rank-bm25"
echo "✓ Testing imports..."

python3 -c "
import sys
sys.path.insert(0, 'src')
from hybrid_search import HybridSearcher, BM25Retriever, SemanticRetriever, reciprocal_rank_fusion
from rag_pipeline import HybridVectorStore, hybrid_retrieve, ask_hybrid
print('✓ All hybrid search modules loaded successfully')
"

echo ""
echo "========================================"
echo "Hybrid Search Setup Complete!"
echo "========================================"
echo ""
echo "Quick Start Guide:"
echo ""
echo "1. Run tests:"
echo "   python3 tests/test_hybrid_search.py"
echo ""
echo "2. Start the Streamlit app (hybrid search is enabled by default):"
echo "   streamlit run app.py"
echo ""
echo "3. In the sidebar under Configuration, you'll see:"
echo "   Hybrid Search (BM25 + Semantic) [checkbox]"
echo ""
echo "4. Ask a question - the system will now use:"
echo "   - BM25 for precise keyword matching (e.g., 'third-party risk')"
echo "   - Semantic embeddings for intent understanding"
echo "   - RRF fusion to merge both signals"
echo ""
echo "Read HYBRID_SEARCH.md for comprehensive documentation"
echo ""
