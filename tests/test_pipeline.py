"""
tests/test_pipeline.py
----------------------
Unit tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline import (
    VectorStore,
    chunk_text,
    generate_answer,
)


# ── Chunking Tests ────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short document."
        chunks = chunk_text(text, "test_doc")
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["source"] == "test_doc"

    def test_long_text_multiple_chunks(self):
        # 2000 chars should produce multiple chunks (CHUNK_SIZE=800, OVERLAP=150)
        text = "This is a sentence with real content. " * 60
        chunks = chunk_text(text, "long_doc")
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        text = "word " * 500
        chunks = chunk_text(text, "source")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_same_source_same_id(self):
        """Deterministic: same input always yields same chunk IDs."""
        text = "consistent text " * 100
        chunks_a = chunk_text(text, "doc_a")
        chunks_b = chunk_text(text, "doc_a")
        assert [c["chunk_id"] for c in chunks_a] == [c["chunk_id"] for c in chunks_b]

    def test_different_sources_different_ids(self):
        text = "same text content " * 100
        chunks_a = chunk_text(text, "source_a")
        chunks_b = chunk_text(text, "source_b")
        ids_a = set(c["chunk_id"] for c in chunks_a)
        ids_b = set(c["chunk_id"] for c in chunks_b)
        assert ids_a.isdisjoint(ids_b)

    def test_empty_text(self):
        chunks = chunk_text("", "empty")
        assert chunks == []

    def test_whitespace_normalized(self):
        text = "word   with   extra    spaces  " * 50
        chunks = chunk_text(text, "ws_test")
        for chunk in chunks:
            assert "  " not in chunk["text"]


# ── VectorStore Tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    def test_is_empty_initially(self, tmp_path):
        store_file = tmp_path / "embeddings.json"
        store = VectorStore(store_path=store_file)
        assert store.is_empty

    def test_add_and_search(self, tmp_path):
        store_file = tmp_path / "embeddings.json"
        store = VectorStore(store_path=store_file)

        chunks = [
            {"text": "Capital requirements for banks", "source": "B-1", "chunk_id": "aaa"},
            {"text": "Outsourcing risk management framework", "source": "B-10", "chunk_id": "bbb"},
            {"text": "Climate change financial risks", "source": "B-15", "chunk_id": "ccc"},
        ]
        # 3 simple 4-dim embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        store.add_chunks(chunks, embeddings)
        assert not store.is_empty
        assert len(store.chunks) == 3

        # Query closest to second chunk
        query_emb = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32)
        results = store.search(query_emb, top_k=1)
        assert results[0]["chunk_id"] == "bbb"
        assert results[0]["score"] > 0.8

    def test_persistence(self, tmp_path):
        store_file = tmp_path / "embeddings.json"
        chunks = [{"text": "Test chunk", "source": "src", "chunk_id": "xyz"}]
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # Write
        store1 = VectorStore(store_path=store_file)
        store1.add_chunks(chunks, embeddings)

        # Read back
        store2 = VectorStore(store_path=store_file)
        assert not store2.is_empty
        assert store2.chunks[0]["chunk_id"] == "xyz"

    def test_deduplication(self, tmp_path):
        store_file = tmp_path / "embeddings.json"
        store = VectorStore(store_path=store_file)

        chunks = [{"text": "Duplicate chunk", "source": "src", "chunk_id": "dup1"}]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

        store.add_chunks(chunks, embeddings)
        store.add_chunks(chunks, embeddings)  # add same chunk again

        assert len(store.chunks) == 1  # should not duplicate

    def test_search_empty_store(self, tmp_path):
        store_file = tmp_path / "empty.json"
        store = VectorStore(store_path=store_file)
        query_emb = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query_emb, top_k=3)
        assert results == []


# ── Generation Tests ──────────────────────────────────────────────────────────

class TestGenerateAnswer:
    def test_empty_context(self):
        result = generate_answer("What is OSFI?", [])
        assert "answer" in result
        assert "could not find" in result["answer"].lower()
        assert result["sources"] == []

    @patch("rag_pipeline.requests.post")
    def test_with_context(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OSFI is the Office of the Superintendent of Financial Institutions."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        chunks = [
            {
                "text": "OSFI supervises federally regulated financial institutions.",
                "source": "OSFI Overview",
                "chunk_id": "abc",
                "score": 0.95,
            }
        ]

        result = generate_answer("What is OSFI?", chunks)
        assert "answer" in result
        assert "OSFI Overview" in result["sources"]
        mock_post.assert_called_once()

    @patch("rag_pipeline.requests.post")
    def test_sources_deduplicated(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer here."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Two chunks from the same source
        chunks = [
            {"text": "Passage A", "source": "B-10", "chunk_id": "a1", "score": 0.9},
            {"text": "Passage B", "source": "B-10", "chunk_id": "a2", "score": 0.8},
        ]

        result = generate_answer("Question?", chunks)
        assert result["sources"].count("B-10") == 1
