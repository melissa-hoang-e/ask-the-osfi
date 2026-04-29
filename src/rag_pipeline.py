"""
rag_pipeline.py
---------------
Core RAG pipeline: document ingestion → chunking → embedding → retrieval → generation.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks
TOP_K = 5                 # number of chunks retrieved per query
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CACHE_DIR = Path("data/cache")
EMBEDDINGS_FILE = Path("data/embeddings.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks. Attempts to split on sentence
    boundaries to avoid cutting mid-sentence.

    Returns a list of dicts: {text, source, chunk_id}
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end < len(text):
            # Try to break on a sentence boundary (. ! ?)
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        chunk_text_content = text[start:end].strip()
        if chunk_text_content:
            chunk_id = hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:8]
            chunks.append({
                "text": chunk_text_content,
                "source": source,
                "chunk_id": chunk_id,
            })

        start = end - CHUNK_OVERLAP
        chunk_index += 1

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings using OpenAI's embedding API.
    Returns a (N, D) numpy array.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"input": texts, "model": EMBED_MODEL}
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    vectors = [item["embedding"] for item in data["data"]]
    return np.array(vectors, dtype=np.float32)


# ── Vector Store (lightweight, file-backed) ───────────────────────────────────

class VectorStore:
    """
    Simple in-memory vector store backed by a JSON file.
    Suitable for the document sizes in this project (~50-200 chunks).
    For production scale, swap for ChromaDB or Pinecone.
    """

    def __init__(self, store_path: Path = EMBEDDINGS_FILE):
        self.store_path = store_path
        self.chunks: list[dict] = []       # [{text, source, chunk_id}]
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        if self.store_path.exists():
            with open(self.store_path) as f:
                saved = json.load(f)
            self.chunks = saved["chunks"]
            self.embeddings = np.array(saved["embeddings"], dtype=np.float32)
            print(f"[VectorStore] Loaded {len(self.chunks)} chunks from cache.")

    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunks,
                    "embeddings": self.embeddings.tolist(),
                },
                f,
            )

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray):
        """Append new chunks and embeddings, then persist."""
        existing_ids = {c["chunk_id"] for c in self.chunks}
        new_chunks, new_embeddings = [], []

        for chunk, emb in zip(chunks, embeddings):
            if chunk["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)
                new_embeddings.append(emb)

        if not new_chunks:
            print("[VectorStore] No new chunks to add (all already indexed).")
            return

        self.chunks.extend(new_chunks)
        if self.embeddings is None:
            self.embeddings = np.array(new_embeddings, dtype=np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self._save()
        print(f"[VectorStore] Added {len(new_chunks)} new chunks. Total: {len(self.chunks)}.")

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        """Return top-k most similar chunks to the query embedding."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        sims = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self.chunks[idx],
                "score": float(sims[idx]),
            })
        return results

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ── Document Ingestion ────────────────────────────────────────────────────────

def ingest_document(text: str, source_name: str, store: VectorStore):
    """Chunk a document, embed the chunks, and add to the vector store."""
    print(f"[Ingest] Processing: {source_name}")
    chunks = chunk_text(text, source_name)
    print(f"[Ingest] {len(chunks)} chunks created.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    store.add_chunks(chunks, embeddings)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, store: VectorStore, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and retrieve the top-k relevant chunks."""
    query_emb = embed_texts([query])[0]
    results = store.search(query_emb, top_k=top_k)
    return results


# ── Generation ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Canadian financial regulation, specifically OSFI (Office of the Superintendent of Financial Institutions) guidelines and policies.

Your answers must:
1. Be grounded ONLY in the provided context passages
2. Cite the source document for every claim (e.g., "According to [Guideline B-10]...")
3. Clearly state when the context does not contain enough information to answer
4. Avoid speculation or drawing on knowledge outside the provided context
5. Be precise and professional, suitable for compliance and audit teams

If the question is outside the scope of Canadian financial regulation, politely redirect the user."""


def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Call the LLM with the retrieved context and return the answer + sources.
    """
    if not context_chunks:
        return {
            "answer": "I could not find relevant information in the OSFI documents to answer your question.",
            "sources": [],
        }

    # Build context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Passage {i} | Source: {chunk['source']} | Relevance: {chunk['score']:.2f}]\n{chunk['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    user_message = f"""Context passages from OSFI documents:

{context_block}

---

Question: {query}

Please answer the question using ONLY the context above. Cite the source for each claim."""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,  # low temp for factual regulatory Q&A
        "max_tokens": 800,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]

    sources = list({chunk["source"] for chunk in context_chunks})

    return {"answer": answer, "sources": sources}


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def ask(query: str, store: VectorStore) -> dict:
    """End-to-end RAG: retrieve → generate."""
    context = retrieve(query, store)
    result = generate_answer(query, context)
    result["retrieved_chunks"] = context
    return result
