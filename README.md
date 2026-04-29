# 🏦 Ask the OSFI

**RAG-powered Q&A over Canadian financial regulation guidelines**

A production-style Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions against publicly available OSFI (Office of the Superintendent of Financial Institutions) guidelines. Built from scratch — no LangChain, no LlamaIndex — to demonstrate the core mechanics of semantic search and LLM-grounded generation.

---

## Why This Exists

Compliance and audit teams routinely need to cross-reference multiple OSFI guidelines (B-10, B-13, E-23, B-15...) to answer questions like *"What controls does OSFI require for cloud outsourcing?"* or *"How does model risk overlap with third-party risk?"*. These documents are dense, long, and interlocking. This tool surfaces the right passages instantly and generates cited, grounded answers.

---

## Architecture

```
User Query
    │
    ▼
[ Embedding ]  ──── OpenAI text-embedding-3-small
    │
    ▼
[ Vector Search ]  ──── Cosine similarity over in-memory numpy index
    │                    (file-backed JSON, swappable for ChromaDB/Pinecone)
    ▼
[ Top-K Chunks ]  ──── Scored passages with source attribution
    │
    ▼
[ LLM Generation ]  ──── GPT-4o-mini with strict grounding prompt
    │                     (cites sources, refuses to speculate)
    ▼
[ Answer + Sources ]
```

**Key design decisions:**
- **No vector database dependency** — uses numpy cosine similarity backed by a JSON file. For this document set (~200 chunks), this is faster and simpler than spinning up ChromaDB. Swap `VectorStore` for production scale.
- **Low temperature (0.1)** — regulatory Q&A demands precision over creativity.
- **Sentence-boundary chunking** — avoids splitting mid-sentence, which degrades retrieval quality on dense legal text.
- **Overlap between chunks (150 chars)** — ensures context at chunk boundaries is never lost.
- **Source citations enforced via system prompt** — the LLM is instructed to refuse if context is insufficient, preventing hallucination on regulatory content.

---

## Indexed Documents

| Guideline | Topic |
|-----------|-------|
| **B-10** | Third-Party Risk Management |
| **B-13** | Technology and Cyber Risk Management |
| **E-23** | Model Risk Management |
| **B-15** | Climate Risk Management |
| **I&S** | Integrity and Security |

All documents are fetched directly from [osfi-bsif.gc.ca](https://www.osfi-bsif.gc.ca) and cached locally on first run.

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/melissa-hoang-e/ask-the-osfi.git
cd ask-the-osfi
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

### 3a. Run the web app

```bash
streamlit run app.py
```

Then open `http://localhost:8501`. Click **"Index OSFI Guidelines"** in the sidebar to fetch and embed the documents (takes ~60–90s on first run, cached afterward).

### 3b. Use the CLI

```bash
# Index documents
python cli.py index

# Ask a question
python cli.py ask "What are OSFI's expectations for third-party cloud providers?"

# Verbose mode — shows retrieved passages + relevance scores
python cli.py ask "What is model risk?" -v

# Index a custom document
python cli.py index --file /path/to/internal_policy.pdf
```

---

## Example Questions

```
What controls does OSFI require for managing third-party vendors?
How should a bank categorize a material outsourcing arrangement?
What are the cybersecurity incident reporting requirements under B-13?
How does OSFI define model risk, and what validation is expected?
What climate risk disclosures are required for federally regulated insurers?
What is OSFI's stance on the use of AI/ML models?
```

---

## Project Structure

```
ask-the-osfi/
├── app.py                  # Streamlit web app
├── cli.py                  # Command-line interface
├── requirements.txt
├── src/
│   ├── rag_pipeline.py     # Core RAG: chunking, embedding, retrieval, generation
│   └── document_loader.py  # Fetches and parses OSFI HTML/PDF documents
├── data/
│   ├── raw_docs/           # Cached raw documents (auto-created)
│   └── embeddings.json     # Persisted vector index (auto-created)
└── tests/
    └── test_pipeline.py    # Unit tests for chunking, vector store, generation
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Extending

**Add more documents:**
```python
# In src/document_loader.py, append to OSFI_DOCUMENTS:
{
    "name": "OSFI Guideline E-21: Operational Risk",
    "url": "https://www.osfi-bsif.gc.ca/en/...",
    "type": "html",
    "short_name": "E-21",
}
```

**Swap the vector store for ChromaDB:**
```python
# Replace VectorStore in rag_pipeline.py with:
import chromadb
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("osfi_docs")
```

**Use a different LLM:**
```python
# In generate_answer(), replace the OpenAI call with any API-compatible endpoint.
# Works with Anthropic Claude, local Ollama, or Azure OpenAI with minimal changes.
```

---

## Disclaimer

> This tool is for **research and educational purposes only**. It is not legal or compliance advice. Answers are generated from publicly available OSFI documents and may not reflect the most current regulatory guidance. Always consult official OSFI publications and qualified legal counsel for compliance decisions.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![NumPy](https://img.shields.io/badge/VectorSearch-NumPy-013243)
![scikit-learn](https://img.shields.io/badge/Similarity-scikit--learn-F7931E)

---

*Built by [Melissa Hoang](https://github.com/melissa-hoang-e)*
