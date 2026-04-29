"""
app.py
------
Streamlit UI for the OSFI RAG assistant.
Run with: streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import VectorStore, ask, ingest_document
from document_loader import load_all_documents, load_custom_document

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ask the OSFI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .source-chip {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.82rem;
        margin: 2px;
        font-weight: 500;
    }
    .chunk-card {
        background: #f8f9fa;
        border-left: 3px solid #1a56db;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #333;
    }
    .score-badge {
        float: right;
        background: #1a56db;
        color: white;
        border-radius: 3px;
        padding: 1px 6px;
        font-size: 0.75rem;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #664d03;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────

if "store" not in st.session_state:
    st.session_state.store = VectorStore()

if "history" not in st.session_state:
    st.session_state.history = []

if "indexed" not in st.session_state:
    st.session_state.indexed = not st.session_state.store.is_empty


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://www.osfi-bsif.gc.ca/images/osfi-bsif-logo.png", width=180)
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required to embed documents and generate answers.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 📚 Document Index")

    if st.session_state.indexed:
        n_chunks = len(st.session_state.store.chunks)
        sources = list({c["source"] for c in st.session_state.store.chunks})
        st.success(f"✅ {n_chunks} chunks indexed")
        with st.expander("Indexed sources"):
            for s in sources:
                st.markdown(f"- {s}")
    else:
        st.warning("No documents indexed yet.")

    if st.button("🔄 Index OSFI Guidelines", use_container_width=True):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Loading and indexing OSFI documents... (this may take 1-2 min)"):
                docs = load_all_documents()
                for doc in docs:
                    ingest_document(doc["text"], doc["name"], st.session_state.store)
            st.session_state.indexed = True
            st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Add Custom Document")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded and st.button("Index uploaded file", use_container_width=True):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please enter your OpenAI API key first.")
        else:
            tmp_path = Path(f"data/raw_docs/{uploaded.name}")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded.read())
            doc = load_custom_document(str(tmp_path))
            if doc:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    ingest_document(doc["text"], doc["name"], st.session_state.store)
                st.session_state.indexed = True
                st.success(f"Indexed: {uploaded.name}")
                st.rerun()

    st.markdown("---")
    st.markdown("### 🗑️ Clear")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if st.button("Reset index", use_container_width=True):
        index_file = Path("data/embeddings.json")
        if index_file.exists():
            index_file.unlink()
        st.session_state.store = VectorStore()
        st.session_state.indexed = False
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small>Built with LangChain-free RAG · OpenAI · Streamlit<br>"
        "Data source: <a href='https://www.osfi-bsif.gc.ca' target='_blank'>osfi-bsif.gc.ca</a></small>",
        unsafe_allow_html=True,
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🏦 Ask the OSFI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">RAG-powered Q&A over Canadian financial regulation guidelines</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="warning-box">⚠️ <strong>Disclaimer:</strong> This tool is for research and educational purposes only. '
    'It is not legal or compliance advice. Always consult official OSFI publications and qualified legal counsel.</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


# ── Example Questions ─────────────────────────────────────────────────────────

EXAMPLES = [
    "What are OSFI's expectations for managing third-party risk?",
    "How does OSFI define model risk and what controls are required?",
    "What are the climate risk disclosure requirements for financial institutions?",
    "What cybersecurity controls does OSFI require under B-13?",
    "How should a bank handle a third-party cloud service provider relationship?",
    "What is OSFI's definition of a material outsourcing arrangement?",
]

st.markdown("**💡 Try an example question:**")
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill"] = example

# ── Chat History ──────────────────────────────────────────────────────────────

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])

    with st.chat_message("assistant"):
        st.write(entry["answer"])

        if entry.get("sources"):
            st.markdown("**Sources:**")
            source_html = " ".join(
                f'<span class="source-chip">{s}</span>' for s in entry["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)

        with st.expander("🔍 View retrieved passages"):
            for chunk in entry.get("retrieved_chunks", []):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<span class="score-badge">score: {chunk["score"]:.2f}</span>'
                    f'<strong>{chunk["source"]}</strong><br><br>{chunk["text"][:400]}...</div>',
                    unsafe_allow_html=True,
                )

# ── Input ─────────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Ask a question about OSFI guidelines...", key="query_input")

if prefill and not query:
    query = prefill

if query:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.indexed:
        st.error("Please index the OSFI documents first using the sidebar button.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating answer..."):
                start = time.time()
                result = ask(query, st.session_state.store)
                elapsed = time.time() - start

            st.write(result["answer"])

            if result.get("sources"):
                st.markdown("**Sources:**")
                source_html = " ".join(
                    f'<span class="source-chip">{s}</span>' for s in result["sources"]
                )
                st.markdown(source_html, unsafe_allow_html=True)

            st.caption(f"⏱ {elapsed:.1f}s · {len(result.get('retrieved_chunks', []))} passages retrieved")

            with st.expander("🔍 View retrieved passages"):
                for chunk in result.get("retrieved_chunks", []):
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<span class="score-badge">score: {chunk["score"]:.2f}</span>'
                        f'<strong>{chunk["source"]}</strong><br><br>{chunk["text"][:400]}...</div>',
                        unsafe_allow_html=True,
                    )

        st.session_state.history.append({
            "query": query,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        })
