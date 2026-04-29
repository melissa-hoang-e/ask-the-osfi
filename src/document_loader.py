"""
document_loader.py
------------------
Fetches and parses publicly available OSFI guideline documents.
Supports PDF (via pdfplumber) and HTML (via BeautifulSoup).
"""

import re
import time
from pathlib import Path
from typing import Optional
import requests

# Optional dependencies — graceful fallback if not installed
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False


CACHE_DIR = Path("data/raw_docs")
REQUEST_DELAY = 1.0  # seconds between requests — be a polite scraper


# ── Public OSFI Documents ─────────────────────────────────────────────────────
# All documents below are freely available on osfi-bsif.gc.ca

OSFI_DOCUMENTS = [
    {
        "name": "OSFI Guideline B-10: Third-Party Risk Management",
        "url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/third-party-risk-management-guideline",
        "type": "html",
        "short_name": "B-10",
    },
    {
        "name": "OSFI Guideline B-13: Technology and Cyber Risk Management",
        "url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/technology-cyber-risk-management-guideline",
        "type": "html",
        "short_name": "B-13",
    },
    {
        "name": "OSFI Guideline E-23: Model Risk Management",
        "url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/model-risk-management-guideline",
        "type": "html",
        "short_name": "E-23",
    },
    {
        "name": "OSFI Guideline B-15: Climate Risk Management",
        "url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/climate-risk-management-guideline",
        "type": "html",
        "short_name": "B-15",
    },
    {
        "name": "OSFI Integrity and Security Guideline",
        "url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/integrity-security-guideline",
        "type": "html",
        "short_name": "Integrity-Security",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_html_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not HTML_SUPPORT:
        raise RuntimeError("Install beautifulsoup4: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")

    # Remove nav, footer, script, style blocks
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    # Get main content — OSFI pages use <main> or article-like divs
    main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|body"))
    target = main if main else soup

    text = target.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_html(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        print(f"  [Cache] {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")

    headers = {"User-Agent": "OSFI-RAG-Research-Tool/1.0 (educational project)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    time.sleep(REQUEST_DELAY)
    return response.text


def _fetch_pdf(url: str, cache_path: Path) -> str:
    if not PDF_SUPPORT:
        raise RuntimeError("Install pdfplumber: pip install pdfplumber")

    if not cache_path.exists():
        headers = {"User-Agent": "OSFI-RAG-Research-Tool/1.0 (educational project)"}
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        time.sleep(REQUEST_DELAY)

    text_parts = []
    with pdfplumber.open(cache_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


# ── Public API ────────────────────────────────────────────────────────────────

def load_document(doc: dict) -> Optional[str]:
    """
    Fetch and parse a single OSFI document.
    Returns the cleaned text content, or None on failure.
    """
    ext = "html" if doc["type"] == "html" else "pdf"
    cache_path = CACHE_DIR / f"{doc['short_name']}.{ext}"

    print(f"[Loader] Fetching: {doc['name']}")
    try:
        if doc["type"] == "html":
            raw = _fetch_html(doc["url"], cache_path)
            return _clean_html_text(raw)
        elif doc["type"] == "pdf":
            return _fetch_pdf(doc["url"], cache_path)
    except Exception as e:
        print(f"  [Error] Failed to load {doc['name']}: {e}")
        return None


def load_all_documents() -> list[dict]:
    """
    Load all OSFI documents. Returns list of {name, text} dicts.
    Skips documents that fail to load.
    """
    loaded = []
    for doc in OSFI_DOCUMENTS:
        text = load_document(doc)
        if text and len(text) > 200:  # sanity check: skip empty/failed loads
            loaded.append({"name": doc["name"], "text": text})
        else:
            print(f"  [Skip] {doc['name']} — insufficient content")

    print(f"\n[Loader] Successfully loaded {len(loaded)}/{len(OSFI_DOCUMENTS)} documents.")
    return loaded


def load_custom_document(filepath: str) -> Optional[dict]:
    """
    Load a user-supplied PDF or text file.
    Useful for adding proprietary internal policy documents.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"[Error] File not found: {filepath}")
        return None

    if path.suffix.lower() == ".pdf":
        if not PDF_SUPPORT:
            print("[Error] pdfplumber not installed. Run: pip install pdfplumber")
            return None
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        text = "\n".join(text_parts)

    elif path.suffix.lower() in [".txt", ".md"]:
        text = path.read_text(encoding="utf-8")

    else:
        print(f"[Error] Unsupported file type: {path.suffix}")
        return None

    return {"name": path.stem, "text": text}
