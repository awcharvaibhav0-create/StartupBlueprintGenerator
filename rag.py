"""
rag.py — Lightweight FAISS-based RAG pipeline for Startup Blueprint Generator.

Loads PDF and TXT documents from the ./data directory, splits them into
chunks, embeds them with a local sentence-transformer model, and exposes a
retrieve() function that returns the most relevant context for a given query.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded fallback knowledge base
# Ensures the app is useful even when no files are placed in ./data
# ---------------------------------------------------------------------------
FALLBACK_KNOWLEDGE: list[str] = [
    # Startup India
    ("Startup India is a flagship initiative of the Government of India, intended to build a "
     "strong eco-system for nurturing innovation and startups in India. DPIIT (Department for "
     "Promotion of Industry and Internal Trade) is the nodal department for Startup India. "
     "Recognized startups can avail income-tax exemption for 3 years under Section 80IAC."),
    ("To be recognized as a startup under DPIIT, the entity must be incorporated as a private "
     "limited company, LLP, or partnership firm; be not more than 10 years old from its date of "
     "incorporation; have an annual turnover not exceeding ₹100 crore; and be working towards "
     "innovation, development, deployment of new products/services/processes."),
    # MSME Schemes
    ("MSME stands for Micro, Small and Medium Enterprises. Micro: investment up to ₹1 crore, "
     "turnover up to ₹5 crore. Small: investment up to ₹10 crore, turnover up to ₹50 crore. "
     "Medium: investment up to ₹50 crore, turnover up to ₹250 crore. MSME registration provides "
     "access to government subsidies, priority sector lending, and credit guarantee schemes."),
    ("CGTMSE (Credit Guarantee Fund Trust for Micro and Small Enterprises) provides collateral-free "
     "loans up to ₹5 crore to MSMEs. The scheme covers up to 85% of credit for micro enterprises "
     "and 75% for other enterprises. Banks and NBFCs are member lending institutions."),
    # Funding Stages
    ("Startup funding stages: Bootstrapping (self-funded); Pre-Seed (friends, family, angel "
     "investors, typically ₹5L–50L); Seed Stage (angel networks, seed funds, ₹50L–5Cr); Series A "
     "(venture capital, ₹5Cr–50Cr for proven traction); Series B/C (growth-stage VC, ₹50Cr+); "
     "IPO (public listing). Each stage requires stronger proof of product-market fit and revenue."),
    ("Angel investors typically invest ₹25 lakh to ₹5 crore in exchange for 10–25% equity. "
     "Key Indian angel networks include Indian Angel Network (IAN), Mumbai Angels, Chennai Angels, "
     "and Let's Venture. Venture capital firms active in India include Sequoia (Peak XV), Accel, "
     "Blume Ventures, Elevation Capital, and Matrix Partners."),
    # Business Model Canvas
    ("The Business Model Canvas has 9 building blocks: Key Partners, Key Activities, Key Resources, "
     "Value Propositions, Customer Relationships, Channels, Customer Segments, Cost Structure, and "
     "Revenue Streams. It was developed by Alexander Osterwalder and is a strategic management "
     "tool for developing new business models or documenting existing ones."),
    ("Common revenue models for startups: SaaS Subscription (monthly/annual recurring revenue); "
     "Marketplace/Commission (take-rate on transactions); Freemium (free tier + paid upgrades); "
     "Advertising (CPM/CPC); Licensing; Direct Sales; Usage-based/Pay-per-use; "
     "Consulting/Services. SaaS typically yields higher LTV and is preferred by investors."),
    # Incubators & Accelerators
    ("Top Indian startup incubators and accelerators: IIM Ahmedabad CIIE, IIT Bombay SINE, "
     "T-Hub Hyderabad, NASSCOM 10000 Startups, Atal Innovation Mission (AIM), SIDBI iSTART, "
     "Microsoft Accelerator, Google for Startups Accelerator India, Y Combinator (global), "
     "Techstars (global). Incubators provide mentorship, workspace, and seed funding in exchange "
     "for equity or a fee."),
    # Government Schemes
    ("Atal Innovation Mission (AIM) runs Atal Incubation Centres (AIC) and Atal Tinkering Labs "
     "(ATL). AIM provides grant funding of up to ₹10 crore to AIC. SIDBI Fund of Funds Scheme "
     "invests in SEBI-registered AIFs that then invest in startups. MUDRA Yojana provides loans "
     "up to ₹10 lakh for non-corporate small businesses under Shishu, Kishor, and Tarun categories."),
    ("Production Linked Incentive (PLI) Scheme provides incentives across 14 sectors including "
     "electronics, pharmaceuticals, auto components, and food processing. Startups in these "
     "sectors can benefit from PLI subsidies. Stand-Up India scheme provides loans between "
     "₹10 lakh and ₹1 crore to SC/ST and women entrepreneurs."),
    # Market Sizing
    ("Market sizing frameworks: TAM (Total Addressable Market) — total revenue if 100% market "
     "share; SAM (Serviceable Addressable Market) — segment you can realistically target; "
     "SOM (Serviceable Obtainable Market) — realistic near-term capture. Investors expect "
     "startups to clearly articulate TAM/SAM/SOM with bottom-up calculations."),
    # Pitch Deck
    ("A strong investor pitch deck covers: Problem, Solution, Market Size, Product Demo, "
     "Business Model, Traction, Team, Competition, Financials (3-year projection), Ask "
     "(funding amount and use of funds). Keep it to 10–15 slides. Lead with traction data "
     "and customer testimonials wherever possible."),
    # Unit Economics
    ("Key startup unit economics metrics: CAC (Customer Acquisition Cost) = total sales & "
     "marketing spend / new customers acquired. LTV (Lifetime Value) = ARPU × gross margin % "
     "× avg customer lifespan. LTV:CAC ratio should be ≥ 3 for a healthy SaaS business. "
     "Payback period = CAC / (ARPU × gross margin %). Target < 12 months for venture-scale growth."),
    # SWOT
    ("SWOT Analysis: Strengths (internal positive factors — unique IP, team expertise, cost "
     "advantage); Weaknesses (internal negative factors — limited capital, small team, "
     "unproven product); Opportunities (external positive — large market, regulatory tailwinds, "
     "partnership potential); Threats (external negative — incumbents, regulation changes, "
     "economic downturn). Use SWOT to shape strategic priorities."),
]

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_index = None           # FAISS index
_chunks: list[str] = [] # text chunks parallel to index vectors
_embedder = None        # SentenceTransformer model
_index_data_dir: Optional[str] = None  # data_dir used to build current index


def _get_embedder():
    """Lazily load the sentence-transformer model."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-transformer model loaded: all-MiniLM-L6-v2")
        except Exception as exc:
            logger.error("Failed to load sentence-transformer: %s", exc)
            raise
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    """Return a float32 numpy array of embeddings."""
    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def _load_text_file(path: Path) -> str:
    """Read a plain-text file."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""


def _load_pdf_file(path: Path) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text_parts: list[str] = []
    try:
        import PyPDF2
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
    except Exception as exc:
        logger.warning("Could not parse PDF %s: %s", path, exc)
    return "\n".join(text_parts)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into overlapping word-level chunks.

    The loop runs over the full word list (not ``len(words) - overlap``)
    so the final partial chunk is never dropped.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def _load_documents(data_dir: str = "data") -> list[str]:
    """Load all PDFs and TXTs from the data directory."""
    data_path = Path(data_dir)
    all_chunks: list[str] = []

    if not data_path.exists():
        logger.info("Data directory '%s' not found; using fallback knowledge.", data_dir)
        return []

    files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.txt"))
    if not files:
        logger.info("No PDF/TXT files in '%s'; using fallback knowledge.", data_dir)
        return []

    for file in files:
        logger.info("Loading: %s", file.name)
        if file.suffix.lower() == ".pdf":
            text = _load_pdf_file(file)
        else:
            text = _load_text_file(file)
        all_chunks.extend(_chunk_text(text))

    logger.info("Loaded %d chunks from %d document(s).", len(all_chunks), len(files))
    return all_chunks


def build_index(data_dir: str = "data", force: bool = False) -> None:
    """Build the FAISS index from documents + fallback knowledge.

    Idempotent: skips rebuilding if the index already exists for the same
    ``data_dir``.  Pass ``force=True`` to force a rebuild (e.g. after adding
    new documents).
    """
    global _index, _chunks, _index_data_dir

    # Skip rebuild if already indexed for the same directory
    if not force and _index is not None and _index_data_dir == data_dir:
        logger.info("FAISS index already built for '%s' — skipping rebuild.", data_dir)
        return

    doc_chunks = _load_documents(data_dir)

    # Deduplicate: preserve order, remove exact duplicate chunks
    seen: set[str] = set()
    unique_chunks: list[str] = []
    for chunk in doc_chunks + FALLBACK_KNOWLEDGE:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    _chunks = unique_chunks
    _index_data_dir = data_dir

    logger.info("Building FAISS index over %d unique chunks …", len(_chunks))
    try:
        import faiss
        vectors = _embed(_chunks)
        # Normalise vectors: enables cosine-similarity semantics via IndexFlatIP
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        # IndexFlatIP with normalised vectors = exact cosine similarity search
        _index = faiss.IndexFlatIP(dim)
        _index.add(vectors)
        logger.info("FAISS index ready (cosine/IP). Vectors: %d, Dim: %d", _index.ntotal, dim)
    except Exception as exc:
        logger.error("FAISS index build failed: %s", exc)
        _index = None


def retrieve(query: str, top_k: int = 5, data_dir: str = "data") -> str:
    """
    Retrieve the top-k most relevant chunks for ``query``.

    Returns a single concatenated string ready to be injected into a prompt.
    Falls back to an empty string on any failure — never crashes the caller.
    """
    global _index, _chunks

    if _index is None:
        try:
            build_index(data_dir)
        except Exception as exc:
            logger.error("RAG build_index failed during retrieve: %s", exc)
            return ""

    if _index is None or not _chunks:
        logger.warning("RAG index unavailable; returning empty context.")
        return ""

    try:
        import faiss
        query_vec = _embed([query])
        # Normalise query vector to match IndexFlatIP (cosine similarity)
        faiss.normalize_L2(query_vec)
        actual_k = min(top_k, len(_chunks))
        distances, indices = _index.search(query_vec, actual_k)
        results: list[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(_chunks):
                results.append(_chunks[idx])
        context = "\n\n---\n\n".join(results)
        return context
    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return ""


def get_index_stats() -> dict:
    """Return basic stats about the loaded index."""
    return {
        "total_chunks": len(_chunks),
        "index_ready": _index is not None,
        "fallback_chunks": len(FALLBACK_KNOWLEDGE),
    }
