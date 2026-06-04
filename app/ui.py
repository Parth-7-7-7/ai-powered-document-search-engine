"""
Streamlit UI for AI-Powered Legal Document Search.

Run:
    streamlit run app/ui.py
"""

import re
import sys
from pathlib import Path

import streamlit as st

# ── Path fix so imports work when run from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.search import LegalSearchEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    background: #f8f9fa;
    border-left: 4px solid #dee2e6;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 4px;
    font-size: 0.88rem;
}
.result-card.contract  { border-left-color: #2563eb; }
.result-card.judgment  { border-left-color: #16a34a; }
.result-card.hybrid    { border-left-color: #9333ea; }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 6px;
}
.badge-contract { background:#dbeafe; color:#1d4ed8; }
.badge-judgment { background:#dcfce7; color:#15803d; }
.score-tag  { color:#6b7280; font-size:0.75rem; }
.title-text { font-weight:600; font-size:0.85rem; color:#111827; margin:4px 0 6px 0; }
.chunk-text { color:#374151; line-height:1.55; }
mark        { background:#fef08a; padding:0 2px; border-radius:2px; }
.col-header { text-align:center; padding:8px 0 4px 0; border-radius:6px; margin-bottom:12px; }
.col-kw   { background:#eff6ff; color:#1d4ed8; }
.col-sem  { background:#f0fdf4; color:#15803d; }
.col-hyb  { background:#faf5ff; color:#7e22ce; }
/* tighten the View Full Text button */
div[data-testid="stButton"] > button[kind="secondary"] {
    font-size: 0.72rem;
    padding: 2px 10px;
    margin-bottom: 10px;
    height: auto;
}
</style>
""", unsafe_allow_html=True)


# ── Engine (cached so it loads only once) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading search engine…")
def get_engine():
    return LegalSearchEngine()


# ── Full-document dialog ──────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "documents"

@st.dialog("Full Document", width="large")
def show_full_chunk(result: dict) -> None:
    dtype     = result["doc_type"]
    badge_bg  = "#dbeafe" if dtype == "contract" else "#dcfce7"
    badge_col = "#1d4ed8" if dtype == "contract" else "#15803d"
    border    = "#2563eb" if dtype == "contract" else "#16a34a"

    # Header
    st.markdown(
        f'<span style="background:{badge_bg};color:{badge_col};padding:3px 12px;'
        f'border-radius:999px;font-size:0.75rem;font-weight:700;text-transform:uppercase">'
        f'{dtype}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {result['title']}")
    st.caption(f"File: `{result['filename']}` &nbsp;·&nbsp; Source: {result['source']}")

    # Matched chunk callout
    st.markdown("**Matched chunk** (what the search found):")
    st.markdown(
        f"<div style='background:{badge_bg};border-left:4px solid {border};"
        f"border-radius:6px;padding:12px 16px;font-size:0.85rem;line-height:1.6;"
        f"color:#1f2937;margin-bottom:16px'>{result['text']}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Full document:**")

    # Read entire document from disk
    doc_path = DOCS_DIR / result["filename"]
    if doc_path.exists():
        full_text = doc_path.read_text(encoding="utf-8")
        char_count = len(full_text)
        st.caption(f"{char_count:,} characters · scroll to read")
        st.text_area(
            label="full_doc",
            value=full_text,
            height=500,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.error(f"Document file not found: {result['filename']}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def highlight(text: str, query: str) -> str:
    for term in re.findall(r'\b\w+\b', query):
        if len(term) < 3:
            continue
        text = re.sub(f'(?i)({re.escape(term)})', r'<mark>\1</mark>', text)
    return text


def truncate(text: str, max_chars: int = 380) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def deduplicate(results: list[dict], top_k: int) -> list[dict]:
    """Keep only the highest-scoring chunk per document, then re-number ranks from 1."""
    seen: set[str] = set()
    out = []
    for r in results:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            out.append(r)
    # re-number ranks sequentially so display shows 1,2,3... not 1,2,4,5...
    for i, r in enumerate(out):
        r = dict(r)          # shallow copy so we don't mutate the original
        r["rank"] = i + 1
        out[i] = r
    return out[:top_k]


def render_card(result: dict, key_suffix: str, show_highlight: bool = False, query: str = "") -> None:
    dtype   = result["doc_type"]
    badge   = f'<span class="badge badge-{dtype}">{dtype}</span>'
    score   = f'<span class="score-tag">score: {result["score"]:.4f} &nbsp;·&nbsp; rank #{result["rank"]}</span>'
    title   = result["title"][:70] + ("…" if len(result["title"]) > 70 else "")
    snippet = truncate(result["text"])
    if show_highlight and query:
        snippet = highlight(snippet, query)

    st.markdown(f"""
    <div class="result-card {dtype}">
        {badge}{score}
        <div class="title-text">{title}</div>
        <div class="chunk-text">{snippet}</div>
    </div>
    """, unsafe_allow_html=True)

    # "View full text" button — unique key per chunk + column
    if st.button("📄 View full text", key=f"view_{result['chunk_id']}_{key_suffix}",
                 use_container_width=False):
        show_full_chunk(result)


def render_column_header(label: str, css_class: str, count: int) -> None:
    st.markdown(
        f'<div class="col-header {css_class}"><strong>{label}</strong> &nbsp;·&nbsp; {count} result(s)</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal Search")
    st.markdown("---")

    top_k = st.slider("Results per mode", min_value=3, max_value=15, value=5, step=1)

    st.markdown("---")
    st.markdown("**Document corpus**")
    st.markdown("- 20 commercial contracts (CUAD)")
    st.markdown("- 15 Indian SC judgments")
    st.markdown("- 3,532 indexed chunks")

    st.markdown("---")
    dedup = st.toggle("One result per document", value=True,
                      help="ON: show only the best-matching chunk per document. OFF: show all chunks (same doc may appear multiple times).")

    st.markdown("---")
    st.markdown("**Search modes**")
    st.markdown("🔵 **Keyword** — BM25 exact matching")
    st.markdown("🟢 **Semantic** — MiniLM cosine similarity")
    st.markdown("🟣 **Hybrid** — RRF fusion of both")

    st.markdown("---")
    st.caption("Trilegal Assignment · AI-Powered Legal Search")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("## AI-Powered Legal Document Search")
st.markdown(
    "Search across **35 legal documents** (contracts + Supreme Court judgments) "
    "using keyword matching, semantic understanding, or hybrid fusion."
)

# ── Search bar ────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="Search query",
        placeholder="e.g.  termination clause breach  /  income tax deduction property  /  software license",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("Search", use_container_width=True, type="primary")

# ── Sample queries ────────────────────────────────────────────────────────────
st.markdown(
    "<div style='font-size:0.8rem;color:#6b7280;margin-top:-8px;margin-bottom:16px'>"
    "Try: &nbsp;"
    "<code>termination breach</code> &nbsp;·&nbsp; "
    "<code>income tax deduction</code> &nbsp;·&nbsp; "
    "<code>indemnification liability</code> &nbsp;·&nbsp; "
    "<code>software license intellectual property</code> &nbsp;·&nbsp; "
    "<code>limitation act fraud decree</code>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Run search ────────────────────────────────────────────────────────────────
if query and (search_clicked or query):
    engine = get_engine()

    with st.spinner("Searching…"):
        # fetch 3× candidates when deduplicating so we still get top_k unique docs
        fetch_k = top_k * 3 if dedup else top_k
        kw_results  = engine.keyword_search(query,  top_k=fetch_k)
        sem_results = engine.semantic_search(query, top_k=fetch_k)
        hyb_results = engine.hybrid_search(query,   top_k=fetch_k)

    if dedup:
        kw_results  = deduplicate(kw_results,  top_k)
        sem_results = deduplicate(sem_results, top_k)
        hyb_results = deduplicate(hyb_results, top_k)

    st.markdown("---")

    # ── Three-column layout ───────────────────────────────────────────────────
    col_kw, col_sem, col_hyb = st.columns(3)

    with col_kw:
        render_column_header("🔵 Keyword (BM25)", "col-kw", len(kw_results))
        if kw_results:
            for r in kw_results:
                render_card(r, key_suffix="kw", show_highlight=True, query=query)
        else:
            st.info("No keyword matches found.")

    with col_sem:
        render_column_header("🟢 Semantic (MiniLM)", "col-sem", len(sem_results))
        if sem_results:
            for r in sem_results:
                render_card(r, key_suffix="sem")
        else:
            st.info("No semantic matches found.")

    with col_hyb:
        render_column_header("🟣 Hybrid (RRF)", "col-hyb", len(hyb_results))
        if hyb_results:
            for r in hyb_results:
                render_card(r, key_suffix="hyb")
        else:
            st.info("No hybrid results found.")

    # ── Difference callout ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 How do the results differ?", expanded=False):
        # Unique documents per mode (deduplicated by filename)
        kw_docs  = {r["filename"]: r["title"] for r in kw_results}
        sem_docs = {r["filename"]: r["title"] for r in sem_results}
        hyb_docs = {r["filename"]: r["title"] for r in hyb_results}

        only_kw  = {f for f in kw_docs  if f not in sem_docs}
        only_sem = {f for f in sem_docs if f not in kw_docs}
        shared   = {f for f in kw_docs  if f in sem_docs}

        # Totals per method
        total_kw  = len(kw_docs)
        total_sem = len(sem_docs)
        total_hyb = len(hyb_docs)

        c1, c2, c3 = st.columns(3)
        for col, label, total, exclusive, color, tip in [
            (c1, "BM25 (Keyword)",   total_kw,  len(only_kw),  "#2563eb",
             f"{len(only_kw)} doc(s) exclusive to BM25 · {len(shared)} shared with semantic"),
            (c2, "Semantic (MiniLM)", total_sem, len(only_sem), "#16a34a",
             f"{len(only_sem)} doc(s) exclusive to semantic · {len(shared)} shared with BM25"),
            (c3, "Found by both",    len(shared), len(shared),  "#9333ea",
             f"Same {len(shared)} doc(s) retrieved by both BM25 and semantic"),
        ]:
            col.markdown(
                f"<div style='background:#1e293b;border-left:4px solid {color};"
                f"border-radius:8px;padding:14px 16px;text-align:center'>"
                f"<div style='font-size:0.75rem;color:#94a3b8;margin-bottom:4px'>{label}</div>"
                f"<div style='font-size:2.2rem;font-weight:700;color:#f1f5f9'>{total}</div>"
                f"<div style='font-size:0.7rem;color:#94a3b8;margin-bottom:2px'>unique docs retrieved</div>"
                f"<div style='font-size:0.7rem;color:#64748b;margin-top:4px'>{tip}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

        # Document breakdown table
        all_files = set(kw_docs) | set(sem_docs)
        rows = []
        for fname in all_files:
            title = kw_docs.get(fname) or sem_docs.get(fname)
            in_kw  = "✅" if fname in kw_docs  else "—"
            in_sem = "✅" if fname in sem_docs  else "—"
            in_hyb = "✅" if fname in hyb_docs  else "—"
            rows.append((title[:65], in_kw, in_sem, in_hyb))
        rows.sort(key=lambda x: (x[1] != "✅", x[2] != "✅"))

        # Header
        h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
        h1.markdown("**Document**")
        h2.markdown("**BM25**")
        h3.markdown("**Semantic**")
        h4.markdown("**Hybrid**")
        st.markdown("<hr style='margin:4px 0 8px 0;border-color:#334155'>", unsafe_allow_html=True)

        for title, kw_v, sem_v, hyb_v in rows:
            r1, r2, r3, r4 = st.columns([5, 1, 1, 1])
            r1.markdown(f"<span style='font-size:0.82rem'>{title}</span>", unsafe_allow_html=True)
            r2.markdown(kw_v)
            r3.markdown(sem_v)
            r4.markdown(hyb_v)

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            "**Keyword (BM25)** matches exact words. "
            "**Semantic (MiniLM)** matches meaning — finds relevant docs even with different wording. "
            "**Hybrid (RRF)** combines both signals — documents appearing in both lists rank highest."
        )
        st.info(
            "**Why does Hybrid sometimes show a document not in the top-5 of either column?**\n\n"
            "Hybrid runs both searches on a larger pool (up to 50 candidates), not just the top-5 shown. "
            "A document ranked #12 in BM25 and #9 in semantic gets an RRF score of "
            "`1/(60+12) + 1/(60+9) = 0.028`, which can beat a document ranked #3 in only one method "
            "`1/(60+3) = 0.016`. This is the core insight of RRF — **consistent relevance across both methods "
            "beats being the best in just one**.",
            icon="💡",
        )

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;padding:48px 0;font-size:1rem;'>"
        "Enter a search query above to get started."
        "</div>",
        unsafe_allow_html=True,
    )
