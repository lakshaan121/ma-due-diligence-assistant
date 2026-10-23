"""
app.py

Streamlit entry point for the AI-Powered M&A Due Diligence Assistant.

This file's job is ORCHESTRATION ONLY -- it calls into src/ modules for
every actual piece of logic (extraction, chunking, embedding, search,
LLM calls, classification). Keeping app.py thin like this means each
concept can be explained, tested, and demonstrated independently of
the UI, which is exactly what an interviewer will want to see.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import extract_text_by_page, PDFProcessingError
from src.chunker import chunk_pages
from src.embeddings import embed_texts, EmbeddingError
from src.vector_store import build_index, retrieve, get_retrieval_settings
from src.rag_pipeline import answer_question, RAGPipelineError
from src.utils import classify_question, safe_error_message, build_summary_report
from src.financial_analysis import calculate_ratios, parse_amount
from src.risk_analysis import flag_risks

load_dotenv()

st.set_page_config(page_title="M&A Due Diligence Assistant", page_icon="📄")

# --------------------------------------------------------------------
# White & Green theme -- kept to plain CSS overrides, no dark mode,
# no gradients, no extra accent colors, per the project's UI spec.
# --------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    /* Body text: dark colour so it stays readable on the white background
       even when the browser/OS or Streamlit is in dark mode. */
    .stApp p, .stApp li, .stApp label, .stApp span,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] { color: #1F2937; }
    /* Input boxes: white background with dark text. */
    .stApp input, .stApp textarea {
        background-color: #FFFFFF; color: #1F2937;
    }
    /* Keep button text white on the green buttons. */
    .stButton > button p, .stButton > button span,
    .stDownloadButton > button p, .stDownloadButton > button span { color: #FFFFFF; }
    h1, h2, h3 { color: #1E7A34; }
    .stButton > button {
        background-color: #1E7A34;
        color: #FFFFFF;
        border: 1px solid #1E7A34;
    }
    .stButton > button:hover {
        background-color: #16602A;
        border: 1px solid #16602A;
    }
    div[data-testid="stFileUploader"] section {
        border: 1px solid #1E7A34;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered M&A Due Diligence Assistant")
st.caption("RAG-Based Document Analysis for M&A Due Diligence")
st.caption(
    "Prototype — for initial document review only. "
    "Not a substitute for professional M&A, legal, or financial advisors."
)

# --------------------------------------------------------------------
# Session state: everything built from the uploaded PDFs needs to
# survive across Streamlit reruns (Streamlit re-executes this whole
# script on every interaction, so without session_state we'd re-process
# the PDFs on every single question).
# --------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []       # list of {"document","page","chunk_text"}
if "index" not in st.session_state:
    st.session_state.index = None      # FAISS index built from chunk embeddings
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []  # [{"name": str, "pages": int}, ...]
if "processed_signature" not in st.session_state:
    # (name, size) of the files the current index was built from, so we can
    # tell when the uploader now holds different files than the index.
    st.session_state.processed_signature = None


def _upload_signature(files) -> tuple:
    return tuple((f.name, f.size) for f in (files or []))


def _index_is_stale(files) -> bool:
    """True if the uploader no longer matches the documents behind the index."""
    return _upload_signature(files) != st.session_state.processed_signature

# --------------------------------------------------------------------
# 1. DOCUMENT UPLOAD
# --------------------------------------------------------------------
st.header("1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload company PDF documents (annual report, financials, legal filings)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Process Documents"):
    all_chunks = []
    processed_docs = []

    # Clear everything derived from the PREVIOUS documents first, so a failed
    # run can never leave an old index/results active.
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.processed_docs = []
    st.session_state.processed_signature = None
    st.session_state.risk_flags = []
    st.session_state.report_text = None

    with st.spinner("Extracting text, chunking, and building the search index..."):
        used_names = set()
        for uploaded_file in uploaded_files:
            # Two uploads with the same filename would be indistinguishable
            # in citations and chunk counts, so make the second one unique.
            doc_name = uploaded_file.name
            n = 2
            while doc_name in used_names:
                doc_name = f"{uploaded_file.name} ({n})"
                n += 1
            used_names.add(doc_name)

            try:
                pages = extract_text_by_page(uploaded_file, doc_name)
            except PDFProcessingError as exc:
                st.error(safe_error_message(exc, f"Could not process {uploaded_file.name}."))
                continue

            doc_chunks = chunk_pages(pages)
            all_chunks.extend(doc_chunks)
            processed_docs.append({"name": doc_name, "pages": len(pages)})

        if all_chunks:
            try:
                chunk_texts = [c["chunk_text"] for c in all_chunks]
                embeddings = embed_texts(chunk_texts)
                index = build_index(embeddings)
            except (EmbeddingError, ValueError) as exc:
                st.error(safe_error_message(exc, "Could not build the search index."))
            else:
                st.session_state.chunks = all_chunks
                st.session_state.index = index
                st.session_state.processed_docs = processed_docs
                st.session_state.processed_signature = _upload_signature(uploaded_files)

if st.session_state.processed_docs:
    st.subheader("Processed Documents")
    for doc in st.session_state.processed_docs:
        st.write(f"✅ **{doc['name']}** — {doc['pages']} pages, "
                  f"{sum(1 for c in st.session_state.chunks if c['document'] == doc['name'])} chunks")

# --------------------------------------------------------------------
# 2. ASK QUESTIONS
# --------------------------------------------------------------------
st.header("2. Ask a Question")
question = st.text_input("Ask a question about the uploaded documents")

if st.button("Analyze"):
    if st.session_state.index is None:
        st.warning("Please upload and process at least one document first.")
        result = None
    elif _index_is_stale(uploaded_files):
        st.warning("The uploaded files have changed since they were last processed. "
                   "Please click Process Documents again.")
        result = None
    elif not question.strip():
        st.warning("Please enter a question.")
        result = None
    else:
        with st.spinner("Searching documents and generating answer..."):
            top_k, min_score = get_retrieval_settings()
            result = None
            try:
                query_embedding = embed_texts([question])
                retrieved_chunks = retrieve(st.session_state.index, st.session_state.chunks,
                                            query_embedding, top_k=top_k, min_score=min_score)
                result = answer_question(question, retrieved_chunks)
            except (EmbeddingError, RAGPipelineError, ValueError) as exc:
                st.error(safe_error_message(exc, "Could not get an answer."))

        if result:
            # --- ANSWER ---
            st.subheader("Answer")
            st.write(result["answer"])

            # --- DUE DILIGENCE CATEGORY ---
            category = classify_question(question)
            st.subheader("Due Diligence Category")
            st.write(f"🏷️ **{category}**")

            # --- SOURCES ---
            st.subheader("Sources")
            if result["sources"]:
                for i, source in enumerate(result["sources"], start=1):
                    with st.expander(f"Source {i}: {source['document']} — Page {source['page']}"):
                        st.write(source["chunk_text"])
            else:
                st.write("No matching sources were found in the uploaded documents.")

# --------------------------------------------------------------------
# 3. FINANCIAL ANALYSIS
# --------------------------------------------------------------------
# These figures are entered manually rather than auto-extracted: the
# spec explicitly avoids building an advanced structured-data parser
# for this prototype, and manual entry keeps every number traceable
# to what the user actually typed in, not a guess by the system.
st.header("3. Financial Analysis")
st.caption("Enter any figures you have from the documents (in the same currency unit, e.g. ₹ crore). Leave blank if unavailable.")

col1, col2 = st.columns(2)
with col1:
    revenue_input = st.text_input("Revenue", key="revenue_input")
    net_profit_input = st.text_input("Net Profit", key="net_profit_input")
    total_debt_input = st.text_input("Total Debt", key="total_debt_input")
with col2:
    total_assets_input = st.text_input("Total Assets", key="total_assets_input")
    total_equity_input = st.text_input("Total Equity (optional)", key="total_equity_input")


if st.button("Calculate Ratios"):
    raw_inputs = {
        "revenue": revenue_input,
        "net_profit": net_profit_input,
        "total_debt": total_debt_input,
        "total_assets": total_assets_input,
        "total_equity": total_equity_input,
    }
    figures, input_errors = {}, {}
    for field, raw in raw_inputs.items():
        # Malformed text is reported as "invalid", not silently treated as blank.
        figures[field], input_errors[field] = parse_amount(raw, field)
    ratios = calculate_ratios(figures, input_errors)

    ratio_labels = {
        "net_profit_margin": "Net Profit Margin",
        "debt_to_asset": "Debt-to-Asset Ratio",
        "debt_to_equity": "Debt-to-Equity Ratio",
    }

    for key, label in ratio_labels.items():
        result = ratios[key]
        if result["value"] is not None:
            unit_suffix = "%" if result["unit"] == "%" else ""
            st.write(f"✅ **{label}:** {result['value']}{unit_suffix} "
                     f"(from {result['inputs_used']})")
        else:
            st.write(f"⚪ **{label}:** Not calculated — {result['reason']}")

    # Stash the last-calculated ratios in session state so the report
    # section below can include them without recalculating.
    st.session_state.last_ratios = ratios

if "last_ratios" not in st.session_state:
    st.session_state.last_ratios = None

# --------------------------------------------------------------------
# 4. KEY AREAS FOR FURTHER INVESTIGATION (risk analysis)
# --------------------------------------------------------------------
st.header("4. Key Areas for Further Investigation")
st.caption("Scans ALL processed document chunks for known risk-indicator keywords "
           "(high debt, declining revenue, losses, litigation, etc.). "
           "This is evidence-based flagging, not a predictive risk score.")

if "risk_flags" not in st.session_state:
    st.session_state.risk_flags = []

if st.button("Scan for Risks"):
    if not st.session_state.chunks:
        st.warning("Please upload and process at least one document first.")
    elif _index_is_stale(uploaded_files):
        st.warning("The uploaded files have changed since they were last processed. "
                   "Please click Process Documents again.")
    else:
        st.session_state.risk_flags = flag_risks(st.session_state.chunks)

if st.session_state.risk_flags:
    for flag in st.session_state.risk_flags:
        with st.expander(f"⚠️ {flag['label']}: {flag['category']} "
                          f"— {flag['document']}, Page {flag['page']}"):
            st.write(flag["evidence"])
elif st.session_state.chunks:
    st.write("No risk keywords found yet — click **Scan for Risks** to check the processed documents.")

# --------------------------------------------------------------------
# 5. DUE DILIGENCE SUMMARY REPORT
# --------------------------------------------------------------------
st.header("5. Due Diligence Summary Report")
st.caption("Generates a downloadable Markdown report from everything processed so far "
           "in this session (documents, last calculated ratios, risk flags).")

if st.button("Generate Report"):
    report_text = build_summary_report(
        processed_docs=st.session_state.processed_docs,
        ratios=st.session_state.last_ratios,
        risk_flags=st.session_state.risk_flags,
    )
    st.session_state.report_text = report_text

if st.session_state.get("report_text"):
    st.text_area("Report Preview", st.session_state.report_text, height=300)
    st.download_button(
        "Download Report (Markdown)",
        data=st.session_state.report_text,
        file_name="due_diligence_summary.md",
        mime="text/markdown",
    )
