"""
test_basic.py

Basic tests for the pipeline. Filled in incrementally as each module
is built (PDF extraction here in Phase 3; chunking, embeddings, etc.
will add their own test functions in later phases).

Run with: python -m pytest tests/test_basic.py -v
"""

import sys
import os

# Allow running these tests without installing the project as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.pdf_processor import extract_text_by_page, PDFProcessingError
from src.chunker import chunk_pages
from src.embeddings import embed_texts
from src.vector_store import build_index, search
from src.rag_pipeline import build_context_and_prompt, answer_question, RAGPipelineError
from src.utils import classify_question, build_summary_report
from src.financial_analysis import calculate_ratios
from src.risk_analysis import flag_risks
import numpy as np
from src.embeddings import get_embedding_model, EmbeddingError

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents")


def test_extract_text_from_valid_pdf():
    """A PDF with real text should return one dict per page with text."""
    path = os.path.join(SAMPLE_DIR, "sample_annual_report.pdf")
    with open(path, "rb") as f:
        pages = extract_text_by_page(f, "sample_annual_report.pdf")

    assert len(pages) == 2
    assert pages[0]["document"] == "sample_annual_report.pdf"
    assert pages[0]["page"] == 1
    assert "liabilities" in pages[0]["text"].lower()
    assert pages[1]["page"] == 2
    assert "litigation" in pages[1]["text"].lower()


def test_blank_pdf_raises_no_text_error():
    """A PDF with pages but no text should raise PDFProcessingError."""
    path = os.path.join(SAMPLE_DIR, "sample_test.pdf")
    with open(path, "rb") as f:
        with pytest.raises(PDFProcessingError):
            extract_text_by_page(f, "sample_test.pdf")


def test_invalid_pdf_raises_error():
    """A non-PDF file should raise PDFProcessingError, not crash."""
    import io
    fake_file = io.BytesIO(b"this is not a real pdf")
    with pytest.raises(PDFProcessingError):
        extract_text_by_page(fake_file, "fake.pdf")


def test_chunking_splits_long_page_into_multiple_chunks():
    """A page longer than chunk_size should produce more than one chunk,
    and every chunk should keep the original document/page tags."""
    long_text = "This is a sample sentence about company revenue. " * 60  # ~3000 chars
    pages = [{"document": "test.pdf", "page": 7, "text": long_text}]

    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["document"] == "test.pdf"
        assert chunk["page"] == 7
        assert len(chunk["chunk_text"]) <= 1000


def test_chunking_short_page_stays_one_chunk():
    """A page shorter than chunk_size should not be split at all."""
    pages = [{"document": "test.pdf", "page": 1, "text": "Short page text."}]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == "Short page text."


def test_chunking_overlap_preserves_boundary_text():
    """Consecutive chunks from the same long page should share some
    overlapping text, so a sentence split across the boundary isn't lost."""
    long_text = "Sentence number {}. ".format(0)
    long_text = "".join(f"Sentence number {i}. " for i in range(200))  # long text
    pages = [{"document": "test.pdf", "page": 3, "text": long_text}]

    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1

    # The end of chunk 1 and the start of chunk 2 should share some text
    # (that's what overlap guarantees).
    tail_of_first = chunks[0]["chunk_text"][-50:]
    assert any(word in chunks[1]["chunk_text"] for word in tail_of_first.split())


@pytest.fixture
def embedding_model():
    """Skip (not fail) when the real model can't be loaded, e.g. offline or
    sentence-transformers not installed. These tests need the real model."""
    try:
        return get_embedding_model()
    except EmbeddingError as exc:
        pytest.skip(f"embedding model unavailable: {exc}")


def test_embedding_shape_is_correct(embedding_model):
    """Each embedding should be a 384-dimensional vector (MiniLM's output size)."""
    embeddings = embed_texts(["The company has high debt."])
    assert embeddings.shape == (1, 384)


def test_embedding_batch_matches_input_count(embedding_model):
    """Embedding N texts should return exactly N vectors."""
    texts = ["Revenue increased.", "Legal case pending.", "Operations expanded."]
    embeddings = embed_texts(texts)
    assert embeddings.shape[0] == len(texts)


def test_similar_sentences_have_high_cosine_similarity(embedding_model):
    """Two sentences with similar MEANING but different words should
    end up with embeddings that are close together -- this is the
    entire point of using semantic embeddings instead of keyword match."""
    sentence_a = "The company has high debt."
    sentence_b = "The company has significant borrowings."
    sentence_c = "The weather was sunny yesterday."

    embeddings = embed_texts([sentence_a, sentence_b, sentence_c])

    # Vectors are already unit-normalized, so dot product == cosine similarity.
    similarity_ab = float(embeddings[0] @ embeddings[1])
    similarity_ac = float(embeddings[0] @ embeddings[2])

    # "debt" vs "borrowings" should be far more similar than "debt" vs "weather".
    assert similarity_ab > similarity_ac
    assert similarity_ab > 0.5


def _unit_vector(vec):
    """Helper: normalize a plain list into a unit-length float32 vector,
    matching how embeddings.py normalizes real embeddings."""
    arr = np.array(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


def test_build_index_and_search_returns_closest_match():
    """Using hand-crafted vectors (no model download needed), confirm
    that search() returns the vector closest in direction to the query."""
    # Three synthetic 4-dim "chunk" vectors, deliberately pointing in
    # different directions.
    chunk_a = _unit_vector([1, 0, 0, 0])   # e.g. "debt" chunk
    chunk_b = _unit_vector([0, 1, 0, 0])   # e.g. "legal" chunk
    chunk_c = _unit_vector([0.9, 0.1, 0, 0])  # very close to chunk_a

    embeddings = np.array([chunk_a, chunk_b, chunk_c], dtype="float32")
    index = build_index(embeddings)

    # Query points almost exactly like chunk_a/chunk_c, nothing like chunk_b.
    query = np.array([_unit_vector([1, 0, 0, 0])], dtype="float32")
    scores, indices = search(index, query, top_k=2)

    # The two closest chunks should be index 0 (chunk_a) and 2 (chunk_c),
    # in that order, not index 1 (chunk_b).
    assert list(indices[0]) == [0, 2]
    assert scores[0][0] > scores[0][1]


def test_search_top_k_does_not_exceed_available_chunks():
    """Asking for more results than exist should not crash -- it should
    just return however many chunks are actually in the index."""
    embeddings = np.array([_unit_vector([1, 0]), _unit_vector([0, 1])], dtype="float32")
    index = build_index(embeddings)
    query = np.array([_unit_vector([1, 0])], dtype="float32")

    scores, indices = search(index, query, top_k=10)
    assert indices.shape[1] == 2  # capped at the 2 vectors actually stored


def test_build_index_rejects_empty_embeddings():
    """Building an index from zero embeddings should raise a clear error,
    not fail deep inside FAISS with a cryptic message."""
    with pytest.raises(ValueError):
        build_index(np.array([]))


def test_build_context_and_prompt_includes_sources_and_question():
    """The prompt sent to the LLM should clearly label each chunk's
    source/page and include the original question -- no network call."""
    chunks = [
        {"document": "Annual_Report.pdf", "page": 45, "chunk_text": "Total liabilities increased."},
        {"document": "Annual_Report.pdf", "page": 12, "chunk_text": "Revenue grew 10% year over year."},
    ]
    prompt = build_context_and_prompt("What are the liabilities?", chunks)

    assert "Annual_Report.pdf" in prompt
    assert "Page 45" in prompt
    assert "Total liabilities increased." in prompt
    assert "What are the liabilities?" in prompt


def test_build_context_and_prompt_handles_no_retrieved_chunks():
    """If retrieval found nothing, the prompt should say so explicitly
    rather than silently sending an empty context to the LLM."""
    prompt = build_context_and_prompt("Any legal issues?", [])
    assert "No relevant chunks" in prompt
    assert "Any legal issues?" in prompt


def test_answer_question_raises_clear_error_without_api_key(monkeypatch):
    """Calling the LLM with no API key configured should raise our own
    RAGPipelineError with a helpful message, not a raw library exception."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RAGPipelineError):
        answer_question("What is the revenue?", [])


@pytest.mark.parametrize("question,expected_category", [
    ("What are the company's total liabilities?", "Financial"),
    ("Is there any pending litigation?", "Legal/Regulatory"),
    ("Who are the major suppliers?", "Business/Operational"),
    ("What red flags exist in this deal?", "Risk"),
])
def test_classify_question_matches_expected_category(question, expected_category):
    assert classify_question(question) == expected_category


def test_classify_question_defaults_when_no_keyword_matches():
    """A question with no matching keyword should fall back to the
    documented default category rather than raising an error."""
    assert classify_question("Tell me something interesting.") == "Business/Operational"


def test_streamlit_app_loads_without_exceptions():
    """Smoke test: the app should render its title and both section
    headers on initial load, with no unhandled exception -- before any
    file is uploaded or any button is clicked."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"),
                            default_timeout=20)
    at.run()

    assert len(at.exception) == 0
    assert at.title[0].value == "AI-Powered M&A Due Diligence Assistant"
    header_texts = [h.value for h in at.header]
    assert "1. Upload Documents" in header_texts
    assert "2. Ask a Question" in header_texts
    assert "3. Financial Analysis" in header_texts
    assert "4. Key Areas for Further Investigation" in header_texts
    assert "5. Due Diligence Summary Report" in header_texts


def test_build_summary_report_includes_all_sections_with_data():
    """A fully-populated report should include doc names, ratio values,
    and risk evidence in their respective sections."""
    processed_docs = [{"name": "Annual_Report.pdf", "pages": 50}]
    ratios = {
        "net_profit_margin": {"value": 18.5, "unit": "%", "inputs_used": {"net_profit": 185, "revenue": 1000}},
        "debt_to_asset": {"value": None, "reason": "total_debt and/or total_assets not provided"},
        "debt_to_equity": {"value": None, "reason": "total_debt and/or total_equity not provided"},
    }
    risk_flags = [{
        "category": "Legal Disputes", "label": "Potential Concern",
        "evidence": "The company faces ongoing litigation.",
        "document": "Annual_Report.pdf", "page": 12,
    }]

    report = build_summary_report(processed_docs, ratios, risk_flags)

    assert "Annual_Report.pdf" in report
    assert "18.5" in report
    assert "Legal Disputes" in report
    assert "ongoing litigation" in report
    assert "## 4. Limitations" in report


def test_build_summary_report_handles_empty_session_honestly():
    """With nothing processed, the report should say so explicitly in
    each section rather than fabricating content or crashing."""
    report = build_summary_report(processed_docs=[], ratios=None, risk_flags=[])

    assert "No documents were processed" in report
    assert "No financial ratios were calculated" in report
    assert "No risk-indicator keywords were found" in report


def test_calculate_ratios_all_inputs_provided():
    """With every input given, all three ratios should compute correctly."""
    figures = {
        "revenue": 1000,
        "net_profit": 185,
        "total_debt": 400,
        "total_assets": 1000,
        "total_equity": 500,
    }
    results = calculate_ratios(figures)

    assert results["net_profit_margin"]["value"] == 18.5
    assert results["debt_to_asset"]["value"] == 0.4
    assert results["debt_to_equity"]["value"] == 0.8


def test_calculate_ratios_missing_inputs_are_skipped_with_reason():
    """Ratios should never be fabricated -- missing inputs should produce
    value=None plus a clear reason, not a crash or a guessed number."""
    figures = {"revenue": 1000}  # net_profit, debt, assets, equity all missing

    results = calculate_ratios(figures)

    assert results["net_profit_margin"]["value"] is None
    assert "reason" in results["net_profit_margin"]
    assert results["debt_to_asset"]["value"] is None
    assert results["debt_to_equity"]["value"] is None


def test_calculate_ratios_handles_zero_denominator_safely():
    """A zero revenue/assets/equity should not raise ZeroDivisionError --
    it should be treated the same as 'not available'."""
    figures = {"revenue": 0, "net_profit": 50, "total_debt": 100, "total_assets": 0}
    results = calculate_ratios(figures)

    assert results["net_profit_margin"]["value"] is None
    assert results["debt_to_asset"]["value"] is None


def test_flag_risks_detects_known_keyword():
    """A chunk containing a known risk keyword should produce a flagged
    item with correct category, label, evidence, and source tagging."""
    chunks = [
        {"document": "Annual_Report.pdf", "page": 45,
         "chunk_text": "The company reported significant borrowings this year."},
    ]
    flags = flag_risks(chunks)

    assert len(flags) == 1
    assert flags[0]["category"] == "High Debt"
    assert flags[0]["label"] == "Potential Concern"
    assert flags[0]["document"] == "Annual_Report.pdf"
    assert flags[0]["page"] == 45
    assert "borrowings" in flags[0]["evidence"].lower()


def test_flag_risks_no_keywords_returns_empty_list():
    """A chunk with no risk keywords should simply produce no flags --
    not an error, not a fabricated low-confidence flag."""
    chunks = [
        {"document": "Annual_Report.pdf", "page": 2,
         "chunk_text": "The company's headquarters are located in Chennai."},
    ]
    assert flag_risks(chunks) == []


def test_flag_risks_detects_multiple_categories_in_different_chunks():
    """Risks from different chunks/categories should all be captured
    independently."""
    chunks = [
        {"document": "A.pdf", "page": 1, "chunk_text": "The company faces ongoing litigation."},
        {"document": "A.pdf", "page": 2, "chunk_text": "Revenue decreased compared to last year."},
    ]
    flags = flag_risks(chunks)
    categories = {f["category"] for f in flags}
    assert "Legal Disputes" in categories
    assert "Declining Revenue" in categories
