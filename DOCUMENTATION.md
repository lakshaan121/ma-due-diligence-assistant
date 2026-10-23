# Project Documentation
## RAG-Powered Automated Due Diligence System for M&A (India-focused)

---

## Abstract

Mergers and acquisitions require reviewing large volumes of unstructured
documents — annual reports, financial statements, legal filings — to
identify risks and key facts before a deal proceeds. This project
implements a prototype AI assistant that lets a user upload such
documents and ask natural-language questions, using Retrieval-Augmented
Generation (RAG) to ground every answer in the actual uploaded text and
cite its exact source and page. The system additionally provides
rule-based question categorization, basic financial ratio calculation,
and keyword-based risk flagging, combined into a downloadable summary
report. The project demonstrates practical, end-to-end use of document
processing, embeddings, vector search, and LLMs at a scope appropriate
for a B.Tech final-year academic prototype.

## Introduction

Due diligence is the investigative process undertaken before a merger
or acquisition to verify facts and uncover risks about a target
company. It traditionally requires manually reading large volumes of
documents — a slow, labor-intensive process. This project explores
whether a Retrieval-Augmented Generation pipeline, combined with simple
rule-based analysis, can meaningfully assist that first-pass document
review, while remaining fully explainable and auditable (every answer
traces back to an exact source page).

## Problem Statement

Manually reviewing company documents during due diligence is
time-consuming and risks missing relevant information buried across
many pages. There is a need for a tool that can quickly surface
relevant passages in response to specific questions, without requiring
the reviewer to read every page, while still showing exactly where
each answer came from.

## Existing System

Currently, due diligence document review is performed manually by
analysts, lawyers, and accountants, often using simple keyword search
(Ctrl+F) within PDF readers. Keyword search fails to find semantically
related information phrased differently from the search term (e.g.
searching "debt" misses a sentence that only says "borrowings").
Commercial due-diligence software exists but is enterprise-priced and
not accessible for academic study of the underlying techniques.

## Limitations of Existing System

- Keyword search misses semantically similar but differently-worded
  content.
- No automatic categorization of findings into financial/legal/
  business/risk themes.
- No automatic financial ratio calculation tied to source evidence.
- Commercial tools are closed-source and not available for a student
  to study, modify, or demonstrate understanding of.

## Proposed System

A document-grounded Q&A system using RAG: uploaded PDFs are extracted,
chunked, embedded, and indexed with FAISS. A user's question is
embedded and matched against the chunk index via cosine similarity;
the top-k matching chunks are passed to an LLM, which is instructed to
answer only from that context and cite its source. The same uploaded
documents also feed a rule-based risk scanner and an optional
manually-entered financial ratio calculator, with everything combined
into a Markdown summary report.

## Aim

To build and demonstrate a simple, fully-explainable RAG-based
document analysis assistant for M&A due diligence, suitable for
discussion in a fresher-level technical interview.

## Objectives

1. Extract text from uploaded PDF documents with page-level tracking.
2. Split extracted text into overlapping chunks for embedding.
3. Generate semantic embeddings using a pre-trained sentence-transformer
   model.
4. Build a FAISS vector index for fast similarity search.
5. Implement a RAG pipeline that answers questions using only
   retrieved context, with source/page citation.
6. Classify questions into Financial / Legal / Business / Risk
   categories using keyword rules.
7. Calculate basic financial ratios from user-supplied figures.
8. Flag potential risk indicators with evidence, not a numeric score.
9. Generate a downloadable due diligence summary report.
10. Provide a simple, clean (white/green) Streamlit interface.

## Scope

**In scope:** PDF-based document Q&A, source citation, basic rule-based
classification and risk flagging, manually-entered ratio calculation,
summary report generation.

**Out of scope:** OCR for scanned documents, automatic extraction of
structured financial data, predictive risk scoring, multi-agent or
autonomous systems, enterprise infrastructure (databases, auth,
microservices, cloud deployment).

## Methodology

The system follows a linear RAG pipeline: PDF → text extraction (pypdf)
→ chunking (LangChain RecursiveCharacterTextSplitter, 1000 chars /
150 overlap) → embedding (Sentence-Transformers, all-MiniLM-L6-v2) →
FAISS indexing → similarity search at question time → LLM generation
(Groq, OpenAI-compatible API) with a context-restricted system prompt
→ answer + citations displayed in Streamlit. Risk flagging and
financial ratios run as separate, simpler rule-based modules over the
same processed chunks/user input, feeding into a combined report.

## System Architecture

See `DIAGRAMS.md` for the overall architecture, RAG pipeline, data
flow, use case, and sequence diagrams.

## Module Description

| Module | Responsibility |
|---|---|
| `pdf_processor.py` | Extracts text per page from uploaded PDFs; raises clear errors for invalid/empty/text-less PDFs |
| `chunker.py` | Splits page text into overlapping chunks, preserving document/page tags |
| `embeddings.py` | Loads all-MiniLM-L6-v2 and converts text to 384-dim normalized vectors |
| `vector_store.py` | Builds/searches a FAISS IndexFlatIP for cosine-similarity retrieval |
| `rag_pipeline.py` | Builds the context-restricted prompt and calls the LLM; raises clean errors on failure |
| `utils.py` | Rule-based question classification; report-building helper |
| `financial_analysis.py` | Calculates Net Profit Margin, Debt-to-Asset, Debt-to-Equity from user input |
| `risk_analysis.py` | Flags risk-indicator keywords in processed chunks with evidence |
| `app.py` | Streamlit UI orchestrating all of the above |

## Technology Stack

Python, Streamlit, pypdf, LangChain (text splitter only), Sentence-
Transformers (all-MiniLM-L6-v2), FAISS, Groq (OpenAI-compatible LLM
API), python-dotenv, pytest.

## Implementation

Implemented incrementally, module by module, with each module unit
tested in isolation before wiring into the Streamlit UI. See the
project's `src/` directory for full source and inline comments
explaining each design decision.

## Testing

29 automated tests in `tests/test_basic.py` cover PDF extraction,
chunking, embedding shape/similarity, FAISS indexing/search, prompt
construction, question classification, financial ratio calculation,
risk flagging, report generation, and a Streamlit UI smoke test.
See `tests/MANUAL_TEST_CHECKLIST.md` for the manual, LLM-dependent test
cases that require a live API key and browser session.

## Results

The prototype successfully answers questions grounded in uploaded
sample documents, correctly cites source document/page for each
answer, declines to answer when context is insufficient, classifies
questions into the four due-diligence categories, calculates financial
ratios only when inputs are available, flags risk keywords with
evidence, and generates a combined Markdown summary report.

*(Student note: replace this section with your own screenshots and
specific example Q&A pairs from your actual demo run before
submission.)*

## Limitations

- No OCR — scanned/image-only PDFs are not analyzed.
- Financial ratios depend entirely on user-entered figures; no
  automatic extraction from financial statements.
- Risk flagging is keyword-based evidence matching, not a predictive
  or validated risk score.
- Question classification is rule-based, not a trained model.
- Tested only on sample/test documents, not a large corpus of real
  company filings.
- Retrieval quality depends on chunk size/overlap and top-k settings,
  which are fixed defaults rather than tuned per document type.

## Future Enhancements

- OCR support for scanned documents.
- Automatic extraction of structured financial figures from tables.
- Multi-document comparison (compare two companies side by side).
- PDF export of the summary report (currently Markdown only).
- A trained (rather than rule-based) question classifier, if a labeled
  dataset becomes available.

## Conclusion

This project demonstrates a complete, explainable RAG pipeline applied
to a realistic academic use case, built with an intentionally small,
well-understood technology stack. Every design decision — from chunk
overlap to the absence of a fabricated risk score — was made to keep
the system both honest about its limitations and fully defensible in
a technical interview setting.

## References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.*
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence
  Embeddings using Siamese BERT-Networks.*
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity
  search with GPUs* (FAISS).
- Official documentation: Streamlit, LangChain, Sentence-Transformers,
  FAISS, Groq API.

*(Student note: add full citation details / URLs before submission.)*
