# AI-Powered M&A Due Diligence Assistant

RAG-based document analysis assistant for initial M&A due diligence.
Built as a B.Tech final-year academic project.

**Status:** Prototype — all core phases implemented and unit tested.
See `DOCUMENTATION.md` for the full academic write-up, `DIAGRAMS.md`
for architecture diagrams, and `INTERVIEW_PREP.md` for rehearsed
answers to likely interview questions.

## What it does

Upload company PDFs (annual report, financial statements, legal filings)
and ask natural-language questions. The system retrieves relevant
passages using semantic search, answers using only that retrieved
context, and always shows the source document and page number.

It also provides:
- Rule-based classification of questions into Financial / Legal /
  Business / Risk categories
- Simple financial ratio calculations (net profit margin,
  debt-to-asset, debt-to-equity)
- Evidence-based "potential concern" flags
- A downloadable due diligence summary report

This is a document-analysis aid, not a substitute for professional
M&A, legal, or financial advisors.

## Architecture

```
PDF -> Text Extraction -> Chunking -> Embeddings -> FAISS Index
                                                         |
Question -> Question Embedding -> Similarity Search -----
                                                         |
                                              Retrieved Chunks -> LLM -> Answer + Source + Page
```

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF extraction | pypdf |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| LLM | Groq (OpenAI-compatible API) |
| Config | python-dotenv |

## Project structure

```
project/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .env.example
├── data/sample_documents/    # sample/test PDFs
├── src/
│   ├── pdf_processor.py      # PDF text + page extraction
│   ├── chunker.py            # text splitting
│   ├── embeddings.py         # embedding generation
│   ├── vector_store.py       # FAISS index build/search
│   ├── rag_pipeline.py       # retrieval + prompt + LLM call
│   ├── financial_analysis.py # ratio calculations
│   ├── risk_analysis.py      # keyword-evidence risk flags
│   └── utils.py              # classification, formatting helpers
└── tests/test_basic.py
```

## Setup

Requires **Python 3.10+** (developed/tested on 3.12). The first run downloads
the `all-MiniLM-L6-v2` embedding model (~80 MB) from huggingface.co.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then add your LLM_API_KEY
streamlit run app.py
```

## Testing

```bash
python -m pytest tests -v
```
Tests that need the real embedding model are *skipped* (not failed) if it
cannot be loaded; all LLM calls in tests are faked, so no API key is needed.

## Retrieval settings (optional, in `.env`)

- `TOP_K` — chunks retrieved per question (default 4).
- `MIN_RELEVANCE_SCORE` — cosine-similarity floor; **disabled by default**
  because a good value must be calibrated on your documents. Use
  `python scripts/calibrate_threshold.py --help`. When enabled and nothing
  passes, the app answers "Insufficient information in the provided
  documents." without calling the LLM.
- `MAX_UPLOAD_MB` (default 50), `LLM_TIMEOUT_SECONDS` (default 30).

## Risk detection behaviour

Sentence-level keyword matching. Negated sentences ("no litigation is
pending") are not flagged; hedged ones ("may", "could") are flagged as
"Requires Further Investigation"; overlapping chunks no longer produce
duplicate findings. Negation detection is a heuristic and can miss complex
phrasing. No LLM is used for risk classification.

## Limitations (honest, as of this prototype)

- No RAG accuracy benchmark has been run yet; no accuracy claims are made
- Every upload is processed in memory per session; nothing is persisted

- No OCR — scanned/image-only PDFs will not extract text
- Financial ratios are only as accurate as the figures entered/found
- Question classification is keyword-based, not a trained model
- Not tested on a large corpus of real company filings — built and
  validated on sample/test documents only

## Development phases

1. ✅ Project skeleton
2. ✅ PDF extraction
3. ✅ Text chunking
4. ✅ Embeddings
5. ✅ FAISS vector search
6. ✅ RAG pipeline
7. ✅ Streamlit UI
8. ✅ Source/page citations
9. ✅ Financial analysis
10. ✅ Risk analysis
11. ✅ Due diligence summary report
12. ✅ Automated tests (29) + manual test checklist
13. ✅ Documentation + diagrams + interview prep

## Known gap to verify yourself

This project was built in a sandboxed environment with no access to
`huggingface.co` or the Groq API, so three things could **not** be
tested end-to-end here and need your confirmation on your own machine:
1. The embedding model download (`sentence-transformers` first run)
2. A real Groq LLM API call
3. The full upload → ask → answer flow in a live browser

Everything else (29 of the test suite's assertions, all pure Python
logic, and a Streamlit smoke test with zero exceptions) has been
verified in this environment.
