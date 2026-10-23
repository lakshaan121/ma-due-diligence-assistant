# Manual Test Checklist

Run these by hand after `streamlit run app.py`, with a real Groq API
key in `.env`. The automated suite (`pytest tests/test_basic.py`)
covers the pure logic; this checklist covers the parts that need a
live LLM call and a real browser session.

| # | Test | Steps | Expected Result |
|---|------|-------|------------------|
| 1 | Upload valid PDF | Upload an annual report PDF, click "Process Documents" | Green checkmark with document name, page count, chunk count shown |
| 2 | Upload invalid file | Try uploading a renamed .txt file as .pdf | Clear error message, no crash |
| 3 | Upload scanned/image-only PDF | Upload a PDF with no text layer | Error: "contains no extractable text" |
| 4 | Ask answerable question | Ask "What is the revenue?" after processing a doc that mentions revenue | Answer generated, at least one source shown with correct document/page |
| 5 | Ask unanswerable question | Ask something unrelated to the uploaded docs, e.g. "What is the capital of France?" | Answer states "Insufficient information in the provided documents." |
| 6 | Question before upload | Click "Analyze" with no document processed | Warning: "Please upload and process at least one document first." |
| 7 | Empty question | Click "Analyze" with an empty question box | Warning: "Please enter a question." |
| 8 | Category classification | Ask a question containing "litigation" | Category shown as "Legal/Regulatory" |
| 9 | Financial ratios — full inputs | Enter Revenue, Net Profit, Total Debt, Total Assets, Total Equity, click "Calculate Ratios" | All three ratios shown with correct values and inputs used |
| 10 | Financial ratios — partial inputs | Leave Total Equity blank, calculate | Debt-to-Equity shows "Not calculated" with a reason; other two compute normally |
| 11 | Risk scan with matches | Click "Scan for Risks" on a doc mentioning e.g. "litigation" or "net loss" | At least one flagged item appears, labeled "Potential Concern", with evidence and source |
| 12 | Risk scan with no matches | Click "Scan for Risks" on a doc with no risk keywords | Message stating no risk keywords were found — not an error |
| 13 | Report generation | Click "Generate Report" after processing docs, calculating ratios, and scanning risks | Markdown report preview appears with all four sections filled; download button works |
| 14 | Report with nothing processed | Click "Generate Report" immediately (no docs/ratios/risks) | Report still generates, each section honestly states nothing was processed |
| 15 | Missing API key | Remove `LLM_API_KEY` from `.env`, restart app, ask a question | Clear error message about the missing key — no raw traceback shown |

## Notes for demo day

- Have 1–2 sample PDFs ready in advance (a real annual report works
  better than a generated sample — more realistic retrieval results).
- Pre-fill `.env` with a working Groq key before the demo starts.
- Know in advance one question your sample PDF CAN answer and one it
  CANNOT — tests #4 and #5 above make a strong live demonstration of
  grounded answering and hallucination control in the same minute.
