"""
risk_analysis.py

Responsibility: Identify evidence-based "potential concern / requires
further investigation" items by matching retrieved chunks against a
fixed set of risk-indicator keywords (high debt, declining revenue,
losses, legal disputes, etc.).

IMPORTANT -- this is NOT a risk score:
    There is no "Risk Score: 87/100" anywhere in this module, and there
    never should be, per the project spec. A number like that implies
    a calibrated, validated model behind it, which this keyword-matching
    approach is not and does not claim to be.

    Instead, every flagged item is a transparent (keyword, evidence,
    source, page) tuple. The LABEL used is always one of:
        "Potential Concern"
        "Requires Further Investigation"
    and a flagged item is simply "a known risk keyword appeared in a
    retrieved chunk" -- nothing more sophisticated is claimed.

How a chunk gets flagged:
    Each chunk is split into sentences and every sentence is checked for
    RISK_KEYWORDS (whole-word match). For each match we record the risk
    category, the matching SENTENCE as evidence, and the chunk's
    document/page.

    Reliability rules (added after review):
      * Negation: "no litigation is pending" / "not subject to any penalty"
        is a statement DENYING the risk, so it is not flagged.
      * Hedged wording ("may", "could", "potential risk of ...") is flagged
        with label "Requires Further Investigation" and status "uncertain"
        instead of "Potential Concern" -- we do not know whether it
        actually happened.
      * Deduplication: chunks overlap (150 chars), so the same sentence can
        appear in two chunks. A finding is kept once per
        (document, page, category, sentence).
    No LLM is used here; classification is fully deterministic. Negation
    detection is a simple look-behind heuristic and can miss complex
    phrasing -- it is not a language-understanding model.
"""

import re

RISK_KEYWORDS = {
    "High Debt": ["high debt", "significant borrowings", "increased debt", "leverage"],
    "Declining Revenue": ["decline in revenue", "revenue decreased", "lower sales", "revenue fell"],
    "Losses": ["net loss", "operating loss", "reported a loss"],
    "Increasing Liabilities": ["liabilities increased", "rise in liabilities", "growing liabilities"],
    "Negative Cash Flow": ["negative cash flow", "cash outflow", "cash flow deficit"],
    "Legal Disputes": ["litigation", "lawsuit", "legal proceedings", "legal dispute"],
    "Regulatory Concerns": ["regulatory action", "non-compliance", "penalty", "regulatory concern"],
}



_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NEGATION = re.compile(
    r"\b(no|not|never|none|nil|neither|nor|without|free of|absence of|"
    r"did not|does not|has not|have not|had not|there (?:is|are|was|were) no)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(may|might|could|can|would|potential|potentially|possible|possibly|"
    r"if|whether|risk of|exposure to)\b", re.IGNORECASE)
# How many characters before a keyword we look at for a negation cue.
_NEGATION_WINDOW = 60


def _keyword_regex(keyword: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)


_COMPILED = {
    cat: [(kw, _keyword_regex(kw)) for kw in kws]
    for cat, kws in RISK_KEYWORDS.items()
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def flag_risks(chunks: list[dict]) -> list[dict]:
    """
    Scan chunks for risk-indicator keywords (sentence level).

    Args:
        chunks: [{"document": str, "page": int, "chunk_text": str}, ...]

    Returns:
        A list of flagged-item dicts (deduplicated, negations removed):
            {
                "category": "High Debt",
                "label": "Potential Concern" | "Requires Further Investigation",
                "status": "keyword_match" | "uncertain",
                "keyword": "significant borrowings",
                "evidence": "<the matching sentence>",
                "document": "ABC_Annual_Report.pdf",
                "page": 45,
            }
        "category", "label", "evidence", "document", "page" are the same
        keys the UI and report already use; "status" and "keyword" are
        additive. An empty list means no (non-negated) keyword was found.
    """
    flagged_items = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("chunk_text") or ""
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            for category, patterns in _COMPILED.items():
                for keyword, pattern in patterns:
                    match = pattern.search(sentence)
                    if not match:
                        continue
                    before = sentence[max(0, match.start() - _NEGATION_WINDOW):match.start()]
                    if _NEGATION.search(before):
                        # Denial of the risk, not a finding -- skip category.
                        break
                    key = (chunk["document"], chunk["page"], category, _normalize(sentence))
                    if key in seen:
                        break
                    seen.add(key)
                    uncertain = bool(_HEDGE.search(sentence))
                    flagged_items.append({
                        "category": category,
                        "label": "Requires Further Investigation" if uncertain else "Potential Concern",
                        "status": "uncertain" if uncertain else "keyword_match",
                        "keyword": keyword,
                        "evidence": sentence,
                        "document": chunk["document"],
                        "page": chunk["page"],
                    })
                    break  # one flag per category per sentence

    return flagged_items
