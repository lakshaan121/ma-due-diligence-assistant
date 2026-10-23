"""
utils.py

Responsibility: Small shared helpers used by the UI layer.
  - Rule-based question classification into one of four categories:
    Financial, Legal/Regulatory, Business/Operational, Risk.
  - A safe error-message wrapper so the UI never leaks raw exception
    text (which could include stack traces or, in rare library bugs,
    fragments of an API key) to the end user.

Why rule-based classification instead of a trained classifier?
    The project spec deliberately keeps this simple: a small
    keyword-to-category lookup is transparent, needs no training data,
    and is easy to explain and defend in an interview ("why did this
    question get tagged Financial?" -> "because it contained the word
    'debt'"). A trained ML classifier would add real complexity for a
    problem this small doesn't need.
"""

CATEGORY_KEYWORDS = {
    "Financial": ["debt", "revenue", "profit", "liabilities", "assets", "margin",
                  "financial position", "income", "cash flow"],
    "Legal/Regulatory": ["lawsuit", "litigation", "compliance", "regulation",
                          "regulatory", "legal"],
    "Business/Operational": ["operations", "customers", "suppliers",
                              "business model", "supply chain", "operational"],
    "Risk": ["risk", "concern", "issue", "red flag", "exposure", "investigation"],
}

DEFAULT_CATEGORY = "Business/Operational"


def classify_question(question: str) -> str:
    """
    Classify a question into one of the four due-diligence categories
    using simple keyword matching.

    Args:
        question: the user's natural-language question.

    Returns:
        One of "Financial", "Legal/Regulatory", "Business/Operational",
        or "Risk". If no keyword matches, defaults to
        "Business/Operational" as the most general catch-all category
        (documented here so the choice isn't a hidden surprise).

    Note on approach:
        Categories are checked in a fixed order (Financial, Legal,
        Business, Risk). If a question matches keywords from more than
        one category, the first matching category in this order wins.
        This is a simple, explainable tie-breaking rule -- appropriate
        for a rule-based classifier at this scope.
    """
    question_lower = question.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                return category

    return DEFAULT_CATEGORY


def safe_error_message(exc: Exception, fallback: str) -> str:
    """
    Return a user-facing error message, using the exception's own
    message when it's one of our own custom exception types (which we
    write to already be safe/clean), or a generic fallback otherwise
    (to avoid ever showing a raw library traceback or leaking internal
    details like an API key to the user).
    """
    known_message = str(exc)
    if known_message:
        return known_message
    return fallback


def build_summary_report(processed_docs: list[dict], ratios: dict | None,
                          risk_flags: list[dict]) -> str:
    """
    Build a simple Markdown due diligence summary report from whatever
    has been processed in the current session.

    Args:
        processed_docs: [{"name": str, "pages": int}, ...] -- from the
                        document upload step.
        ratios: output of financial_analysis.calculate_ratios(), or
                None if the user never calculated any ratios this session.
        risk_flags: output of risk_analysis.flag_risks() -- may be an
                    empty list if no scan was run or nothing was found.

    Returns:
        A Markdown string covering: overview, financial findings, risk
        findings, evidence/sources, and limitations -- per the project
        spec's report requirements. This is a prototype-level report:
        plain Markdown, not a styled PDF, which keeps it simple to
        generate, inspect, and explain.

    Note:
        This function only describes what was ACTUALLY processed in
        this session -- it never claims analysis that didn't happen
        (e.g. it says "no ratios were calculated" rather than omitting
        the section silently, per the project's academic-honesty
        requirement).
    """
    lines = ["# Due Diligence Summary Report", ""]
    lines.append("_Prototype output — AI-assisted document review only. "
                  "Not a substitute for professional M&A, legal, or "
                  "financial advisors._")
    lines.append("")

    # --- Document overview ---
    lines.append("## 1. Document Overview")
    if processed_docs:
        for doc in processed_docs:
            lines.append(f"- **{doc['name']}** — {doc['pages']} pages")
    else:
        lines.append("_No documents were processed in this session._")
    lines.append("")

    # --- Financial findings ---
    lines.append("## 2. Financial Findings")
    if ratios:
        ratio_labels = {
            "net_profit_margin": "Net Profit Margin",
            "debt_to_asset": "Debt-to-Asset Ratio",
            "debt_to_equity": "Debt-to-Equity Ratio",
        }
        for key, label in ratio_labels.items():
            result = ratios.get(key, {})
            if result.get("value") is not None:
                unit = "%" if result.get("unit") == "%" else ""
                lines.append(f"- **{label}:** {result['value']}{unit} "
                              f"(inputs: {result['inputs_used']})")
            else:
                lines.append(f"- **{label}:** Not calculated "
                              f"({result.get('reason', 'inputs not provided')})")
    else:
        lines.append("_No financial ratios were calculated in this session._")
    lines.append("")

    # --- Risk / areas for further investigation ---
    lines.append("## 3. Potential Concerns / Areas for Further Investigation")
    if risk_flags:
        for flag in risk_flags:
            lines.append(f"- **{flag['label']} — {flag['category']}** "
                          f"(_{flag['document']}, Page {flag['page']}_): "
                          f"{flag['evidence']}")
    else:
        lines.append("_No risk-indicator keywords were found, or the risk scan was not run._")
    lines.append("")

    # --- Limitations ---
    lines.append("## 4. Limitations")
    lines.append("- This report reflects only the documents processed in the current session.")
    lines.append("- Risk flagging is keyword-based evidence matching, not a predictive risk score.")
    lines.append("- Financial ratios are only as accurate as the figures entered by the user.")
    lines.append("- The system does not perform OCR; scanned/image-only PDFs are not analyzed.")
    lines.append("- This is an academic prototype and has not been validated on real transactions.")

    return "\n".join(lines)
