"""
financial_analysis.py

Responsibility: Calculate a small set of well-defined financial ratios
from user-supplied figures. Deliberately NOT a full financial modeling
engine -- just three standard, easily-defended ratios.

Ratios implemented:
    1. Net Profit Margin (%) = Net Profit / Revenue * 100
       -> "Of every rupee of revenue, how much ends up as profit?"
    2. Debt-to-Asset Ratio   = Total Debt / Total Assets
       -> "What fraction of the company's assets are funded by debt?"
    3. Debt-to-Equity Ratio  = Total Debt / Total Equity
       -> "How much debt is used relative to shareholder-funded equity?"
       (only calculated if equity is provided/available)

Why these three and nothing fancier:
    Each has a simple, universally-accepted formula with no hidden
    assumptions -- easy to state, easy to defend, easy to recompute by
    hand in front of an interviewer. A more elaborate financial model
    (DCF, WACC, comparable-company analysis) would be well beyond what
    a document-QA prototype like this should attempt.

Units and definitions (all inputs must use the SAME currency unit, e.g.
INR crore; ratios are unit-free so the unit cancels out):
    revenue       total revenue for the period; must be > 0 to be a divisor
    net_profit    profit after tax; may legitimately be NEGATIVE (a loss)
    total_debt    total borrowings; must be >= 0
    total_assets  must be > 0 to be a divisor
    total_equity  shareholders' equity; may legitimately be NEGATIVE
                  (the ratio is then negative); must be non-zero as divisor

Missing vs invalid vs zero-denominator are reported as distinct reasons:
    "<field> not provided"                 -> blank / None
    "<field> is invalid: ..."              -> malformed text, NaN/inf, or a
                                              value that is impossible (e.g.
                                              negative revenue/debt/assets)
    "<field> is zero, cannot divide by it" -> legitimate number, bad divisor

Golden rule: never fabricate a ratio.
    If a required input is missing, we skip that ratio entirely and
    say why -- we never substitute a guessed or default value. Always
    show exactly which inputs were used, so the number is traceable.
"""


import math

FIELD_LABELS = {
    "revenue": "Revenue",
    "net_profit": "Net Profit",
    "total_debt": "Total Debt",
    "total_assets": "Total Assets",
    "total_equity": "Total Equity",
}
# Fields that cannot be negative as a financial quantity.
_NON_NEGATIVE = {"revenue", "total_debt", "total_assets"}


def parse_amount(raw, field: str = "value"):
    """Parse one text-box value.

    Returns (value, error): (None, None) when blank (= not provided),
    (None, message) when malformed, otherwise (float, None).
    Thousands separators like "1,250.5" are accepted; nothing else is
    guessed (no currency symbols, no "crore" suffixes).
    """
    label = FIELD_LABELS.get(field, field)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, None
    try:
        value = float(str(raw).strip().replace(",", ""))
    except ValueError:
        return None, f"{label} is invalid: '{raw}' is not a number"
    if not math.isfinite(value):
        return None, f"{label} is invalid: '{raw}' is not a finite number"
    return value, None


def _check(figures: dict, errors: dict, field: str):
    """Return (value, problem) for one field. problem is None when usable."""
    label = FIELD_LABELS[field]
    if field in errors and errors[field]:
        return None, errors[field]
    value = figures.get(field)
    if value is None:
        return None, f"{label} not provided"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{label} is invalid: expected a number"
    if not math.isfinite(value):
        return None, f"{label} is invalid: not a finite number"
    if field in _NON_NEGATIVE and value < 0:
        return None, f"{label} is invalid: cannot be negative"
    return value, None


def _ratio(name_inputs: tuple, figures: dict, errors: dict, compute, unit: str):
    """Shared logic: validate the two inputs, guard the divisor, compute."""
    numerator_field, divisor_field = name_inputs
    num, num_problem = _check(figures, errors, numerator_field)
    div, div_problem = _check(figures, errors, divisor_field)
    problems = [p for p in (num_problem, div_problem) if p]
    if problems:
        return {"value": None, "reason": "; ".join(problems)}
    if div == 0:
        return {"value": None,
                "reason": f"{FIELD_LABELS[divisor_field]} is zero, cannot divide by it"}
    return {
        "value": compute(num, div),
        "unit": unit,
        "inputs_used": {numerator_field: num, divisor_field: div},
    }


def calculate_ratios(figures: dict, errors: dict | None = None) -> dict:
    """
    Calculate financial ratios from user-supplied figures.

    Args:
        figures: dict with any of the optional numeric keys "revenue",
            "net_profit", "total_debt", "total_assets", "total_equity".
            Missing / None = "not provided" (never raises, never guessed).
        errors: optional {field: message} from parse_amount() for fields
            whose text could not be parsed, so the reason shown to the user
            says "invalid" rather than the misleading "not provided".

    Returns:
        {"net_profit_margin" | "debt_to_asset" | "debt_to_equity":
            {"value": float, "unit": str, "inputs_used": {...}}   # computed
            {"value": None, "reason": str}                        # not computed
        }
        Output keys are unchanged from earlier versions. The original input
        values are echoed unrounded in "inputs_used"; only the ratio is
        rounded (2 dp), so results are reproducible by hand.
    """
    errors = errors or {}
    return {
        "net_profit_margin": _ratio(
            ("net_profit", "revenue"), figures, errors,
            lambda n, d: round((n / d) * 100, 2), "%"),
        "debt_to_asset": _ratio(
            ("total_debt", "total_assets"), figures, errors,
            lambda n, d: round(n / d, 2), "ratio"),
        "debt_to_equity": _ratio(
            ("total_debt", "total_equity"), figures, errors,
            lambda n, d: round(n / d, 2), "ratio"),
    }
