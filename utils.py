"""
utils.py — Shared utilities for Startup Blueprint Generator.

Responsibilities
----------------
• Python-based financial calculations (scores, revenue, break-even)
• Plotly chart builders
• PDF report generation (ReportLab)
• Markdown report builder
• Miscellaneous helper functions
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ===========================================================================
# SECTION 1 — Financial Calculations
# ===========================================================================


def _safe_int(value, default: int = 0) -> int:
    """Return *value* coerced to int, or *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    """Return *value* coerced to float, or *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_breakeven_index(be_raw, n_labels: int) -> Optional[int]:
    """
    Robustly convert a break-even value of any type to a zero-based integer
    index safe for Plotly add_vline on a categorical x-axis.

    Accepts: int, float, "18", "Month 18", "M18".
    Returns None when the value is missing / out of range (no vline drawn).
    Always returns None or a value in [0, n_labels-1].
    """
    if be_raw is None:
        return None
    be_str = str(be_raw).strip()
    m = re.search(r"\d+", be_str)
    if not m:
        return None
    month_1based = int(m.group())
    if month_1based <= 0:
        return None
    idx = month_1based - 1          # convert to 0-based index
    if idx >= n_labels:
        return None
    return idx



# ---------------------------------------------------------------------------
# Business Model Canvas parser
# ---------------------------------------------------------------------------

# The 9 canonical BMC blocks
_BMC_BLOCKS: list[tuple[str, str]] = [
    ("customer_segments",       "Customer Segments"),
    ("value_propositions",      "Value Propositions"),
    ("channels",                "Channels"),
    ("customer_relationships",  "Customer Relationships"),
    ("revenue_streams",         "Revenue Streams"),
    ("key_resources",           "Key Resources"),
    ("key_activities",          "Key Activities"),
    ("key_partnerships",        "Key Partnerships"),
    ("cost_structure",          "Cost Structure"),
]

# Alternate names the LLM may use for each block (lowercase for matching)
_BMC_ALIASES: dict[str, list[str]] = {
    "customer_segments":      ["customer segment", "customers", "target customer", "user segment"],
    "value_propositions":     ["value proposition", "value prop", "usp", "unique value"],
    "channels":               ["channel", "distribution", "delivery"],
    "customer_relationships": ["customer relationship", "relationship", "retention"],
    "revenue_streams":        ["revenue stream", "revenue", "income", "monetisation", "monetization"],
    "key_resources":          ["key resource", "resource"],
    "key_activities":         ["key activit", "activit"],
    "key_partnerships":       ["key partner", "partner"],
    "cost_structure":         ["cost structure", "cost", "expenses", "expenditure"],
}


def parse_bmc(raw: str) -> dict[str, str]:
    """
    Parse the LLM's business_model_canvas value into a 9-block dict.

    Handles:
      1. JSON object   {"Customer Segments": "...", ...}
      2. JSON array    [{"block": "...", "content": "..."}, ...]
      3. Labelled text "Customer Segments: ...\nValue Propositions: ..."
      4. Plain prose   stored under value_propositions as fallback

    Returns dict[canonical_key -> text].  Missing blocks get "".
    Never raises.
    """
    result: dict[str, str] = {k: "" for k, _ in _BMC_BLOCKS}
    if not raw or not raw.strip():
        return result
    text = raw.strip()

    # Attempt 1: JSON object or array
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return _map_bmc_dict(parsed, result)
            if isinstance(parsed, list):
                merged: dict[str, str] = {}
                for item in parsed:
                    if isinstance(item, dict):
                        block = (item.get("block") or item.get("name") or
                                 item.get("title") or item.get("key") or "")
                        content = (item.get("content") or item.get("description") or
                                   item.get("value") or item.get("text") or "")
                        if block:
                            merged[str(block)] = str(content)
                if merged:
                    return _map_bmc_dict(merged, result)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Attempt 2: JSON embedded inside prose
    json_match = re.search(r"\{[\s\S]+\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                mapped = _map_bmc_dict(parsed, {k: "" for k, _ in _BMC_BLOCKS})
                if any(v for v in mapped.values()):
                    return mapped
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Attempt 3: labelled text blocks
    mapped = _parse_bmc_text(text, {k: "" for k, _ in _BMC_BLOCKS})
    if any(v for v in mapped.values()):
        return mapped

    # Fallback: store full text as overview
    result["value_propositions"] = text
    logger.debug("parse_bmc: falling back to plain-text storage")
    return result


def _map_bmc_dict(source: dict, result: dict[str, str]) -> dict[str, str]:
    """Map arbitrary-key dict onto the 9 canonical BMC block keys."""
    for raw_key, raw_val in source.items():
        val_str = str(raw_val).strip() if raw_val else ""
        matched = _match_bmc_key(str(raw_key))
        if matched and val_str:
            result[matched] = val_str
    return result


def _match_bmc_key(raw_key: str) -> Optional[str]:
    """Return canonical BMC key for raw_key using exact then alias matching."""
    norm = raw_key.lower().replace("_", " ").replace("-", " ").strip()
    for canon_key, _ in _BMC_BLOCKS:
        if norm == canon_key.replace("_", " "):
            return canon_key
    for canon_key, aliases in _BMC_ALIASES.items():
        for alias in aliases:
            if alias in norm or norm in alias:
                return canon_key
    return None


def _parse_bmc_text(text: str, result: dict[str, str]) -> dict[str, str]:
    """Parse labelled text like 'Customer Segments: SMBs...' into BMC blocks."""
    lines = text.splitlines()
    current_key: Optional[str] = None
    buffer: list[str] = []

    def _flush() -> None:
        if current_key and buffer:
            result[current_key] = " ".join(l.strip() for l in buffer if l.strip())

    for line in lines:
        label_m = re.match(
            r"^[*#\-\d.\s]*\*{0,2}([A-Za-z][A-Za-z\s/&]{2,40})\*{0,2}\s*[:]\s*(.*)",
            line,
        )
        if label_m:
            candidate = label_m.group(1).strip()
            tail      = label_m.group(2).strip()
            canon = _match_bmc_key(candidate)
            if canon:
                _flush()
                buffer = [tail] if tail else []
                current_key = canon
                continue
        header_m = re.match(
            r"^[*#\-\d.\s]*\*{0,2}([A-Za-z][A-Za-z\s/&]{2,40})\*{0,2}\s*$", line
        )
        if header_m:
            canon = _match_bmc_key(header_m.group(1).strip())
            if canon:
                _flush()
                buffer = []
                current_key = canon
                continue
        if current_key is not None:
            buffer.append(line)

    _flush()
    return result


# ---------------------------------------------------------------------------
# Final-recommendations parser
# ---------------------------------------------------------------------------

def parse_recommendations(raw: str) -> list[str]:
    """
    Convert any representation of final_recommendations into a list[str].

    Handles Python list repr, JSON array, JSON object, numbered/bulleted text,
    and plain prose.  Never raises.  Always returns at least one item.
    """
    if not raw or not raw.strip():
        return ["No recommendations available."]

    text = raw.strip()

    # Attempt 1: JSON array
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                items = [str(i).strip() for i in parsed if str(i).strip()]
                if items:
                    return items
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 2: Python list literal (ast.literal_eval)
    if text.startswith("[") and text.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                items = [str(i).strip() for i in parsed if str(i).strip()]
                if items:
                    return items
        except (ValueError, SyntaxError):
            pass

    # Attempt 3: JSON object with ordinal keys
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                items = [str(v).strip() for v in parsed.values() if str(v).strip()]
                if items:
                    return items
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 4: numbered or bulleted lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-•*+]\s*)", "", line).strip()
        if cleaned and len(cleaned) > 4:
            items.append(cleaned)
    if items:
        return items

    return [text]



def calculate_scores(
    idea: str,
    industry: str,
    stage: str,
    team_size: int,
    monthly_burn: float,
    target_market: str,
    revenue_model: str,
) -> dict:
    """
    Derive three KPI scores deterministically from the user inputs.
    Returns a dict with float scores (0–100).

    Improvements over the original:
      • Readiness base floor raised to 40 (the "Idea" stage penalty was too harsh)
      • Idea word-count bonus scales smoothly: +2 per 5 words up to +10
      • Monthly burn signal: a meaningful burn (>50k) adds up to +5 readiness
      • Health score: target-market specificity bonus increased; burn signal added
      • Funding: industry match list widened; team-size bonus scales more
      • All three scores are floored at 20.0 so the gauge never shows a red/empty ring
        for a legitimate startup — scores only go to zero if inputs are all missing

    All numeric parameters are coerced at entry — never raises on bad input types.
    """
    # --- Coerce inputs at entry ---
    team_size_n     = max(_safe_int(team_size,     default=1), 1)
    monthly_burn_n  = max(_safe_float(monthly_burn, default=0.0), 0.0)
    idea_s          = str(idea)          if idea          is not None else ""
    industry_s      = str(industry)      if industry      is not None else ""
    stage_s         = str(stage)         if stage         is not None else ""
    target_market_s = str(target_market) if target_market is not None else ""
    revenue_model_s = str(revenue_model) if revenue_model is not None else ""

    idea_words = len(idea_s.split())

    # --- Startup Readiness Score (0–100) ---
    # Base varies with stage; floor 40 so even a bare "Idea" stage isn't punishing
    stage_map = {
        "idea": 40, "validation": 50, "mvp": 60,
        "early traction": 72, "growth": 82, "scaling": 90,
    }
    readiness = 40.0
    for k, v in stage_map.items():
        if k in stage_s.lower():
            readiness = float(v)
            break

    readiness += min(team_size_n * 2, 12)           # team size bonus (max +12)
    readiness += min((idea_words // 5) * 2, 10)     # idea detail bonus (max +10)
    if monthly_burn_n >= 50_000:                    # meaningful committed budget
        readiness += 3
    readiness = max(min(readiness, 100.0), 20.0)    # floor 20, cap 100

    # --- Business Health Score (0–100) ---
    rm_scores = {
        "saas": 18, "subscription": 16, "marketplace": 14,
        "freemium": 11, "advertising": 7, "licensing": 13,
        "direct sales": 9, "consulting": 7, "usage": 10,
    }
    health = 52.0
    for k, v in rm_scores.items():
        if k in revenue_model_s.lower():
            health += v
            break
    else:
        health += 5                                  # unknown model still gets default

    tm_words = len(target_market_s.split())
    health += min(tm_words * 1.0, 8.0)              # specificity bonus (max +8)
    health += min(team_size_n * 2, 10)              # team bonus (max +10)
    if monthly_burn_n >= 100_000:
        health += 4                                  # funded commitment signal
    health = max(min(health, 100.0), 20.0)

    # --- Funding Eligibility Score (0–100) ---
    funding_stage_map = {
        "idea": 35, "validation": 48, "mvp": 60,
        "early traction": 72, "growth": 83, "scaling": 92,
    }
    funding = 35.0
    for k, v in funding_stage_map.items():
        if k in stage_s.lower():
            funding = float(v)
            break

    if team_size_n >= 2:
        funding += min((team_size_n - 1) * 3, 12)  # scales with team (max +12)
    high_funding_industries = [
        "tech", "ai", "fintech", "health", "saas", "edtech",
        "agri", "clean", "cyber", "logistic", "d2c",
    ]
    if any(w in industry_s.lower() for w in high_funding_industries):
        funding += 10
    if monthly_burn_n >= 200_000:
        funding += 5                                 # serious committed capital
    funding = max(min(funding, 100.0), 20.0)

    return {
        "startup_readiness_score": round(readiness, 1),
        "business_health_score":   round(health, 1),
        "funding_eligibility_score": round(funding, 1),
    }


def calculate_financials(
    monthly_burn: float,
    team_size: int,
    price_per_unit: float = 999.0,
    units_month1: int = 10,
    growth_rate: float = 0.20,
    months: int = 24,
) -> dict:
    """
    Project revenue, costs, and compute break-even for `months` periods.
    Returns a dict with lists (for charting) and scalar summary values.

    All parameters are coerced to safe numeric types at entry.  Strings,
    None, negative values, and zero are handled without raising so the
    function is safe regardless of what session state or LLM output provides.

    Safe ranges applied:
      monthly_burn   ≥ 0           (negative burn makes no financial sense)
      team_size      ≥ 1           (at least one person)
      price_per_unit ≥ 0           (free product is valid)
      units_month1   ≥ 0
      growth_rate    clamped 0–10  (0 % to 1000 % monthly; guards "20" string)
      months         1–120         (1 month minimum, 10-year maximum)
    """
    # --- Coerce all parameters at entry — never assume numeric types -------
    monthly_burn_n   = max(_safe_float(monthly_burn,   default=0.0), 0.0)
    team_size_n      = max(_safe_int(team_size,        default=1),   1)
    price_per_unit_n = max(_safe_float(price_per_unit, default=999.0), 0.0)
    units_month1_n   = max(_safe_int(units_month1,     default=10),  0)
    months_n         = max(min(_safe_int(months, default=24), 120),  1)

    # growth_rate may arrive as "0.20", "20", "20%", or a proper float.
    # Strip a trailing "%" and parse; then clamp to [0, 10] so the
    # exponentiation (1 + rate)^23 never overflows.
    _gr_raw = growth_rate
    if isinstance(_gr_raw, str):
        _gr_raw = _gr_raw.strip().rstrip("%")
    _gr = _safe_float(_gr_raw, default=0.20)
    # A value > 1 is almost certainly a percentage expressed as a whole number
    # (e.g. 20 meaning 20%) — normalise it to a fraction.
    if _gr > 1.0:
        _gr = _gr / 100.0
    growth_rate_n = max(min(_gr, 10.0), 0.0)

    monthly_costs: list[float] = []
    monthly_revenues: list[float] = []
    cumulative_profit: list[float] = []
    labels: list[str] = []

    # Salary is the dominant cost component; scale with team size
    salary_cost = team_size_n * 35_000
    infra_cost  = max(monthly_burn_n * 0.15, 5_000.0)
    marketing   = max(monthly_burn_n * 0.20, 8_000.0)
    overhead    = max(monthly_burn_n * 0.10, 3_000.0)
    base_monthly_cost = salary_cost + infra_cost + marketing + overhead

    cum_profit = 0.0
    breakeven_month: Optional[int] = None

    for m in range(1, months_n + 1):
        units   = int(units_month1_n * (1 + growth_rate_n) ** (m - 1))
        revenue = units * price_per_unit_n
        cost    = base_monthly_cost * (1 + 0.02) ** (m - 1)
        profit  = revenue - cost
        cum_profit += profit

        monthly_revenues.append(round(revenue, 2))
        monthly_costs.append(round(cost, 2))
        cumulative_profit.append(round(cum_profit, 2))
        labels.append(f"M{m}")

        if breakeven_month is None and cum_profit >= 0:
            breakeven_month = m

    # sum([]) returns int 0, not float 0.0 — force float so downstream
    # format strings (e.g. f"{v:,.0f}") never receive an unexpected type.
    annual_revenue_y1 = float(sum(monthly_revenues[:12]))
    annual_revenue_y2 = float(sum(monthly_revenues[12:24]))
    total_investment  = float(sum(c for c in monthly_costs if c > 0))

    return {
        "labels": labels,
        "monthly_revenues": monthly_revenues,
        "monthly_costs": monthly_costs,
        "cumulative_profit": cumulative_profit,
        "breakeven_month": breakeven_month,
        "annual_revenue_y1": round(annual_revenue_y1, 2),
        "annual_revenue_y2": round(annual_revenue_y2, 2),
        "total_investment_24m": round(total_investment, 2),
        "base_monthly_cost": round(base_monthly_cost, 2),
        "monthly_burn_used": round(base_monthly_cost, 2),
    }


def build_calc_summary(scores: dict, financials: dict) -> str:
    """Format scores + financials into a human-readable string for LLM Call 2 context.

    All values are coerced before formatting so a stale or partially-populated
    dict (e.g. from an interrupted run) never produces a format-string crash.
    """
    be = _safe_int(financials.get("breakeven_month"), default=0)
    be_str = f"Month {be}" if be else "Beyond 24 months"
    lines = [
        "=== PYTHON CALCULATIONS SUMMARY ===",
        f"Startup Readiness Score   : {_safe_float(scores.get('startup_readiness_score'), 0.0):.1f}/100",
        f"Business Health Score     : {_safe_float(scores.get('business_health_score'),   0.0):.1f}/100",
        f"Funding Eligibility Score : {_safe_float(scores.get('funding_eligibility_score'),0.0):.1f}/100",
        "",
        f"Projected Year-1 Revenue  : Rs{_safe_float(financials.get('annual_revenue_y1'),  0.0):,.0f}",
        f"Projected Year-2 Revenue  : Rs{_safe_float(financials.get('annual_revenue_y2'),  0.0):,.0f}",
        f"Total 24-month Investment : Rs{_safe_float(financials.get('total_investment_24m'),0.0):,.0f}",
        f"Base Monthly Burn Rate    : Rs{_safe_float(financials.get('base_monthly_cost'),   0.0):,.0f}",
        f"Estimated Break-even      : {be_str}",
    ]
    return "\n".join(lines)


# ===========================================================================
# SECTION 2 — Plotly Chart Builders
# ===========================================================================

IBM_BLUE   = "#0f62fe"
IBM_CYAN   = "#1192e8"
IBM_PURPLE = "#8a3ffc"
IBM_GREEN  = "#24a148"
IBM_RED    = "#da1e28"
IBM_GRAY   = "#8d8d8d"
CHART_BG   = "#161616"
PAPER_BG   = "#161616"
FONT_COLOR = "#f4f4f4"

_BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=FONT_COLOR, family="IBM Plex Sans, Segoe UI, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)


def chart_revenue_vs_cost(financials: dict) -> go.Figure:
    """Line chart: Monthly Revenue vs Monthly Cost over 24 months."""
    try:
        labels = financials.get("labels") or []
        revenues = financials.get("monthly_revenues") or []
        costs    = financials.get("monthly_costs")    or []
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels, y=revenues,
            name="Revenue", mode="lines+markers",
            line=dict(color=IBM_GREEN, width=2.5),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=costs,
            name="Cost", mode="lines+markers",
            line=dict(color=IBM_RED, width=2.5),
            marker=dict(size=5),
        ))
        # _safe_breakeven_index handles int, float, "18", "Month 18", "M18"
        # and returns a validated 0-based index or None — never crashes add_vline.
        be_idx = _safe_breakeven_index(financials.get("breakeven_month"), len(labels))
        if be_idx is not None:
            fig.add_vline(
                x=be_idx,
                line_dash="dash", line_color=IBM_CYAN,
                annotation_text="Break-even", annotation_position="top right",
                annotation_font_color=IBM_CYAN,
            )
    except Exception as exc:
        logger.error("chart_revenue_vs_cost: render error — %s", exc)
        fig = go.Figure()
        fig.update_layout(**_BASE_LAYOUT, title="Revenue vs Cost (data unavailable)")
        return fig
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Revenue vs Cost (24-Month Projection)",
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def chart_cumulative_profit(financials: dict) -> go.Figure:
    """Area chart: Cumulative Profit/Loss over 24 months."""
    labels = financials["labels"]
    profits = financials["cumulative_profit"]
    colors = [IBM_GREEN if p >= 0 else IBM_RED for p in profits]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=profits,
        fill="tozeroy",
        mode="lines",
        line=dict(color=IBM_BLUE, width=2.5),
        fillcolor="rgba(15,98,254,0.15)",
        name="Cumulative P&L",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=IBM_GRAY)
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Cumulative Profit / Loss",
        xaxis_title="Month",
        yaxis_title="Cumulative Amount (₹)",
    )
    return fig


def chart_score_gauges(scores: dict) -> go.Figure:
    """Three gauge charts side-by-side for the three KPI scores."""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
    )

    def _gauge(value, title, color):
        return go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 14, "color": FONT_COLOR}},
            number={"suffix": "/100", "font": {"size": 18, "color": FONT_COLOR}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": IBM_GRAY},
                "bar": {"color": color},
                "bgcolor": "#262626",
                "bordercolor": "#393939",
                "steps": [
                    {"range": [0, 40],  "color": "#3d1414"},
                    {"range": [40, 70], "color": "#2c2c00"},
                    {"range": [70, 100],"color": "#0a2e0a"},
                ],
            },
        )

    fig.add_trace(_gauge(_safe_float(scores.get("startup_readiness_score"),   0.0), "Startup Readiness",   IBM_BLUE),   row=1, col=1)
    fig.add_trace(_gauge(_safe_float(scores.get("business_health_score"),     0.0), "Business Health",     IBM_GREEN),  row=1, col=2)
    fig.add_trace(_gauge(_safe_float(scores.get("funding_eligibility_score"), 0.0), "Funding Eligibility", IBM_PURPLE), row=1, col=3)

    fig.update_layout(
        **_BASE_LAYOUT,
        height=260,
        title="Startup KPI Scores",
    )
    return fig


def chart_swot(swot_text: str) -> go.Figure:
    """4-quadrant SWOT visualization using a 2x2 table figure.

    Accepts a free-text SWOT string (from the JSON swot_analysis value).
    Parses it into four buckets supporting all common LLM output formats:

    Format A - newline-separated paragraphs:
        Strengths include...
        Weaknesses include...

    Format B - single continuous paragraph (all four on one line):
        Strengths include... Weaknesses include... Opportunities... Threats...

    Format C - markdown bold headers:
        **Strengths**
        text

    Format D - markdown headings:
        ## Strengths
        text

    Format E - colon + bullet lists:
        Strengths:
        - item

    The previous line-by-line parser failed for Formats B-D because it used
    a ``continue`` after detecting a keyword, which discarded the rest of the
    line.  When IBM Granite writes all four SWOT sections as a single paragraph
    with no newlines between sections (Format B), the Strengths keyword fires
    once, the entire paragraph is stored under 'strengths', and the other three
    keywords are never seen as section boundaries -- leaving weaknesses,
    opportunities, and threats all empty.

    Fix: use a regex section-split approach.  Build a pattern that matches the
    START of any SWOT keyword anywhere in the text.  Split the entire string on
    those boundaries.  Each resulting chunk belongs to exactly one bucket.
    This correctly handles all formats regardless of newline placement.
    """
    import re

    # (canonical_key, stem) -- stems match singular, plural, and connective
    # forms without word-boundary anchors so they work in mid-sentence too
    _STEMS: list[tuple[str, str]] = [
        ("strengths",     "strength"),
        ("weaknesses",    "weakness"),
        ("opportunities", "opportun"),
        ("threats",       "threat"),
    ]

    # Lookahead boundary: split just BEFORE each SWOT keyword occurrence.
    # The (?:^|[^\w]) guard prevents matching inside unrelated words.
    _BOUNDARY = re.compile(
        r"(?=(?:^|[^\w])(?:" +
        "|".join(stem for _, stem in _STEMS) +
        r")[a-z]*\b)",
        re.IGNORECASE | re.MULTILINE,
    )

    def _chunk_to_items(chunk: str) -> list:
        """Extract displayable items from one SWOT section chunk."""
        items: list = []
        for raw in chunk.splitlines():
            s = raw.strip()
            if not s:
                continue
            # Skip pure keyword header lines (e.g. "Strengths:", "**Strengths**")
            if re.match(
                r"^[\s*_#\-]*(?:strength|weakness|opportun|threat)[a-z]*[\s*_#\-:]*$",
                s, re.IGNORECASE,
            ):
                continue
            # Bullet items: detect BEFORE stripping so the marker is present
            if re.match(r"^[-*+]\s+", s):
                item = re.sub(r"^[-*+]\s+", "", s).strip()
                if item:
                    items.append(item)
                continue
            if re.match(r"^\d+[.)]\s+", s):
                item = re.sub(r"^\d+[.)]\s+", "", s).strip()
                if item:
                    items.append(item)
                continue
            # Prose lines: strip markdown decoration, then split into sentences
            s = re.sub(r"^[\s*_#+\-]+", "", s).strip()
            if not s:
                continue
            # Strip colon-prefixed heading remnant (e.g. "Strengths: text")
            colon = s.find(":")
            if colon != -1 and colon < 30:
                s = s[colon + 1:].strip()
            if not s:
                continue
            if len(s.split()) >= 3:
                sentences = [
                    x.strip()
                    for x in re.split(r"(?<=[.!?])\s+", s)
                    if x.strip() and len(x.split()) >= 2
                ]
                items.extend(sentences if sentences else [s])
        return items

    buckets: dict[str, list[str]] = {k: [] for k, _ in _STEMS}

    for part in _BOUNDARY.split(swot_text):
        if not part.strip():
            continue
        # Identify which bucket owns this chunk
        part_l = part.lower()
        for key, stem in _STEMS:
            if stem in part_l:
                buckets[key].extend(_chunk_to_items(part))
                break

    def _fmt(items: list[str]) -> str:
        return "<br>".join(f"• {i}" for i in items[:6]) or "No information available"

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>💪 Strengths</b>", "<b>⚠️ Weaknesses</b>",
                    "<b>🚀 Opportunities</b>", "<b>🔴 Threats</b>"],
            fill_color=[IBM_GREEN, IBM_RED, IBM_BLUE, IBM_PURPLE],
            font=dict(color="white", size=13),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[
                [_fmt(buckets["strengths"])],
                [_fmt(buckets["weaknesses"])],
                [_fmt(buckets["opportunities"])],
                [_fmt(buckets["threats"])],
            ],
            fill_color="#1c1c1c",
            font=dict(color=FONT_COLOR, size=12),
            align="left",
            height=160,
        ),
    )])
    fig.update_layout(
        **_BASE_LAYOUT,
        title="SWOT Analysis",
        height=240,
    )
    return fig


def chart_cost_breakdown(financials: dict) -> go.Figure:
    """Donut chart: monthly cost breakdown estimate."""
    base = max(_safe_float(financials.get("base_monthly_cost", 0), default=0.0), 0.0)
    # Reconstruct approximate split (must match calculate_financials logic)
    labels = ["Salaries", "Marketing", "Infrastructure", "Overhead", "Legal & Misc"]
    salary     = base * 0.60
    marketing  = base * 0.20
    infra      = base * 0.10
    overhead   = base * 0.06
    legal      = base * 0.04
    values = [salary, marketing, infra, overhead, legal]
    colors = [IBM_BLUE, IBM_PURPLE, IBM_CYAN, IBM_GRAY, IBM_GREEN]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#161616", width=2)),
        textfont=dict(color=FONT_COLOR),
    )])
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Monthly Cost Breakdown",
        height=300,
        legend=dict(orientation="v", x=1.0, y=0.5),
    )
    return fig


def chart_funding_stages() -> go.Figure:
    """Horizontal bar chart showing typical Indian startup funding stages."""
    stages = ["Bootstrapping", "Pre-Seed", "Seed", "Series A", "Series B", "Series C+"]
    low    = [0,     5,    50,    500,   5000,  20000]   # ₹ Lakhs
    high   = [5,     50,   500,   5000,  20000, 100000]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Typical Range (₹ Lakhs)",
        y=stages,
        x=[h - l for h, l in zip(high, low)],
        base=low,
        orientation="h",
        marker_color=[IBM_GRAY, IBM_GREEN, IBM_CYAN, IBM_BLUE, IBM_PURPLE, IBM_RED],
        text=[f"₹{l}L – ₹{h}L" for l, h in zip(low, high)],
        textposition="inside",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Startup Funding Stages (India)",
        xaxis_title="Amount (₹ Lakhs)",
        showlegend=False,
        height=300,
    )
    return fig


# ===========================================================================
# SECTION 3 — PDF Report Generation (ReportLab)
# ===========================================================================

# Canonical section order for PDF and Markdown reports.
# Each tuple: (json_key, display_title)
SECTION_ORDER: list[tuple[str, str]] = [
    ("startup_validation",          "Startup Validation"),
    ("problem_statement",           "Problem Statement"),
    ("solution_description",        "Solution Description"),
    ("target_customer_analysis",    "Target Customer Analysis"),
    ("competitor_analysis",         "Competitor Analysis"),
    ("swot_analysis",               "SWOT Analysis"),
    ("market_opportunity_analysis", "Market Opportunity Analysis"),
    ("business_model_canvas",       "Business Model Canvas"),
    ("revenue_model",               "Revenue Model"),
    ("pricing_strategy",            "Pricing Strategy"),
    ("cost_estimation",             "Cost Estimation"),
    ("break_even_analysis",         "Break-even Analysis"),
    ("funding_recommendations",     "Funding Recommendations"),
    ("government_schemes",          "Government Startup Schemes"),
    ("investor_pitch",              "Investor Pitch"),
    ("startup_roadmap",             "Startup Roadmap"),
    ("risk_assessment",             "Risk Assessment"),
    ("executive_summary",           "Executive Summary"),
    ("future_scope",                "Future Scope"),
    ("final_recommendations",       "Final Recommendations"),
]


def generate_pdf_report(
    idea: str,
    industry: str,
    stage: str,
    scores: dict,
    financials: dict,
    sections: Optional[dict] = None,
    # Legacy kwargs kept for backward compatibility — ignored when sections is given
    call1_sections: Optional[dict] = None,
    call2_sections: Optional[dict] = None,
) -> bytes:
    """
    Generate a professional PDF Blueprint Report from the JSON blueprint dict.

    Preferred call (new):
        generate_pdf_report(..., sections=blueprint["sections"])

    Legacy call (still accepted):
        generate_pdf_report(..., call1_sections=c1, call2_sections=c2)

    Returns the PDF as raw bytes.
    """
    # Resolve the sections dict: prefer the explicit merged arg, fall back to
    # merging the legacy split-dict args so old callers still work.
    if sections is None:
        from agents import merge_blueprint_sections
        sections = merge_blueprint_sections(call1_sections, call2_sections)
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    # Styles
    styles = getSampleStyleSheet()
    IBM_BLUE_HEX = colors.HexColor("#0f62fe")
    IBM_DARK     = colors.HexColor("#161616")
    IBM_LIGHT    = colors.HexColor("#f4f4f4")

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"],
        fontSize=24, textColor=IBM_BLUE_HEX,
        spaceAfter=6, alignment=TA_CENTER,
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=16, textColor=IBM_BLUE_HEX,
        spaceBefore=14, spaceAfter=4,
        borderPad=4,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#0043ce"),
        spaceBefore=10, spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=6,
    )
    muted_style = ParagraphStyle(
        "Muted", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#6f6f6f"),
        spaceAfter=4,
    )

    story = []

    # ---- Cover ----
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("🚀 Startup Blueprint Report", title_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"<b>Startup Idea:</b> {idea}", h2_style))
    story.append(Paragraph(
        f"<b>Industry:</b> {industry} &nbsp;&nbsp; <b>Stage:</b> {stage}", body_style
    ))
    story.append(Paragraph(
        f"<i>Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} "
        f"using IBM Granite 4 H Small via IBM watsonx.ai</i>", muted_style
    ))
    story.append(HRFlowable(width="100%", color=IBM_BLUE_HEX, thickness=1.5))
    story.append(Spacer(1, 0.5*cm))

    # ---- KPI Scores ----
    _rs  = _safe_float(scores.get("startup_readiness_score"),   0.0)
    _bh  = _safe_float(scores.get("business_health_score"),     0.0)
    _fe  = _safe_float(scores.get("funding_eligibility_score"), 0.0)
    story.append(Paragraph("Key Performance Scores", h1_style))
    score_data = [
        ["Metric", "Score", "Status"],
        ["Startup Readiness",   f"{_rs:.1f}/100", _score_label(_rs)],
        ["Business Health",     f"{_bh:.1f}/100", _score_label(_bh)],
        ["Funding Eligibility", f"{_fe:.1f}/100", _score_label(_fe)],
    ]
    score_table = Table(score_data, colWidths=[7*cm, 4*cm, 5*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), IBM_BLUE_HEX),
        ("TEXTCOLOR",  (0, 0), (-1, 0), IBM_LIGHT),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f4f4f4"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # ---- Financial Highlights ----
    _ry1 = _safe_float(financials.get("annual_revenue_y1"),    0.0)
    _ry2 = _safe_float(financials.get("annual_revenue_y2"),    0.0)
    _inv = _safe_float(financials.get("total_investment_24m"), 0.0)
    _bmc = _safe_float(financials.get("base_monthly_cost"),    0.0)
    _be  = _safe_int(financials.get("breakeven_month"),        0)
    story.append(Paragraph("Financial Highlights", h1_style))
    fin_data = [
        ["Metric", "Value"],
        ["Projected Year-1 Revenue", f"Rs{_ry1:,.0f}"],
        ["Projected Year-2 Revenue", f"Rs{_ry2:,.0f}"],
        ["Total 24-Month Investment", f"Rs{_inv:,.0f}"],
        ["Monthly Burn Rate",         f"Rs{_bmc:,.0f}"],
        ["Estimated Break-even",      f"Month {_be}" if _be else "Beyond 24 months"],
    ]
    fin_table = Table(fin_data, colWidths=[8*cm, 8*cm])
    fin_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), IBM_BLUE_HEX),
        ("TEXTCOLOR",     (0, 0), (-1, 0), IBM_LIGHT),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f4f4f4"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ("ALIGN",         (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(fin_table)
    story.append(PageBreak())

    # ---- LLM Sections — direct JSON key lookup from merged dict ----
    # bullet_style has a left indent so bullet items are visually offset
    bullet_style = ParagraphStyle(
        "Bullet", parent=body_style,
        leftIndent=14, firstLineIndent=-10, spaceAfter=3,
    )
    for json_key, display_title in SECTION_ORDER:
        content = sections.get(json_key, "")
        if not content or not content.strip():
            continue
        story.append(Paragraph(display_title, h1_style))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#e0e0e0"), thickness=0.5))
        story.append(Spacer(1, 0.15*cm))

        # ── Special case 1: Business Model Canvas ─────────────────────────
        if json_key == "business_model_canvas":
            bmc = parse_bmc(content)
            bmc_has_content = any(v.strip() for v in bmc.values())
            if bmc_has_content:
                bmc_data = [["BMC Block", "Description"]]
                for bk, bt in _BMC_BLOCKS:
                    val = bmc.get(bk, "").strip()
                    if val:
                        bmc_data.append([bt, _clean_md(val)])
                if len(bmc_data) > 1:
                    bmc_table = Table(bmc_data, colWidths=[4.5*cm, 12*cm])
                    bmc_table.setStyle(TableStyle([
                        ("BACKGROUND",    (0, 0), (-1, 0), IBM_BLUE_HEX),
                        ("TEXTCOLOR",     (0, 0), (-1, 0), IBM_LIGHT),
                        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE",      (0, 0), (-1, -1), 9),
                        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
                         [colors.HexColor("#f4f4f4"), colors.white]),
                        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
                        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING",    (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
                    ]))
                    story.append(bmc_table)
                    story.append(Spacer(1, 0.3*cm))
                    continue
            # If BMC parsing yielded nothing, fall through to plain-text render

        # ── Special case 2: Final Recommendations ─────────────────────────
        if json_key == "final_recommendations":
            recs = parse_recommendations(content)
            for i, rec in enumerate(recs, 1):
                clean_rec = _clean_md(rec)
                story.append(Paragraph(f"{i}. {clean_rec}", bullet_style))
            story.append(Spacer(1, 0.3*cm))
            continue

        # ── Default: plain-text with paragraph and bullet support ──────────
        clean = _clean_md(content)
        for block in clean.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if "\n" in block and any(
                ln.strip().startswith("• ") for ln in block.splitlines()
            ):
                for line in block.splitlines():
                    line = line.strip()
                    if line:
                        story.append(Paragraph(line, bullet_style if line.startswith("• ") else body_style))
            else:
                story.append(Paragraph(block, body_style))
        story.append(Spacer(1, 0.3*cm))

    # ---- Footer page ----
    story.append(PageBreak())
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph(
        "Generated by <b>Startup Blueprint Generator</b>", title_style
    ))
    story.append(Paragraph(
        "Powered by IBM Granite 4 H Small · IBM watsonx.ai · FAISS RAG",
        muted_style,
    ))
    story.append(Paragraph(
        "IBM SkillsBuild AICTE 2026",
        ParagraphStyle("Footer2", parent=muted_style, alignment=TA_CENTER),
    ))

    doc.build(story)
    return buffer.getvalue()


def _score_label(score: float) -> str:
    if score >= 75: return "Excellent ✓"
    if score >= 55: return "Good"
    if score >= 35: return "Moderate"
    return "Needs Work"


def _clean_md(text: str) -> str:
    """Strip Markdown syntax for ReportLab Paragraph plain-text rendering.

    • Removes heading markers, bold/italic markers, and backtick code spans.
    • Normalises list prefixes to "• ".
    • Escapes XML special characters (&, <, >) so ReportLab never raises
      a parse error on content that contains stray angle-brackets.
    • Collapses 3+ blank lines to 2.
    """
    if not text:
        return ""
    text = re.sub(r"#{1,6}\s+", "", text)                            # headings
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)                     # bold
    text = re.sub(r"\*(.*?)\*",     r"\1", text)                     # italic
    text = re.sub(r"`(.*?)`",       r"\1", text)                     # inline code
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"\n{3,}", "\n\n", text)                           # spacing
    # Escape XML chars that would break ReportLab's XML parser
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.strip()


# ===========================================================================
# SECTION 4 — Markdown Report Builder
# ===========================================================================

# Maps JSON key → Markdown section heading with emoji
_MD_SECTION_HEADINGS: dict[str, str] = {
    "startup_validation":          "## ✅ Startup Validation",
    "problem_statement":           "## 🎯 Problem Statement",
    "solution_description":        "## 💡 Solution Description",
    "target_customer_analysis":    "## 👥 Target Customer Analysis",
    "competitor_analysis":         "## ⚔️ Competitor Analysis",
    "swot_analysis":               "## 🔷 SWOT Analysis",
    "market_opportunity_analysis": "## 🌍 Market Opportunity Analysis",
    "business_model_canvas":       "## 🏗️ Business Model Canvas",
    "revenue_model":               "## 💵 Revenue Model",
    "pricing_strategy":            "## 🏷️ Pricing Strategy",
    "cost_estimation":             "## 📋 Cost Estimation",
    "break_even_analysis":         "## 📉 Break-even Analysis",
    "funding_recommendations":     "## 🏦 Funding Recommendations",
    "government_schemes":          "## 🏛️ Government Startup Schemes",
    "investor_pitch":              "## 🎤 Investor Pitch",
    "startup_roadmap":             "## 🗺️ Startup Roadmap",
    "risk_assessment":             "## ⚠️ Risk Assessment",
    "executive_summary":           "## 📄 Executive Summary",
    "future_scope":                "## 🔭 Future Scope",
    "final_recommendations":       "## ✨ Final Recommendations",
}


def generate_markdown_report(
    idea: str,
    industry: str,
    stage: str,
    scores: dict,
    financials: dict,
    sections: Optional[dict] = None,
    # Legacy kwargs kept for backward compatibility — ignored when sections is given
    call1_sections: Optional[dict] = None,
    call2_sections: Optional[dict] = None,
) -> str:
    """
    Return the full blueprint as a well-structured Markdown string.

    Preferred call (new):
        generate_markdown_report(..., sections=blueprint["sections"])

    Legacy call (still accepted):
        generate_markdown_report(..., call1_sections=c1, call2_sections=c2)
    """
    if sections is None:
        from agents import merge_blueprint_sections
        sections = merge_blueprint_sections(call1_sections, call2_sections)

    _rs  = _safe_float(scores.get("startup_readiness_score"),   0.0)
    _bh  = _safe_float(scores.get("business_health_score"),     0.0)
    _fe  = _safe_float(scores.get("funding_eligibility_score"), 0.0)
    _ry1 = _safe_float(financials.get("annual_revenue_y1"),     0.0)
    _ry2 = _safe_float(financials.get("annual_revenue_y2"),     0.0)
    _inv = _safe_float(financials.get("total_investment_24m"),  0.0)
    _bmc = _safe_float(financials.get("base_monthly_cost"),     0.0)
    _be  = _safe_int(financials.get("breakeven_month"),         0)
    _be_str = f"Month {_be}" if _be else "Beyond 24 months"
    # Guard str inputs for header fields
    idea_s     = str(idea)     if idea     is not None else ""
    industry_s = str(industry) if industry is not None else ""
    stage_s    = str(stage)    if stage    is not None else ""
    lines = [
        "# Startup Blueprint Report",
        f"> **Startup Idea:** {idea_s}  ",
        f"> **Industry:** {industry_s} | **Stage:** {stage_s}  ",
        f"> *Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} using IBM Granite 4 H Small*",
        "",
        "---",
        "",
        "## Key Performance Scores",
        "",
        "| Metric | Score | Status |",
        "|--------|-------|--------|",
        f"| Startup Readiness | {_rs:.1f}/100 | {_score_label(_rs)} |",
        f"| Business Health | {_bh:.1f}/100 | {_score_label(_bh)} |",
        f"| Funding Eligibility | {_fe:.1f}/100 | {_score_label(_fe)} |",
        "",
        "---",
        "",
        "## Financial Highlights",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Year-1 Revenue Projection | Rs{_ry1:,.0f} |",
        f"| Year-2 Revenue Projection | Rs{_ry2:,.0f} |",
        f"| Total 24-Month Investment | Rs{_inv:,.0f} |",
        f"| Monthly Burn Rate | Rs{_bmc:,.0f} |",
        f"| Estimated Break-even | {_be_str} |",
        "",
        "---",
        "",
    ]

    for json_key, display_title in SECTION_ORDER:
        heading = _MD_SECTION_HEADINGS.get(json_key, f"## {display_title}")
        content = sections.get(json_key, "")
        lines.append(heading)
        lines.append("")

        if not content or not content.strip():
            lines.append("*This section was not generated.*")
        elif json_key == "business_model_canvas":
            # Render BMC as a Markdown table instead of raw JSON or prose
            bmc = parse_bmc(content)
            bmc_has_content = any(v.strip() for v in bmc.values())
            if bmc_has_content:
                lines.append("| Block | Description |")
                lines.append("|-------|-------------|")
                for bk, bt in _BMC_BLOCKS:
                    val = bmc.get(bk, "").strip()
                    if val:
                        # Escape pipe chars in cell content
                        safe_val = val.replace("|", "\\|").replace("\n", " ")
                        lines.append(f"| **{bt}** | {safe_val} |")
            else:
                lines.append(content)
        elif json_key == "final_recommendations":
            # Render as Markdown bullet list regardless of input format
            recs = parse_recommendations(content)
            for rec in recs:
                lines.append(f"- {rec}")
        else:
            lines.append(content)

        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("*Report generated by Startup Blueprint Generator — Powered by IBM watsonx.ai*")
    return "\n".join(lines)
