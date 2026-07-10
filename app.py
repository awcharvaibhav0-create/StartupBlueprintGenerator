"""
app.py — Streamlit frontend for Startup Blueprint Generator.

IBM SkillsBuild AICTE 2026 · Powered by IBM Granite 4 H Small · IBM watsonx.ai
"""

import logging
import os
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Module-level imports — avoids repeated import overhead on every function call
from rag import build_index, retrieve as _rag_retrieve, get_index_stats
from agents import (
    AgentOrchestrator,
    CALL1_REQUIRED_KEYS, CALL2_REQUIRED_KEYS,
    ALL_REQUIRED_KEYS, merge_blueprint_sections,
    _call_llm,
)
from utils import (
    calculate_scores, calculate_financials, build_calc_summary,
    chart_score_gauges, chart_revenue_vs_cost, chart_cumulative_profit,
    chart_swot, chart_cost_breakdown, chart_funding_stages,
    generate_pdf_report, generate_markdown_report,
    _safe_int, _safe_float,
    parse_bmc, parse_recommendations, _BMC_BLOCKS,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Startup Blueprint Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.ibm.com/watsonx",
        "About": "Startup Blueprint Generator · IBM SkillsBuild AICTE 2026",
    },
)

# ===========================================================================
# Custom CSS — IBM Design-inspired dark theme
# ===========================================================================
st.markdown(
    """
<style>
/* ---- Base — system font stack; IBM Plex Sans if available locally ---- */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', Arial,
                 system-ui, -apple-system, sans-serif;
}

/* ---- App background ---- */
.stApp { background-color: #161616; color: #f4f4f4; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #1c1c1c;
    border-right: 1px solid #393939;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #78a9ff;
}

/* ---- Main header ---- */
.sbg-hero {
    background: linear-gradient(135deg, #0f62fe 0%, #0043ce 60%, #8a3ffc 100%);
    border-radius: 12px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
}
.sbg-hero h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
.sbg-hero p  { color: #a6c8ff; font-size: 1rem; margin: 0; }

/* ---- KPI cards ---- */
.kpi-card {
    background: #262626;
    border: 1px solid #393939;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #0f62fe; }
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #78a9ff;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-label {
    font-size: 0.82rem;
    color: #8d8d8d;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---- Section cards ---- */
.section-card {
    background: #1c1c1c;
    border: 1px solid #393939;
    border-left: 4px solid #0f62fe;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.section-card h4 {
    color: #78a9ff;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.6rem 0;
}

/* ---- Chat bubbles ---- */
.chat-user {
    background: #1d3461;
    border-radius: 8px 8px 2px 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #e8f1ff;
}
.chat-ai {
    background: #1c1c1c;
    border: 1px solid #393939;
    border-radius: 8px 8px 8px 2px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #f4f4f4;
}

/* ---- Buttons ---- */
.stButton > button {
    background: #0f62fe;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    font-size: 0.95rem;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover { background: #0043ce; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ---- Inputs ---- */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: #262626 !important;
    color: #f4f4f4 !important;
    border: 1px solid #525252 !important;
    border-radius: 6px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #0f62fe !important;
    box-shadow: 0 0 0 2px rgba(15,98,254,0.25) !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { background: #1c1c1c; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #8d8d8d; }
.stTabs [aria-selected="true"] { color: #78a9ff !important; border-bottom-color: #0f62fe !important; }

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: #1c1c1c;
    border-radius: 6px;
    color: #78a9ff !important;
}

/* ---- Progress ---- */
.stProgress > div > div { background-color: #0f62fe; }

/* ---- Divider ---- */
hr { border-color: #393939; }

/* ---- Code ---- */
code { background: #262626 !important; color: #78a9ff !important; border-radius: 4px; }

/* ---- Tag badges ---- */
.badge {
    display: inline-block;
    background: #0f62fe22;
    color: #78a9ff;
    border: 1px solid #0f62fe55;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ===========================================================================
# Session State Initialisation
# ===========================================================================
_DEFAULTS = {
    "blueprint": None,          # full result dict from orchestrator
    "scores": None,
    "financials": None,
    "rag_context": "",          # cached RAG context from last generation
    "chat_history": [],         # list of {"role": "user"|"ai", "content": str}
    "rag_ready": False,
    "generation_done": False,
    "idea": "",
    "industry": "Technology",
    "stage": "Idea",
    "target_market": "",
    "revenue_model": "SaaS Subscription",
    "team_size": 3,
    "monthly_burn": 150000.0,
    "price_per_unit": 999.0,
    "units_month1": 10,
    "growth_rate": 0.20,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===========================================================================
# Helpers
# ===========================================================================

def _check_env() -> tuple[bool, str]:
    """Validate required environment variables."""
    api_key    = os.getenv("IBM_API_KEY", "")
    project_id = os.getenv("IBM_PROJECT_ID", "")
    wx_url     = os.getenv("IBM_WATSONX_URL", "")
    if not api_key:
        return False, "IBM_API_KEY is missing from your .env file."
    if not project_id:
        return False, "IBM_PROJECT_ID is missing from your .env file."
    if not wx_url:
        return False, "IBM_WATSONX_URL is missing from your .env file."
    return True, "OK"


def _init_rag():
    """Initialise RAG index once per session (idempotent)."""
    if not st.session_state.rag_ready:
        with st.spinner("📚 Loading knowledge base …"):
            try:
                build_index("data")
                st.session_state.rag_ready = True
            except Exception as exc:
                logger.warning("RAG init warning: %s", exc)
                st.session_state.rag_ready = True  # continue with fallback knowledge


def _retrieve(query: str) -> str:
    """Retrieve RAG context; always returns a string, never raises."""
    try:
        return _rag_retrieve(query, top_k=5)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return ""


def _kpi_card(label: str, value: str, color: str = "#78a9ff") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{color};">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


def _score_color(score) -> str:
    s = _safe_float(score, default=0.0)
    if s >= 75: return "#24a148"
    if s >= 55: return "#f1c21b"
    if s >= 35: return "#ff832b"
    return "#da1e28"


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            "## 🚀 Startup Blueprint\n"
            "<span class='badge'>IBM Granite 4 H Small</span> "
            "<span class='badge'>watsonx.ai</span> "
            "<span class='badge'>FAISS RAG</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "⚙️ Generator", "📊 Dashboard", "💬 AI Chat", "📄 Reports"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ENV status
        ok, msg = _check_env()
        if ok:
            st.success("✅ IBM credentials loaded")
        else:
            st.error(f"❌ {msg}")

        # RAG status (uses module-level import)
        stats = get_index_stats()
        if stats["index_ready"]:
            st.info(f"📚 RAG: {stats['total_chunks']} chunks indexed")
        else:
            st.warning("📚 RAG index not built yet")

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#6f6f6f;'>"
            "IBM SkillsBuild AICTE 2026<br>"
            "Powered by IBM watsonx.ai<br>"
            "Model: ibm/granite-4-h-small"
            "</div>",
            unsafe_allow_html=True,
        )

        return page


# ===========================================================================
# Page: Home / Landing
# ===========================================================================

def page_home():
    st.markdown(
        """
<div class="sbg-hero">
  <h1>🚀 Startup Blueprint Generator</h1>
  <p>Transform your startup idea into a complete, investor-ready business blueprint
     powered by IBM Granite 4 H Small on IBM watsonx.ai.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🤖", "Multi-Agent AI", "7 specialised agents coordinated by an orchestrator"),
        ("📡", "FAISS RAG", "Retrieval from startup, DPIIT & MSME knowledge base"),
        ("📊", "Live Dashboards", "Plotly charts: revenue, break-even, SWOT, scores"),
        ("📄", "PDF Export",  "Download a professional business blueprint report"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f'<div class="kpi-card" style="text-align:left;">'
                f'<div style="font-size:1.8rem;margin-bottom:0.5rem;">{icon}</div>'
                f'<div style="font-weight:600;color:#78a9ff;font-size:0.95rem;">{title}</div>'
                f'<div style="color:#8d8d8d;font-size:0.82rem;margin-top:0.3rem;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
**Two-LLM Optimised Pipeline**

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│           Agent Orchestrator                │
│  ┌─────────────────────────────────────┐   │
│  │  LLM Call 1 — IBM Granite           │   │
│  │  • Startup Validation Agent         │   │
│  │  • Market Research Agent            │   │
│  │  • Business Model Canvas Agent      │   │
│  │  • Financial Planning Agent         │   │
│  └─────────────────────────────────────┘   │
│         │                                   │
│         ▼                                   │
│  Python Calculations + FAISS RAG Query      │
│         │                                   │
│         ▼                                   │
│  ┌─────────────────────────────────────┐   │
│  │  LLM Call 2 — IBM Granite           │   │
│  │  • Funding & Govt Scheme Agent      │   │
│  │  • Investor Pitch Agent             │   │
│  │  • Startup Roadmap Agent            │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
    │
    ▼
Streamlit Dashboard · PDF Export · AI Chat
```
""")

    with col_r:
        st.markdown("### 📋 What You Get")
        sections = [
            "✅ Startup Validation", "🎯 Problem & Solution", "👥 Customer Analysis",
            "⚔️ Competitor Analysis", "🔷 SWOT Analysis", "🌍 Market Opportunity",
            "🏗️ Business Model Canvas", "💵 Revenue & Pricing", "💰 Cost & Break-even",
            "🏦 Funding Recommendations", "🏛️ Govt Schemes", "🎤 Investor Pitch",
            "🗺️ Startup Roadmap", "⚠️ Risk Assessment", "📄 Executive Summary",
        ]
        for s in sections:
            st.markdown(f"- {s}")

    st.markdown("---")
    st.info("👉 Go to **⚙️ Generator** in the sidebar to create your blueprint.")


# ===========================================================================
# Page: Generator (Input Form)
# ===========================================================================

def page_generator():
    st.markdown("## ⚙️ Blueprint Generator")
    st.markdown("Fill in your startup details and click **Generate Blueprint**.")

    env_ok, env_msg = _check_env()
    if not env_ok:
        st.error(f"⚠️ Configuration Error: {env_msg}")
        st.code(
            "# Add your credentials to a .env file:\n"
            "IBM_API_KEY=...\nIBM_PROJECT_ID=...\nIBM_WATSONX_URL=...\nMODEL_ID=ibm/granite-4-h-small",
            language="bash",
        )
        return

    with st.form("blueprint_form", clear_on_submit=False):
        st.markdown("### 💡 Startup Idea")
        idea = st.text_area(
            "Describe your startup idea",
            value=st.session_state.idea,
            height=110,
            placeholder=(
                "e.g. An AI-powered platform that helps small farmers in India "
                "predict crop disease using smartphone photos and provides "
                "personalised treatment recommendations in regional languages."
            ),
        )

        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox(
                "Industry / Sector",
                [
                    "Technology", "FinTech", "HealthTech", "EdTech", "AgriTech",
                    "E-Commerce", "SaaS", "CleanTech", "LogisticsTech", "D2C / Retail",
                    "AI / ML", "Cybersecurity", "Gaming", "Media & Entertainment",
                    "Real Estate Tech", "HR Tech", "LegalTech", "Travel & Hospitality",
                    "Food & Beverage", "Manufacturing", "Other",
                ],
                index=0,
            )
            target_market = st.text_input(
                "Target Market",
                value=st.session_state.target_market,
                placeholder="e.g. Small & medium farmers in rural India, 18–55 years",
            )
            revenue_model = st.selectbox(
                "Revenue Model",
                [
                    "SaaS Subscription", "Freemium", "Marketplace / Commission",
                    "Direct Sales", "Advertising", "Licensing",
                    "Usage-based / Pay-per-use", "Consulting / Services",
                ],
            )

        with col2:
            stage = st.selectbox(
                "Current Stage",
                ["Idea", "Validation", "MVP", "Early Traction", "Growth", "Scaling"],
            )
            team_size = st.slider("Team Size", 1, 50, st.session_state.team_size)
            monthly_burn = st.number_input(
                "Monthly Burn Rate (₹)",
                min_value=10_000,
                max_value=50_000_000,
                value=int(st.session_state.monthly_burn),
                step=10_000,
                format="%d",
            )

        st.markdown("### 📈 Revenue Projection Parameters")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            price_per_unit = st.number_input(
                "Price per Unit / Customer (₹)", min_value=1.0,
                value=st.session_state.price_per_unit, step=100.0,
            )
        with pc2:
            units_month1 = st.number_input(
                "Customers / Units — Month 1", min_value=1,
                value=st.session_state.units_month1, step=1,
            )
        with pc3:
            growth_rate_pct = st.slider(
                "Monthly Growth Rate (%)", 1, 50,
                int(st.session_state.growth_rate * 100),
            )

        submitted = st.form_submit_button("🚀 Generate Blueprint", use_container_width=True)

    if submitted:
        if not idea.strip():
            st.error("Please enter your startup idea before generating.")
            return

        # Persist inputs to session state
        st.session_state.update({
            "idea": idea, "industry": industry, "stage": stage,
            "target_market": target_market, "revenue_model": revenue_model,
            "team_size": team_size, "monthly_burn": float(monthly_burn),
            "price_per_unit": price_per_unit, "units_month1": int(units_month1),
            "growth_rate": growth_rate_pct / 100.0,
        })

        _run_generation()


def _run_generation():
    """Execute the full pipeline: RAG → calculations → LLM1 → LLM2."""
    s = st.session_state

    # 1. Init RAG (idempotent — skips if already built)
    _init_rag()

    progress = st.progress(0, text="Starting generation …")

    try:
        # 2. Python Calculations (no LLM cost)
        progress.progress(10, text="⚙️ Running financial calculations …")
        scores = calculate_scores(
            s.idea, s.industry, s.stage, s.team_size,
            s.monthly_burn, s.target_market, s.revenue_model,
        )
        financials = calculate_financials(
            monthly_burn=s.monthly_burn,
            team_size=s.team_size,
            price_per_unit=s.price_per_unit,
            units_month1=s.units_month1,
            growth_rate=s.growth_rate,
            months=24,
        )
        calc_summary = build_calc_summary(scores, financials)

        # 3. RAG retrieval (no LLM cost)
        progress.progress(20, text="📚 Retrieving knowledge base context …")
        rag_query   = f"{s.idea} {s.industry} startup funding government schemes India"
        rag_context = _retrieve(rag_query)

        # 4. LLM Call 1 — Validation, Market Research, BMC, Financial Overview
        progress.progress(35, text="🤖 IBM Granite · Call 1 — Validation & Market Research …")
        orchestrator = AgentOrchestrator()
        # Run only LLM Call 1 portion — orchestrator.run does both calls internally
        result = orchestrator.run(
            idea=s.idea,
            industry=s.industry,
            stage=s.stage,
            target_market=s.target_market,
            revenue_model=s.revenue_model,
            team_size=s.team_size,
            monthly_burn=s.monthly_burn,
            rag_context=rag_context,
            calc_summary=calc_summary,
        )
        # Progress update between the two calls is inside orchestrator.run;
        # we advance here after the full orchestrator returns.
        progress.progress(85, text="🤖 IBM Granite · Call 2 complete — assembling blueprint …")

        # result["sections"] is the guaranteed-complete 20-key merged dict
        # produced by merge_blueprint_sections() inside orchestrator.run().
        # Re-merge here only so app.py has an authoritative local reference
        # for the warning check; no second LLM call is made.
        sections = result["sections"]
        missing = [k for k in ALL_REQUIRED_KEYS if not sections.get(k)]
        if missing:
            logger.warning("Blueprint incomplete — empty sections: %s", missing)

        st.session_state.blueprint        = result
        st.session_state.scores           = scores
        st.session_state.financials       = financials
        st.session_state.rag_context      = rag_context   # cache for chat page
        st.session_state.generation_done  = True

        progress.progress(100, text="✅ Blueprint ready!")
        time.sleep(0.4)
        progress.empty()

        # Warn only when there are genuinely empty sections (not just short ones)
        truly_missing = [k for k in missing
                         if not sections.get(k, "").strip()]
        if truly_missing:
            st.warning(
                f"⚠️ {len(truly_missing)} section(s) came back empty from IBM Granite "
                f"({', '.join(truly_missing[:4])}{'…' if len(truly_missing) > 4 else ''}). "
                "The blueprint is still usable — try regenerating for fuller output."
            )
        else:
            st.success("✅ Blueprint generated! Go to **📊 Dashboard** or **📄 Reports**.")

    except ConnectionError as exc:
        # Must come before EnvironmentError — ConnectionError is a subclass of
        # EnvironmentError in Python 3, so catching EnvironmentError first would
        # swallow this clause silently.
        progress.empty()
        logger.error("Network error during generation: %s", exc)
        st.error("🌐 Network Error: Could not reach IBM watsonx.ai.")
        st.info("Check your internet connection and verify IBM_WATSONX_URL in your .env file.")
    except TimeoutError as exc:
        # TimeoutError is also a subclass of EnvironmentError — must come before it.
        progress.empty()
        logger.error("Timeout during generation: %s", exc)
        st.error("⏱️ Request timed out while calling IBM watsonx.ai.")
        st.info("IBM API may be temporarily slow. Please try again in a moment.")
    except EnvironmentError as exc:
        progress.empty()
        st.error(f"🔐 Configuration Error: {exc}")
        st.info("Ensure IBM_API_KEY, IBM_PROJECT_ID, and IBM_WATSONX_URL are set in your .env file.")
    except Exception as exc:
        progress.empty()
        logger.exception("Blueprint generation failed")
        st.error(f"❌ Generation failed: {type(exc).__name__}: {exc}")
        st.info("Check your IBM credentials, internet connection, and try again.")


# ===========================================================================
# Page: Dashboard
# ===========================================================================

def page_dashboard():
    st.markdown("## 📊 Startup Dashboard")

    if not st.session_state.generation_done:
        st.info("👉 Generate a blueprint first from the **⚙️ Generator** page.")
        return

    scores     = st.session_state.scores
    financials = st.session_state.financials
    blueprint  = st.session_state.blueprint

    # Guard: blueprint must be fully populated before rendering
    if not blueprint or "sections" not in blueprint:
        st.warning(
            "Blueprint data is incomplete. Please regenerate from the **⚙️ Generator** page."
        )
        return

    # Fetch the canonical merged 20-key dict once — every read below uses this.
    # blueprint["sections"] is built by merge_blueprint_sections() inside
    # orchestrator.run(), so every key in ALL_REQUIRED_KEYS is guaranteed present.
    all_secs = blueprint["sections"]

    # ---- Coerce KPI values once — guards str/None from stale session state ----
    _rs  = _safe_float(scores.get("startup_readiness_score"),   0.0)
    _bh  = _safe_float(scores.get("business_health_score"),     0.0)
    _fe  = _safe_float(scores.get("funding_eligibility_score"), 0.0)
    _ry1 = _safe_float(financials.get("annual_revenue_y1"),     0.0)
    _be  = _safe_int(financials.get("breakeven_month"),         0)

    # ---- KPI Row ----
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(
            _kpi_card("Startup Readiness", f"{_rs:.1f}", _score_color(_rs)),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            _kpi_card("Business Health", f"{_bh:.1f}", _score_color(_bh)),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            _kpi_card("Funding Eligibility", f"{_fe:.1f}", _score_color(_fe)),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            _kpi_card("Year-1 Revenue", f"Rs{_ry1/100000:.1f}L", "#78a9ff"),
            unsafe_allow_html=True,
        )
    with k5:
        be_str = f"M{_be}" if _be else "24M+"
        st.markdown(
            _kpi_card("Break-even", be_str, "#24a148"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Score Gauges ----
    st.plotly_chart(chart_score_gauges(scores), use_container_width=True)

    # ---- Revenue / Cost / SWOT tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Revenue vs Cost",
        "💹 Cumulative P&L",
        "🔷 SWOT",
        "🍩 Cost Breakdown",
        "🏦 Funding Stages",
    ])

    with tab1:
        try:
            st.plotly_chart(chart_revenue_vs_cost(financials), use_container_width=True)
        except Exception as _e:
            st.error(f"Chart error: {_e}")
            logger.error("chart_revenue_vs_cost failed: %s", _e)

    with tab2:
        try:
            st.plotly_chart(chart_cumulative_profit(financials), use_container_width=True)
        except Exception as _e:
            st.error(f"Chart error: {_e}")
            logger.error("chart_cumulative_profit failed: %s", _e)

    with tab3:
        swot_text = all_secs.get("swot_analysis", "")
        if swot_text.strip():
            try:
                st.plotly_chart(chart_swot(swot_text), use_container_width=True)
            except Exception as _e:
                logger.error("chart_swot failed: %s", _e)
            with st.expander("📄 Full SWOT Text"):
                st.markdown(swot_text)
        else:
            st.info(
                "IBM Granite did not return a SWOT analysis for this run. "
                "Try regenerating the blueprint for a fuller response."
            )

    with tab4:
        try:
            st.plotly_chart(chart_cost_breakdown(financials), use_container_width=True)
        except Exception as _e:
            st.error(f"Chart error: {_e}")
            logger.error("chart_cost_breakdown failed: %s", _e)

    with tab5:
        try:
            st.plotly_chart(chart_funding_stages(), use_container_width=True)
        except Exception as _e:
            st.error(f"Chart error: {_e}")
            logger.error("chart_funding_stages failed: %s", _e)

    st.markdown("---")

    # ---- Blueprint Sections — direct key lookup from merged JSON dict ----
    # all_secs is blueprint["sections"], fetched once at the top of this function.
    # Every key in ALL_REQUIRED_KEYS is guaranteed present (value may be "").
    # No Markdown parsing, no fuzzy matching, no get_section() indirection.
    st.markdown("### 📋 Blueprint Sections")

    # Ordered display list — keys match ALL_REQUIRED_KEYS exactly.
    DISPLAY_SECTIONS = [
        ("startup_validation",          "✅ Startup Validation"),
        ("problem_statement",           "🎯 Problem Statement"),
        ("solution_description",        "💡 Solution Description"),
        ("target_customer_analysis",    "👥 Target Customer Analysis"),
        ("competitor_analysis",         "⚔️ Competitor Analysis"),
        ("swot_analysis",               "🔷 SWOT Analysis"),
        ("market_opportunity_analysis", "🌍 Market Opportunity"),
        ("business_model_canvas",       "🏗️ Business Model Canvas"),
        ("revenue_model",               "💵 Revenue Model"),
        ("pricing_strategy",            "🏷️ Pricing Strategy"),
        ("cost_estimation",             "📋 Cost Estimation"),
        ("break_even_analysis",         "📉 Break-even Analysis"),
        ("funding_recommendations",     "🏦 Funding Recommendations"),
        ("government_schemes",          "🏛️ Government Schemes"),
        ("investor_pitch",              "🎤 Investor Pitch"),
        ("startup_roadmap",             "🗺️ Startup Roadmap"),
        ("risk_assessment",             "⚠️ Risk Assessment"),
        ("executive_summary",           "📄 Executive Summary"),
        ("future_scope",                "🔭 Future Scope"),
        ("final_recommendations",       "✨ Final Recommendations"),
    ]

    # Validate: ensure all 20 required keys are present with placeholders
    for _k in ALL_REQUIRED_KEYS:
        if _k not in all_secs:
            all_secs[_k] = ""

    for json_key, title in DISPLAY_SECTIONS:
        content = all_secs.get(json_key, "")
        with st.expander(title, expanded=False):
            if not content or not content.strip():
                st.caption(
                    "IBM Granite did not return content for this section. "
                    "Regenerate the blueprint for a fuller response."
                )

            elif json_key == "business_model_canvas":
                # Render as professional 9-block grid — never show raw JSON
                try:
                    bmc = parse_bmc(content)
                    bmc_has_content = any(v.strip() for v in bmc.values())
                    if bmc_has_content:
                        # Display in a 3-column grid matching standard BMC layout
                        col_groups = [
                            [("key_partnerships", "🤝 Key Partnerships"),
                             ("key_activities",   "⚙️ Key Activities"),
                             ("value_propositions","💎 Value Propositions"),
                             ("customer_relationships","🤗 Customer Relationships"),
                             ("customer_segments","👥 Customer Segments")],
                            [("key_resources",    "🔑 Key Resources"),
                             ("channels",         "📢 Channels")],
                            [("cost_structure",   "💰 Cost Structure"),
                             ("revenue_streams",  "💵 Revenue Streams")],
                        ]
                        # Row 1: 5 blocks side by side (standard BMC top row)
                        top_blocks = col_groups[0]
                        cols = st.columns(len(top_blocks))
                        for col, (bk, bt) in zip(cols, top_blocks):
                            val = bmc.get(bk, "").strip()
                            with col:
                                st.markdown(
                                    f'<div class="section-card">'
                                    f'<h4>{bt}</h4>'
                                    f'{val if val else "<i>Not specified</i>"}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        # Row 2: Key Resources (left) + Channels (right)
                        c1, c2 = st.columns(2)
                        for col, (bk, bt) in zip([c1, c2], col_groups[1]):
                            val = bmc.get(bk, "").strip()
                            with col:
                                st.markdown(
                                    f'<div class="section-card">'
                                    f'<h4>{bt}</h4>'
                                    f'{val if val else "<i>Not specified</i>"}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        # Row 3: Cost Structure (left) + Revenue Streams (right)
                        c1, c2 = st.columns(2)
                        for col, (bk, bt) in zip([c1, c2], col_groups[2]):
                            val = bmc.get(bk, "").strip()
                            with col:
                                st.markdown(
                                    f'<div class="section-card">'
                                    f'<h4>{bt}</h4>'
                                    f'{val if val else "<i>Not specified</i>"}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                    else:
                        # parse_bmc returned all empty — render raw as plain text
                        st.markdown(content)
                except Exception as _e:
                    logger.error("BMC render error: %s", _e)
                    st.markdown(content)

            elif json_key == "final_recommendations":
                # Always render as a clean bullet list — never show Python list repr
                try:
                    recs = parse_recommendations(content)
                    for rec in recs:
                        st.markdown(f"- {rec}")
                except Exception as _e:
                    logger.error("Recommendations render error: %s", _e)
                    st.markdown(content)

            else:
                st.markdown(content)


# ===========================================================================
# Page: AI Chat
# ===========================================================================

def page_chat():
    st.markdown("## 💬 AI Chat Assistant")
    st.markdown("Ask anything about your startup blueprint, funding, or business strategy.")

    env_ok, env_msg = _check_env()
    if not env_ok:
        st.error(f"⚠️ {env_msg}")
        return

    # Ensure RAG index is ready — idempotent, safe to call on every page load
    _init_rag()

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 <b>You</b><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-ai">🤖 <b>Blueprint AI</b><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_msg = st.text_input(
                "Your question",
                placeholder="e.g. How should I approach Series A funding for my EdTech startup?",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.spinner("🤖 Thinking …"):
            try:
                # ── 1. Blueprint context ─────────────────────────────────────
                # Include the full merged 20-key JSON dict.  A generous per-key
                # cap (800 chars) keeps the prompt within token budget while
                # preserving multi-paragraph and bullet-list structure.
                # Internal newlines are preserved so the model reads the content
                # in the same form it was generated.
                _KEY_CAP = 800   # chars per section — raised from 300
                blueprint_ctx = ""
                if st.session_state.generation_done and st.session_state.blueprint:
                    bp       = st.session_state.blueprint
                    all_secs = bp["sections"]  # guaranteed-complete 20-key dict
                    bp_lines = [
                        "--- STARTUP BLUEPRINT ---",
                        f"Idea     : {st.session_state.idea}",
                        f"Industry : {st.session_state.industry}",
                        f"Stage    : {st.session_state.stage}",
                        "",
                    ]
                    for key in ALL_REQUIRED_KEYS:
                        value = all_secs.get(key, "")
                        if not value.strip():
                            continue
                        heading = key.replace("_", " ").title()
                        # Trim only if genuinely long; keep newlines intact
                        if len(value) > _KEY_CAP:
                            value = value[:_KEY_CAP] + "…"
                        bp_lines.append(f"[{heading}]")
                        bp_lines.append(value)
                        bp_lines.append("")
                    blueprint_ctx = "\n".join(bp_lines)

                # ── 2. RAG context ───────────────────────────────────────────
                # Reuse the context cached during blueprint generation so no
                # extra RAG call is made for questions about the same startup.
                # When no blueprint has been generated yet, retrieve once for
                # the current question only.
                cached_rag = st.session_state.get("rag_context", "")
                if cached_rag:
                    rag_ctx = cached_rag
                else:
                    rag_ctx = _retrieve(user_msg)

                # ── 3. Conversation history ──────────────────────────────────
                # Include previous turns so the model can refer back to earlier
                # answers without the user having to repeat context.
                # Cap at the 6 most recent exchanges (12 messages) to stay
                # well within the token budget.
                _HISTORY_TURNS = 6
                history = st.session_state.chat_history[:-1]  # exclude current message
                recent  = history[-(2 * _HISTORY_TURNS):]      # last N user+ai pairs
                history_block = ""
                if recent:
                    history_lines = []
                    for msg in recent:
                        role_label = "User" if msg["role"] == "user" else "Assistant"
                        history_lines.append(f"{role_label}: {msg['content']}")
                    history_block = "--- CONVERSATION HISTORY ---\n" + "\n".join(history_lines)

                # ── 4. Assemble prompt ───────────────────────────────────────
                system = (
                    "You are an expert startup advisor and business consultant. "
                    "Answer the question clearly and practically. "
                    "Use the blueprint and knowledge base provided. "
                    "Focus on actionable, India-specific advice where relevant."
                )
                parts = [system, ""]
                if rag_ctx:
                    parts += [f"--- KNOWLEDGE BASE ---", rag_ctx, ""]
                if blueprint_ctx:
                    parts += [blueprint_ctx, ""]
                if history_block:
                    parts += [history_block, ""]
                parts += [f"User: {user_msg}", "", "Assistant:"]
                prompt = "\n".join(parts)

                response = _call_llm(prompt)
                st.session_state.chat_history.append({"role": "ai", "content": response})

            except Exception as exc:
                err = f"Sorry, I encountered an error: {exc}"
                st.session_state.chat_history.append({"role": "ai", "content": err})
                logger.error("Chat error: %s", exc)

        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Suggested questions — shown only when no conversation has started
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What are the best government schemes for a HealthTech startup in India?",
            "How do I calculate the right valuation for my seed round?",
            "What metrics should I track to be Series A ready?",
            "Explain the Business Model Canvas for an EdTech platform.",
            "What is the difference between DPIIT recognition and MSME registration?",
            "How do I find angel investors for my startup in India?",
        ]
        col_a, col_b = st.columns(2)
        for i, sug in enumerate(suggestions):
            with (col_a if i % 2 == 0 else col_b):
                st.markdown(f'<div class="section-card"><h4>💬</h4>{sug}</div>', unsafe_allow_html=True)


# ===========================================================================
# Page: Reports
# ===========================================================================

def page_reports():
    st.markdown("## 📄 Reports & Exports")

    if not st.session_state.generation_done:
        st.info("👉 Generate a blueprint first from the **⚙️ Generator** page.")
        return

    s          = st.session_state
    scores     = s.scores
    financials = s.financials
    blueprint  = s.blueprint

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📥 PDF Blueprint Report")
        st.markdown(
            "Download a professionally formatted PDF report containing "
            "the complete startup blueprint, KPI scores, financial highlights, "
            "and all AI-generated sections."
        )
        if st.button("📄 Generate & Download PDF", use_container_width=True):
            with st.spinner("Generating PDF …"):
                try:
                    pdf_bytes = generate_pdf_report(
                        idea=s.idea,
                        industry=s.industry,
                        stage=s.stage,
                        scores=scores,
                        financials=financials,
                        sections=blueprint["sections"],
                    )
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"startup_blueprint_{s.industry.lower().replace(' ','_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")
                    logger.exception("PDF generation error")

    with col2:
        st.markdown("### 📝 Markdown Report")
        st.markdown(
            "Download the full blueprint as a Markdown (.md) file — "
            "perfect for pasting into Notion, Confluence, or GitHub."
        )
        try:
            md_content = generate_markdown_report(
                idea=s.idea,
                industry=s.industry,
                stage=s.stage,
                scores=scores,
                financials=financials,
                sections=blueprint["sections"],
            )
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=md_content.encode("utf-8"),
                file_name=f"startup_blueprint_{s.industry.lower().replace(' ','_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Markdown generation failed: {exc}")

    st.markdown("---")

    # ---- Section previews — direct key lookup from merged JSON dict ----
    # No Markdown parsing, no fuzzy matching, no get_section() indirection.
    all_secs = blueprint["sections"]

    st.markdown("### 👁️ Preview — Executive Summary")
    exec_summ = all_secs.get("executive_summary", "")
    if exec_summ.strip():
        st.markdown(exec_summ)
    else:
        st.info(
            "IBM Granite did not return an executive summary for this run. "
            "Regenerate the blueprint for a fuller response."
        )

    st.markdown("### 🎤 Investor Pitch Preview")
    pitch = all_secs.get("investor_pitch", "")
    if pitch.strip():
        with st.expander("View Investor Pitch", expanded=True):
            st.markdown(pitch)
    else:
        st.info(
            "IBM Granite did not return an investor pitch for this run. "
            "Regenerate the blueprint for a fuller response."
        )

    st.markdown("### 🗺️ Startup Roadmap Preview")
    roadmap = all_secs.get("startup_roadmap", "")
    if roadmap.strip():
        with st.expander("View Roadmap", expanded=True):
            st.markdown(roadmap)
    else:
        st.info(
            "IBM Granite did not return a startup roadmap for this run. "
            "Regenerate the blueprint for a fuller response."
        )


# ===========================================================================
# Main Router
# ===========================================================================

def main():
    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "⚙️ Generator":
        page_generator()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "💬 AI Chat":
        page_chat()
    elif page == "📄 Reports":
        page_reports()


if __name__ == "__main__":
    main()
