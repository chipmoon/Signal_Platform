"""
Trading Intelligence Platform - Streamlit Dashboard
===================================================
Multi-page web application for professional trading system analysis.

Pages:
    1. Dashboard - Backtest overview and equity curve
    2. Strategy Analysis - Individual strategy breakdown
    3. AI Forecast - Multi-horizon predictions
    4. Risk Monitor - Drawdown, VaR, position sizing
    5. Settings - Parameter configuration

Usage:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from loguru import logger

# Page configuration
st.set_page_config(
    page_title="Trading Intelligence Platform",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FRESHNESS_FILES = [
    PROJECT_ROOT / ".cache" / "meta.json",
    PROJECT_ROOT / "data" / "nightly_scan_meta.json",
]


def _data_fingerprint() -> int:
    """Return a cheap fingerprint for local files written by nightly jobs."""
    stamps = [p.stat().st_mtime_ns for p in DATA_FRESHNESS_FILES if p.exists()]
    return max(stamps) if stamps else 0


def _sync_external_data_updates() -> None:
    """Drop session quote caches after local nightly/cache files change."""
    fingerprint = _data_fingerprint()
    previous = st.session_state.get("_data_fingerprint")
    if previous is not None and fingerprint and fingerprint != previous:
        for key in list(st.session_state.keys()):
            if key.startswith("rt_quote_"):
                st.session_state.pop(key, None)
    st.session_state["_data_fingerprint"] = fingerprint


def _install_periodic_refresh() -> None:
    """Keep a locally opened dashboard in sync with files updated outside Streamlit."""
    try:
        interval_sec = int(os.getenv("TRADING_SYSTEM_AUTO_REFRESH_SEC", "120"))
    except ValueError:
        interval_sec = 120
    if interval_sec <= 0:
        return
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            const url = new URL(window.parent.location.href);
            url.searchParams.set("_market_refresh", Date.now().toString());
            window.parent.location.replace(url.toString());
        }}, {interval_sec * 1000});
        </script>
        """,
        height=0,
    )


_sync_external_data_updates()
_install_periodic_refresh()

# Custom Premium UI - High Contrast Theme
st.markdown(
    """
    <style>
    /* ========================================
       PREMIUM DARK THEME - HIGH CONTRAST
       ======================================== */
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a15 0%, #141428 100%);
    }
    
    /* ========================================
       SIDEBAR - HIGH CONTRAST
       ======================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar Title */
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        padding: 1rem 0 !important;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem !important;
    }
    
    /* Radio Button Labels (Menu Items) - HIGH CONTRAST */
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Radio Button Container */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.75rem !important;
    }
    
    /* Individual Radio Options */
    [data-testid="stSidebar"] .stRadio label {
        color: #b0b0b0 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1.2rem !important;
        margin: 0.2rem 0 !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Hover State */
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
        color: #ffffff !important;
        transform: translateX(4px) !important;
    }
    
    /* Selected/Active State */
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transform: scale(1.02) translateX(4px) !important;
    }
    
    /* Sidebar Text Elements */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Markdown */
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar Caption */
    [data-testid="stSidebar"] .caption {
        color: #9fa8da !important;
        font-size: 0.85rem !important;
    }
    
    /* ========================================
       HEADERS - HIGH CONTRAST
       ======================================== */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #f0f0f0 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
    }
    
    h3 {
        color: #e8eaf6 !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
    }
    
    /* ========================================
       METRIC CARDS - MAX READABILITY
       ======================================== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00e6ff, #00ff88) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    
    /* ========================================
       BUTTONS - PREMIUM STYLE
       ======================================== */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0px) scale(1);
    }
    
    /* Primary Button */
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        color: #0a0a15 !important;
        font-weight: 800;
    }
    
    /* ========================================
       INPUTS - DARK THEME OVERRIDE
       ======================================== */
    /* Text inputs */
    input, textarea {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        border: 1px solid rgba(100, 120, 200, 0.5) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-weight: 500 !important;
    }
    
    input:focus, textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
        background-color: #1e1e3a !important;
    }
    
    /* Select dropdowns */
    [data-baseweb="select"], .stSelectbox [data-baseweb="select"] {
        background-color: #1a1a2e !important;
    }
    
    [data-baseweb="select"] > div, .stSelectbox [data-baseweb="select"] > div {
        background-color: #1a1a2e !important;
        border-color: rgba(100, 120, 200, 0.8) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    [data-baseweb="select"] span, [data-baseweb="select"] div {
        color: #ffffff !important;
    }
    
    /* Dropdown menu and Popover (Important for readability) */
    div[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
        background-color: #16213e !important;
        background: #16213e !important;
        border: 2px solid rgba(100, 120, 200, 0.8) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.8) !important;
    }
    
    div[data-baseweb="popover"] li, [data-baseweb="menu"] li, ul[role="listbox"] li {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        background-color: transparent !important;
        padding: 12px 20px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    div[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover, ul[role="listbox"] li:hover {
        background-color: #667eea !important;
        color: #ffffff !important;
        cursor: pointer !important;
    }

    /* Fix for "All Markets" text visibility */
    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    /* Number input */
    [data-testid="stNumberInput"] input {
        background-color: rgba(30, 30, 55, 0.9) !important;
        color: #f0f0f0 !important;
    }
    
    /* Date input */
    [data-testid="stDateInput"] input {
        background-color: rgba(30, 30, 55, 0.9) !important;
        color: #f0f0f0 !important;
    }
    
    /* Slider */
    [data-testid="stSlider"] div[data-baseweb="slider"] div {
        color: #e0e0e0 !important;
    }
    
    /* Checkbox */
    [data-testid="stCheckbox"] label span {
        color: #e0e0e0 !important;
    }
    
    /* Input labels - global */
    label, .stTextInput label, .stSelectbox label {
        color: #c5cae9 !important;
        font-weight: 500 !important;
    }
    
    /* ========================================
       DATA TABLES
       ======================================== */
    .dataframe {
        background: rgba(30, 30, 50, 0.6);
        border-radius: 8px;
        color: #e0e0e0 !important;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        color: #e8eaf6 !important;
    }
    
    /* ========================================
       ALERTS & INFO BOXES
       ======================================== */
    .stAlert {
        background: rgba(255, 193, 7, 0.15);
        border-left: 4px solid #ffc107;
        border-radius: 6px;
        color: #ffffff !important;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4caf50;
        color: #ffffff !important;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.15);
        border-left: 4px solid #f44336;
        color: #ffffff !important;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.15);
        border-left: 4px solid #2196f3;
        color: #ffffff !important;
    }
    
    /* ========================================
       GENERAL TEXT
       ======================================== */
    p, div, span, label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Links */
    a {
        color: #00d4ff !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #00ff88 !important;
        text-decoration: underline !important;
    }
    
    /* Code Blocks */
    code {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #00ff88 !important;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    /* ========================================
       TABS
       ======================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        color: #c5cae9 !important;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.15);
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* ========================================
       EXPANDER
       ======================================== */
    .streamlit-expanderHeader, [data-testid="stExpander"] details summary {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stExpander"] details summary p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.15);
    }
    
    /* ========================================
       CALENDAR / DATE PICKER POPUP - FIX BRIGHTNESS
       ======================================== */
    /* Target the calendar popup container */
    div[data-baseweb="datepicker"] ~ div[role="dialog"],
    div[data-baseweb="datepicker"] + div,
    div[data-baseweb="calendar"] {
        background-color: #1a1a2e !important;
        color: #f0f0f0 !important;
        border: 1px solid rgba(100, 120, 200, 0.5) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
    }
    
    /* Calendar Header (Month/Year) */
    div[data-baseweb="calendar"] header,
    div[data-baseweb="calendar"] header ~ div {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    /* Day Labels (Su, Mo, Tu...) */
    div[data-baseweb="calendar"] div[role="gridcell"] {
        color: #c5cae9 !important;
        font-weight: 600 !important;
    }
    
    /* Actual Day Numbers */
    div[data-baseweb="calendar"] div[aria-label^="Choose"] {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px !important;
    }
    
    /* Hover on Days */
    div[data-baseweb="calendar"] div[aria-label^="Choose"]:hover {
        background-color: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Selected Day */
    div[data-baseweb="calendar"] div[aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #0a0a15 !important;
        font-weight: 700 !important;
    }
    
    /* Out of month days */
    div[data-baseweb="calendar"] div[aria-disabled="true"] {
        color: rgba(255, 255, 255, 0.2) !important;
        background-color: transparent !important;
    }

    /* ========================================
       RHS WIDGETS - PREMIUM CARDS
       ======================================== */
    .rhs-card {
        background: rgba(20, 20, 35, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
    }
    
    .rhs-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .rhs-badge {
        font-size: 0.75rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    
    .badge-ready { background: rgba(255, 152, 0, 0.2); color: #FF9800; border: 1px solid #FF9800; }
    .badge-bull { background: rgba(0, 230, 118, 0.1); color: #00E676; }
    .badge-bear { background: rgba(255, 82, 82, 0.1); color: #FF5252; }
    .badge-neutral { background: rgba(144, 164, 174, 0.1); color: #B0BEC5; }
    
    .price-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .price-subcard {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: left;
    }
    
    .price-label { font-size: 0.75rem; color: #94a3b8; margin-bottom: 2px; }
    .price-value { font-size: 1.1rem; font-weight: 800; color: #ffffff; }
    .price-value-highlight { color: #64ffda; }
    
    .distance-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    
    .progress-outer {
        background: rgba(255, 255, 255, 0.05);
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    .progress-inner {
        height: 100%;
        background: linear-gradient(90deg, #64ffda, #00bfa5);
        border-radius: 3px;
    }
    
    .rec-box {
        background: rgba(100, 255, 218, 0.03);
        border-left: 3px solid #64ffda;
        padding: 0.75rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
        color: #e2e8f0;
        line-height: 1.4;
    }
    
    .level-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .level-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.5rem;
        text-align: center;
        border-radius: 6px;
    }
    
    .level-price { font-size: 0.85rem; font-weight: 700; color: #ffffff; }
    .level-marker { width: 100%; height: 3px; background: #667eea; margin-top: 4px; border-radius: 1px; }
    .level-marker.red { background: #FF5252; }
    .level-marker.blue { background: #00d4ff; }
    
    .sentiment-label {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .move-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .move-index {
        width: 24px;
        height: 24px;
        background: rgba(102, 126, 234, 0.2);
        color: #764ba2;
        border-radius: 4px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: 800;
        margin-right: 0.75rem;
        font-size: 0.8rem;
    }
    
    .move-info { flex: 1; }
    .move-price { font-size: 0.9rem; font-weight: 700; color: #ffffff; }
    .move-desc { font-size: 0.75rem; color: #94a3b8; }
    .move-prob { font-size: 0.9rem; font-weight: 800; color: #00ff88; }

    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
# Keep navigation labels ASCII-safe to avoid mojibake on Windows/local Streamlit.
st.sidebar.title("Trading Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Pulse",
        "Dashboard",
        "Sentiment Hub",
        "Alpha Scanner",
        "Hidden Gem Council",
        "AI Forecast",
        "Geo-OSINT Lab",
        "Risk Monitor",
        "Asset Manager",
        "Settings",
    ],
    index=0,
    label_visibility="collapsed",
)

# Learning Center
st.sidebar.markdown("---")
with st.sidebar.expander("Learning Center"):
    st.markdown("""
    **ATR (Average True Range)**
    Measures market volatility. Higher = more price movement.
    
    **COT (Commitment of Traders)**
    Shows what big institutional funds are doing.
    
    **RSI (Rel. Strength Index)**
    Values >70 are overbought; <30 are oversold.
    
    **Bias (AI Forecast)**
    The AI's predicted direction for the next 24 hours.
    """)
    if st.button("Read Full Guide"):
        st.session_state["show_guide"] = True

# Global Stock Selector
st.sidebar.markdown("---")
st.sidebar.markdown("### Active Stock")

current_sym = st.session_state.get("global_symbol", "")
current_mkt = st.session_state.get("global_market", "")

if current_sym:
    st.sidebar.success(f"**{current_sym}** ({current_mkt})")
    try:
        from src.plugins import registry as _reg_live
        _provider = _reg_live.get(current_mkt)
        if _provider:
            _quote = _provider.get_realtime_quote(current_sym)
            if _quote:
                _delta_color = "normal" if _quote.change >= 0 else "inverse"
                st.sidebar.metric(
                    "Live Price",
                    f"{_quote.price:,.2f}",
                    f"{_quote.change:+.2f}%",
                    delta_color=_delta_color,
                )
                st.sidebar.caption(
                    f"Source: {_quote.source} | Updated: {_quote.timestamp[:19].replace('T', ' ')}"
                )
            else:
                st.sidebar.caption("Live Price: unavailable")
    except Exception as _ex:
        logger.debug(f"Sidebar live quote failed for {current_sym}: {_ex}")
else:
    st.sidebar.info("No stock selected")

new_sym = st.sidebar.text_input(
    "Quick Symbol",
    placeholder="AAPL, NVDA, GOLD, BTC-USD...",
    key="sidebar_quick_sym",
    label_visibility="collapsed",
)

if new_sym:
    new_sym_upper = new_sym.upper().strip()
    if new_sym_upper != current_sym:
        from src.plugins import registry as _reg
        hits = _reg.search_all(new_sym_upper, limit=1)
        if hits:
            st.session_state["global_symbol"] = hits[0].symbol
            st.session_state["global_market"] = hits[0].market
            st.session_state["selected_asset"] = hits[0]
        else:
            st.session_state["global_symbol"] = new_sym_upper
            st.session_state["global_market"] = "US"

# Quick pick row
qp_cols = st.sidebar.columns(3)
for i, s in enumerate(["AAPL", "NVDA", "XAG"]):
    with qp_cols[i]:
        if st.button(s, key=f"sb_qp_{s}", use_container_width=True):
            from src.plugins import registry as _reg2
            # Handle XAG mapping specifically if needed, or search_all will find it
            hits2 = _reg2.search_all(s, limit=1)
            if hits2:
                st.session_state["global_symbol"] = hits2[0].symbol
                st.session_state["global_market"] = hits2[0].market
            st.rerun()

qp_cols2 = st.sidebar.columns(3)
for i, s in enumerate(["TSLA", "SPY", "BTC-USD"]):
    with qp_cols2[i]:
        if st.button(s, key=f"sb_qp_{s}", use_container_width=True):
            from src.plugins import registry as _reg3
            hits3 = _reg3.search_all(s, limit=1)
            if hits3:
                st.session_state["global_symbol"] = hits3[0].symbol
                st.session_state["global_market"] = hits3[0].market
            st.rerun()

st.sidebar.markdown("---")

# Page Routing
if st.session_state.get("show_guide", False):
    st.markdown("## Trading Intelligence Guide")
    if st.button("Back to Dashboard"):
        st.session_state["show_guide"] = False
        st.rerun()
    
    with open("docs/TRADING_GUIDE.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())
    
    if st.button("Back to Top"):
        st.rerun()

elif page == "Market Pulse":
    from views import market_pulse
    market_pulse.render()

elif page == "Dashboard":
    from views import dashboard
    dashboard.render()

elif page == "Sentiment Hub":
    from views import sentiment_hub
    sentiment_hub.render()



elif page == "AI Forecast":
    from views import ai_forecast
    ai_forecast.render()

elif page == "Geo-OSINT Lab":
    from views import geo_osint_lab
    geo_osint_lab.render()

elif page == "Risk Monitor":
    from views import risk_monitor
    risk_monitor.render()

elif page == "Asset Manager":
    from views import asset_manager
    asset_manager.render()

elif page == "Alpha Scanner":
    from views import alpha_scanner
    alpha_scanner.render()

elif page == "Hidden Gem Council":
    from views import hidden_gem
    hidden_gem.render()

elif page == "Settings":
    from views import settings
    settings.render()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Trading Intelligence Platform v2.0")
st.sidebar.caption("Built with Streamlit - Powered by Machine Learning")
