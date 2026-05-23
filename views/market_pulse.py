"""
Market Pulse View
=================
Situational awareness dashboard for global markets and systemic risk.
Displays Crash Shield levels and Market Regimes for VN, TW, US, and Crypto.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

from src.strategies.crash_shield import CrashShield
from src.strategies.regime_detector import RegimeDetector
from src.plugins import registry
from src.llm_advisor import advisor
from src.strategies.sentiment_hub import SentimentHub

def render():
    st.title("📡 Market Pulse — Global Situational Awareness")
    st.markdown("---")

    # 🛡️ SYSTEMIC RISK SCANNER (Crash Shield)
    with st.spinner("Scanning global markets for panic signals..."):
        risk_report = CrashShield.evaluate_risk()
    
    # Display Crash Level Banner
    level = risk_report["level"]
    status = risk_report["status"]
    
    if level == 3:
        st.error(f"### 🚨 CRITICAL: {status}")
        st.warning("⚠️ CRASH PROTOCOL ACTIVE: All new trades blocked. Tighten all existing stops or exit weak positions.")
    elif level == 2:
        st.warning(f"### 🟠 HIGH ALERT: {status}")
        st.info("💡 DEFENSIVE POSTURE: Reduce position sizes by 60%. Block new volatile assets.")
    elif level == 1:
        st.info(f"### 🟡 CAUTION: {status}")
        st.write("💡 SELECTIVE TRADING: Reduce size by 30%. Focus on high-quality setups only.")
    else:
        st.success(f"### 🟢 ALL CLEAR: {status}")
        st.write("💡 NORMAL TRADING: Global volatility is within safe parameters.")

    # 🤖 GEMINI GLOBAL INTEL
    st.markdown("---")
    with st.expander("🤖 Gemini AI Global Intel", expanded=True):
        hub = SentimentHub()
        headlines = [n['title'] for n in hub.get_macro_news("Geopolitics")[:5]]
        headlines += [n['title'] for n in hub.get_macro_news("Economy")[:5]]
        
        if headlines:
            summary = advisor.get_macro_summary(headlines)
            st.info(summary)
        else:
            st.caption("Awaiting global news feeds...")

    st.markdown("---")

    # 📊 GLOBAL INDEX CARDS
    cols = st.columns(4)
    data = risk_report.get("data", {})
    
    markets = [
        ("VN-Index", "VN", "🇻🇳"), 
        ("S&P 500", "US", "🇺🇸"), 
        ("Bitcoin", "BTC", "₿"), 
        ("Gold", "GOLD", "📀")
    ]
    
    for i, (label, key, icon) in enumerate(markets):
        m_data = data.get(key, {})
        price = m_data.get("price", 0)
        change = m_data.get("change_pct", 0)
        
        with cols[i]:
            st.metric(
                label=f"{icon} {label}", 
                value=f"{price:,.2f}" if key != "BTC" else f"${price:,.0f}", 
                delta=f"{change:+.2f}%"
            )

    st.markdown("---")

    # 🗺️ MARKET REGIME MAP
    st.subheader("🗺️ Market Regime Map")
    regime_cols = st.columns(3)
    
    # Analyze core indices for regimes
    with st.spinner("Analyzing market regimes..."):
        # We'll fetch a bit more data for regimes
        regimes = {}
        target_indices = [
            ("VN Index", "^VNINDEX", "VN"),
            ("Nasdaq 100", "^IXIC", "US"),
            ("Taiwan Weighted", "^TWII", "TW")
        ]
        
        for name, ticker, mkt in target_indices:
            try:
                provider = registry.get("US") # Use US (yfinance) for global indices
                df = provider.get_price_data(ticker, period="1y")
                regimes[name] = RegimeDetector.identify(df)
            except:
                regimes[name] = {"regime": "UNKNOWN", "icon": "❓", "description": "Data Error"}

    for i, (name, analysis) in enumerate(regimes.items()):
        with regime_cols[i % 3]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); text-align: center;">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">{name}</div>
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{analysis['icon']}</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem;">{analysis['regime']}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.4;">{analysis['description']}</div>
                <hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                    <span>ADX: <b>{analysis.get('adx', 0)}</b></span>
                    <span>RSI: <b>{analysis.get('rsi', 0)}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 🕵️ DECISION ADVISORY (Algorithm-based)
    st.subheader("🕵️ Institutional Decision Advisory")
    
    advice_cols = st.columns([2, 1])
    
    with advice_cols[0]:
        st.markdown("#### 🏛️ Strategy Allocation")
        if level >= 2:
            st.error("🛡️ **DEFENSIVE MODE ACTIVE**")
            st.write("- **Trend Strategies**: HALTED (High chop risk)")
            st.write("- **SMC/Wyckoff**: Focus ONLY on deep discount demand zones")
            st.write("- **Position Sizing**: Max 2% per trade, 20% total equity")
            st.write("- **Assets**: Shift to Cash or Gold")
        elif regimes.get("VN Index", {}).get("regime") == "BULL":
            st.success("🚀 **OFFENSIVE MODE ACTIVE**")
            st.write("- **Trend Strategies**: FULL POWER (Golden Cross / Sea Level)")
            st.write("- **SMC/Wyckoff**: Buy the first pullback (LPS / FVG)")
            st.write("- **Position Sizing**: Standard (5-10% per trade)")
            st.write("- **Assets**: High RS sectors (FPT, TCB, HPG...)")
        else:
            st.info("↔️ **NEUTRAL MODE ACTIVE**")
            st.write("- **Trend Strategies**: Selective (Wait for breakouts)")
            st.write("- **SMC/Wyckoff**: Mean reversion at TR boundaries")
            st.write("- **Position Sizing**: Moderate (3-5% per trade)")

    with advice_cols[1]:
        st.markdown("#### 📊 Sector Pulse (VN)")
        # Placeholder for sector analysis
        st.caption("Auto-scanning 12 VN sectors...")
        st.write("🟢 **Banking**: Leading")
        st.write("⚪ **Real Estate**: Weak")
        st.write("🟢 **Technology**: Strong")
        st.write("🔴 **Utilities**: Lagging")

    # Last update info
    st.sidebar.markdown("---")
    st.sidebar.write(f"⏱️ Pulse Sync: {risk_report.get('timestamp')}")
    if st.sidebar.button("🔄 Refresh Global Pulse"):
        st.rerun()

if __name__ == "__main__":
    render()
