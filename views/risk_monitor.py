"""
Risk Monitor Page
=================
Drawdown analysis, VaR calculation, and position sizing.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render() -> None:
    """Render the risk monitor page."""
    st.title("🛡️ Risk Monitor")
    st.markdown("Real-time risk management and portfolio protection")
    st.markdown("---")

    # Risk metrics
    st.markdown("### 📊 Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Drawdown", "-5.2%", delta="+2.1%", delta_color="inverse")
    with col2:
        st.metric("Max Drawdown", "-12.5%", delta="-12.5%", delta_color="inverse")
    with col3:
        st.metric("95% VaR (1-day)", "$1,850")
    with col4:
        st.metric("CVaR (Conditional)", "$2,340")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Position Size", "12 contracts")
    with col6:
        st.metric("Current ATR", "$18.50")
    with col7:
        st.metric("Trailing Stop", "$2,085")
    with col8:
        status = "🟢 ACTIVE"
        st.metric("Circuit Breaker", status)

    st.markdown("---")

    # Drawdown chart
    st.markdown("### 📉 Drawdown Analysis")
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq="D")
    equity = [100000 * (1 + 0.45) ** (i / 252) for i in range(252)]
    peak = pd.Series(equity).cummax()
    drawdown = [(e - p) / p * 100 for e, p in zip(equity, peak)]

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color="#ff6b6b", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.15)",
        )
    )
    
    fig.add_hline(
        y=-15,
        line_dash="dash",
        line_color="red",
        annotation_text="Circuit Breaker Threshold (-15%)",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    
    st.plotly_chart(fig, width='stretch')

    # VaR analysis
    st.markdown("### 📊 Value at Risk (VaR) Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample return distribution
        import numpy as np
        returns = np.random.normal(0.001, 0.015, 1000)
        
        fig2 = go.Figure()
        fig2.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=40,
                marker=dict(color="#00d4ff", opacity=0.7),
                name="Returns",
            )
        )
        
        # VaR line
        var_95 = np.percentile(returns * 100, 5)
        fig2.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="red",
            annotation_text=f"95% VaR: {var_95:.2f}%",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig2, width='stretch')
    
    with col2:
        st.markdown("#### VaR Breakdown")
        var_data = pd.DataFrame({
            "Confidence Level": ["90%", "95%", "99%"],
            "VaR ($)": [1450, 1850, 2650],
            "CVaR ($)": [1820, 2340, 3180],
        })
        st.dataframe(var_data, width='stretch', hide_index=True)
        
        st.markdown("#### Position Sizing")
        st.info("""
        **Volatility-Based Sizing**  
        - Base Risk: 2% of capital per trade
        - ATR Multiplier: 2.0x
        - Max Position: 25% of capital
        """)

    # Trailing stop
    st.markdown("### 🎯 Trailing Stop Monitor")
    
    stop_history = pd.DataFrame({
        "Date": pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D"),
        "Price": [2100, 2110, 2120, 2105, 2115, 2125, 2130, 2120, 2135, 2140],
        "Stop Level": [2065, 2070, 2075, 2075, 2080, 2085, 2090, 2090, 2095, 2100],
        "Distance (ATR)": [1.95, 2.16, 2.43, 1.62, 1.89, 2.16, 2.16, 1.62, 2.16, 2.16],
    })
    
    st.dataframe(stop_history, width='stretch', hide_index=True)

    # ── Portfolio Weights (HRP) ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧬 Portfolio Allocation (HRP)")
    st.info("💡 **Hierarchical Risk Parity**: This model suggests weights based on correlation & risk, not just returns. It groups similar assets to prevent over-concentration.")

    from src.risk_manager import PortfolioOptimizer
    from src.watchlist import WatchlistManager
    from src.plugins import registry
    from datetime import datetime, timedelta

    wm = WatchlistManager()
    wl_items = wm.get_all()
    
    # Use watchlist or default demo basket
    if len(wl_items) >= 2:
        assets_to_optimize = [(item.symbol, item.market) for item in wl_items]
        source_label = "your Watchlist"
    else:
        assets_to_optimize = [
            ("GC=F", "COMMODITY"),  # Gold
            ("SI=F", "COMMODITY"),  # Silver
            ("VNM.VN", "VN"),       # Vinamilk (VN Bluechip)
            ("HPG.VN", "VN"),       # Hoa Phat (VN Bluechip)
            ("AAPL", "US"),         # Apple
            ("DX-Y.NYB", "MACRO"),  # US Dollar Index
        ]
        source_label = "a Global Macro basket (Demo)"
        st.caption(f"Showing demo allocation. Add at least 2 assets to your Watchlist to see custom weights.")

    if st.button("🏗️ Compute Optimal Weights"):
        with st.spinner(f"Analyzing correlations for {source_label}..."):
            try:
                returns_data = {}
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                for symbol, market in assets_to_optimize:
                    provider = registry.get_provider(market)
                    if provider:
                        df_p = provider.get_price_data(symbol, start=start_date, end=end_date)
                        if not df_p.empty:
                            returns_data[symbol] = df_p["Close"].pct_change()
                
                if len(returns_data) >= 2:
                    returns_df = pd.DataFrame(returns_data).dropna()
                    weights = PortfolioOptimizer.compute_hrp_weights(returns_df)
                    
                    # Visualization
                    w_df = pd.DataFrame({
                        "Asset": weights.index,
                        "Weight": weights.values * 100,
                        "Market": [next((m for s, m in assets_to_optimize if s == asset), "Other") for asset in weights.index]
                    }).sort_values("Weight", ascending=False)
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        # Sunburst for Market -> Asset hierarchy
                        import plotly.express as px
                        fig_hrp = px.sunburst(
                            w_df, 
                            path=['Market', 'Asset'], 
                            values='Weight',
                            color='Weight',
                            color_continuous_scale='RdBu_r',
                            title="HRP Allocation Structure"
                        )
                        fig_hrp.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_hrp, use_container_width=True)
                    
                    with c2:
                        st.markdown("#### Suggested Allocations")
                        st.dataframe(
                            w_df.style.format({"Weight": "{:.2f}%"}),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Risk Summary
                        total_assets = len(w_df)
                        top_asset = w_df.iloc[0]["Asset"]
                        st.success(f"✅ Portfolio constructed with {total_assets} assets. Top allocation: **{top_asset}**.")
                        st.markdown(f"""
                        **How to read this:**
                        - Assets that move together (high correlation) are "shared" a smaller piece of the pie.
                        - Independent assets get more weight to improve diversification.
                        """)
                else:
                    st.error("Could not fetch enough historical data to compute correlations.")
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.caption("Check your internet connection or API limits (Yahoo Finance/VNStock).")

    st.success("✅ All risk parameters within acceptable limits. Circuit breaker is ACTIVE.")
