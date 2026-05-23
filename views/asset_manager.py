"""
Asset Manager Page
==================
Dedicated page for searching and managing assets across markets.
"""

import streamlit as st
from datetime import datetime, timedelta

from src.components.asset_search import render_asset_search, render_watchlist
from src.plugins import registry
from src.watchlist import WatchlistManager


def render() -> None:
    """Render Asset Manager page."""
    st.title("📊 Multi-Market Asset Manager")
    st.markdown("Search and manage assets across Commodities, Vietnam Stocks, and Taiwan Stocks")
    st.markdown("---")
    
    # Initialize watchlist manager
    if 'watchlist_manager' not in st.session_state:
        st.session_state.watchlist_manager = WatchlistManager()
    
    watchlist_manager = st.session_state.watchlist_manager
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Search Assets", "⭐ My Watchlist"])
    
    # TAB 1: Search Assets
    with tab1:
        selected_asset = render_asset_search(
            watchlist_manager=watchlist_manager,
            show_watchlist_actions=True
        )
        
        # If an asset is selected, show detailed info
        if selected_asset or st.session_state.get('selected_asset'):
            asset = selected_asset or st.session_state.get('selected_asset')
            
            st.markdown("---")
            st.markdown("### 📋 Asset Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Symbol", asset.symbol)
                st.metric("Market", asset.market)
            
            with col2:
                st.metric("Name", asset.name)
                st.metric("Exchange", asset.exchange or "N/A")
            
            with col3:
                st.metric("Sector", asset.sector or "N/A")
                st.metric("Currency", asset.currency)
            
            # ── Price Action Intelligence ─────────────────────────────
            from src.strategies.price_action import PriceActionEngine
            from src.strategies.ai_predictor import AIPredictor
            
            st.markdown("### ⚡ Trend & Price Action Intelligence")
            
            with st.container():
                try:
                    provider = registry.get(asset.market)
                    if provider:
                        # Fetch 180 days for full analysis
                        analysis_end = datetime.now().strftime("%Y-%m-%d")
                        analysis_start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                        
                        df_pa = provider.get_price_data(asset.symbol, analysis_start, analysis_end)
                        
                        if not df_pa.empty:
                            # Get benchmark for RS
                            bench_sym = "^GSPC" if asset.market == "US" else "VNINDEX" if asset.market == "VN" else None
                            bench_df = provider.get_price_data(bench_sym, analysis_start, analysis_end) if bench_sym else None
                            
                            analysis = PriceActionEngine.analyze(df_pa, bench_df)
                            
                            pa_col1, pa_col2, pa_col3 = st.columns(3)
                            
                            with pa_col1:
                                rs = analysis.get("rs_score", 0)
                                rs_color = "green" if rs > 5 else "red" if rs < -5 else None
                                st.metric("RS Score (6m)", f"{rs:+.1f}", delta=f"{rs:+.1f} vs Bench", delta_color="normal")
                                st.caption("Outperformance vs Index")
                                
                            with pa_col2:
                                struc = analysis.get("structure", "N/A")
                                st.metric("Market Structure", struc)
                                st.caption("Last 20 Day Sequence")
                                
                            with pa_col3:
                                v_stat = analysis.get("volume_status", "N/A")
                                st.metric("Volume Momentum", v_stat)
                                st.caption("Activity vs 20D Avg")
                            
                            # ── Intraday AI Pulse (Morning-to-Afternoon Prediction) ──────
                            st.markdown("#### 🕒 Intraday AI Pulse (Morning-to-Afternoon)")
                            with st.expander("🔍 Deep Dive: Morning Session Analysis", expanded=True):
                                try:
                                    # Fetch today's 1H data
                                    intra_start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                                    df_1h = provider.get_price_data(asset.symbol, intra_start, analysis_end, interval="1h")
                                    
                                    if not df_1h.empty:
                                        predictor = AIPredictor()
                                        intra_ai = predictor.predict_afternoon_bias(df_1h)
                                        
                                        ia_col1, ia_col2, ia_col3 = st.columns(3)
                                        with ia_col1:
                                            bias_color = "green" if "Bullish" in intra_ai["bias"] else "red" if "Bearish" in intra_ai["bias"] else "gray"
                                            st.markdown(f"**Afternoon Bias:** :{bias_color}[{intra_ai['bias']}]")
                                            st.progress(intra_ai["confidence"], text=f"Confidence: {intra_ai['confidence']:.0%}")
                                            
                                        with ia_col2:
                                            st.metric("Morning Return", f"{intra_ai.get('morning_return', 0):.2f}%")
                                            
                                        with ia_col3:
                                            st.metric("Volume Delta", f"{intra_ai.get('volume_delta', 0):.1f}%")
                                            
                                        st.caption(f"💡 **AI Logic:** {intra_ai.get('reason', 'Based on session momentum and price-volume clusters.')}")
                                    else:
                                        st.info("Intraday data not available for this session yet.")
                                except Exception as e:
                                    st.warning(f"Intraday AI Pulse failed: {e}")
                                    
                            # ── Institutional Multi-Timeframe Health (1H + 4H) ──────
                            st.markdown("#### 💎 Institutional Multi-Timeframe Health")
                            with st.expander("📊 Fractal Analysis: 1H & 4H Signal Convergence", expanded=True):
                                try:
                                    # Fetch 1H and 4H
                                    df_4h = provider.get_price_data(asset.symbol, intra_start, analysis_end, interval="4h")
                                    
                                    if not df_1h.empty and not df_4h.empty:
                                        mtf_analysis = predictor.analyze_multi_tf(df_1h, df_4h)
                                        
                                        # Visual Dashboard
                                        m_col1, m_col2 = st.columns([2, 1])
                                        
                                        with m_col1:
                                            st.subheader(mtf_analysis["health"])
                                            st.markdown(f"**Sync Status:** {mtf_analysis['bias']}")
                                            st.markdown(f"**Next Day Forecast:** `{mtf_analysis['next_day_forecast']}`")
                                            
                                        with m_col2:
                                            st.metric("1H Momentum", f"{mtf_analysis['1h_mom']:.2f}%")
                                            st.metric("4H Structure", f"{mtf_analysis['4h_struct']:.2f}%")
                                            
                                        st.info("💡 **Expert Recommendation:** Only enter size when both 1H and 4H are in 'Bullish Synergy'. Avoid 'Weak Rebounds' as they often fail at 4H resistance.")
                                    else:
                                        st.info("4H data not supported by provider for this asset.")
                                except Exception as e:
                                    st.warning(f"Multi-TF Health analysis failed: {e}")
                except Exception as e:
                    st.error(f"Price Action analysis failed: {e}")

            # Try to fetch recent price data
            with st.expander("📈 View Data Table & Live Quote"):
                try:
                    provider = registry.get(asset.market)
                    if provider:
                        end_date = datetime.now().strftime("%Y-%m-%d")
                        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                        
                        with st.spinner(f"Fetching {asset.symbol} data..."):
                            df = provider.get_price_data(asset.symbol, start_date, end_date)
                        
                        if not df.empty:
                            st.success(f"Fetched {len(df)} days of data")
                            
                            # Show latest price
                            latest = df.iloc[-1]
                            col_price1, col_price2, col_price3 = st.columns(3)
                            
                            with col_price1:
                                st.metric("Latest Close", f"{latest['Close']:,.2f} {asset.currency}")
                            
                            with col_price2:
                                if len(df) > 1:
                                    prev_close = df.iloc[-2]['Close']
                                    change = ((latest['Close'] / prev_close) - 1) * 100
                                    st.metric("Daily Change", f"{change:+.2f}%")
                            
                            with col_price3:
                                st.metric("Volume", f"{latest['Volume']:,.0f}")
                            
                            # Show data table
                            st.dataframe(
                                df.tail(10),
                                width='stretch',
                                hide_index=True
                            )
                        
                except Exception as e:
                    st.error(f"Could not fetch price data: {e}")
            
            # Fundamentals (if supported)
            provider = registry.get(asset.market)
            if provider and provider.supports_fundamentals():
                with st.expander("💼 View Value Metrics (P/E, ROE...)"):
                    try:
                        with st.spinner("Fetching fundamentals..."):
                            fundamentals = provider.get_fundamentals(asset.symbol)
                        
                        if fundamentals and any(v > 0 for v in fundamentals.values()):
                            col_f1, col_f2, col_f3 = st.columns(3)
                            
                            with col_f1:
                                st.metric("P/E Ratio", f"{fundamentals.get('pe', 0):.2f}")
                                st.metric("EPS", f"{fundamentals.get('eps', 0):.2f}")
                            
                            with col_f2:
                                st.metric("P/B Ratio", f"{fundamentals.get('pb', 0):.2f}")
                                st.metric("ROE", f"{fundamentals.get('roe', 0):.2f}%")
                            
                            with col_f3:
                                st.metric("Book Value", f"{fundamentals.get('bvps', 0):.2f}")
                                st.metric("Div Yield", f"{fundamentals.get('dividend_yield', 0):.2f}%")
                        else:
                            st.info("No fundamental data available")
                    
                    except Exception as e:
                        st.error(f"Could not fetch fundamentals: {e}")
    
    # TAB 2: Watchlist
    with tab2:
        selected_watchlist_asset = render_watchlist(
            watchlist_manager=watchlist_manager,
            show_actions=True
        )
        
        if selected_watchlist_asset:
            st.session_state['selected_asset'] = selected_watchlist_asset
        
        # Watchlist actions
        st.markdown("---")
        st.markdown("### ⚙️ Watchlist Actions")
        
        col_action1, col_action2 = st.columns(2)
        
        with col_action1:
            if st.button("📥 Export Symbols for Backtest"):
                symbols = watchlist_manager.export_symbols()
                st.code(", ".join(symbols))
                st.success(f"Exported {len(symbols)} symbols")
        
        with col_action2:
            if st.button("🗑️ Clear Watchlist", type="secondary"):
                if st.session_state.get('confirm_clear'):
                    watchlist_manager.clear()
                    st.success("Watchlist cleared!")
                    st.session_state['confirm_clear'] = False
                    st.rerun()
                else:
                    st.session_state['confirm_clear'] = True
                    st.warning("Click again to confirm")
    
    # (Removed Market Overview - duplicated by Market Pulse)

if __name__ == "__main__":
    render()
