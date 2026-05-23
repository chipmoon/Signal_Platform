"""
Asset Search Component for Streamlit
=====================================
Reusable component for searching and selecting assets across markets.
"""

import streamlit as st
from typing import List, Optional

from src.plugins import registry
from src.plugins.base import AssetInfo
from src.watchlist import WatchlistManager


# ─── Quick-pick symbols per category ────────────────────────────────
QUICK_PICKS = {
    "🇺🇸 US Stocks": ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN"],
    "🇻🇳 Vietnam": ["VCB", "HPG", "FPT", "MWG", "TCB", "SSI", "GAS", "VHM"],
    "📊 ETFs": ["SPY", "QQQ", "GLD", "VOO", "SOXX", "ARKK"],
    "🪙 Comp / Crypto": ["GOLD", "SILVER", "BTC-USD", "ETH-USD", "SOL-USD", "CRUDE_OIL"],
    "🌏 Asia": ["2330.TW", "2317.TW", "TSM", "BABA", "700.HK", "9988.HK"],
}


def _perform_search(query: str, market_filter: str = "All Markets") -> List[AssetInfo]:
    """Search assets across markets."""
    markets = registry.list_markets()

    if market_filter == "All Markets":
        return registry.search_all(query, limit=20)

    for m in markets:
        if m["name"] == market_filter:
            provider = registry.get(m["id"])
            return provider.search_assets(query, limit=20) if provider else []

    return []


def render_asset_search(
    watchlist_manager: Optional[WatchlistManager] = None,
    show_watchlist_actions: bool = True,
) -> Optional[AssetInfo]:
    """Render asset search with quick-pick buttons and search box."""
    st.markdown("### 🔍 Asset Search")

    # Market filter
    markets = registry.list_markets()
    market_options = ["All Markets"] + [m["name"] for m in markets]
    selected_market_name = st.selectbox(
        "Filter by Market", options=market_options, key="market_filter"
    )

    # ── Handle quick-pick: read pending query BEFORE creating text_input ──
    pending_query = st.session_state.pop("_pending_search", "")

    search_query = st.text_input(
        "Search by symbol or name",
        value=pending_query,
        placeholder="e.g., AAPL, GOLD, NVDA, TSLA, 2330, VNM, Bitcoin...",
        key="asset_search_query",
    )

    selected_asset = None

    if search_query:
        # Deduplicate results by symbol + market
        all_results = _perform_search(search_query, selected_market_name)
        seen = set()
        results = []
        for r in all_results:
            key = f"{r.symbol}_{r.market}"
            if key not in seen:
                results.append(r)
                seen.add(key)

        if results:
            st.success(f"**Found {len(results)} results:**")

            for i, asset in enumerate(results):
                col1, col2, col3 = st.columns([3, 2, 2])

                with col1:
                    st.markdown(f"**{asset.symbol}** — {asset.name}")
                    st.caption(f"📊 {asset.market} | {asset.sector or 'N/A'}")

                with col2:
                    st.caption(f"💱 {asset.currency} | 🏛️ {asset.exchange or 'N/A'}")

                with col3:
                    if st.button("📋 Select", key=f"sel_{asset.symbol}_{asset.market}_{i}"):
                        selected_asset = asset
                        st.session_state["selected_asset"] = asset
                        st.session_state["global_symbol"] = asset.symbol
                        st.session_state["global_market"] = asset.market
                        st.toast(f"✅ Selected {asset.symbol}")
                        st.rerun()

                # Watchlist actions
                if show_watchlist_actions and watchlist_manager:
                    c_add, c_rem = st.columns(2)
                    with c_add:
                        if not watchlist_manager.contains(asset.symbol, asset.market):
                            if st.button("➕ Watchlist", key=f"add_{asset.symbol}_{asset.market}_{i}"):
                                watchlist_manager.add(
                                    symbol=asset.symbol,
                                    market=asset.market,
                                    name=asset.name,
                                )
                                st.success(f"Added {asset.symbol}!")
                                st.rerun()
                    with c_rem:
                        if watchlist_manager.contains(asset.symbol, asset.market):
                            if st.button("➖ Remove", key=f"rm_{asset.symbol}_{asset.market}_{i}"):
                                watchlist_manager.remove(asset.symbol, asset.market)
                                st.rerun()

                st.markdown("---")
        else:
            st.warning(f"No results for '{search_query}'")
            st.info(
                "💡 **Tips:** Try a ticker (AAPL, NVDA), commodity (GOLD), "
                "crypto (BTC-USD), or company name (Apple, Tesla)"
            )
    else:
        # ── Popular Quick Picks ──────────────────────────────────────
        st.markdown("#### 🔥 Popular Assets — Quick Pick")
        st.caption("Click to search, or type above")

        for category, symbols in QUICK_PICKS.items():
            st.markdown(f"**{category}**")
            cols = st.columns(len(symbols))
            for i, sym in enumerate(symbols):
                with cols[i]:
                    if st.button(sym, key=f"qp_{sym}", use_container_width=True):
                        # Store in _pending_search, then rerun
                        st.session_state["_pending_search"] = sym
                        st.rerun()

        # Market summary
        st.markdown("---")
        st.markdown("#### 🌐 Available Markets")
        if markets:
            mkt_cols = st.columns(len(markets))
            for i, m in enumerate(markets):
                with mkt_cols[i]:
                    st.metric(m["name"], f"{m['id']}")

    return selected_asset


def render_watchlist(
    watchlist_manager: WatchlistManager,
    show_actions: bool = True,
) -> Optional[AssetInfo]:
    """Render watchlist display."""
    st.markdown("### ⭐ My Watchlist")

    items = watchlist_manager.get_all()

    if not items:
        st.info("Your watchlist is empty. Search for assets above to add them!")
        return None

    # View Mode Toggle
    view_mode = st.radio("View Mode", ["Detail Cards", "Strategic Dashboard 📊"], horizontal=True, label_visibility="collapsed")

    # Market filter
    watch_markets = watchlist_manager.get_markets()
    if len(watch_markets) > 1:
        market_filter = st.selectbox(
            "Filter by Market",
            options=["All"] + watch_markets,
            key="watchlist_market_filter",
        )
    else:
        market_filter = "All"

    filtered = (
        watchlist_manager.get_by_market(market_filter)
        if market_filter != "All"
        else items
    )

    if view_mode == "Detail Cards":
        st.caption(f"{len(filtered)} assets")
        selected_asset = None

        for item in filtered:
            col1, col2, col3 = st.columns([3, 2, 2])

            with col1:
                st.markdown(f"**{item.symbol}** — {item.name}")
                st.caption(f"📊 {item.market}")
                if item.notes:
                    st.caption(f"📝 {item.notes}")

            with col2:
                if item.alert_price:
                    st.caption(f"🔔 Alert: ${item.alert_price:,.2f}")

            with col3:
                if show_actions:
                    if st.button("📋 Select", key=f"sel_wl_{item.symbol}_{item.market}"):
                        provider = registry.get(item.market)
                        if provider:
                            selected_asset = provider.get_asset_info(item.symbol)
                            if selected_asset:
                                st.session_state["selected_asset"] = selected_asset
                                st.session_state["global_symbol"] = selected_asset.symbol
                                st.session_state["global_market"] = selected_asset.market

                    if st.button("🗑️", key=f"del_wl_{item.symbol}_{item.market}"):
                        watchlist_manager.remove(item.symbol, item.market)
                        st.rerun()

            st.markdown("---")
        return selected_asset

    else:
        # STRATEGIC DASHBOARD VIEW
        from src.strategies.price_action import PriceActionEngine
        from datetime import datetime, timedelta

        st.info("💡 Strategic View: Real-time Price Action & RS Profiling")
        
        # Prepare table headers
        header_cols = st.columns([1.5, 0.8, 0.6, 0.6, 0.6, 1.2, 1.0, 0.4])
        header_cols[0].markdown("**Asset**")
        header_cols[1].markdown("**Price**")
        header_cols[2].markdown("**P/E**")
        header_cols[3].markdown("**P/B**")
        header_cols[4].markdown("**RS**")
        header_cols[5].markdown("**Structure**")
        header_cols[6].markdown("**Volume**")
        st.markdown("---")

        selected_asset = None
        for item in filtered:
            try:
                provider = registry.get(item.market)
                if not provider: continue
                
                # Fetch minimal data for analysis
                df = provider.get_price_data(item.symbol, (datetime.now()-timedelta(days=180)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))
                if df.empty: continue
                
                # Get Benchmark for RS
                bench_sym = "^GSPC" if item.market == "US" else "VNINDEX" if item.market == "VN" else None
                bench_df = provider.get_price_data(bench_sym, (datetime.now()-timedelta(days=180)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")) if bench_sym else None
                
                analysis = PriceActionEngine.analyze(df, bench_df)
                fundamentals = provider.get_fundamentals(item.symbol) if provider.supports_fundamentals() else {}
                
                row = st.columns([1.5, 0.8, 0.6, 0.6, 0.6, 1.2, 1.0, 0.4])
                
                # Column 0: Symbol
                row[0].markdown(f"**{item.symbol}**")
                row[0].caption(item.name[:15] + "...")
                
                # Column 1: Price
                last_price = df.iloc[-1]["Close"]
                change = (last_price / df.iloc[-2]["Close"] - 1) * 100
                row[1].markdown(f"${last_price:,.2f}")
                row[1].caption(f"{change:+.2f}%")
                
                # Column 2: P/E
                pe = fundamentals.get("pe", 0)
                row[2].markdown(f"{pe:.1f}" if pe > 0 else "N/A")

                # Column 3: P/B
                pb = fundamentals.get("pb", 0)
                row[3].markdown(f"{pb:.1f}" if pb > 0 else "N/A")
                
                # Column 4: RS Score
                rs = analysis.get("rs_score", 0)
                rs_color = "green" if rs > 5 else "red" if rs < -5 else "white"
                row[4].markdown(f":{rs_color}[{rs:+.0f}]")
                
                # Column 5: Structure
                row[5].markdown(f"{analysis.get('structure', 'N/A')}")
                
                # Column 6: Volume
                v_stat = analysis.get("volume_status", "N/A")
                row[6].markdown(f"{v_stat}")
                
                # Column 7: Action
                if row[7].button("🎯", key=f"dash_sel_{item.symbol}"):
                    selected_asset = provider.get_asset_info(item.symbol)
                    st.session_state["global_symbol"] = item.symbol
                    st.session_state["global_market"] = item.market

            except Exception as e:
                logger.error(f"Failed to render dashboard row for {item.symbol}: {e}")
                continue

        return selected_asset


def render_global_stock_selector() -> Optional[str]:
    """
    Render a compact stock selector in the sidebar.
    Used by ALL pages for unified stock selection.

    Returns:
        Selected symbol string or None
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Active Stock")

    # Show currently selected stock
    current_symbol = st.session_state.get("global_symbol", "")
    current_market = st.session_state.get("global_market", "")

    if current_symbol:
        st.sidebar.success(f"**{current_symbol}** ({current_market})")
    else:
        st.sidebar.info("No stock selected")

    # Quick symbol input
    new_symbol = st.sidebar.text_input(
        "Enter Symbol",
        value=current_symbol,
        placeholder="AAPL, NVDA, GOLD...",
        key="sidebar_global_symbol",
    )

    if new_symbol and new_symbol != current_symbol:
        # Try to find the asset across all markets
        results = registry.search_all(new_symbol, limit=1)
        if results:
            asset = results[0]
            st.session_state["global_symbol"] = asset.symbol
            st.session_state["global_market"] = asset.market
            st.session_state["selected_asset"] = asset
            st.sidebar.success(f"✅ {asset.name}")
        else:
            # Try as a direct Yahoo Finance symbol
            st.session_state["global_symbol"] = new_symbol.upper()
            st.session_state["global_market"] = "US"
            st.sidebar.caption(f"Using {new_symbol.upper()} (direct symbol)")

    return st.session_state.get("global_symbol", None)
