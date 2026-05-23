"""Dashboard page: market overview + chart workspace."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import urllib.parse
import urllib.request

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from src.plugins import registry


def _safe_pct(curr: float, prev: float) -> float:
    if prev and prev != 0:
        return (curr - prev) / prev * 100.0
    return 0.0


def _fetch_index_close(symbol: str, market: str | None = None) -> tuple[float | None, float | None, str]:
    """Return latest close, previous close, and source label."""
    try:
        if market:
            provider = registry.get(market)
            if provider:
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
                df = provider.get_price_data(symbol, start, end)
                if df is not None and not df.empty and "Close" in df.columns:
                    c = float(df["Close"].iloc[-1])
                    p = float(df["Close"].iloc[-2]) if len(df) >= 2 else c
                    return c, p, f"{market} provider"
    except Exception:
        pass

    # Fallback: yfinance
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1mo", interval="1d", auto_adjust=False)
        if not hist.empty:
            c = float(hist["Close"].iloc[-1])
            p = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else c
            return c, p, "yfinance"
    except Exception:
        pass

    return None, None, "N/A"


def _fetch_price_df(symbol: str, market: str, days: int = 260) -> pd.DataFrame:
    provider = registry.get(market)
    if not provider:
        return pd.DataFrame()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = provider.get_price_data(symbol, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _render_candlestick(df: pd.DataFrame, symbol: str, chart_key: str = "local") -> None:
    if df.empty:
        st.warning("No local OHLC data for this symbol.")
        return
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=symbol,
            )
        ]
    )
    fig.update_layout(template="plotly_dark", height=900, margin=dict(l=8, r=8, t=30, b=8), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch", key=f"candlestick_{chart_key}_{symbol}")


def _tv_symbol_for(symbol: str, market: str, vn_exchange: str) -> str:
    s = symbol.upper().replace(".VN", "").replace(".TW", "").replace(".TWO", "")
    if market == "VN":
        return f"{vn_exchange}:{s}"
    if market == "TW":
        return f"TWSE:{s}"
    return symbol.upper()


def _resolve_vn_tv_symbol(symbol: str) -> tuple[str | None, str]:
    """
    Auto-resolve VN symbol on TradingView by trying HOSE/HNX/UPCOM.
    Returns (tv_symbol, reason).
    """
    base = symbol.upper().replace(".VN", "")
    exchanges = ["HOSE", "HNX", "UPCOM"]
    for ex in exchanges:
        try:
            q = urllib.parse.urlencode({"text": base, "exchange": ex, "hl": "1", "lang": "en", "search_type": "stock"})
            url = f"https://symbol-search.tradingview.com/symbol_search/?{q}"
            with urllib.request.urlopen(url, timeout=6) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            for item in payload:
                sym = str(item.get("symbol", "")).upper()
                exch = str(item.get("exchange", "")).upper()
                if sym == base and exch == ex:
                    return f"{ex}:{base}", f"Resolved by TradingView search ({ex})"
        except Exception:
            continue
    return None, "TradingView symbol not found on HOSE/HNX/UPCOM search API"


def _render_tradingview_widget(tv_symbol: str, interval: str) -> None:
    interval_map = {"1D": "D", "4H": "240", "1H": "60", "30m": "30"}
    tv_interval = interval_map.get(interval, "D")
    html = f"""
    <div class="tradingview-widget-container" style="width:100%; height:88vh; min-height:980px;"> 
      <div id="tradingview_chart" style="width:100%; height:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "{tv_interval}",
        "timezone": "Asia/Taipei",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(html, height=1100, scrolling=False)


def render() -> None:
    # Keep this page visually wide and reduce clipping for embedded widgets
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 100% !important;
            padding-top: 0.8rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        [data-testid="stTabs"] { margin-top: 0.25rem; }
        [data-testid="stTabs"] iframe { width: 100% !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Market Dashboard")
    st.markdown("Vietnam + Taiwan index monitor and chart workspace")

    st.markdown("### Key Market Indices")
    vn_c, vn_p, vn_src = _fetch_index_close("VNINDEX", market="VN")
    tw_c, tw_p, tw_src = _fetch_index_close("^TWII", market=None)

    c1, c2 = st.columns(2)
    with c1:
        if vn_c is not None:
            st.metric("VN-Index", f"{vn_c:,.2f}", delta=f"{_safe_pct(vn_c, vn_p):+.2f}%")
            st.caption(f"Source: {vn_src}")
        else:
            st.warning("VN-Index unavailable")
    with c2:
        if tw_c is not None:
            st.metric("TWSE Weighted (TWII)", f"{tw_c:,.2f}", delta=f"{_safe_pct(tw_c, tw_p):+.2f}%")
            st.caption(f"Source: {tw_src}")
        else:
            st.warning("TWII unavailable")

    st.markdown("---")
    st.markdown("### Stock Chart Workspace")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        symbol = st.text_input("Symbol", value=st.session_state.get("global_symbol", "BSR.VN"))
    with col2:
        market = st.selectbox("Market", ["VN", "TW", "US"], index=0)
    with col3:
        interval = st.selectbox("Interval", ["1D", "4H", "1H", "30m"], index=0)
    with col4:
        vn_exchange = st.selectbox("VN Exchange", ["HOSE", "HNX", "UPCOM"], index=0)

    st.session_state["global_symbol"] = symbol
    st.session_state["global_market"] = market

    tab_local, tab_tv = st.tabs(["Candlestick (Local Data)", "TradingView Live Widget"])
    with tab_local:
        df = _fetch_price_df(symbol, market, days=320)
        _render_candlestick(df, symbol, chart_key="tab_local")
    with tab_tv:
        # Auto resolve VN prefix by trying HOSE/HNX/UPCOM.
        if market == "VN":
            resolved, reason = _resolve_vn_tv_symbol(symbol)
            if resolved:
                st.caption(f"TradingView symbol: {resolved} | {reason}")
                _render_tradingview_widget(resolved, interval)
            else:
                st.warning(f"TradingView unavailable for this VN symbol. {reason}. Fallback to local candlestick.")
                df_fb = _fetch_price_df(symbol, market, days=320)
                _render_candlestick(df_fb, symbol, chart_key="tab_tv_fallback")
        else:
            tv_symbol = _tv_symbol_for(symbol, market, vn_exchange)
            st.caption(f"TradingView symbol: {tv_symbol}")
            _render_tradingview_widget(tv_symbol, interval)
