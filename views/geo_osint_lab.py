"""Geo-OSINT Lab page: production live OSINT -> causal -> execution gate."""

from __future__ import annotations

import streamlit as st

from src.strategies.live_geo_osint import LiveGeoOsintEngine


def render() -> None:
    st.title("Geo-OSINT Lab")
    st.caption("Production runtime: Live OSINT feed -> normalize -> causal score -> execution gate")

    st.graphviz_chart(
        """
digraph G {
    rankdir=LR;
    node [shape=box, style=rounded];
    A [label="OSINT Parser\\n(AIS, wires, sanctions, policy)"];
    B [label="Normalization Layer\\n(schema + confidence + decay)"];
    C [label="Causal Scoring\\n(prob oil-up, expected move)"];
    D [label="Execution Gate\\n(normal/reduce/pause)"];
    E [label="AI Forecast + Backtest Engine"];
    A -> B -> C -> D -> E;
}
"""
    )

    symbol = st.session_state.get("global_symbol", "")
    market = st.session_state.get("global_market", "US")

    c1, c2, c3 = st.columns([1, 1, 2], gap="small")
    with c1:
        force_refresh = st.button("Refresh Live Feed", use_container_width=True)
    with c2:
        auto_sync = st.toggle("Auto Sync Gate", value=True, key="geo_auto_sync")
    with c3:
        st.caption(f"Target: {symbol or 'N/A'} ({market}) | Source: yfinance wire + symbol news")

    cache_key = f"geo_live_state::{market}::{symbol}"
    state = st.session_state.get(cache_key)
    if force_refresh or not isinstance(state, dict):
        with st.spinner("Ingesting live OSINT feed..."):
            state = LiveGeoOsintEngine().build_live_state(symbol=symbol, market=market)
        st.session_state[cache_key] = state

    events = state.get("events")
    osint = state.get("osint", {})
    causal = state.get("causal", {})
    gate = state.get("gate", {})
    status = state.get("status", {})

    if auto_sync and gate:
        st.session_state["geo_gate_state"] = gate
        st.session_state["geo_osint_state"] = osint
        st.session_state["geo_causal_state"] = causal

    st.markdown("### Normalized Event Feed")
    st.dataframe(events, width="stretch", hide_index=True if events is not None else False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Geo Risk Score", f"{float(osint.get('geo_risk_score', 0.0)):.2f}")
    m2.metric("Prob Oil Up (1D)", f"{float(causal.get('prob_oil_up_1d', 0.5)):.2%}")
    m3.metric("Expected Oil Move", f"{float(causal.get('expected_oil_move_1d_pct', 0.0)):+.2f}%")
    m4.metric("Gate Action", str(gate.get("action", "NORMAL")))

    st.markdown(
        f"**Execution decision**: `{gate.get('action', 'NORMAL')}` | "
        f"Multiplier: `{float(gate.get('position_multiplier', 1.0)):.2f}` | "
        f"Reason: {gate.get('reason', 'N/A')}"
    )
    st.caption(
        f"Feed status | stale={bool(status.get('stale', True))} | events={int(status.get('events_count', 0))} | "
        f"freshness(min)={float(status.get('freshness_minutes', -1.0)):.1f} | latest={status.get('latest_event_ts', '') or 'N/A'}"
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Apply Gate To AI Forecast", use_container_width=True):
            st.session_state["geo_gate_state"] = gate
            st.session_state["geo_osint_state"] = osint
            st.session_state["geo_causal_state"] = causal
            st.success("Gate synced to AI Forecast.")
    with b2:
        if st.button("Clear Gate", use_container_width=True):
            for k in ("geo_gate_state", "geo_osint_state", "geo_causal_state"):
                st.session_state.pop(k, None)
            st.info("Gate state cleared for AI Forecast.")
