"""Alpha Scanner View."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.strategies.alpha_scanner import AlphaScannerEngine

_NIGHTLY_JSON = Path(__file__).resolve().parents[1] / "data" / "nightly_scan_results.json"
_NIGHTLY_META = Path(__file__).resolve().parents[1] / "data" / "nightly_scan_meta.json"


def _load_nightly() -> tuple[pd.DataFrame | None, dict]:
    """Load pre-computed nightly scan results."""
    if not _NIGHTLY_JSON.exists():
        return None, {}
    try:
        data = json.loads(_NIGHTLY_JSON.read_text(encoding="utf-8"))
        meta = json.loads(_NIGHTLY_META.read_text(encoding="utf-8")) if _NIGHTLY_META.exists() else {}
        return pd.DataFrame(data), meta
    except Exception:
        return None, {}

def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%" if pd.notna(v) else "N/A"


def _apply_preset_filter(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Apply tactical preset filters on scanner outputs."""
    if df.empty:
        return df
    if preset == "Institutional Balanced":
        out = df.copy()
        out = out[
            (out["recommendation"].isin(["STRONG BUY", "BUY"]))
            | (out["institutional_score"] >= 0.08)
            | ((out["qmf_score"] >= 0.05) & (out["wyckoff_score"] >= 0.05))
        ]
        return out.sort_values(
            by=["veto", "institutional_score", "pred_63d_ret", "confidence_boosted"],
            ascending=[True, False, False, False],
        )
    if preset == "Parabolic Runner (1-3M)":
        # Runner profile: strong 1-3M upside + improving structural/flow context.
        out = df.copy()
        out = out[
            (out["market"].isin(["VN", "TW"]))
            & (out["pred_21d_ret"] >= 8.0)
            & (out["pred_63d_ret"] >= 15.0)
            & (out["institutional_score"] >= 0.08)
            & (out["wyckoff_score"] >= -0.05)
            & (out["qmf_score"] >= -0.25)
            & (out["confidence_boosted"] >= 0.06)
        ]
        # Prefer strong 1M acceleration and positive structure
        out = out.sort_values(
            by=["pred_21d_ret", "institutional_score", "pred_63d_ret", "confidence_boosted"],
            ascending=[False, False, False, False],
        )
        # Fallback if strict runner profile returns empty: keep best relative runners.
        if out.empty:
            out = df[df["market"].isin(["VN", "TW"])].copy()
            out = out[
                (out["pred_21d_ret"] >= 4.0)
                & (out["institutional_score"] >= 0.03)
                & (out["confidence_boosted"] >= 0.03)
            ].sort_values(
                by=["pred_21d_ret", "pred_63d_ret", "institutional_score", "qmf_score"],
                ascending=[False, False, False, False],
            )
        return out
    return df


def render() -> None:
    st.title("Global Alpha Scanner")
    st.markdown("Top ideas for **1-3 month** holding window (Vietnam + Taiwan)")
    st.markdown("---")

    # ── Nightly Pre-Computed Results (fastest path) ────────────────────────────
    nightly_df, meta = _load_nightly()
    if nightly_df is not None and not nightly_df.empty:
        gen_at = meta.get("generated_at_readable", "unknown")
        total = meta.get("total_candidates", len(nightly_df))
        st.success(f"Nightly scan ready — **{total} candidates** | Generated: {gen_at}")

        with st.expander("Top 20 Nightly Picks (pre-computed, full 700-stock universe)", expanded=True):
            show_cols = [
                "symbol", "name", "recommendation", "institutional_score",
                "pred_21d_ret", "pred_63d_ret", "confidence_boosted",
                "wyckoff_phase", "qmf_score", "stoch_state", "veto", "veto_reason",
            ]
            available = [c for c in show_cols if c in nightly_df.columns]
            display = nightly_df[available].copy()

            # Format columns for readability
            for col in ["pred_21d_ret", "pred_63d_ret"]:
                if col in display.columns:
                    display[col] = display[col].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            if "confidence_boosted" in display.columns:
                display["confidence_boosted"] = (display["confidence_boosted"] * 100).round(1).astype(str) + "%"
            if "institutional_score" in display.columns:
                display["institutional_score"] = display["institutional_score"].round(3)
            if "qmf_score" in display.columns:
                display["qmf_score"] = display["qmf_score"].round(2)
            if "veto" in display.columns:
                display["veto"] = display["veto"].map(lambda x: "YES" if x else "NO")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Picks", len(nightly_df))
            if "recommendation" in nightly_df.columns:
                m2.metric("STRONG BUY", int((nightly_df["recommendation"] == "STRONG BUY").sum()))
                m3.metric("BUY", int((nightly_df["recommendation"] == "BUY").sum()))
            if "veto" in nightly_df.columns:
                m4.metric("Veto Active", int(nightly_df["veto"].sum()) if nightly_df["veto"].dtype == bool else 0)

            st.dataframe(display, hide_index=True, use_container_width=True)

            if "recommendation" in nightly_df.columns:
                st.download_button(
                    "Export Nightly Results CSV",
                    nightly_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"nightly_alpha_{meta.get('scan_date', datetime.now().strftime('%Y%m%d'))}.csv",
                    mime="text/csv",
                )
        st.markdown("---")
        st.caption("Run live scan below to refresh with real-time data (slower, needs internet)")
    else:
        st.info("No nightly results yet. Run `python scripts/nightly_full_scan.py --push` locally to generate.")

    st.markdown("---")
    st.subheader("Live Alpha Scan (Real-time)")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
    if "last_scan_time" not in st.session_state:
        st.session_state.last_scan_time = None
    if "scan_both_presets" not in st.session_state:
        st.session_state.scan_both_presets = False

    with st.expander("Scanner Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            extended = st.toggle("Extended VN universe", value=True, key="scanner_extended")
        with c2:
            top_n = st.slider("Top picks", min_value=5, max_value=20, value=10, step=1)
        with c3:
            scope = st.selectbox("Market scope", ["VN_TW", "VN", "TW", "ALL"], index=0)
        preset = st.radio(
            "Preset",
            ["Institutional Balanced", "Parabolic Runner (1-3M)"],
            index=0,
            horizontal=True,
            key="alpha_preset_mode",
            help="Choose filtering mode for alpha picks.",
        )

    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        if st.button("Run Alpha Scan", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0, text="Tier 1: fast scan")

            def update_progress(p: float) -> None:
                if p < 0.33:
                    t = f"Tier 1: RS/structure ({int(p * 100)}%)"
                elif p < 0.66:
                    t = f"Tier 2: Wyckoff pre-filter ({int(p * 100)}%)"
                else:
                    t = f"Tier 3: institutional deep scan ({int(p * 100)}%)"
                progress_bar.progress(min(p, 1.0), text=t)

            engine = AlphaScannerEngine(extended_universe=extended, commodities=False, market_scope=scope)
            results = engine.scan_universe(progress_callback=update_progress)
            st.session_state.scan_results = results
            st.session_state.last_scan_time = datetime.now()
            st.session_state.scan_both_presets = False
            progress_bar.empty()
            st.success(f"Scan complete: {len(results)} candidates")
            st.rerun()
    with b3:
        if st.button("Run Both Presets", use_container_width=True):
            progress_bar = st.progress(0.0, text="Tier 1: fast scan")

            def update_progress(p: float) -> None:
                if p < 0.33:
                    t = f"Tier 1: RS/structure ({int(p * 100)}%)"
                elif p < 0.66:
                    t = f"Tier 2: Wyckoff pre-filter ({int(p * 100)}%)"
                else:
                    t = f"Tier 3: institutional deep scan ({int(p * 100)}%)"
                progress_bar.progress(min(p, 1.0), text=t)

            engine = AlphaScannerEngine(extended_universe=extended, commodities=False, market_scope=scope)
            results = engine.scan_universe(progress_callback=update_progress)
            st.session_state.scan_results = results
            st.session_state.last_scan_time = datetime.now()
            st.session_state.scan_both_presets = True
            progress_bar.empty()
            st.success(f"Scan complete (Both Presets): {len(results)} candidates")
            st.rerun()

    with b2:
        if st.session_state.last_scan_time:
            st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if st.button("Clear", use_container_width=True):
                st.session_state.scan_results = None
                st.session_state.scan_both_presets = False
                st.rerun()

    if st.session_state.scan_results is None:
        st.info("Run scan to get top 1-3 month opportunities.")
        return

    df = pd.DataFrame(st.session_state.scan_results)
    if df.empty:
        st.warning("Scan completed but no candidates passed engine filters. Try scope=ALL or disable strict assumptions.")
        return

    df = df.sort_values("institutional_score", ascending=False).reset_index(drop=True)
    df["Upside 1M"] = df["pred_21d_ret"].map(_fmt_pct)
    df["Upside 3M"] = df["pred_63d_ret"].map(_fmt_pct)
    df["Conf %"] = (df["confidence_boosted"] * 100).round(1).astype(str) + "%"
    df["QMF"] = df["qmf_score"].map(lambda x: f"{x:+.2f}")
    df["Wyckoff"] = df["wyckoff_score"].map(lambda x: f"{x:+.2f}")
    df["Veto"] = df["veto"].map(lambda x: "YES" if x else "NO")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    c2.metric("STRONG BUY", int((df["recommendation"] == "STRONG BUY").sum()))
    c3.metric("BUY", int((df["recommendation"] == "BUY").sum()))
    c4.metric("Veto Active", int(df["veto"].sum()))

    st.markdown("### Tactical Rules")
    st.caption("Ranking ưu tiên theo institutional_score, loại bỏ setup nhiễu: ưu tiên BUY/STRONG BUY và tránh veto.")

    tradable = df[df["market"].isin(["VN", "TW"])].copy()
    total_vn = int((tradable["market"] == "VN").sum())
    total_tw = int((tradable["market"] == "TW").sum())
    tradable = _apply_preset_filter(tradable, preset)

    st.caption(f"Preset active: {preset}")
    st.caption(
        f"Raw candidates: VN={total_vn}, TW={total_tw} | After tactical filter: VN={int((tradable['market']=='VN').sum())}, TW={int((tradable['market']=='TW').sum())}"
    )
    show_cols = [
        "symbol", "name", "market", "recommendation", "institutional_score",
        "Upside 1M", "Upside 3M", "Conf %", "QMF", "Wyckoff", "stoch_state", "Veto", "veto_reason",
    ]
    def _render_regional_tables(data: pd.DataFrame, title: str):
        st.markdown(title)
        t1, t2 = st.tabs(["Vietnam", "Taiwan"])
        with t1:
            vn = data[data["market"] == "VN"].head(top_n)
            st.metric("VN Candidates", len(vn))
            if vn.empty:
                st.warning("No VN candidates passed filter. Showing top raw VN instead.")
                vn_raw = df[df["market"] == "VN"].sort_values("institutional_score", ascending=False).head(top_n)
                st.dataframe(vn_raw[show_cols], hide_index=True, use_container_width=True)
            else:
                st.dataframe(vn[show_cols], hide_index=True, use_container_width=True)
        with t2:
            tw = data[data["market"] == "TW"].head(top_n)
            st.metric("TW Candidates", len(tw))
            if tw.empty:
                st.warning("No TW candidates passed filter. Showing top raw TW instead.")
                tw_raw = df[df["market"] == "TW"].sort_values("institutional_score", ascending=False).head(top_n)
                st.dataframe(tw_raw[show_cols], hide_index=True, use_container_width=True)
            else:
                st.dataframe(tw[show_cols], hide_index=True, use_container_width=True)

    if st.session_state.get("scan_both_presets", False):
        balanced = _apply_preset_filter(df[df["market"].isin(["VN", "TW"])].copy(), "Institutional Balanced")
        runner = _apply_preset_filter(df[df["market"].isin(["VN", "TW"])].copy(), "Parabolic Runner (1-3M)")
        overlap = balanced[balanced["symbol"].isin(set(runner["symbol"]))].copy()
        overlap = overlap.sort_values(
            by=["institutional_score", "pred_63d_ret", "confidence_boosted"],
            ascending=[False, False, False],
        )

        # Keep old behavior: independent VN/TW picks => up to 2 * top_n per preset
        balanced_vn = balanced[balanced["market"] == "VN"].head(top_n)
        balanced_tw = balanced[balanced["market"] == "TW"].head(top_n)
        balanced_2x = pd.concat([balanced_vn, balanced_tw], ignore_index=True)

        runner_vn = runner[runner["market"] == "VN"].head(top_n)
        runner_tw = runner[runner["market"] == "TW"].head(top_n)
        runner_2x = pd.concat([runner_vn, runner_tw], ignore_index=True)

        st.markdown("### Balanced")
        st.caption(f"VN={len(balanced_vn)} | TW={len(balanced_tw)} | Total={len(balanced_2x)}")
        st.dataframe(balanced_2x[show_cols], hide_index=True, use_container_width=True)
        st.markdown("### Runner")
        st.caption(f"VN={len(runner_vn)} | TW={len(runner_tw)} | Total={len(runner_2x)}")
        if runner_2x.empty:
            st.warning("Runner strict filter found no symbols. Showing top relative runners from raw candidates.")
            fallback_runner = df[df["market"].isin(["VN", "TW"])].sort_values(
                by=["pred_21d_ret", "pred_63d_ret", "institutional_score"],
                ascending=[False, False, False],
            )
            fallback_runner_vn = fallback_runner[fallback_runner["market"] == "VN"].head(top_n)
            fallback_runner_tw = fallback_runner[fallback_runner["market"] == "TW"].head(top_n)
            runner_2x = pd.concat([fallback_runner_vn, fallback_runner_tw], ignore_index=True)
        st.dataframe(runner_2x[show_cols], hide_index=True, use_container_width=True)
        st.markdown("### Overlap")
        st.dataframe(overlap.head(top_n)[show_cols], hide_index=True, use_container_width=True)

        cexp1, cexp2, cexp3 = st.columns(3)
        cexp1.download_button("Export Balanced CSV", balanced.to_csv(index=False).encode("utf-8"), file_name=f"alpha_balanced_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        cexp2.download_button("Export Runner CSV", runner.to_csv(index=False).encode("utf-8"), file_name=f"alpha_runner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        cexp3.download_button("Export Overlap CSV", overlap.to_csv(index=False).encode("utf-8"), file_name=f"alpha_overlap_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    else:
        _render_regional_tables(tradable, "### Regional Top 10 (Independent Lists)")

    st.download_button(
        label="Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"alpha_scan_1_3m_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
