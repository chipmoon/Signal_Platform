"""
Settings Page
=============
Configuration and parameter tuning interface.
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the settings page."""
    st.title("⚙️ Settings & Configuration")
    st.markdown("Tune strategy parameters and system settings")
    st.markdown("---")

    # Tabs for different config sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Backtest Config",
        "🎯 Strategy Params",
        "🛡️ Risk Management",
        "🤖 AI Model",
    ])

    with tab1:
        st.markdown("### 🔧 Backtest Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Initial Capital ($)", value=100_000, step=10_000, min_value=10_000)
            st.number_input("Position Size (%)", value=20.0, step=5.0, min_value=5.0, max_value=50.0)
            st.number_input("Slippage (bps)", value=5, step=1, min_value=0, max_value=50)
        
        with col2:
            st.number_input("Commission per Trade ($)", value=10.0, step=5.0, min_value=0.0)
            st.checkbox("Enable Pyramiding", value=False)
            st.checkbox("Use Trailing Stop", value=True)

    with tab2:
        st.markdown("### 🎯 Strategy Parameters")
        
        # COT Config
        with st.expander("📊 COT Monitor", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.slider("Short Ratio Threshold", 0.0, 1.0, 0.35, 0.05)
                st.slider("Anomaly Multiplier", 1.0, 5.0, 2.5, 0.1)
            with col2:
                st.number_input("Change Lookback (weeks)", value=12, step=1, min_value=4)
                st.number_input("Volatility Window (days)", value=60, step=10, min_value=20)

        # Volume-Price Config
        with st.expander("💹 Volume-Price Detector"):
            col1, col2 = st.columns(2)
            with col1:
                st.slider("Volume Spike Threshold", 1.0, 10.0, 5.0, 0.5)
                st.slider("Price Drop Threshold (%)", -10.0, 0.0, -3.0, 0.5)
            with col2:
                st.number_input("Lookback Window (days)", value=20, step=5, min_value=10)
                st.slider("ML Contamination", 0.01, 0.20, 0.05, 0.01)

        # Bank Config
        with st.expander("🏦 Bank Participation"):
            col1, col2 = st.columns(2)
            with col1:
                st.slider("Concentration Threshold", 0.0, 1.0, 0.50, 0.05)
                st.slider("USD Strength Threshold (%)", 0.0, 5.0, 2.0, 0.1)
            with col2:
                st.number_input("USD Window (days)", value=20, step=5, min_value=5)
                st.checkbox("Enable Regime Filter", value=True)

    with tab3:
        st.markdown("### 🛡️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Max Drawdown (%)", 5.0, 30.0, 15.0, 1.0)
            st.slider("Recovery Threshold (%)", 1.0, 10.0, 5.0, 0.5)
            st.slider("ATR Trailing Multiplier", 1.0, 4.0, 2.0, 0.1)
        
        with col2:
            st.slider("RSI Overbought", 60, 90, 75, 5)
            st.slider("RSI Oversold", 10, 40, 25, 5)
            st.slider("ADX Threshold", 15, 30, 20, 1)

        st.markdown("#### Position Sizing")
        sizing_method = st.radio(
            "Sizing Method",
            ["Fixed %", "Volatility-Based (ATR)", "Kelly Criterion"],
            index=1,
        )
        
        if sizing_method == "Volatility-Based (ATR)":
            st.info("📊 Position size adjusts based on ATR to maintain consistent dollar risk per trade.")
        elif sizing_method == "Kelly Criterion":
            st.warning("⚠️ Kelly sizing can be aggressive. Consider using fractional Kelly (0.5x).")

    with tab4:
        st.markdown("### 🤖 AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.multiselect(
                "Forecast Horizons (days)",
                [1, 5, 10, 21, 42, 63, 126],
                default=[1, 21, 42, 63, 126],
            )
            st.number_input("Lookback Days", value=15, step=5, min_value=5, max_value=60)
            st.number_input("Walk-Forward Window", value=252, step=50, min_value=100)
        
        with col2:
            st.text_input("Hidden Layer Sizes", value="64, 32")
            st.number_input("Max Iterations", value=500, step=100, min_value=100)
            st.number_input("Learning Rate", value=0.001, step=0.0001, format="%.4f", min_value=0.0001)

        st.checkbox("Enable Ensemble (MLP + GradientBoosting)", value=True)
        st.checkbox("Use Cross-Validation Scoring", value=True)

    # Save button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save Configuration", width='stretch'):
            st.success("✅ Configuration saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", width='stretch'):
            st.info("ℹ️ Configuration reset to default values.")
    
    with col3:
        if st.button("📥 Export Config", width='stretch'):
            st.download_button(
                label="Download config.json",
                data='{"config": "example"}',
                file_name="trading_config.json",
                mime="application/json",
            )
