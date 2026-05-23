"""
Strategy 3: Bank Participation Report Monitor
==============================================
Tracks bank short concentration (simulated) combined with USD Index trends.

Conforms to ``TradingStrategy`` Protocol.

Signals:
    1. Bank concentration > 50% of total bank shorts → Major bank domination
    2. Combined with USD strength → Double bearish for commodities
    3. Banks covering + weak USD → Reversal BUY
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config import BankConfig
from src.risk_manager import calculate_adx, calculate_atr


class BankParticipationMonitor:
    """Monitor bank-level short concentration + USD index.

    Implements ``TradingStrategy`` Protocol.
    """

    # Concentration bounds for normalization
    _CONC_MIN = 0.15
    _CONC_MAX = 0.75
    _BUY_CONC_THRESHOLD = 0.30

    def __init__(self, config: BankConfig | None = None) -> None:
        cfg = config or BankConfig()
        self.concentration_threshold = cfg.concentration_threshold
        self.usd_strength_window = cfg.usd_strength_window
        self.usd_strength_threshold = cfg.usd_strength_threshold

    def _derive_bank_concentration(self, data: pd.DataFrame) -> pd.DataFrame:
        """Derive bank concentration from real COT-style columns.

        Fail-closed policy:
        - If ``Bank_Short_Concentration`` already exists, use it.
        - Else if ``Commercial_Short`` and ``Open_Interest`` exist, derive deterministic ratio.
        - Else mark data unavailable and keep signals neutral.
        """
        df = data.copy()
        df["bank_data_available"] = False

        if "Bank_Short_Concentration" in df.columns:
            df["Bank_Short_Concentration"] = (
                pd.to_numeric(df["Bank_Short_Concentration"], errors="coerce")
                .clip(lower=self._CONC_MIN, upper=self._CONC_MAX)
            )
            df["bank_data_available"] = df["Bank_Short_Concentration"].notna()
        elif {"Commercial_Short", "Open_Interest"}.issubset(df.columns):
            ratio = (
                pd.to_numeric(df["Commercial_Short"], errors="coerce")
                / pd.to_numeric(df["Open_Interest"], errors="coerce").replace(0, pd.NA)
            )
            # Smooth raw ratio to reduce weekly-step noise after forward-fill
            ratio = ratio.rolling(window=3, min_periods=1).mean()
            df["Bank_Short_Concentration"] = ratio.clip(self._CONC_MIN, self._CONC_MAX)
            df["bank_data_available"] = df["Bank_Short_Concentration"].notna()
        else:
            df["Bank_Short_Concentration"] = pd.NA
            logger.warning(
                "BankParticipation: missing real concentration fields. "
                "Fail-closed mode will keep bank signals neutral."
            )

        df["Bank_Concentration_Alert"] = (
            df["Bank_Short_Concentration"] > self.concentration_threshold
        )
        return df

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Helper to detect Trending vs Ranging regime."""
        if df.empty or len(df) < 20:
            return "Ranging"
            
        adx_series = calculate_adx(df, period=14)
        adx = adx_series.iloc[-1]
        atr = calculate_atr(df, period=14)
        atr_ma = atr.rolling(window=50, min_periods=10).mean()
        
        atr_expanding = False
        if len(atr) >= 1 and len(atr_ma) >= 1:
            atr_expanding = atr.iloc[-1] > atr_ma.iloc[-1]
        
        if adx > 20 and atr_expanding:
            return "Trending"
        return "Ranging"

    def generate_signals(
        self,
        data: pd.DataFrame,
        *,
        usd_data: pd.DataFrame | None = None,
        fail_closed: bool = True,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Generate signals from bank participation + USD trends + regime detection.

        Appends columns:
            ``bank_signal`` (-1/0/1), ``bank_signal_reason``,
            ``Bank_Short_Concentration``, ``USD_Trend``, ``market_regime``.
        """
        df = self._derive_bank_concentration(data)

        # Market regime detection (ADX + ATR)
        df["ADX_bank"] = calculate_adx(df, period=14)
        atr = calculate_atr(df, period=14)
        atr_ma = atr.rolling(window=50, min_periods=10).mean()
        atr_expanding = atr > atr_ma

        # Regime classification
        df["market_regime"] = "RANGING"
        trending_mask = (df["ADX_bank"] > 20) & atr_expanding
        df.loc[trending_mask, "market_regime"] = "TRENDING"

        # USD Index integration
        df["USD_Trend"] = 0.0
        df["USD_Strong"] = False

        if usd_data is not None and not usd_data.empty:
            usd = usd_data[["Date", "Close"]].rename(columns={"Close": "USD_Close"})
            df = pd.merge_asof(
                df.sort_values("Date"),
                usd.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            df["USD_Trend"] = df["USD_Close"].pct_change(self.usd_strength_window)
            df["USD_Strong"] = df["USD_Trend"] > self.usd_strength_threshold
        else:
            # Simulate from inverse price correlation
            df["USD_Trend"] = (
                -df["Close"].pct_change(self.usd_strength_window).fillna(0) * 0.5
            )
            df["USD_Strong"] = df["USD_Trend"] > self.usd_strength_threshold

        # Combined signal
        df["bank_signal"] = 0
        df["bank_signal_reason"] = ""

        # If concentration is unavailable, stay neutral in fail-closed mode
        unavailable = ~df["bank_data_available"].fillna(False)
        if fail_closed:
            df.loc[unavailable, "bank_signal"] = 0
            df.loc[unavailable, "bank_signal_reason"] = "NO_REAL_BANK_FLOW_DATA"

        # STRONG SELL: High concentration + USD strengthening (trending market only)
        strong_sell = df["Bank_Concentration_Alert"] & df["USD_Strong"]
        df.loc[strong_sell, "bank_signal"] = -1
        df.loc[strong_sell, "bank_signal_reason"] = (
            "BANK_DOMINANCE: Top bank >50% shorts + USD strengthening"
        )

        # MODERATE SELL: High concentration alone
        moderate_sell = df["Bank_Concentration_Alert"] & ~df["USD_Strong"]
        df.loc[moderate_sell, "bank_signal"] = -1
        df.loc[moderate_sell, "bank_signal_reason"] = (
            "BANK_WARNING: Top bank concentration high"
        )

        # Regime filter: suppress sell signals in RANGING markets (too noisy)
        ranging_mask = df["market_regime"] == "RANGING"
        ranging_sells = ranging_mask & (df["bank_signal"] == -1) & ~strong_sell
        df.loc[ranging_sells, "bank_signal"] = 0
        df.loc[ranging_sells, "bank_signal_reason"] = ""

        # BUY: Banks covering + weak USD
        buy_mask = (
            (df["Bank_Short_Concentration"] < self._BUY_CONC_THRESHOLD)
            & (df["USD_Trend"] < -self.usd_strength_threshold)
        )
        df.loc[buy_mask, "bank_signal"] = 1
        df.loc[buy_mask, "bank_signal_reason"] = (
            "BANK_COVERING: Banks reducing shorts + weak USD"
        )

        n_sell = (df["bank_signal"] == -1).sum()
        n_buy = (df["bank_signal"] == 1).sum()
        n_unavail = unavailable.sum()
        n_trending = trending_mask.sum()
        n_ranging = ranging_mask.sum()
        logger.info(
            f"Bank Participation: {n_sell} SELL, {n_buy} BUY signals. "
            f"Regime: {n_trending} trending / {n_ranging} ranging days. "
            f"Unavailable bank-flow rows: {n_unavail}"
        )
        return df
