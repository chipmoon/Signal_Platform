"""
Seasonality Filter
==================
Encodes historical seasonal patterns for different asset classes and markets.

Why seasonality matters:
    - VN stocks: Jan-Apr strong (post-Tet effect), Aug-Sep weak (summer lull)
    - Gold: Aug-Sep strong (Indian festival demand), Nov-Dec strong (year-end)
    - Oil: Q1 weak (post-winter), Q2-Q3 seasonal demand peaks
    - US stocks: May-Oct mixed ("Sell in May"), Nov-Apr strong (Santa Rally effect)
    - Taiwan semis: Q4 strong (iPhone cycle), Q2 weak (inventory correction)

Output:
    seasonality_score   — float -1.0 to +1.0 (bearish to bullish seasonal bias)
    seasonality_label   — str description of current seasonal window
    seasonality_month   — int month number
    seasonality_weight  — float recommended position size multiplier (0.5x-1.5x)

Integration:
    - Used as a pre-filter: if score < -0.5 → reduce all position sizes by 50%
    - Fed into AIPredictor._prepare_features() as scalar feature
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Seasonality Calendars
# (Derived from historical backtests and academic seasonal studies)
# Score range: -1.0 (strongly bearish seasonal) to +1.0 (strongly bullish)
# ─────────────────────────────────────────────────────────────────────────────

_GOLD_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  ( 0.2, "Jan: Mildly bullish — post-holiday demand, USD weakness"),
    2:  ( 0.3, "Feb: Bullish — Chinese New Year jewelry demand peak"),
    3:  ( 0.1, "Mar: Neutral — seasonal consolidation"),
    4:  ( 0.0, "Apr: Neutral — spring lull"),
    5:  (-0.2, "May: Mildly bearish — pre-summer softening"),
    6:  (-0.1, "Jun: Neutral — quiet summer"),
    7:  ( 0.1, "Jul: Neutral — early positioning"),
    8:  ( 0.5, "Aug: STRONG BULLISH — Indian wedding season starts, Eid purchases"),
    9:  ( 0.6, "Sep: PEAK BULLISH — Indian festival demand (Navratri/Dussehra)"),
    10: ( 0.3, "Oct: Bullish — Diwali demand, year-end positioning"),
    11: ( 0.4, "Nov: Bullish — USD weakness seasonal, hedge fund rebalancing"),
    12: ( 0.2, "Dec: Mildly bullish — year-end safe haven, tax-loss harvesting done"),
}

_SILVER_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  ( 0.1, "Jan: Neutral"),
    2:  ( 0.3, "Feb: Bullish — follows gold + industrial demand"),
    3:  ( 0.2, "Mar: Mildly bullish — solar panel procurement season"),
    4:  ( 0.1, "Apr: Neutral"),
    5:  (-0.3, "May: Bearish — industrial slowdown seasonal"),
    6:  (-0.2, "Jun: Mildly bearish"),
    7:  ( 0.0, "Jul: Neutral"),
    8:  ( 0.4, "Aug: Bullish — follows gold festival season"),
    9:  ( 0.5, "Sep: Strong bullish — peak industrial + festival"),
    10: ( 0.2, "Oct: Bullish"),
    11: ( 0.3, "Nov: Bullish"),
    12: ( 0.1, "Dec: Neutral"),
}

_CRUDE_OIL_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  (-0.1, "Jan: Neutral — post-winter inventory build"),
    2:  ( 0.1, "Feb: Neutral"),
    3:  ( 0.3, "Mar: Bullish — driving season preparation, refinery maintenance ends"),
    4:  ( 0.4, "Apr: Bullish — US driving season starts"),
    5:  ( 0.5, "May: STRONG BULLISH — peak driving season"),
    6:  ( 0.4, "Jun: Bullish — summer demand peak"),
    7:  ( 0.3, "Jul: Mildly bullish — Atlantic hurricane season risk premium"),
    8:  ( 0.1, "Aug: Neutral — late summer demand"),
    9:  (-0.2, "Sep: Mildly bearish — end of driving season"),
    10: (-0.3, "Oct: Bearish — demand slowdown, inventory rebuild"),
    11: (-0.2, "Nov: Mildly bearish — winter heating oil rotation"),
    12: ( 0.0, "Dec: Neutral — year-end"),
}

_VN_STOCK_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  ( 0.6, "Jan: STRONG BULLISH — Pre-Tet window, institutional year-end buying"),
    2:  ( 0.5, "Feb: Bullish — Post-Tet liquidity return, foreign inflows"),
    3:  ( 0.4, "Mar: Bullish — Q1 earnings season optimism"),
    4:  ( 0.2, "Apr: Mildly bullish — dividend season rally"),
    5:  ( 0.0, "May: Neutral — Sell in May effect starts"),
    6:  (-0.1, "Jun: Neutral to weak — mid-year rebalancing"),
    7:  (-0.2, "Jul: Mildly bearish — summer lull, low volume"),
    8:  (-0.5, "Aug: BEARISH — weakest month historically, foreign selling"),
    9:  (-0.4, "Sep: Bearish — global risk-off, post-summer weakness"),
    10: ( 0.1, "Oct: Neutral — recovery positioning"),
    11: ( 0.3, "Nov: Bullish — year-end rally begins"),
    12: ( 0.4, "Dec: Bullish — Tet preparation, institutional window dressing"),
}

_TW_STOCK_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  ( 0.3, "Jan: Bullish — iPhone supply chain restocking"),
    2:  ( 0.2, "Feb: Mildly bullish — post-CNY return"),
    3:  ( 0.1, "Mar: Neutral — inventory assessment"),
    4:  (-0.1, "Apr: Neutral to weak — earnings caution"),
    5:  (-0.3, "May: Bearish — inventory correction, tech downcycle risk"),
    6:  (-0.2, "Jun: Mildly bearish — mid-year adjustment"),
    7:  ( 0.1, "Jul: Neutral — back-to-school orders starting"),
    8:  ( 0.3, "Aug: Bullish — iPhone production ramp-up"),
    9:  ( 0.5, "Sep: STRONG BULLISH — iPhone launch month, peak orders"),
    10: ( 0.4, "Oct: Bullish — Q4 electronics season"),
    11: ( 0.2, "Nov: Mildly bullish — holiday shipments"),
    12: ( 0.0, "Dec: Neutral — order book slowing"),
}

_US_STOCK_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  ( 0.3, "Jan: Bullish — January effect, new money inflows"),
    2:  ( 0.2, "Feb: Mildly bullish"),
    3:  ( 0.1, "Mar: Neutral — end of Q1 rebalancing"),
    4:  ( 0.2, "Apr: Bullish — Q1 earnings season"),
    5:  (-0.1, "May: Neutral — 'Sell in May' caution"),
    6:  (-0.1, "Jun: Neutral — summer doldrums"),
    7:  ( 0.1, "Jul: Mildly bullish — summer rally"),
    8:  (-0.2, "Aug: Mildly bearish — lowest volume month"),
    9:  (-0.4, "Sep: BEARISH — historically worst month for S&P500"),
    10: ( 0.0, "Oct: Neutral — 'Witching' volatility, can reverse"),
    11: ( 0.4, "Nov: BULLISH — Santa rally starts, Thanksgiving"),
    12: ( 0.5, "Dec: STRONG BULLISH — Santa Claus rally, window dressing"),
}

_CRYPTO_SEASONALITY: dict[int, tuple[float, str]] = {
    1:  ( 0.4, "Jan: Bullish — new year inflows, altcoin season"),
    2:  ( 0.2, "Feb: Mildly bullish"),
    3:  ( 0.1, "Mar: Neutral"),
    4:  ( 0.3, "Apr: Bullish — historically strong (halving cycle effects)"),
    5:  (-0.1, "May: Neutral"),
    6:  (-0.3, "Jun: Bearish — summer crypto lull"),
    7:  ( 0.2, "Jul: Mildly bullish — summer rebound"),
    8:  ( 0.1, "Aug: Neutral"),
    9:  ( 0.0, "Sep: Neutral — 'Rektember' caution"),
    10: ( 0.4, "Oct: Bullish — 'Uptober', historically strong"),
    11: ( 0.5, "Nov: STRONG BULLISH — bull run season"),
    12: ( 0.3, "Dec: Bullish — year-end positioning"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Market Detection Logic
# ─────────────────────────────────────────────────────────────────────────────

def _detect_asset_type(symbol: str, market: str) -> str:
    """Map (symbol, market) to a seasonality key."""
    symbol_upper = symbol.upper()

    # Commodities
    if any(x in symbol_upper for x in ["GC", "GOLD", "XAU"]):
        return "GOLD"
    if any(x in symbol_upper for x in ["SI", "SILVER", "XAG"]):
        return "SILVER"
    if any(x in symbol_upper for x in ["CL", "OIL", "CRUDE", "WTI", "BRENT"]):
        return "CRUDE_OIL"

    # Crypto
    if any(x in symbol_upper for x in ["BTC", "ETH", "SOL", "BNB", "CRYPTO", "-USD"]):
        return "CRYPTO"

    # Markets
    if market == "VN" or symbol_upper.endswith(".VN"):
        return "VN"
    if market == "TW" or symbol_upper.endswith(".TW"):
        return "TW"
    if market in ("US", "NASDAQ", "NYSE") or any(x in symbol_upper for x in ["AAPL", "MSFT", "SPY", "QQQ", "NVDA", "META", "TSLA", "GOOGL", "AMZN"]):
        return "US"

    return "US"  # Default fallback


_SEASONALITY_MAP = {
    "GOLD":      _GOLD_SEASONALITY,
    "SILVER":    _SILVER_SEASONALITY,
    "CRUDE_OIL": _CRUDE_OIL_SEASONALITY,
    "VN":        _VN_STOCK_SEASONALITY,
    "TW":        _TW_STOCK_SEASONALITY,
    "US":        _US_STOCK_SEASONALITY,
    "CRYPTO":    _CRYPTO_SEASONALITY,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main: SeasonalityFilter
# ─────────────────────────────────────────────────────────────────────────────

class SeasonalityFilter:
    """
    Seasonal Context Filter for any asset class.

    Usage:
        sf = SeasonalityFilter()
        state = sf.get_current_season("GC=F", "COMMODITY")
        # Returns: score, label, weight multiplier

    Integration with AI:
        df["seasonality_score"] = sf.get_score_for_df(df, symbol, market)
        # Adds scalar feature for each bar's historical month seasonality
    """

    def get_current_season(
        self,
        symbol: str,
        market: str,
        reference_date: datetime | None = None,
    ) -> dict:
        """
        Return current seasonality state for a symbol.

        Args:
            symbol: Asset ticker (e.g. 'GC=F', 'VCB.VN')
            market: Market code ('VN', 'US', 'COMMODITY', 'TW', 'CRYPTO')
            reference_date: Date to check (defaults to today)

        Returns:
            dict with: score, label, month, weight, asset_type, next_strong_month
        """
        if reference_date is None:
            reference_date = datetime.now()

        month = reference_date.month
        asset_type = _detect_asset_type(symbol, market)
        calendar = _SEASONALITY_MAP.get(asset_type, _US_STOCK_SEASONALITY)

        score, label = calendar.get(month, (0.0, "Unknown seasonal window"))

        # Position size multiplier: max 1.5x in peak season, min 0.5x in adverse
        weight = 1.0 + score * 0.5  # score=1.0 → 1.5x, score=-1.0 → 0.5x
        weight = round(max(0.5, min(1.5, weight)), 2)

        # Find next strong month (score >= 0.4)
        next_strong = None
        for offset in range(1, 13):
            next_m = ((month - 1 + offset) % 12) + 1
            next_score, _ = calendar.get(next_m, (0.0, ""))
            if next_score >= 0.4:
                next_strong = {
                    "month": next_m,
                    "months_away": offset,
                    "score": next_score,
                }
                break

        # Adverse season flag
        adverse = score <= -0.3

        logger.debug(
            f"Seasonality [{asset_type}] Month={month}: "
            f"score={score:+.1f}, weight={weight}x, adverse={adverse}"
        )

        return {
            "score": round(float(score), 2),
            "label": label,
            "month": month,
            "asset_type": asset_type,
            "weight": weight,
            "adverse": adverse,
            "next_strong_month": next_strong,
            "calendar": {m: v[0] for m, v in calendar.items()},
        }

    def get_score_series(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: str,
    ) -> pd.Series:
        """
        Return a Series of seasonality scores per row in df, based on date.

        Useful for adding as an AI feature column.
        """
        if "Date" not in df.columns:
            return pd.Series(0.0, index=df.index)

        asset_type = _detect_asset_type(symbol, market)
        calendar = _SEASONALITY_MAP.get(asset_type, _US_STOCK_SEASONALITY)

        dates = pd.to_datetime(df["Date"])
        months = dates.dt.month
        scores = months.map(lambda m: calendar.get(m, (0.0, ""))[0])
        return scores.fillna(0.0)

    def add_seasonality_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: str,
    ) -> pd.DataFrame:
        """
        Inject seasonality features into a DataFrame.

        Adds:
            seasonality_score     — float per bar
            seasonality_weight    — float position size multiplier per bar
            seasonality_adverse   — bool: avoid trading this month
        """
        result = df.copy()
        scores = self.get_score_series(result, symbol, market)
        result["seasonality_score"] = scores
        result["seasonality_weight"] = (1.0 + scores * 0.5).clip(0.5, 1.5)
        result["seasonality_adverse"] = scores <= -0.3
        return result
