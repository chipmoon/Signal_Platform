"""
US Stock Provider Plugin
========================
Popular US stocks, ETFs, and indices via Yahoo Finance.
Covers S&P 500 top components, major ETFs, and indices.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from .base import AssetInfo, AssetProvider

# ─── Popular US Stocks & ETFs ───────────────────────────────────────
US_ASSETS = {
    # ── Major Indices ─────────────────────────────────────────────
    "^GSPC":  {"name": "S&P 500 Index",          "sector": "Index"},
    "^DJI":   {"name": "Dow Jones Industrial",    "sector": "Index"},
    "^IXIC":  {"name": "NASDAQ Composite",        "sector": "Index"},
    "^RUT":   {"name": "Russell 2000",            "sector": "Index"},
    "^VIX":   {"name": "CBOE Volatility Index",   "sector": "Volatility"},

    # ── Magnificent 7 ─────────────────────────────────────────────
    "AAPL":   {"name": "Apple Inc.",              "sector": "Technology"},
    "MSFT":   {"name": "Microsoft Corp.",         "sector": "Technology"},
    "GOOGL":  {"name": "Alphabet (Google)",       "sector": "Technology"},
    "AMZN":   {"name": "Amazon.com Inc.",         "sector": "Consumer Cyclical"},
    "NVDA":   {"name": "NVIDIA Corp.",            "sector": "Technology"},
    "META":   {"name": "Meta Platforms",          "sector": "Technology"},
    "TSLA":   {"name": "Tesla Inc.",              "sector": "Consumer Cyclical"},

    # ── Top S&P 500 ───────────────────────────────────────────────
    "BRK-B":  {"name": "Berkshire Hathaway B",    "sector": "Financial Services"},
    "JPM":    {"name": "JPMorgan Chase",          "sector": "Financial Services"},
    "V":      {"name": "Visa Inc.",               "sector": "Financial Services"},
    "JNJ":    {"name": "Johnson & Johnson",       "sector": "Healthcare"},
    "UNH":    {"name": "UnitedHealth Group",      "sector": "Healthcare"},
    "XOM":    {"name": "Exxon Mobil",             "sector": "Energy"},
    "PG":     {"name": "Procter & Gamble",        "sector": "Consumer Defensive"},
    "MA":     {"name": "Mastercard Inc.",          "sector": "Financial Services"},
    "HD":     {"name": "Home Depot Inc.",          "sector": "Consumer Cyclical"},
    "ABBV":   {"name": "AbbVie Inc.",             "sector": "Healthcare"},
    "KO":     {"name": "Coca-Cola Company",       "sector": "Consumer Defensive"},
    "PEP":    {"name": "PepsiCo Inc.",            "sector": "Consumer Defensive"},
    "MRK":    {"name": "Merck & Co.",             "sector": "Healthcare"},
    "AVGO":   {"name": "Broadcom Inc.",           "sector": "Technology"},
    "LLY":    {"name": "Eli Lilly & Co.",         "sector": "Healthcare"},
    "COST":   {"name": "Costco Wholesale",        "sector": "Consumer Defensive"},
    "WMT":    {"name": "Walmart Inc.",            "sector": "Consumer Defensive"},
    "MCD":    {"name": "McDonald's Corp.",        "sector": "Consumer Cyclical"},
    "DIS":    {"name": "Walt Disney Co.",         "sector": "Communication Services"},
    "ADBE":   {"name": "Adobe Inc.",              "sector": "Technology"},
    "CRM":    {"name": "Salesforce Inc.",         "sector": "Technology"},
    "AMD":    {"name": "Advanced Micro Devices",  "sector": "Technology"},
    "INTC":   {"name": "Intel Corp.",             "sector": "Technology"},
    "NFLX":   {"name": "Netflix Inc.",            "sector": "Communication Services"},
    "CSCO":   {"name": "Cisco Systems",           "sector": "Technology"},
    "ORCL":   {"name": "Oracle Corp.",            "sector": "Technology"},
    "BA":     {"name": "Boeing Company",          "sector": "Industrials"},
    "CAT":    {"name": "Caterpillar Inc.",        "sector": "Industrials"},
    "GS":     {"name": "Goldman Sachs",           "sector": "Financial Services"},
    "PYPL":   {"name": "PayPal Holdings",         "sector": "Financial Services"},
    "SQ":     {"name": "Block Inc. (Square)",     "sector": "Financial Services"},
    "SHOP":   {"name": "Shopify Inc.",            "sector": "Technology"},
    "PLTR":   {"name": "Palantir Technologies",   "sector": "Technology"},
    "SNOW":   {"name": "Snowflake Inc.",          "sector": "Technology"},
    "COIN":   {"name": "Coinbase Global",         "sector": "Financial Services"},
    "UBER":   {"name": "Uber Technologies",       "sector": "Technology"},
    "ABNB":   {"name": "Airbnb Inc.",             "sector": "Consumer Cyclical"},
    "RIVN":   {"name": "Rivian Automotive",       "sector": "Consumer Cyclical"},
    "NIO":    {"name": "NIO Inc.",                "sector": "Consumer Cyclical"},
    "BABA":   {"name": "Alibaba Group",           "sector": "Consumer Cyclical"},
    "TSM":    {"name": "Taiwan Semiconductor (ADR)", "sector": "Technology"},
    "MRVL":   {"name": "Marvell Technology",      "sector": "Technology"},

    # ── Popular ETFs ──────────────────────────────────────────────
    "SPY":    {"name": "SPDR S&P 500 ETF",        "sector": "ETF - Index"},
    "QQQ":    {"name": "Invesco QQQ (NASDAQ 100)","sector": "ETF - Index"},
    "DIA":    {"name": "SPDR Dow Jones ETF",      "sector": "ETF - Index"},
    "IWM":    {"name": "iShares Russell 2000 ETF","sector": "ETF - Index"},
    "GLD":    {"name": "SPDR Gold Shares ETF",    "sector": "ETF - Commodity"},
    "SLV":    {"name": "iShares Silver Trust ETF","sector": "ETF - Commodity"},
    "USO":    {"name": "United States Oil Fund",  "sector": "ETF - Commodity"},
    "TLT":    {"name": "iShares 20+ Year Treasury","sector": "ETF - Bond"},
    "VTI":    {"name": "Vanguard Total Stock Market","sector": "ETF - Index"},
    "VOO":    {"name": "Vanguard S&P 500 ETF",   "sector": "ETF - Index"},
    "ARKK":   {"name": "ARK Innovation ETF",     "sector": "ETF - Thematic"},
    "XLF":    {"name": "Financial Select SPDR",   "sector": "ETF - Sector"},
    "XLE":    {"name": "Energy Select SPDR",      "sector": "ETF - Sector"},
    "XLK":    {"name": "Technology Select SPDR",  "sector": "ETF - Sector"},
    "SOXX":   {"name": "iShares Semiconductor ETF","sector": "ETF - Sector"},
    "VNQ":    {"name": "Vanguard Real Estate ETF","sector": "ETF - REIT"},
    "EEM":    {"name": "iShares Emerging Markets","sector": "ETF - International"},
    "EFA":    {"name": "iShares EAFE",            "sector": "ETF - International"},

    # ── Crypto-related Stocks ─────────────────────────────────────
    "MSTR":   {"name": "MicroStrategy Inc.",      "sector": "Technology/Crypto"},
    "MARA":   {"name": "Marathon Digital",        "sector": "Technology/Crypto"},
    "RIOT":   {"name": "Riot Platforms",          "sector": "Technology/Crypto"},

    # ── Top 30 Crypto (Institutional & High Volume) ───────────────
    "BTC-USD":  {"name": "Bitcoin",               "sector": "Crypto - L1"},
    "ETH-USD":  {"name": "Ethereum",              "sector": "Crypto - L1"},
    "SOL-USD":  {"name": "Solana",                "sector": "Crypto - L1"},
    "BNB-USD":  {"name": "Binance Coin",          "sector": "Crypto - L1"},
    "XRP-USD":  {"name": "XRP",                   "sector": "Crypto - Payment"},
    "ADA-USD":  {"name": "Cardano",               "sector": "Crypto - L1"},
    "AVAX-USD": {"name": "Avalanche",             "sector": "Crypto - L1"},
    "DOGE-USD": {"name": "Dogecoin",              "sector": "Crypto - Meme"},
    "DOT-USD":  {"name": "Polkadot",              "sector": "Crypto - L1"},
    "TRX-USD":  {"name": "Tron",                  "sector": "Crypto - L1"},
    "LINK-USD": {"name": "Chainlink",             "sector": "Crypto - Oracle"},
    "MATIC-USD":{"name": "Polygon",               "sector": "Crypto - L2"},
    "SHIB-USD": {"name": "Shiba Inu",             "sector": "Crypto - Meme"},
    "LTC-USD":  {"name": "Litecoin",              "sector": "Crypto - L1"},
    "BCH-USD":  {"name": "Bitcoin Cash",          "sector": "Crypto - L1"},
    "ATOM-USD": {"name": "Cosmos",                "sector": "Crypto - L1"},
    "UNI-USD":  {"name": "Uniswap",               "sector": "Crypto - DEX"},
    "ETC-USD":  {"name": "Ethereum Classic",      "sector": "Crypto - L1"},
    "XLM-USD":  {"name": "Stellar",               "sector": "Crypto - Payment"},
    "IMX-USD":  {"name": "Immutable X",           "sector": "Crypto - L2/Gaming"},
    "NEAR-USD": {"name": "Near Protocol",         "sector": "Crypto - L1"},
    "APT-USD":  {"name": "Aptos",                 "sector": "Crypto - L1"},
    "OP-USD":   {"name": "Optimism",              "sector": "Crypto - L2"},
    "TIA-USD":  {"name": "Celestia",              "sector": "Crypto - L1"},
    "KAS-USD":  {"name": "Kaspa",                 "sector": "Crypto - L1"},
    "RNDR-USD": {"name": "Render Token",          "sector": "Crypto - AI/GPU"},
    "STX-USD":  {"name": "Stacks",                "sector": "Crypto - L2"},
    "INJ-USD":  {"name": "Injective",             "sector": "Crypto - DeFi L1"},
    "FIL-USD":  {"name": "Filecoin",              "sector": "Crypto - Storage"},
    "LDO-USD":  {"name": "Lido DAO",              "sector": "Crypto - LSD"},
}


# ─── Noisy US Tickers to Exclude ────────────────────────────────────
# These often clash with popular international tickers (VN / TW)
US_BLACKLIST = {
    # VN stocks that exist as obscure US tickers on yfinance
    "BSR", "HAG", "GAS", "OIL", "PVS", "PVD", "POW", "DCM", "DPM",
    "SSI", "STB", "HPG", "PLX", "BID", "CTG", "ACB", "TCB", "MBB",
    "PAN", "VNM", "FPT", "VCB", "VHM", "VIC", "MSN", "MWG", "REE",
    "VRE", "SAB", "GEX", "KDH", "NLG", "PDR", "DIG", "IJC", "KBC",
    "PHR", "SBT", "HSG", "NKG", "TLG", "GMD", "VTP", "VOS", "PPC",
    "HDB", "LPB", "VIB", "TPB", "SHB", "EIB", "OCB", "MSB", "KLB",
    "VND", "HCM", "VCI", "SHS", "AGR", "BSC", "CTS", "FTS", "TVS",
}


class USStockProvider(AssetProvider):
    """Provider for US Stocks, ETFs, Indices, and Crypto via Yahoo Finance."""

    @property
    def market_id(self) -> str:
        return "US"

    @property
    def market_name(self) -> str:
        return "US Stocks & ETFs"

    def search_assets(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """Search US stocks by symbol, name, or sector.
        
        For short queries (<=4 chars), only match on symbol prefix to avoid
        false positives like 'PAN' matching 'Company' in stock names.
        """
        if not query:
            # Return top stocks by default
            top = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ", "TSLA", "META", "BTC-USD"]
            return [self._to_asset_info(s) for s in top[:limit]]

        query_upper = query.upper().strip()
        query_lower = query.lower().strip()
        
        # Skip blacklisted symbols entirely
        if query_upper in US_BLACKLIST:
            return []
        
        results = []
        is_short_query = len(query_lower) <= 3

        for symbol, info in US_ASSETS.items():
            # Exact symbol match always works
            if query_upper == symbol.upper():
                results.append(self._to_asset_info(symbol))
                continue
            
            if is_short_query:
                # Short queries: only match symbol prefix (not name substring)
                # This prevents 'PAN' from matching 'Coca-Cola Company'
                if symbol.lower().startswith(query_lower):
                    results.append(self._to_asset_info(symbol))
            else:
                # Longer queries: full substring search on symbol + name + sector
                if (
                    query_lower in symbol.lower()
                    or query_lower in info["name"].lower()
                    or query_lower in info["sector"].lower()
                ):
                    results.append(self._to_asset_info(symbol))

        return results[:limit]

    def _to_asset_info(self, symbol: str) -> AssetInfo:
        """Convert symbol to AssetInfo."""
        info = US_ASSETS.get(symbol, {"name": symbol, "sector": "Unknown"})
        
        # Determine exchange
        if symbol.endswith("-USD"):
            exchange = "Crypto"
        elif symbol.startswith("^"):
            exchange = "Index"
        else:
            exchange = "NYSE/NASDAQ"

        return AssetInfo(
            symbol=symbol,
            name=info["name"],
            market=self.market_id,
            sector=info["sector"],
            exchange=exchange,
            currency="USD",
        )

    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Get asset info by symbol.
        
        Only returns info for symbols explicitly in US_ASSETS or confirmed
        US equities via yfinance. Blacklisted symbols (common VN/TW tickers)
        are skipped to prevent false matches.
        """
        symbol = symbol.upper().strip()

        if symbol in US_ASSETS:
            return self._to_asset_info(symbol)

        if symbol in US_BLACKLIST:
            return None

        # Ignore purely numeric symbols (likely TW/HK/VN/etc.)
        if symbol.isdigit():
            return None
        
        # Ignore symbols with market suffixes (e.g., PAN.VN, 2330.TW)
        if '.' in symbol:
            return None

        # For short uppercase-only symbols (2-4 chars), skip yfinance fallback
        # These are very likely Asian market tickers, not US stocks.
        # Only do yfinance lookup for known US patterns or longer queries.
        if len(symbol) <= 4 and symbol.isalpha():
            # Conservative: don't claim random 2-4 letter tickers as US stocks
            # unless they're explicitly in US_ASSETS (checked above)
            return None

        # Try with Yahoo Finance lookup for remaining unknown symbols
        # (e.g., BTC-USD, BRK-B, ^GSPC — these have special chars)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            name = info.get("shortName") or info.get("longName") or symbol
            sector = info.get("sector") or "Unknown"
            return AssetInfo(
                symbol=symbol,
                name=name,
                market=self.market_id,
                sector=sector,
                exchange=info.get("exchange", "NYSE/NASDAQ"),
                currency=info.get("currency", "USD"),
            )
        except Exception:
            return None

    def get_price_data(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance."""
        try:
            # Map common internal intervals to yfinance formats
            yf_interval = interval
            if interval == "4h":
                yf_interval = "90m" # yfinance doesn't natively do 4h well, 90m or 1h is better
                
            logger.info(f"Fetching {symbol} ({interval}) data from {start} to {end}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data for {symbol}")

            df = df.reset_index()
            if "Date" not in df.columns and "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            elif "Date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "Date"})
            logger.success(f"Fetched {len(df)} rows for {symbol}")
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise ValueError(f"Could not fetch data for {symbol}: {e}")

    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        """Get fundamental data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "pe": info.get("trailingPE", 0) or 0,
                "pb": info.get("priceToBook", 0) or 0,
                "eps": info.get("trailingEps", 0) or 0,
                "roe": (info.get("returnOnEquity", 0) or 0) * 100,
                "bvps": info.get("bookValue", 0) or 0,
                "dividend_yield": (info.get("dividendYield", 0) or 0) * 100,
                "market_cap": info.get("marketCap", 0) or 0,
                "revenue": info.get("totalRevenue", 0) or 0,
            }
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")
            return {}

    def get_realtime_quote(self, symbol: str):
        """Get real-time quote for US stock/crypto via yfinance fast_info."""
        from datetime import datetime
        from .base import RealtimeQuote

        try:
            ticker = yf.Ticker(symbol)
            fi = ticker.fast_info

            price = float(fi.get('lastPrice', 0) or fi.get('last_price', 0))
            prev_close = float(fi.get('previousClose', 0) or fi.get('previous_close', 0))
            day_high = float(fi.get('dayHigh', 0) or fi.get('day_high', price))
            day_low = float(fi.get('dayLow', 0) or fi.get('day_low', price))
            volume = float(fi.get('lastVolume', 0) or fi.get('last_volume', 0))

            if price <= 0:
                # Fallback: get from history
                hist = ticker.history(period='2d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    if len(hist) >= 2:
                        prev_close = float(hist['Close'].iloc[-2])

            change = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            # Crypto markets are 24/7
            is_crypto = symbol.endswith('-USD')
            is_market_open = is_crypto  # For stocks, yfinance doesn't reliably tell us

            return RealtimeQuote(
                symbol=symbol,
                price=price,
                change=round(change, 2),
                prev_close=prev_close,
                volume=volume,
                high=day_high,
                low=day_low,
                timestamp=datetime.now().isoformat(),
                source='yfinance',
                is_market_open=is_market_open,
            )
        except Exception as e:
            logger.warning(f"Realtime quote failed for {symbol}: {e}")
            return super().get_realtime_quote(symbol)

    def supports_fundamentals(self) -> bool:
        return True

    def supports_cot_data(self) -> bool:
        return False


# Auto-register
from . import registry  # noqa: E402

_us_provider = USStockProvider()
registry.register(_us_provider)
logger.success(f"Registered {_us_provider.market_name} provider ({len(US_ASSETS)} assets)")
