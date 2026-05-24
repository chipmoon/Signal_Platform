"""
Taiwan Stock Provider Plugin
=============================
Provides data for Taiwanese stocks using Yahoo Finance.

Extracts from D:\Python_VS\stock\stock_tw_rs.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from .base import AssetInfo, AssetProvider, RealtimeQuote


# Taiwan stock list (Expanded to TWSE 50 + Required + Hot AI/Semis)
TAIWAN_STOCKS = {
    # --- Semiconductor & IC Design (Crucial) ---
    "2330.TW": {"Name": "TSMC", "Name_CN": "台積電", "Sector": "Semiconductors - Foundry"},
    "2454.TW": {"Name": "MediaTek", "Name_CN": "聯發科", "Sector": "IC Design - Mobile/AI"},
    "2303.TW": {"Name": "UMC", "Name_CN": "聯電", "Sector": "Semiconductors - Foundry"},
    "3711.TW": {"Name": "ASE Technology", "Name_CN": "日月光", "Sector": "Semiconductors - OSAT"},
    "3034.TW": {"Name": "Novatek", "Name_CN": "聯詠", "Sector": "IC Design - Display"},
    "2379.TW": {"Name": "Realtek", "Name_CN": "瑞昱", "Sector": "IC Design - Network"},
    "4919.TW": {"Name": "Nuvoton", "Name_CN": "新唐", "Sector": "IC Design - MCU"},
    "5274.TWO": {"Name": "ASPEED", "Name_CN": "信驊", "Sector": "IC Design - AI BMC"},
    "3661.TW": {"Name": "Alchip", "Name_CN": "世芯-KY", "Sector": "IC Design - AI ASIC"},
    "3443.TW": {"Name": "GUC", "Name_CN": "創意", "Sector": "IC Design - ASIC"},
    "3035.TW": {"Name": "Faraday", "Name_CN": "智原", "Sector": "IC Design - ASIC"},
    "5269.TW": {"Name": "ASMedia", "Name_CN": "祥碩", "Sector": "IC Design - HS Interface"},
    "6415.TW": {"Name": "Silergy-KY", "Name_CN": "矽力-KY", "Sector": "Power Management IC"},
    "5347.TWO": {"Name": "Vanguard (VIS)", "Name_CN": "世界先進", "Sector": "Semiconductors - Foundry"},
    "2408.TW": {"Name": "Nanya Tech", "Name_CN": "南亞科", "Sector": "Memory - DRAM"},
    "2337.TW": {"Name": "旺宏", "Name_CN": "旺宏", "Sector": "Memory - Flash"},
    "2344.TW": {"Name": "華邦電", "Name_CN": "華邦電", "Sector": "Memory - DRAM/Flash"},
    "8299.TWO": {"Name": "Phison", "Name_CN": "群聯", "Sector": "Flash Controller"},

    # --- AI Servers, Computing & OEM ---
    "2317.TW": {"Name": "Foxconn (Hon Hai)", "Name_CN": "鴻海", "Sector": "Electronics OEM/AI"},
    "2382.TW": {"Name": "Quanta Computer", "Name_CN": "廣達", "Sector": "AI Server/OEM"},
    "3231.TW": {"Name": "Wistron", "Name_CN": "緯創", "Sector": "AI Server/OEM"},
    "6669.TW": {"Name": "Wiwynn", "Name_CN": "緯穎", "Sector": "Cloud Infrastructure"},
    "2376.TW": {"Name": "GIGABYTE", "Name_CN": "技嘉", "Sector": "AI Server/Motherboard"},
    "2357.TW": {"Name": "Asustek", "Name_CN": "華碩", "Sector": "Computing - PC/AI"},
    "2353.TW": {"Name": "Acer", "Name_CN": "宏碁", "Sector": "Computing - PC"},
    "2324.TW": {"Name": "Compal", "Name_CN": "仁寶", "Sector": "Computing - PC"},
    "2356.TW": {"Name": "Inventec", "Name_CN": "英業達", "Sector": "Server/OEM"},
    "4938.TW": {"Name": "Pegatron", "Name_CN": "和碩", "Sector": "Electronics OEM"},
    "2395.TW": {"Name": "Advantech", "Name_CN": "研華", "Sector": "Industrial IoT"},
    "2377.TW": {"Name": "MSI", "Name_CN": "微星", "Sector": "Computing - Gaming"},

    # --- Components (AI Cooling, PCB, Optics) ---
    "3017.TW": {"Name": "AVC", "Name_CN": "奇鋐", "Sector": "AI Cooling - Thermal"},
    "3653.TW": {"Name": "Jentech", "Name_CN": "健策", "Sector": "AI Cooling - Heat Spreaders"},
    "3037.TW": {"Name": "Unimicron", "Name_CN": "欣興", "Sector": "PCB - ABF Substrate"},
    "8046.TW": {"Name": "Nan Ya PCB", "Name_CN": "南電", "Sector": "PCB - ABF Substrate"},
    "2368.TW": {"Name": "GCE", "Name_CN": "金像電", "Sector": "PCB - AI Server"},
    "2313.TW": {"Name": "Compeq", "Name_CN": "華通", "Sector": "PCB - HDI"},
    "2383.TW": {"Name": "Tripod", "Name_CN": "健鼎", "Sector": "PCB"},
    "6213.TW": {"Name": "ITEQ", "Name_CN": "聯茂", "Sector": "CCL - High Speed"},
    "2308.TW": {"Name": "Delta Electronics", "Name_CN": "台達電", "Sector": "Power Electronics"},
    "2301.TW": {"Name": "Lite-On", "Name_CN": "光寶科", "Sector": "Power/Cloud"},
    "2449.TW": {"Name": "KYEC", "Name_CN": "京元電子", "Sector": "Semis Testing"},
    "3008.TW": {"Name": "Largan Precision", "Name_CN": "大立光", "Sector": "Optics - Mobile"},
    "3406.TW": {"Name": "GSEO", "Name_CN": "玉晶光", "Sector": "Optics"},
    "4908.TWO": {"Name": "APAC Opto", "Name_CN": "前鼎", "Sector": "Optics - Fiber Networking"},
    "2345.TW": {"Name": "Accton", "Name_CN": "智邦", "Sector": "Networking / Switch"},
    "3533.TW": {"Name": "BizLink", "Name_CN": "貿聯-KY", "Sector": "Connectivity"},

    # --- Financials (TWSE 50 Leaders) ---
    "2881.TW": {"Name": "Fubon Financial", "Name_CN": "富邦金", "Sector": "Financial - Insurance"},
    "2882.TW": {"Name": "Cathay Financial", "Name_CN": "國泰金", "Sector": "Financial - Insurance"},
    "2891.TW": {"Name": "CTBC Financial", "Name_CN": "中信金", "Sector": "Financial - Bank"},
    "2886.TW": {"Name": "Mega Financial", "Name_CN": "兆豐金", "Sector": "Financial - Bank"},
    "2884.TW": {"Name": "E.SUN Financial", "Name_CN": "玉山金", "Sector": "Financial - Bank"},
    "5880.TW": {"Name": "TCFH", "Name_CN": "合庫金", "Sector": "Financial - Bank"},
    "2885.TW": {"Name": "Yuanta Financial", "Name_CN": "元大金", "Sector": "Financial - Brokerage"},

    # --- Industrial, Materials & Display ---
    "2002.TW": {"Name": "China Steel", "Name_CN": "中鋼", "Sector": "Materials - Steel"},
    "1101.TW": {"Name": "Taiwan Cement", "Name_CN": "台泥", "Sector": "Materials - Cement"},
    "1303.TW": {"Name": "Nan Ya Plastics", "Name_CN": "南亞", "Sector": "Materials - Petrochemicals"},
    "1301.TW": {"Name": "Formosa Plastics", "Name_CN": "台塑", "Sector": "Materials - Petrochemicals"},
    "2327.TW": {"Name": "Yageo", "Name_CN": "國巨", "Sector": "Passive Components"},
    "2409.TW": {"Name": "AUO", "Name_CN": "友達", "Sector": "Display - LCD"},
    "3481.TW": {"Name": "Innolux", "Name_CN": "群創", "Sector": "Display - LCD"},
    "8069.TWO": {"Name": "E Ink", "Name_CN": "元太", "Sector": "Display - ePaper"},

    # --- Shipping & Trans ---
    "2603.TW": {"Name": "Evergreen Marine", "Name_CN": "長榮", "Sector": "Shipping"},
    "2609.TW": {"Name": "Yang Ming", "Name_CN": "陽明", "Sector": "Shipping"},
    "2615.TW": {"Name": "Wan Hai Lines", "Name_CN": "萬海", "Sector": "Shipping"},
    "2618.TW": {"Name": "EVA Air", "Name_CN": "長榮航", "Sector": "Airline"},

    # --- Consumer & Telecomm ---
    "1216.TW": {"Name": "Uni-President", "Name_CN": "統一", "Sector": "Consumer Goods"},
    "2412.TW": {"Name": "Chunghwa Telecom", "Name_CN": "中華電", "Sector": "Telecomm"},
    "3045.TW": {"Name": "Taiwan Mobile", "Name_CN": "台灣大", "Sector": "Telecomm"},
    "2912.TW": {"Name": "PCSC (7-Eleven)", "Name_CN": "統一超", "Sector": "Retail"},
}


from ..cache_manager import cache

PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


@contextmanager
def _safe_yfinance_env():
    """Provide writable runtime env for yfinance cache and bypass broken loopback proxy."""
    backup: Dict[str, Optional[str]] = {}
    project_root = str(Path(__file__).resolve().parents[2])
    yf_cache_dir = Path(project_root) / ".cache" / "yfinance"
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for home_key in ["HOME", "USERPROFILE"]:
            backup[home_key] = os.environ.get(home_key)
            os.environ[home_key] = project_root
        backup["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(Path(project_root) / ".cache")

        for key in PROXY_ENV_KEYS:
            value = os.environ.get(key)
            if not value:
                continue
            low = value.lower()
            if "127.0.0.1:9" in low or "localhost:9" in low:
                backup[key] = value
                os.environ.pop(key, None)
        # Ensure yfinance peewee DB is created in a writable location.
        yf.set_tz_cache_location(str(yf_cache_dir))
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TaiwanStockProvider(AssetProvider):
    """Asset provider for Taiwanese stocks via Yahoo Finance."""
    
    def __init__(self):
        self._stock_list: List[Dict[str, str]] = []
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load stock list from cache or fallback list."""
        if self._initialized:
            return
            
        # Try cache first (full listing: ~40k stocks)
        cached = cache.get_cached_stock_list("TW", max_age_hours=168) # 7 days
        if cached:
            self._stock_list = cached
            self._initialized = True
            logger.success(f"Loaded {len(cached)} TW stocks from cache")
            return
            
        # Fallback to local hardcoded list
        for symbol, info in TAIWAN_STOCKS.items():
            self._stock_list.append({
                "symbol": symbol,
                "name": info["Name"],
                "name_cn": info.get("Name_CN", ""),
                "sector": info["Sector"],
                "exchange": "TWSE" if symbol.endswith(".TW") else "TPEx",
                "currency": "TWD"
            })
        self._initialized = True
        logger.info(f"Loaded {len(self._stock_list)} TW stocks from fallback list")
    
    @property
    def market_id(self) -> str:
        return "TW"
    
    @property
    def market_name(self) -> str:
        return "Taiwan Stocks"
    
    def search_assets(self, query: str, limit: int = 10) -> List[AssetInfo]:
        """
        Search Taiwan stocks by symbol, name, or sector.
        """
        self._ensure_initialized()
        query_lower = query.lower()
        q_clean = query_lower.replace('.tw', '').replace('.two', '').strip()
        results = []
        
        # Priority 1: Hot Stocks from hardcoded list (if they match)
        for symbol, info in TAIWAN_STOCKS.items():
            stock_code = symbol.replace('.TW', '').replace('.TWO', '')
            if (q_clean and q_clean == stock_code.lower()) or (query_lower and query_lower in info['Name'].lower()):
                results.append(
                    AssetInfo(
                        symbol=symbol,
                        name=f"{info['Name']} ({info.get('Name_CN', '')})",
                        market=self.market_id,
                        sector=info['Sector'],
                        exchange="TWSE/TPE",
                        currency="TWD",
                    )
                )
                if len(results) >= limit: return results

        # Priority 2: Full listing from cache
        seen_symbols = {r.symbol for r in results}
        for stock in self._stock_list:
            if stock['symbol'] in seen_symbols: continue
            
            stock_code = stock['symbol'].split('.')[0]
            if (
                (q_clean and q_clean == stock_code.lower())
                or (q_clean and q_clean in stock['name'].lower())
                or (q_clean and q_clean in stock.get('name_cn', '').lower())
            ):
                results.append(
                    AssetInfo(
                        symbol=stock['symbol'],
                        name=f"{stock['name']}",
                        market=self.market_id,
                        sector=stock.get('sector', 'Taiwan Stock'),
                        exchange=stock.get('exchange', 'TWSE/TPE'),
                        currency="TWD",
                    )
                )
            if len(results) >= limit: break
            
        return results
    
    def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Get Taiwan stock info by symbol (full list search)."""
        self._ensure_initialized()
        symbol = symbol.upper().strip()

        # 1. Exact match in full list
        # Handle formats like "8096", "8096.TW", "8096.TWO"
        clean_code = symbol.replace('.TW', '').replace('.TWO', '')

        # 0. Prefer curated list metadata for canonical naming consistency.
        for curated_symbol, info in TAIWAN_STOCKS.items():
            curated_code = curated_symbol.replace('.TW', '').replace('.TWO', '')
            if curated_symbol == symbol or curated_code.upper() == clean_code.upper():
                return AssetInfo(
                    symbol=curated_symbol,
                    name=f"{info['Name']} ({info.get('Name_CN', '')})".strip(),
                    market=self.market_id,
                    sector=info.get('Sector', 'Taiwan Stock'),
                    exchange="TWSE" if curated_symbol.endswith('.TW') else "TPEx",
                    currency="TWD",
                )
        
        for stock in self._stock_list:
            stock_clean = stock['symbol'].split('.')[0].upper()
            if stock['symbol'] == symbol or stock_clean == symbol:
                return AssetInfo(
                    symbol=stock['symbol'],
                    name=stock['name'],
                    market=self.market_id,
                    sector=stock.get('sector', 'Taiwan Stock'),
                    exchange=stock.get('exchange', 'TWSE/TPE'),
                    currency="TWD",
                )
        
        # 2. Dynamic Detection via yfinance for fallback
        if clean_code.isdigit() and len(clean_code) == 4:
            try:
                # Try both .TW and .TWO
                for suffix in ['.TW', '.TWO']:
                    target = f"{clean_code}{suffix}"
                    with _safe_yfinance_env():
                        ticker = yf.Ticker(target)
                        inf = ticker.info
                    if inf and (inf.get('shortName') or inf.get('longName')):
                        return AssetInfo(
                            symbol=target,
                            name=inf.get('shortName') or inf.get('longName') or target,
                            market=self.market_id,
                            sector=inf.get('sector', 'Taiwan Stock'),
                            exchange="TWSE/TPE",
                            currency="TWD",
                        )
            except:
                pass
        
        return None

    def get_price_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch Taiwan stock price data — cache-first, yfinance fallback."""
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'):
            yahoo_symbol = f"{symbol}.TW"
        else:
            yahoo_symbol = symbol

        clean_code = yahoo_symbol.replace('.TW', '').replace('.TWO', '')

        # ── 1. Try local OHLCV cache first (populated by nightly scan) ──
        try:
            cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "ohlcv"
            for fname in [f"{yahoo_symbol}.parquet", f"{clean_code}.TW.parquet",
                          f"{yahoo_symbol}.csv", f"{clean_code}.TW.csv"]:
                cache_path = cache_dir / fname
                if cache_path.exists():
                    if fname.endswith('.parquet'):
                        df_c = pd.read_parquet(cache_path)
                    else:
                        df_c = pd.read_csv(cache_path)
                    if not df_c.empty:
                        logger.info(f"TW cache hit: {yahoo_symbol} ({len(df_c)} rows)")
                        # Filter by date range
                        if "Date" in df_c.columns:
                            df_c["Date"] = pd.to_datetime(df_c["Date"])
                            df_c = df_c[
                                (df_c["Date"] >= pd.to_datetime(start)) &
                                (df_c["Date"] <= pd.to_datetime(end))
                            ]
                        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
                        if all(c in df_c.columns for c in required):
                            return df_c[required]
        except Exception as _ce:
            logger.debug(f"TW cache lookup failed: {_ce}")

        # ── 2. yfinance (works locally, may be blocked on Streamlit Cloud) ──
        try:
            yf_interval = interval
            if interval == "4h":
                yf_interval = "90m"

            logger.info(f"Fetching {yahoo_symbol} ({interval}) from yfinance {start}→{end}")
            with _safe_yfinance_env():
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=True)
    
    def get_fundamentals(self, symbol: str) -> Dict[str, float]:
        """Get fundamental data for Taiwan stock from Yahoo Finance."""
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'):
            # Default to checking local list first
            info = self.get_asset_info(symbol)
            yahoo_symbol = info.symbol if info else f"{symbol}.TW"
        else:
            yahoo_symbol = symbol
        
        fundamentals = {'eps': 0.0, 'pe': 0.0, 'pb': 0.0, 'roe': 0.0, 'bvps': 0.0, 'dividend_yield': 0.0}
        try:
            with _safe_yfinance_env():
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
            fundamentals['pe'] = info.get('trailingPE', 0) or 0
            fundamentals['pb'] = info.get('priceToBook', 0) or 0
            fundamentals['roe'] = (info.get('returnOnEquity', 0) or 0) * 100
            fundamentals['eps'] = info.get('trailingEps', 0) or 0
            fundamentals['bvps'] = info.get('bookValue', 0) or 0
            fundamentals['dividend_yield'] = (info.get('dividendYield', 0) or 0) * 100
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {yahoo_symbol}: {e}")
        return fundamentals
    
    def get_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Fetch latest price for Taiwan stocks from yfinance fast_info."""
        if not symbol.endswith('.TW') and not symbol.endswith('.TWO'):
            asset = self.get_asset_info(symbol)
            yahoo_symbol = asset.symbol if asset else f"{symbol}.TW"
        else:
            yahoo_symbol = symbol

        try:
            with _safe_yfinance_env():
                ticker = yf.Ticker(yahoo_symbol)
            
            # fast_info is much faster than history()
            with _safe_yfinance_env():
                info = ticker.fast_info
            price = info.get('last_price')
            prev_close = info.get('previous_close')
            
            if price is None:
                # Fallback to history if fast_info fails
                with _safe_yfinance_env():
                    df = ticker.history(period="2d")
                if df.empty: return None
                price = float(df.iloc[-1]['Close'])
                prev_close = float(df.iloc[-2]['Close']) if len(df) >= 2 else price

            change = ((price - prev_close) / prev_close * 100) if prev_close and prev_close > 0 else 0.0

            # Taiwan Market Hours (UTC+8): 09:00 - 13:30, Mon-Fri
            now_tw = datetime.now() # Assuming local is UTC+8 or logic handles it
            is_open = (9 <= now_tw.hour < 14) and (now_tw.weekday() < 5)
            if now_tw.hour == 13 and now_tw.minute > 30: is_open = False

            return RealtimeQuote(
                symbol=symbol,
                price=price,
                change=round(change, 2),
                prev_close=prev_close,
                volume=float(info.get('last_volume', 0)),
                high=float(info.get('day_high', price)),
                low=float(info.get('day_low', price)),
                timestamp=datetime.now().isoformat(),
                source='yfinance_fast',
                is_market_open=is_open
            )
        except Exception as e:
            logger.warning(f"Failed to fetch realtime quote for {yahoo_symbol}: {e}")
            return None

    def supports_fundamentals(self) -> bool: return True
    def supports_cot_data(self) -> bool: return False


# Auto-register this provider
from . import registry

_tw_provider = TaiwanStockProvider()
registry.register(_tw_provider)
logger.success(f"Registered {_tw_provider.market_name} provider ({len(TAIWAN_STOCKS)} stocks)")
