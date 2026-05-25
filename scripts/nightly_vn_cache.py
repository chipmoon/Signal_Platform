"""
Nightly VN Cache Pipeline — Option A
======================================
Chạy mỗi tối local để fetch dữ liệu VN từ vnstock → lưu parquet cache
Sáng hôm sau Streamlit đọc từ cache thay vì gọi API bị chặn.

Cách dùng:
    python scripts/nightly_vn_cache.py              # Fetch + save cache
    python scripts/nightly_vn_cache.py --push       # Fetch + save + git push

Tự động hóa (Windows Task Scheduler):
    Trigger: Daily lúc 18:00 (sau khi thị trường đóng 15:00 UTC+7)
    Action:  "d:\Python_VS\trading_system\venv\Scripts\python.exe"
             scripts/nightly_vn_cache.py --push
             Start in: d:\Python_VS\trading_system
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

# ── Top 95 VN stocks confirmed available on yfinance ──────────────────────────
TOP_VN_STOCKS = [
    # Banks
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "HDB",
    "LPB", "SSB", "EIB", "OCB", "TPB", "SHB", "EVF",
    # Real Estate
    "VIC", "VHM", "VRE", "KDH", "NVL", "PDR", "DXG", "IJC", "TDC",
    "SIP", "HDC", "HAG", "LCG", "VPI", "CII", "HDG", "PC1", "VCG",
    # Industry & Materials
    "HPG", "GAS", "PLX", "GVR", "BSR", "DGC", "PHR", "DPR", "TRC",
    "BMP", "AAA", "LSS", "PPC", "NT2", "POW", "GEG", "BWE", "KHP",
    "REE", "GEX", "HHV", "PVT",
    # Consumer & Retail
    "SAB", "MSN", "VNM", "MWG", "PNJ", "TLG", "DHC", "DBC", "PAN",
    "VHC", "HAX", "HAH", "VTO", "ASM", "CSV",
    # Technology
    "FPT", "CMG", "VNE",
    # Aviation & Transport
    "VJC", "GMD",
    # Financials / Securities
    "SSI", "VCI", "HCM", "VND", "BSI", "ORS", "CTS", "FTS", "VDS",
    # Construction
    "CTD", "FCN", "VCG",
    # Agriculture
    "AGR", "DPM", "DCM",
    # Others
    "BVH", "BCM", "SCS", "TDM", "DBC", "TV2",
]

# Deduplicate
SYMBOLS = list(dict.fromkeys(TOP_VN_STOCKS))


def fetch_with_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch using vnstock (works local, blocked on cloud)."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df = stock.quote.history(start=start, end=end, interval="1D")
        if df is None or df.empty:
            return None
        col_map = {"time": "Date", "open": "Open", "high": "High",
                   "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_map)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.debug(f"vnstock failed for {symbol}: {e}")
        return None


def fetch_with_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch using yfinance (fallback, works everywhere)."""
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.VN").history(
            start=start, end=end, interval="1d", auto_adjust=False
        )
        if df.empty:
            return None
        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.debug(f"yfinance failed for {symbol}: {e}")
        return None


def save_cache(symbol: str, df: pd.DataFrame, cache_dir: Path) -> None:
    """Save parquet to .cache/prices/SYMBOL_VN.parquet."""
    prices_dir = cache_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    path = prices_dir / f"{symbol}_VN.parquet"

    # Merge with existing cache
    if path.exists():
        try:
            existing = pd.read_parquet(path, engine="pyarrow")
            existing["Date"] = pd.to_datetime(existing["Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = pd.concat([existing, df]).drop_duplicates("Date", keep="last")
            df = df.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass

    df.to_parquet(path, index=False, engine="pyarrow")


def update_stock_list_cache(symbols: list[str], cache_dir: Path) -> None:
    """Save the hardcoded stock list as parquet so app can load without vnstock API."""
    stock_list = [
        {"symbol": s, "name": s, "exchange": "HOSE", "sector": "Other"}
        for s in symbols
    ]
    df = pd.DataFrame(stock_list)
    path = cache_dir / "stock_list_VN.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved stock list: {len(symbols)} symbols → {path}")


def git_push(project_root: Path) -> bool:
    """Commit cache files and push to GitHub."""
    try:
        cache_dir = project_root / ".cache"
        subprocess.run(
            ["git", "add", str(cache_dir)],
            cwd=project_root, check=True, capture_output=True
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        result = subprocess.run(
            ["git", "commit", "-m", f"data: nightly VN cache update {timestamp}"],
            cwd=project_root, capture_output=True, text=True
        )
        if "nothing to commit" in result.stdout:
            logger.info("No cache changes to push")
            return True
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=project_root, check=True, capture_output=True
        )
        logger.success(f"Pushed cache to GitHub at {timestamp}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git push failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Nightly VN cache fetcher")
    parser.add_argument("--push", action="store_true", help="Push cache to GitHub after fetching")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch (default: 365)")
    parser.add_argument("--symbols", nargs="+", help="Override symbol list")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / ".cache"

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    symbols = args.symbols or SYMBOLS

    logger.info(f"Nightly VN Cache — {len(symbols)} symbols | {start} → {end}")
    logger.info(f"Cache dir: {cache_dir}")

    # Save hardcoded stock list first
    update_stock_list_cache(symbols, cache_dir)

    ok_vnstock, ok_yfinance, failed = [], [], []

    for i, sym in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Fetching {sym}...")

        # 1. Fetch prices
        price_ok = False
        df = fetch_with_vnstock(sym, start, end)
        if df is not None and not df.empty:
            save_cache(sym, df, cache_dir)
            ok_vnstock.append(sym)
            logger.success(f"  {sym} prices via vnstock — {len(df)} rows")
            price_ok = True
        else:
            df = fetch_with_yfinance(sym, start, end)
            if df is not None and not df.empty:
                save_cache(sym, df, cache_dir)
                ok_yfinance.append(sym)
                logger.warning(f"  {sym} prices via yfinance fallback — {len(df)} rows")
                price_ok = True

        if not price_ok:
            failed.append(sym)
            logger.error(f"  {sym} prices FAILED all sources")

        # 2. Fetch and cache fundamentals
        try:
            from src.analytics.fundamental_score import _fetch_vn_fundamentals_from_vnstock
            from src.cache_manager import cache as cm
            fund_data = _fetch_vn_fundamentals_from_vnstock(sym)
            if any(v is not None for v in fund_data.values()):
                cm.cache_fundamentals(sym, "VN", fund_data)
                logger.success(f"  {sym} fundamentals cached successfully")
            else:
                logger.warning(f"  {sym} fundamentals empty")
        except Exception as e:
            logger.error(f"  {sym} fundamentals fetch error: {e}")

        # Throttle to stay under rate limits (e.g. 20 requests/min)
        import time
        time.sleep(6)

    # Summary
    logger.info("=" * 50)
    logger.success(f"vnstock: {len(ok_vnstock)} | yfinance: {len(ok_yfinance)} | failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed symbols: {failed}")

    # Push to GitHub if requested
    if args.push:
        logger.info("Pushing cache to GitHub...")
        git_push(project_root)

    logger.success("Nightly cache update complete!")


if __name__ == "__main__":
    main()
