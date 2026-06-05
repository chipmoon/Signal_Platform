r"""
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

# ── Top VN stocks (expanded to ~115, includes oil&gas, steel, aviation) ──────────
TOP_VN_STOCKS = [
    # Banks (17)
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "HDB",
    "LPB", "SSB", "EIB", "OCB", "TPB", "SHB", "EVF", "VBB",
    # Real Estate (20)
    "VIC", "VHM", "VRE", "KDH", "NVL", "PDR", "DXG", "IJC", "TDC",
    "SIP", "HDC", "HAG", "LCG", "VPI", "CII", "HDG", "PC1", "VCG",
    "NLG", "SZC",
    # Oil & Gas (6) — NEWLY ADDED
    "PVS", "PVD", "PVT", "PVB", "PVI", "GAS",
    # Industry & Materials (24)
    "HPG", "PLX", "GVR", "BSR", "DGC", "PHR", "DPR", "TRC",
    "BMP", "AAA", "LSS", "PPC", "NT2", "POW", "GEG", "BWE", "KHP",
    "REE", "GEX", "HHV",
    "NKG", "TLH", "SMC", "HSG",
    # Consumer & Retail (16)
    "SAB", "MSN", "VNM", "MWG", "PNJ", "TLG", "DHC", "DBC", "PAN",
    "VHC", "HAX", "HAH", "VTO", "ASM", "CSV", "BHN",
    # Technology (4)
    "FPT", "CMG", "VNE", "SGT",
    # Aviation & Transport (3)
    "VJC", "GMD", "HVN",
    # Financials / Securities (10)
    "SSI", "VCI", "HCM", "VND", "BSI", "ORS", "CTS", "FTS", "VDS", "MBS",
    # Construction (3)
    "CTD", "FCN", "HBC",
    # Agriculture & Fertilizer (4)
    "AGR", "DPM", "DCM", "DDV",
    # Pharma (3)
    "DHG", "IMP", "TRA",
    # Others (6)
    "BVH", "BCM", "SCS", "TDM", "TV2", "VGC",
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
    from src.cache_manager import CacheManager

    cm = CacheManager(str(cache_dir))
    cm.cache_price_data(symbol, "VN", df)


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
    """Commit cache files and push to GitHub (both master and main branches)."""
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
        # Push to master first
        subprocess.run(
            ["git", "push", "origin", "master"],
            cwd=project_root, check=True, capture_output=True
        )
        # Sync master → main (Streamlit Cloud deploys from main)
        subprocess.run(["git", "checkout", "main"], cwd=project_root, check=True, capture_output=True)
        subprocess.run(["git", "merge", "master", "--ff-only"], cwd=project_root, check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=project_root, check=True, capture_output=True)
        subprocess.run(["git", "checkout", "master"], cwd=project_root, check=True, capture_output=True)
        logger.success(f"Pushed cache to GitHub (master + main) at {timestamp}")
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
