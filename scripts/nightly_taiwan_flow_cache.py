"""
Nightly Taiwan Flow Cache — scripts/nightly_taiwan_flow_cache.py
==============================================================
Chạy sau khi thị trường đóng cửa (18:30 UTC+7) để fetch dữ liệu
Khối ngoại từng phiên của Đài Loan từ FinMind API và merge vào parquet cache.

Cách dùng:
    python scripts/nightly_taiwan_flow_cache.py              # Tất cả cổ phiếu TW
    python scripts/nightly_taiwan_flow_cache.py --symbols 2330,2454
    python scripts/nightly_taiwan_flow_cache.py --dry-run    # Chỉ test, không ghi file
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf
from loguru import logger
from FinMind.data import DataLoader

from src.plugins.taiwan import TAIWAN_STOCKS

CACHE_DIR = ROOT / ".cache" / "prices"

# Free/community token (optional environment variable FINMIND_API_TOKEN)
_FINMIND_DEFAULT_TOKEN = ""


def _fetch_foreign_flow_taiwan(stock_code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch daily foreign buy/sell data for a Taiwan stock via FinMind API.
    Returns: Date, Foreign_Buy, Foreign_Sell, Block_Trade_Volume
    """
    import os
    token = os.environ.get("FINMIND_API_TOKEN", _FINMIND_DEFAULT_TOKEN)
    
    try:
        api = DataLoader()
        if token:
            api.login_by_token(api_token=token)
            
        # Download Institutional Investors Buy/Sell dataset
        df = api.taiwan_stock_institutional_investors(
            stock_id=stock_code,
            start_date=start,
            end_date=end
        )
        
        if df is None or df.empty:
            return None
            
        # FinMind returned columns: date, stock_id, buy, sell, name
        # Keep only "Foreign_Investor" (外資)
        df_foreign = df[df["name"] == "Foreign_Investor"].copy()
        if df_foreign.empty:
            return None
            
        # Rename columns to standard schema
        df_foreign["Date"] = pd.to_datetime(df_foreign["date"]).dt.normalize().dt.tz_localize(None)
        df_foreign["Foreign_Buy"] = pd.to_numeric(df_foreign["buy"], errors="coerce").fillna(0.0)
        df_foreign["Foreign_Sell"] = pd.to_numeric(df_foreign["sell"], errors="coerce").fillna(0.0)
        df_foreign["Block_Trade_Volume"] = 0.0  # default since block trade volume is not separated
        
        return df_foreign[["Date", "Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]]
        
    except Exception as e:
        logger.debug(f"  {stock_code}: FinMind fetch failed — {e}")
        return None


def _merge_flow_into_parquet(symbol: str, stock_code: str, flow_df: pd.DataFrame, dry_run: bool) -> bool:
    """
    Merge foreign flow columns into the TW price parquet file.
    """
    parquet_path = CACHE_DIR / f"{stock_code}_TW.parquet"
    
    try:
        existing = None
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path, engine="pyarrow")
            logger.info(f"  {symbol}: Loaded existing price cache ({len(existing)} rows)")
        else:
            # Bootstrap using yfinance
            logger.info(f"  {symbol}: Parquet not found. Bootstrapping price history via yfinance...")
            ticker = yf.Ticker(symbol)
            df_price = ticker.history(period="2y", interval="1d")
            if df_price.empty:
                logger.warning(f"  {symbol}: yfinance returned empty price history. Cannot bootstrap.")
                return False
                
            df_price = df_price.reset_index()
            if "Date" not in df_price.columns and "Datetime" in df_price.columns:
                df_price = df_price.rename(columns={"Datetime": "Date"})
            elif "Date" not in df_price.columns and "index" in df_price.columns:
                df_price = df_price.rename(columns={"index": "Date"})
                
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            existing = df_price[required].copy()
            
        # Normalize Date column for reliable merge
        if "Date" in existing.columns:
            existing["Date"] = pd.to_datetime(existing["Date"]).dt.normalize().dt.tz_localize(None)
        elif existing.index.name == "Date":
            existing = existing.reset_index()
            existing["Date"] = pd.to_datetime(existing["Date"]).dt.normalize().dt.tz_localize(None)
            
        # Normalize flow dates
        flow_df["Date"] = pd.to_datetime(flow_df["Date"]).dt.normalize().dt.tz_localize(None)
        
        # ── Save today's flow to sidecar JSON ──
        sidecar_path = parquet_path.with_suffix(".flow.json")
        try:
            sidecar_records = flow_df.copy()
            sidecar_records["Date"] = sidecar_records["Date"].astype(str)
            sidecar_records.to_json(sidecar_path, orient="records", date_format="iso")
        except Exception as _e:
            logger.debug(f"  {symbol}: Sidecar write failed — {_e}")
            
        # ── Also apply any previously saved sidecar rows ──
        if sidecar_path.exists():
            try:
                sc = pd.read_json(sidecar_path, orient="records")
                sc["Date"] = pd.to_datetime(sc["Date"]).dt.normalize().dt.tz_localize(None)
                flow_df = pd.concat([sc, flow_df]).drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
            except Exception as _e:
                logger.debug(f"  {symbol}: Sidecar read failed — {_e}")
                
        # Left join with existing price data
        merged = existing.merge(
            flow_df[["Date", "Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]],
            on="Date", how="left", suffixes=("_old", "")
        )
        
        # Resolve column conflicts
        for col in ["Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]:
            if f"{col}_old" in merged.columns:
                merged[col] = merged[col].fillna(merged[f"{col}_old"])
                merged.drop(columns=[f"{col}_old"], inplace=True)
            if col not in merged.columns:
                merged[col] = 0.0
                
        # Fill NaN values in flow columns with 0.0
        merged["Foreign_Buy"] = merged["Foreign_Buy"].fillna(0.0)
        merged["Foreign_Sell"] = merged["Foreign_Sell"].fillna(0.0)
        merged["Block_Trade_Volume"] = merged["Block_Trade_Volume"].fillna(0.0)
        
        # Compute Foreign_Net_TWD (in millions of TWD) and Foreign_Net_VND (in millions of VND)
        # Convert TWD to VND dynamically (1 TWD ~ 800 VND)
        if "Close" in merged.columns:
            net_vol = merged["Foreign_Buy"] - merged["Foreign_Sell"]
            close = pd.to_numeric(merged["Close"], errors="coerce").fillna(0)
            merged["Foreign_Net_TWD"] = (net_vol * close / 1_000_000).round(3)
            merged["Foreign_Net_VND"] = (net_vol * close * 800 / 1_000_000).round(3)  # approximate conversion
        else:
            merged["Foreign_Net_TWD"] = 0.0
            merged["Foreign_Net_VND"] = 0.0
            
        merged = merged.sort_values("Date").reset_index(drop=True)
        
        # Show today's flow in logs
        today_row = merged[merged["Date"] == merged["Date"].max()]
        if not today_row.empty:
            kb = float(today_row["Foreign_Buy"].iloc[0] or 0)
            ks = float(today_row["Foreign_Sell"].iloc[0] or 0)
            net_twd = float(today_row.get("Foreign_Net_TWD", pd.Series([0])).iloc[0] or 0)
            logger.info(
                f"  {symbol}: {len(merged)} rows | "
                f"Today Foreign Flow: Buy={kb:,.0f} Sell={ks:,.0f} Net={net_twd:,.2f}M TWD"
            )
            
        if not dry_run:
            merged.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.success(f"  {symbol}: ✅ parquet updated successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"  {symbol}: Merge failed — {e}")
        return False


def run_taiwan_flow_cache(symbols: list[str], dry_run: bool = False) -> dict:
    """
    Main runner for Taiwan institutional flow data update.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d") # 60 days to catch all recent days
    
    logger.info(f"🌊 Taiwan Foreign Flow Cache: {len(symbols)} symbols | {start} → {end}")
    if dry_run:
        logger.warning("🔍 DRY RUN — no files will be written")
        
    success, failed, no_data = [], [], []
    
    for i, symbol in enumerate(symbols, 1):
        stock_code = symbol.replace(".TW", "").replace(".TWO", "").strip()
        logger.info(f"[{i:3d}/{len(symbols)}] Fetching {symbol} ({stock_code})...")
        
        try:
            flow_df = _fetch_foreign_flow_taiwan(stock_code, start, end)
            if flow_df is None or flow_df.empty:
                logger.warning(f"  {symbol}: No institutional flow data returned (market closed or API limits)")
                no_data.append(symbol)
                continue
                
            ok = _merge_flow_into_parquet(symbol, stock_code, flow_df, dry_run)
            if ok:
                success.append(symbol)
            else:
                failed.append(symbol)
                
        except Exception as e:
            logger.error(f"  {symbol}: Unexpected error — {e}")
            failed.append(symbol)
            
        # Throttle to respect API limits
        time.sleep(0.5)
        
    logger.info("=" * 60)
    logger.success(f"✅ Success: {len(success)} symbols")
    logger.info(f"⚪ No data: {len(no_data)} symbols (API limitation)")
    if failed:
        logger.warning(f"❌ Failed:  {len(failed)} symbols: {', '.join(failed[:20])}")
        
    return {"success": success, "no_data": no_data, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Nightly Taiwan Foreign Flow Cache — fetch TW institutional data → parquet",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols to process (default: all registered Taiwan stocks)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch data but do not write to disk"
    )
    args = parser.parse_args()
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        # Ensure correct suffix format
        symbols_with_suffix = []
        for s in symbols:
            if not s.endswith(".TW") and not s.endswith(".TWO"):
                if s in TAIWAN_STOCKS:
                    symbols_with_suffix.append(s)
                else:
                    symbols_with_suffix.append(f"{s}.TW")
            else:
                symbols_with_suffix.append(s)
        symbols = symbols_with_suffix
    else:
        symbols = list(TAIWAN_STOCKS.keys())
        
    run_taiwan_flow_cache(symbols=symbols, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
