"""
Nightly Foreign Flow Cache — scripts/nightly_foreign_flow_cache.py
===================================================================
Chạy sau khi thị trường đóng cửa (18:30 UTC+7) để fetch dữ liệu
Khối ngoại từng phiên từ vnstock và merge vào parquet cache hiện có.

Cách dùng:
    python scripts/nightly_foreign_flow_cache.py              # Tất cả VN
    python scripts/nightly_foreign_flow_cache.py --symbols BSR,PHR,VCB
    python scripts/nightly_foreign_flow_cache.py --dry-run    # Chỉ test, không ghi file

Tự động hóa (Windows Task Scheduler):
    Trigger: Daily lúc 18:30 (sau ATC 15:00 UTC+7)
    Action:  venv\\Scripts\\python.exe scripts\\nightly_foreign_flow_cache.py
    Start in: d:\\Python_VS\\trading_system

Dữ liệu ghi thêm vào parquet (merge theo Date):
    Foreign_Buy     : Khối lượng KN mua (cổ phiếu)
    Foreign_Sell    : Khối lượng KN bán (cổ phiếu)
    Block_Trade_Volume : Khối lượng thỏa thuận (cổ phiếu)
    Foreign_Net_VND : Giá trị mua ròng KN (triệu VND, ước tính = net_vol × close)
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
from loguru import logger

CACHE_DIR = ROOT / ".cache" / "prices"

# ── Symbols to update (same list as nightly_vn_cache.py) ─────────────────────
DEFAULT_VN_SYMBOLS = [
    # Banks
    "VCB","BID","CTG","TCB","MBB","ACB","VPB","STB","HDB",
    "LPB","SSB","EIB","OCB","TPB","SHB","EVF","VBB",
    # Real Estate
    "VIC","VHM","VRE","KDH","NVL","PDR","DXG","IJC","TDC",
    "SIP","HDC","HAG","LCG","VPI","CII","HDG","PC1","VCG","NLG","SZC",
    # Oil & Gas
    "PVS","PVD","PVT","PVB","PVI","GAS",
    # Industry & Materials
    "HPG","PLX","GVR","BSR","DGC","PHR","DPR","TRC",
    "BMP","AAA","LSS","PPC","NT2","POW","GEG","BWE","KHP",
    "REE","GEX","HHV","NKG","TLH","SMC","HSG",
    # Consumer & Retail
    "SAB","MSN","VNM","MWG","PNJ","TLG","DHC","DBC","PAN",
    "VHC","HAX","HAH","VTO","ASM","CSV","BHN",
    # Technology
    "FPT","CMG","VNE","SGT",
    # Aviation & Transport
    "VJC","GMD","HVN",
    # Financials / Securities
    "SSI","VCI","HCM","VND","BSI","ORS","CTS","FTS","VDS","MBS",
    # Construction
    "CTD","FCN","HBC",
    # Agriculture & Fertilizer
    "AGR","DPM","DCM","DDV",
    # Pharma
    "DHG","IMP","TRA",
    # Others
    "BVH","BCM","SCS","TDM","TV2","VGC",
]



# ── Foreign Flow Fetcher — vnstock Trading.price_board() ─────────────────────
# Confirmed working: Trading.price_board() returns MultiIndex DataFrame with:
#   ('match', 'foreign_buy_volume'), ('match', 'foreign_sell_volume'),
#   ('match', 'foreign_buy_value'), ('match', 'foreign_sell_value')
# This is the ONLY vnstock endpoint with foreign flow on Community/Free tier.
# It returns TODAY's intraday snapshot — suitable for nightly cache runs.

_VNSTOCK_API_KEY = "vnstock_ee8c180549c43fab65ea2396660d2051"


def _fetch_foreign_flow_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch today's foreign buy/sell snapshot via vnstock Trading.price_board().

    - Uses the system's VNSTOCK_API_KEY (Community tier, 60 req/min).
    - Returns a single-row DataFrame for TODAY's session.
    - Appended nightly: run every session close (21:00) to build history.
    - Returns None on failure (fail-closed — caller skips symbol silently).

    Columns returned: Date, Foreign_Buy, Foreign_Sell, Block_Trade_Volume
    """
    import os, warnings
    import warnings as _warn

    # Ensure API key is set
    os.environ.setdefault("VNSTOCK_API_KEY", _VNSTOCK_API_KEY)

    try:
        from vnstock import Trading, Vnstock
    except ImportError:
        logger.warning("vnstock not installed — cannot fetch foreign flow")
        return None

    today = pd.Timestamp(datetime.now().date())

    # ── Method 1: Trading.price_board() — has ('match','foreign_buy_volume') ──
    try:
        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            t = Trading(symbol=symbol, source="VCI")
            board = t.price_board()

        if board is not None and not board.empty:
            # Flatten MultiIndex columns: ('match', 'foreign_buy_volume') → 'match_foreign_buy_volume'
            if isinstance(board.columns, pd.MultiIndex):
                board.columns = [
                    f"{g}_{n}" if g != "" else n
                    for g, n in board.columns
                ]

            # Normalize column names (case-insensitive key lookup)
            col_lower = {c.lower(): c for c in board.columns}

            def _get(key_variants):
                for k in key_variants:
                    mapped = col_lower.get(k.lower())
                    if mapped and mapped in board.columns:
                        val = board[mapped].iloc[0]
                        return float(val) if pd.notna(val) else 0.0
                return 0.0

            f_buy  = _get(["match_foreign_buy_volume",  "foreign_buy_volume",  "foreignBuyVolume"])
            f_sell = _get(["match_foreign_sell_volume", "foreign_sell_volume", "foreignSellVolume"])
            pt_vol = _get(["match_accumulated_volume",  "match_vol",           "ptMatchVolume", "accumulated_volume"])

            logger.debug(f"  {symbol}: price_board → buy={f_buy:,.0f} sell={f_sell:,.0f}")

            if f_buy > 0 or f_sell > 0:
                return pd.DataFrame([{
                    "Date":               today,
                    "Foreign_Buy":        f_buy,
                    "Foreign_Sell":       f_sell,
                    "Block_Trade_Volume": pt_vol,
                }])
            else:
                logger.debug(f"  {symbol}: price_board returned zero foreign flow (market may be closed)")
                # Still return a row with zeros so we know the script ran
                return pd.DataFrame([{
                    "Date":               today,
                    "Foreign_Buy":        0.0,
                    "Foreign_Sell":       0.0,
                    "Block_Trade_Volume": pt_vol,
                }])

    except Exception as e:
        logger.debug(f"  {symbol}: Trading.price_board() failed — {e}")

    # ── Method 2: quote.history fallback — no foreign cols but confirms symbol ok ──
    try:
        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            stock = Vnstock().stock(symbol=symbol, source="VCI")
            df = stock.quote.history(start=start, end=end, interval="1D")

        if df is not None and not df.empty:
            # Check for foreign cols (would be present if API upgrades in future)
            col_map = {
                "foreignBuyVolume": "Foreign_Buy", "foreign_buy_volume": "Foreign_Buy",
                "fBuyVol": "Foreign_Buy",
                "foreignSellVolume": "Foreign_Sell", "foreign_sell_volume": "Foreign_Sell",
                "fSellVol": "Foreign_Sell",
                "ptMatchVolume": "Block_Trade_Volume",
            }
            df = df.rename(columns=col_map)
            flow_cols = [c for c in ["Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"] if c in df.columns]
            if flow_cols:
                df["Date"] = pd.to_datetime(df.get("Date", df.get("time"))).dt.tz_localize(None)
                for col in ["Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]:
                    if col not in df.columns:
                        df[col] = 0.0
                df_out = df[["Date", "Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]]
                df_out = df_out[(df_out["Date"] >= pd.Timestamp(start)) & (df_out["Date"] <= pd.Timestamp(end))]
                if not df_out.empty:
                    return df_out
    except Exception as e:
        logger.debug(f"  {symbol}: vnstock history fallback failed — {e}")

    return None







def _merge_flow_into_parquet(symbol: str, flow_df: pd.DataFrame, dry_run: bool) -> bool:

    """
    Merge foreign flow columns into existing price parquet file.
    Adds/updates columns: Foreign_Buy, Foreign_Sell, Block_Trade_Volume, Foreign_Net_VND.
    """
    parquet_path = CACHE_DIR / f"{symbol}_VN.parquet"
    if not parquet_path.exists():
        logger.warning(f"  {symbol}: parquet not found — run nightly_vn_cache.py first")
        return False

    try:
        existing = pd.read_parquet(parquet_path, engine="pyarrow")

        # Normalize Date column — strip time component for reliable join
        if "Date" in existing.columns:
            existing["Date"] = pd.to_datetime(existing["Date"]).dt.normalize().dt.tz_localize(None)
        elif existing.index.name == "Date":
            existing = existing.reset_index()
            existing["Date"] = pd.to_datetime(existing["Date"]).dt.normalize().dt.tz_localize(None)

        # Normalize flow dates too
        flow_df["Date"] = pd.to_datetime(flow_df["Date"]).dt.normalize().dt.tz_localize(None)

        # ── Save today's flow to sidecar JSON so it survives until price cache updates ──
        # Sidecar: .cache/prices/{symbol}_VN.flow.json
        sidecar_path = parquet_path.with_suffix(".flow.json")
        try:
            import json as _json
            sidecar_records = flow_df.copy()
            sidecar_records["Date"] = sidecar_records["Date"].astype(str)
            sidecar_records.to_json(sidecar_path, orient="records", date_format="iso")
        except Exception as _e:
            logger.debug(f"  {symbol}: sidecar write failed — {_e}")

        # ── Also apply any previously saved sidecar rows that now have price data ──
        if sidecar_path.exists():
            try:
                import json as _json
                sc = pd.read_json(sidecar_path, orient="records")
                sc["Date"] = pd.to_datetime(sc["Date"]).dt.normalize().dt.tz_localize(None)
                # Union: merge sidecar + today's flow, keep latest values per date
                flow_df = pd.concat([sc, flow_df]).drop_duplicates(subset=["Date"], keep="last")
                flow_df = flow_df.reset_index(drop=True)
            except Exception as _e:
                logger.debug(f"  {symbol}: sidecar read failed — {_e}")

        # LEFT join — only update rows that already have price data
        # Rows without price (e.g. today before nightly_vn_cache) are NOT added
        # This prevents NaN-price rows that break candlestick charts
        merged = existing.merge(
            flow_df[["Date", "Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]],
            on="Date", how="left", suffixes=("_old", "")
        )

        # Resolve column conflicts after merge
        for col in ["Foreign_Buy", "Foreign_Sell", "Block_Trade_Volume"]:
            if f"{col}_old" in merged.columns:
                # Prefer new data, fall back to old
                merged[col] = merged[col].fillna(merged[f"{col}_old"])
                merged.drop(columns=[f"{col}_old"], inplace=True)
            if col not in merged.columns:
                merged[col] = 0.0


        # Compute Foreign_Net_VND (million VND): net_vol × avg_price / 1e6
        if "Close" in merged.columns:
            net_vol = merged["Foreign_Buy"].fillna(0) - merged["Foreign_Sell"].fillna(0)
            close = pd.to_numeric(merged["Close"], errors="coerce").fillna(0)
            merged["Foreign_Net_VND"] = (net_vol * close / 1_000_000).round(3)
        else:
            merged["Foreign_Net_VND"] = 0.0

        merged = merged.sort_values("Date").reset_index(drop=True)

        # Count rows with actual flow data (buy or sell > 0)
        n_flow = int(((merged["Foreign_Buy"].fillna(0) > 0) | (merged["Foreign_Sell"].fillna(0) > 0)).sum())
        # Count rows where flow columns exist (not NaN)
        n_recorded = int(merged["Foreign_Buy"].notna().sum())

        # Show today's KN data in the log
        today_row = merged[merged["Date"] == merged["Date"].max()]
        if not today_row.empty:
            kb = float(today_row["Foreign_Buy"].iloc[0] or 0)
            ks = float(today_row["Foreign_Sell"].iloc[0] or 0)
            kn_net = float(today_row.get("Foreign_Net_VND", pd.Series([0])).iloc[0] or 0)
            logger.info(
                f"  {symbol}: {len(merged)} rows | {n_recorded} recorded | "
                f"Today KN: buy={kb:,.0f} sell={ks:,.0f} net={kn_net:,.1f}M VND"
            )
        else:
            logger.info(f"  {symbol}: {len(merged)} rows total, {n_recorded} with KN data recorded")


        if not dry_run:
            merged.to_parquet(parquet_path, engine="pyarrow", index=False)
            logger.success(f"  {symbol}: ✅ parquet updated")

        return True

    except Exception as e:
        logger.error(f"  {symbol}: merge failed — {e}")
        return False


def run_foreign_flow_cache(symbols: list[str], dry_run: bool = False) -> dict:
    """
    Main runner: fetch foreign flow for each symbol and merge into parquet cache.
    Returns summary dict.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    logger.info(f"🌊 Foreign Flow Cache: {len(symbols)} symbols | {start} → {end}")
    if dry_run:
        logger.warning("🔍 DRY RUN — no files will be written")

    success, failed, no_data = [], [], []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i:3d}/{len(symbols)}] Fetching {symbol}...")
        try:
            flow_df = _fetch_foreign_flow_vnstock(symbol, start, end)
            if flow_df is None or flow_df.empty:
                logger.warning(f"  {symbol}: no foreign flow data returned (API may not support it)")
                no_data.append(symbol)
                continue

            ok = _merge_flow_into_parquet(symbol, flow_df, dry_run)
            if ok:
                success.append(symbol)
            else:
                failed.append(symbol)

        except Exception as e:
            logger.error(f"  {symbol}: unexpected error — {e}")
            failed.append(symbol)

        # Throttle to avoid rate limits
        time.sleep(0.5)

    logger.info("=" * 60)
    logger.success(f"✅ Success: {len(success)} symbols")
    logger.info(f"⚪ No data: {len(no_data)} symbols (API limitation)")
    if failed:
        logger.warning(f"❌ Failed:  {len(failed)} symbols: {', '.join(failed[:20])}")

    return {"success": success, "no_data": no_data, "failed": failed}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nightly Foreign Flow Cache — fetch KN data from vnstock → parquet",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols to process (default: all VN stocks)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch data but do not write to disk"
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = DEFAULT_VN_SYMBOLS

    run_foreign_flow_cache(symbols=symbols, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
