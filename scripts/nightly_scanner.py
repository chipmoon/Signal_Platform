"""
Nightly Stock Scanner — Composite 6-Month Opportunity Ranker
=============================================================
Quét toàn bộ cổ phiếu TW + VN, tính điểm Composite 6-tháng,
xuất Excel ranking để tìm cổ phiếu tiềm năng tăng giá.

Cách dùng:
    python scripts/nightly_scanner.py              # Scan tất cả
    python scripts/nightly_scanner.py --market VN  # Chỉ VN
    python scripts/nightly_scanner.py --market TW  # Chỉ TW
    python scripts/nightly_scanner.py --top 30     # Top 30 cổ phiếu
    python scripts/nightly_scanner.py --out results/scan_today.xlsx

Composite Score (6-tháng):
    35% Fundamental Quality   (ROE, D/E, EPS growth, P/E, FCF)
    30% Wyckoff + SMC         (phase, structure, signal)
    20% MTF Confluence        (Weekly/Daily/4H alignment)
    15% Elliott Wave Target   (upside potential vs. current price)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import pandas as pd
from loguru import logger

# ── Optional rich progress bar ────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

# ── Analytics engines ─────────────────────────────────────────────────────────
from src.analytics.fundamental_score import compute_fundamental_score
from src.analytics.mtf_confluence import compute_mtf_confluence
from src.strategies.wyckoff_analyzer import WyckoffAnalyzer, WyckoffConfig
from src.strategies.smc_analyzer import SmcAnalyzer, SmcConfig
from src.cache_manager import cache as cm

try:
    from src.strategies.elliott_wave import ElliottWaveAnalyzer
    EW_AVAILABLE = True
except ImportError:
    EW_AVAILABLE = False
    logger.warning("Elliott Wave module not available — EW score will be 0")

# ── Stock universes ───────────────────────────────────────────────────────────
from src.plugins.taiwan import TAIWAN_STOCKS  # dict: symbol -> {Name, Sector}

VN_SYMBOLS = [
    # Banks
    "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "STB", "HDB",
    "LPB", "SSB", "EIB", "OCB", "TPB", "SHB", "EVF",
    # Real Estate
    "VIC", "VHM", "VRE", "KDH", "NVL", "PDR", "DXG", "IJC", "TDC",
    "SIP", "HDC", "HAG", "LCG", "VPI", "CII", "HDG", "PC1", "VCG", "NLG", "SZC",
    # Oil & Gas
    "PVS", "PVD", "PVT", "PVB", "PVI", "GAS",
    # Industry & Materials
    "HPG", "PLX", "GVR", "BSR", "DGC", "PHR", "DPR", "TRC",
    "BMP", "AAA", "LSS", "PPC", "NT2", "POW", "GEG", "BWE", "KHP",
    "REE", "GEX", "HHV", "NKG", "TLH", "SMC", "HSG",
    # Consumer & Retail
    "SAB", "MSN", "VNM", "MWG", "PNJ", "TLG", "DHC", "DBC", "PAN",
    "VHC", "HAX", "HAH", "VTO", "ASM", "CSV", "BHN",
    # Technology
    "FPT", "CMG", "VNE", "SGT",
    # Aviation & Transport
    "VJC", "GMD", "HVN",
    # Financials / Securities
    "SSI", "VCI", "HCM", "VND", "BSI", "ORS", "CTS", "FTS", "VDS", "MBS",
    # Construction
    "CTD", "FCN", "HBC",
    # Agriculture & Fertilizer
    "AGR", "DPM", "DCM", "DDV",
    # Pharma
    "DHG", "IMP", "TRA",
    # Others
    "BVH", "BCM", "SCS", "TDM", "TV2", "VGC",
]

# ── Composite weights (6-month horizon) ───────────────────────────────────────
W_FUNDAMENTAL = 0.35
W_WYCKOFF_SMC = 0.30
W_MTF         = 0.20
W_ELLIOTT     = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load price data from cache
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_df(symbol: str, market: str) -> Optional[pd.DataFrame]:
    """Load price DataFrame from .cache/prices/ parquet."""
    safe = symbol.replace(".", "_").replace("-", "_")
    path = ROOT / ".cache" / "prices" / f"{safe}_{market}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        # Normalize columns
        col_map = {c: c.capitalize() for c in df.columns}
        col_map.update({"Adj close": "Close", "Adj Close": "Close"})
        df = df.rename(columns=col_map)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        return df if len(df) >= 60 else None
    except Exception as e:
        logger.debug(f"Price load failed for {symbol}: {e}")
        return None


def _fetch_price_from_api(symbol: str, market: str) -> Optional[pd.DataFrame]:
    """Fallback: fetch from yfinance if cache missing."""
    try:
        import yfinance as yf
        if market == "VN":
            ticker_str = f"{symbol}.VN"
        elif market == "TW":
            ticker_str = symbol  # already contains .TW or .TWO
        else:
            ticker_str = symbol

        df = yf.Ticker(ticker_str).history(period="1y", interval="1d", auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]] if len(df) >= 60 else None
    except Exception:
        return None


def _get_price_df(symbol: str, market: str) -> Optional[pd.DataFrame]:
    """Cache-first price loading."""
    df = _load_price_df(symbol, market)
    if df is not None:
        return df
    logger.debug(f"  {symbol}: no cache, trying API...")
    return _fetch_price_from_api(symbol, market)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring engines
# ─────────────────────────────────────────────────────────────────────────────

def score_fundamental(symbol: str, market: str) -> tuple[float, int, str]:
    """
    Returns (normalized_score 0-1, raw_score 0-100, grade).
    Uses cache-first strategy inside compute_fundamental_score.
    """
    try:
        fs = compute_fundamental_score(symbol, market)
        return fs.total_score / 100.0, fs.total_score, fs.grade
    except Exception as e:
        logger.debug(f"  {symbol} fundamental error: {e}")
        return 0.0, 0, "N/A"


def score_wyckoff_smc(df: pd.DataFrame) -> tuple[float, str, float]:
    """
    Returns (normalized_score 0-1, wyckoff_phase, smc_score_raw).
    Wyckoff: Accumulation/Markup → bullish, Distribution/Markdown → bearish.
    SMC: smc_score in [-1, +1] range.
    """
    try:
        wa = WyckoffAnalyzer(WyckoffConfig())
        wy = wa.analyze_current_state(df)
        w_score = float(wy.get("score", 0.0))        # typically -1 to +1
        w_phase = str(wy.get("phase", "Unknown"))

        smc = SmcAnalyzer(SmcConfig()).get_current_state(df)
        smc_raw = float(smc.get("smc_score", 0.0))   # typically -1 to +1

        # Combine: 60% Wyckoff + 40% SMC, normalize to 0-1
        combined = (w_score * 0.6 + smc_raw * 0.4 + 1.0) / 2.0  # map [-1,1] → [0,1]
        combined = max(0.0, min(1.0, combined))
        return combined, w_phase, smc_raw
    except Exception as e:
        logger.debug(f"  Wyckoff/SMC error: {e}")
        return 0.5, "Unknown", 0.0  # neutral default


def score_mtf(symbol: str, market: str, df_daily: pd.DataFrame) -> tuple[float, str, int]:
    """
    Returns (normalized_score 0-1, confluence_label, alignment 0-3).
    confluence_score range is -9 to +9 → normalize to 0-1.
    """
    try:
        mtf = compute_mtf_confluence(symbol, market, df_daily=df_daily)
        raw = float(mtf.get("confluence_score", 0.0))  # -9 to +9
        normalized = (raw + 9.0) / 18.0                # → 0 to 1
        normalized = max(0.0, min(1.0, normalized))
        label = mtf.get("confluence_label", "Neutral")
        alignment = int(mtf.get("alignment", 0))
        return normalized, label, alignment
    except Exception as e:
        logger.debug(f"  MTF error: {e}")
        return 0.5, "Unknown", 0


def score_elliott(df: pd.DataFrame) -> tuple[float, float]:
    """
    Returns (normalized_score 0-1, upside_pct).
    Measures how much upside the Elliott Wave target implies vs. current price.
    """
    if not EW_AVAILABLE or df is None or len(df) < 50:
        return 0.5, 0.0  # neutral default

    try:
        ew = ElliottWaveAnalyzer()
        # Try weekly resample first
        df_ew = df.copy()
        if "Date" in df_ew.columns:
            df_ew = df_ew.set_index("Date")
        df_weekly = df_ew.resample("W").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna().reset_index()
        df_weekly.columns = [str(c) for c in df_weekly.columns]

        state = ew.get_current_state(df_weekly) if len(df_weekly) >= 20 else {}
        target = float(state.get("target_price", 0.0))
        curr = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
        bias = str(state.get("bias", "Neutral"))
        confidence = float(state.get("confidence", 0)) / 100.0

        if target > 0 and curr > 0:
            upside_pct = (target - curr) / curr * 100.0
        else:
            upside_pct = 0.0

        # Score: bullish bias + upside potential, weighted by confidence
        if "Bullish" in bias and upside_pct > 0:
            raw = min(upside_pct / 50.0, 1.0) * confidence  # 50% upside = max score
            score = 0.5 + raw * 0.5   # range 0.5-1.0 for bullish
        elif "Bearish" in bias:
            score = max(0.0, 0.5 - confidence * 0.3)
        else:
            score = 0.5

        return score, upside_pct
    except Exception as e:
        logger.debug(f"  Elliott Wave error: {e}")
        return 0.5, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_symbol(symbol: str, market: str, name: str, sector: str) -> Optional[dict]:
    """Run full composite scoring for one symbol. Returns result dict or None."""
    # 1. Load price data
    df = _get_price_df(symbol, market)
    if df is None:
        logger.warning(f"  {symbol}: no price data — skipping")
        return None

    current_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0.0
    if current_price <= 0:
        return None

    # 2. Score each component
    fund_norm, fund_raw, fund_grade = score_fundamental(symbol, market)
    wy_smc_norm, wy_phase, smc_raw = score_wyckoff_smc(df)
    mtf_norm, mtf_label, mtf_align = score_mtf(symbol, market, df)
    ew_norm, upside_pct = score_elliott(df)

    # 3. Composite score (0-100)
    composite = (
        fund_norm  * W_FUNDAMENTAL +
        wy_smc_norm * W_WYCKOFF_SMC +
        mtf_norm   * W_MTF +
        ew_norm    * W_ELLIOTT
    ) * 100.0

    # 4. Opportunity rating
    if composite >= 72:
        rating = "⭐⭐⭐ Strong Buy"
    elif composite >= 60:
        rating = "⭐⭐ Watch"
    elif composite >= 45:
        rating = "⭐ Neutral"
    else:
        rating = "⚠️ Avoid"

    return {
        "Symbol":           symbol,
        "Name":             name,
        "Market":           market,
        "Sector":           sector,
        "Price":            round(current_price, 2),
        # Composite
        "Composite Score":  round(composite, 1),
        "Rating":           rating,
        # Fundamental
        "Fund Score":       fund_raw,
        "Fund Grade":       fund_grade,
        # Wyckoff + SMC
        "Wyckoff Phase":    wy_phase,
        "SMC Score":        round(smc_raw, 3),
        "W+SMC Score":      round(wy_smc_norm * 100, 1),
        # MTF
        "MTF Label":        mtf_label.replace("🟢 ", "").replace("🔴 ", "").replace("⚪ ", ""),
        "MTF Alignment":    mtf_align,
        "MTF Score":        round(mtf_norm * 100, 1),
        # Elliott
        "EW Upside %":      round(upside_pct, 1),
        "EW Score":         round(ew_norm * 100, 1),
    }


def _build_universe(market_filter: Optional[str]) -> list[tuple[str, str, str, str]]:
    """Returns list of (symbol, market, name, sector)."""
    universe = []

    if market_filter in (None, "TW"):
        for sym, info in TAIWAN_STOCKS.items():
            universe.append((sym, "TW", info.get("Name", sym), info.get("Sector", "Other")))

    if market_filter in (None, "VN"):
        for sym in VN_SYMBOLS:
            universe.append((sym, "VN", sym, "VN Stock"))

    return universe


def _style_excel(writer: pd.ExcelWriter, df_all: pd.DataFrame, df_vn: pd.DataFrame, df_tw: pd.DataFrame):
    """Apply conditional formatting and column widths to Excel sheets."""
    try:
        wb = writer.book
        green_fill = {"type": "3_color_scale", "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B"}

        for sheet_name, df in [("All Stocks", df_all), ("VN Ranking", df_vn), ("TW Ranking", df_tw)]:
            if df.empty or sheet_name not in writer.sheets:
                continue
            ws = writer.sheets[sheet_name]
            # Auto-width columns
            for col_idx, col in enumerate(df.columns, 1):
                max_len = max(len(str(col)), df[col].astype(str).str.len().max())
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 2, 30)

            # Color Composite Score column
            if "Composite Score" in df.columns:
                score_col_idx = list(df.columns).index("Composite Score") + 1
                col_letter = ws.cell(1, score_col_idx).column_letter
                ws.conditional_formatting.add(
                    f"{col_letter}2:{col_letter}{len(df)+1}",
                    __import__("openpyxl.formatting.rule", fromlist=["ColorScaleRule"]).ColorScaleRule(
                        start_type="min", start_color="F8696B",
                        mid_type="percentile", mid_value=50, mid_color="FFEB84",
                        end_type="max", end_color="63BE7B",
                    )
                )
    except Exception as e:
        logger.debug(f"Excel styling error (non-critical): {e}")


def run_scan(market_filter: Optional[str], top_n: Optional[int], output_path: Path) -> pd.DataFrame:
    universe = _build_universe(market_filter)
    total = len(universe)
    results = []
    failed = []

    logger.info(f"🔍 Starting scan: {total} symbols | Market: {market_filter or 'ALL'}")
    logger.info(f"   Weights — Fundamental: {W_FUNDAMENTAL*100:.0f}% | Wyckoff+SMC: {W_WYCKOFF_SMC*100:.0f}% | MTF: {W_MTF*100:.0f}% | Elliott: {W_ELLIOTT*100:.0f}%")

    if RICH_AVAILABLE:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        )
    else:
        progress_ctx = None

    def _process():
        for i, (symbol, market, name, sector) in enumerate(universe, 1):
            desc = f"[{i:3d}/{total}] {symbol:<12} ({market})"
            if RICH_AVAILABLE:
                task = progress_ctx.add_task(desc, total=1)

            try:
                result = scan_symbol(symbol, market, name, sector)
                if result:
                    results.append(result)
                    if RICH_AVAILABLE:
                        progress_ctx.update(task, advance=1, description=f"✅ {desc} → {result['Composite Score']:.0f}")
                else:
                    failed.append(symbol)
                    if RICH_AVAILABLE:
                        progress_ctx.update(task, advance=1, description=f"⚠️  {desc} → no data")
            except Exception as e:
                failed.append(symbol)
                logger.debug(f"  SCAN ERROR {symbol}: {e}")
                if RICH_AVAILABLE:
                    progress_ctx.update(task, advance=1, description=f"❌ {desc}")

            # Small throttle to avoid rate limiting
            time.sleep(0.3)

    if RICH_AVAILABLE:
        with progress_ctx:
            _process()
    else:
        _process()

    if not results:
        logger.error("No results collected — check cache or connectivity")
        return pd.DataFrame()

    # Sort by Composite Score descending
    df = pd.DataFrame(results).sort_values("Composite Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    df_vn = df[df["Market"] == "VN"].reset_index(drop=True)
    df_vn.insert(0, "VN Rank", range(1, len(df_vn) + 1))

    df_tw = df[df["Market"] == "TW"].reset_index(drop=True)
    df_tw.insert(0, "TW Rank", range(1, len(df_tw) + 1))

    df_top = df.head(top_n) if top_n else df

    # ── Export to Excel ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Top Picks across all markets
        df_top.to_excel(writer, sheet_name="🏆 Top Picks", index=False)

        # Sheet 2: VN Ranking
        df_vn.to_excel(writer, sheet_name="VN Ranking", index=False)

        # Sheet 3: TW Ranking
        df_tw.to_excel(writer, sheet_name="TW Ranking", index=False)

        # Sheet 4: All Stocks (full)
        df.to_excel(writer, sheet_name="All Stocks", index=False)

        # Sheet 5: Metadata
        meta = pd.DataFrame([{
            "Run Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Total Scanned": total,
            "Successful": len(results),
            "Failed": len(failed),
            "Failed Symbols": ", ".join(failed[:30]),
            "Fundamental Weight": f"{W_FUNDAMENTAL*100:.0f}%",
            "Wyckoff+SMC Weight": f"{W_WYCKOFF_SMC*100:.0f}%",
            "MTF Weight": f"{W_MTF*100:.0f}%",
            "Elliott Wave Weight": f"{W_ELLIOTT*100:.0f}%",
            "Score Threshold ⭐⭐⭐": "≥ 72",
            "Score Threshold ⭐⭐": "≥ 60",
        }])
        meta.T.reset_index().rename(columns={"index": "Parameter", 0: "Value"}).to_excel(
            writer, sheet_name="Scan Info", index=False
        )

        _style_excel(writer, df, df_vn, df_tw)

    logger.success(f"✅ Scan complete! {len(results)}/{total} symbols scored")
    logger.success(f"📊 Excel saved → {output_path}")

    # ── Print top 15 to terminal ──────────────────────────────────────────────
    top15 = df.head(15)
    if RICH_AVAILABLE:
        table = Table(title=f"🏆 Top 15 — Composite Score [{datetime.now().strftime('%Y-%m-%d')}]",
                      show_header=True, header_style="bold cyan")
        for col in ["Rank", "Symbol", "Market", "Composite Score", "Rating",
                    "Fund Grade", "Wyckoff Phase", "MTF Label", "EW Upside %"]:
            table.add_column(col, justify="right" if col in ("Rank", "Composite Score", "EW Upside %") else "left")
        for _, row in top15.iterrows():
            table.add_row(
                str(int(row["Rank"])), row["Symbol"], row["Market"],
                f"{row['Composite Score']:.1f}", row["Rating"],
                row["Fund Grade"], str(row["Wyckoff Phase"]),
                row["MTF Label"], f"{row['EW Upside %']:+.1f}%",
            )
        console.print(table)
    else:
        print(f"\n{'='*80}")
        print(f"TOP 15 — Composite Score [{datetime.now().strftime('%Y-%m-%d')}]")
        print(f"{'='*80}")
        print(top15[["Rank", "Symbol", "Market", "Composite Score", "Rating",
                      "Fund Grade", "Wyckoff Phase", "MTF Label", "EW Upside %"]].to_string(index=False))
        print(f"{'='*80}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nightly Stock Scanner — Composite 6-Month Ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--market", choices=["VN", "TW"], default=None,
                        help="Filter to specific market (default: both)")
    parser.add_argument("--top", type=int, default=None,
                        help="Show/export only top N stocks in 'Top Picks' sheet")
    parser.add_argument("--out", type=str, default=None,
                        help="Output Excel path (default: output/scan_YYYYMMDD.xlsx)")
    parser.add_argument("--no-excel", action="store_true",
                        help="Skip Excel export, print to terminal only")
    args = parser.parse_args()

    # Default output path
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    if args.out:
        output_path = Path(args.out)
    else:
        output_path = ROOT / "output" / f"scan_{date_str}.xlsx"

    start = time.time()
    df = run_scan(
        market_filter=args.market,
        top_n=args.top,
        output_path=output_path,
    )
    elapsed = time.time() - start

    if not df.empty:
        strong_buys = df[df["Rating"].str.contains("Strong Buy", na=False)]
        watch = df[df["Rating"].str.contains("Watch", na=False)]
        logger.info(f"⏱️  Total time: {elapsed/60:.1f} min")
        logger.info(f"⭐⭐⭐ Strong Buy candidates: {len(strong_buys)}")
        logger.info(f"⭐⭐  Watch list: {len(watch)}")

        if not strong_buys.empty:
            syms = strong_buys["Symbol"].tolist()
            logger.success(f"💡 Strong Buy symbols: {', '.join(syms[:20])}")
            logger.info("→ Copy any symbol above and paste into AI-Forecast to deep-dive analysis")


if __name__ == "__main__":
    main()
