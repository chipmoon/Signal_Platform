"""
Fundamental Quality Score
==========================
Warren Buffett / Lynch style quality scoring (0–100).
Uses publicly available data from Yahoo Finance (yfinance ticker.info).

Score Components (total 100 pts):
    ROE ≥ 15%             → +20 pts   (capital efficiency)
    Debt/Equity ≤ 0.5     → +15 pts   (financial health)
    EPS Growth 3Y ≥ 10%   → +20 pts   (earnings momentum)
    P/E ≤ sector avg      → +15 pts   (reasonable valuation)
    FCF Yield ≥ 3%        → +15 pts   (real cash generation)
    Revenue Growth ≥ 8%   → +10 pts   (topline expansion)
    Profit Margin ≥ 10%   → +5 pts    (operational efficiency)

Interpretation:
    90-100: Exceptional quality (Buffett-grade)
    70-89:  High quality
    50-69:  Average quality
    30-49:  Below average
    0-29:   Low quality / speculative

Notes:
    - VN stocks: Limited data from yfinance — will fallback gracefully
    - TW stocks: Moderate coverage
    - US stocks: Full coverage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf
from loguru import logger


# ─── Data Structure ───────────────────────────────────────────────────────────

@dataclass
class FundamentalScore:
    """Fundamental quality assessment result."""
    total_score: int = 0                     # 0-100
    grade: str = "N/A"                       # A+, A, B, C, D, F
    label: str = "Unknown"                   # "High Quality", etc.

    # Individual scores
    roe_score: int = 0
    debt_score: int = 0
    eps_growth_score: int = 0
    valuation_score: int = 0
    fcf_score: int = 0
    revenue_growth_score: int = 0
    margin_score: int = 0

    # Raw values fetched
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    eps_growth: Optional[float] = None
    pe_ratio: Optional[float] = None
    fcf_yield: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None

    # Data availability
    data_coverage: int = 0                   # 0-100% how much data was available
    caveats: list[str] = field(default_factory=list)


# ─── Scorer ───────────────────────────────────────────────────────────────────

def _to_yf_ticker(symbol: str, market: str) -> str:
    if market == "VN":
        base_symbol = symbol.replace('.VN', '').upper()
        return f"{base_symbol}.VN"
    elif market == "TW":
        base_symbol = symbol.replace('.TW', '').replace('.TWO', '').upper()
        try:
            from src.plugins.taiwan import TAIWAN_STOCKS
            for curated_symbol in TAIWAN_STOCKS:
                if curated_symbol.replace('.TW', '').replace('.TWO', '').upper() == base_symbol:
                    return curated_symbol
        except Exception:
            pass
        return f"{base_symbol}.TW"
    return symbol


def _safe_float(val, default=None) -> Optional[float]:
    try:
        f = float(val)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def _fetch_vn_fundamentals_from_vnstock(symbol: str) -> dict:
    """Fetch and calculate all 7 Buffett-style fundamental metrics for VN stocks."""
    base_symbol = symbol.replace('.VN', '').upper()
    data = {
        "roe": None,
        "debt_to_equity": None,
        "eps_growth": None,
        "pe_ratio": None,
        "fcf_yield": None,
        "revenue_growth": None,
        "profit_margin": None
    }
    
    import time
    import io
    import contextlib

    def _safe_err(e: Exception) -> str:
        try:
            return str(e).encode("ascii", errors="ignore").decode("ascii") or e.__class__.__name__
        except Exception:
            return e.__class__.__name__

    for attempt in range(3):
        try:
            from vnstock import Vnstock
            import pandas as pd
            
            # Temporarily redirect stdout/stderr to suppress vnstock banner print and avoid UnicodeEncodeError on Windows
            f_out = io.StringIO()
            f_err = io.StringIO()
            with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                stock = Vnstock().stock(symbol=base_symbol, source="VCI")
                inc = stock.finance.income_statement(period="quarter", lang="vi")
                bs = stock.finance.balance_sheet(period="quarter", lang="vi")
                cf = stock.finance.cash_flow(period="quarter", lang="vi")
            
            # Verify we got valid dataframes (not empty/None)
            if inc is not None and not inc.empty and bs is not None and not bs.empty and cf is not None and not cf.empty:
                break
            else:
                raise ValueError("One or more financial statements returned empty")
        except BaseException as e:
            err_str = _safe_err(e)
            logger.warning(f"Attempt {attempt+1} failed for {base_symbol} (possibly rate limit/blocked): {err_str}")
            if attempt < 2:
                logger.info("Waiting 15 seconds to reset rate limit...")
                time.sleep(15)
            else:
                return data

    try:
        meta_cols = {'item_id', 'item_en', 'item'}
        data_cols = [c for c in inc.columns if c not in meta_cols]
        data_cols = sorted(data_cols, reverse=True)
        if not data_cols:
            return data

        def get_row_values(df, item_id):
            row = df[df['item_id'] == item_id]
            if row.empty:
                return None
            return {col: float(row[col].values[0]) for col in data_cols if pd.notna(row[col].values[0])}

        net_sales = get_row_values(inc, 'isa3') or get_row_values(inc, 'isa1')
        net_profit = get_row_values(inc, 'isa20')
        eps = get_row_values(inc, 'isa23')
        total_assets = get_row_values(bs, 'bsa53')
        equity = get_row_values(bs, 'bsa78')
        ocf = get_row_values(cf, 'cfa18')
        capex = get_row_values(cf, 'cfa19')

        def ttm_sum(values_dict):
            if not values_dict:
                return None
            vals = [values_dict[q] for q in data_cols if q in values_dict]
            if len(vals) < 4:
                return sum(vals) * (4.0 / len(vals)) if vals else None
            return sum(vals[:4])

        ttm_sales = ttm_sum(net_sales)
        ttm_profit = ttm_sum(net_profit)
        ttm_ocf = ttm_sum(ocf)
        ttm_capex = ttm_sum(capex)

        latest_q = data_cols[0]
        oldest_q = data_cols[-1]

        latest_equity = equity.get(latest_q) if equity else None
        latest_assets = total_assets.get(latest_q) if total_assets else None

        # 1. ROE = TTM Profit / Latest Equity
        if ttm_profit is not None and latest_equity:
            data["roe"] = ttm_profit / latest_equity

        # 2. Debt to Equity = (Total Assets - Equity) / Equity
        if latest_assets and latest_equity:
            data["debt_to_equity"] = (latest_assets - latest_equity) / latest_equity

        # 3. EPS Growth = Profit Growth between oldest and latest quarter
        if net_profit and latest_q in net_profit and oldest_q in net_profit:
            p_latest = net_profit[latest_q]
            p_oldest = net_profit[oldest_q]
            if p_oldest != 0:
                data["eps_growth"] = (p_latest - p_oldest) / abs(p_oldest)

        # 4. Revenue Growth
        if net_sales and latest_q in net_sales and oldest_q in net_sales:
            s_latest = net_sales[latest_q]
            s_oldest = net_sales[oldest_q]
            if s_oldest != 0:
                data["revenue_growth"] = (s_latest - s_oldest) / abs(s_oldest)

        # 5. Profit Margin
        if ttm_profit is not None and ttm_sales:
            data["profit_margin"] = ttm_profit / ttm_sales

        # 6. Free Cash Flow
        fcf = None
        if ttm_ocf is not None and ttm_capex is not None:
            fcf = ttm_ocf + ttm_capex # CapEx is negative in CF report

        # 7. FCF Yield & PE Ratio need marketCap / price from yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{base_symbol}.VN")
            mkt_cap = ticker.info.get("marketCap")
            price = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose")
            
            if fcf is not None and mkt_cap and mkt_cap > 0:
                data["fcf_yield"] = fcf / mkt_cap
                
            if price and ttm_profit and mkt_cap:
                shares = mkt_cap / price
                if shares > 0:
                    ttm_eps = ttm_profit / shares
                    if ttm_eps > 0:
                        data["pe_ratio"] = price / ttm_eps
        except Exception as e:
            logger.debug(f"Failed to get price/marketcap from yfinance for {base_symbol}: {e}")

    except Exception as e:
        logger.warning(f"Error computing vnstock fundamentals for {base_symbol}: {e}")

    return data


def compute_fundamental_score(symbol: str, market: str) -> FundamentalScore:
    """
    Compute Fundamental Quality Score for a given symbol.
    Returns FundamentalScore dataclass.
    """
    fs = FundamentalScore()
    yf_ticker = _to_yf_ticker(symbol, market)
    
    # 1. Try local cache first
    from src.cache_manager import cache
    cached = cache.get_cached_fundamentals(symbol, market)
    
    if cached:
        fs.roe = _safe_float(cached.get("roe"))
        fs.debt_to_equity = _safe_float(cached.get("debt_to_equity"))
        fs.eps_growth = _safe_float(cached.get("eps_growth"))
        fs.pe_ratio = _safe_float(cached.get("pe_ratio"))
        fs.fcf_yield = _safe_float(cached.get("fcf_yield"))
        fs.revenue_growth = _safe_float(cached.get("revenue_growth"))
        fs.profit_margin = _safe_float(cached.get("profit_margin"))
        logger.info(f"Loaded fundamentals from cache for {symbol} ({market})")
    else:
        # 2. Fetch fresh data
        data = None
        if market == "VN":
            data = _fetch_vn_fundamentals_from_vnstock(symbol)
            # Check if we successfully fetched anything
            if any(v is not None for v in data.values()):
                cache.cache_fundamentals(symbol, market, data)
        else:
            try:
                from src.plugins.taiwan import _safe_yfinance_env
                with _safe_yfinance_env():
                    ticker = yf.Ticker(yf_ticker)
                    info = ticker.info
                if info and info.get("symbol") is not None:
                    # Normalize yfinance values
                    roe = _safe_float(info.get("returnOnEquity"))
                    debt_to_equity = _safe_float(info.get("debtToEquity"))
                    if debt_to_equity is not None and debt_to_equity > 10:
                        debt_to_equity /= 100.0  # Convert % to ratio
                        
                    pe_ratio = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
                    profit_margin = _safe_float(info.get("profitMargins"))
                    revenue_growth = _safe_float(info.get("revenueGrowth"))
                    
                    eps_growth = _safe_float(info.get("earningsGrowth"))
                    if eps_growth is None:
                        trailing = _safe_float(info.get("trailingEps"))
                        forward  = _safe_float(info.get("forwardEps"))
                        if trailing and forward and trailing != 0:
                            eps_growth = (forward - trailing) / abs(trailing)
                            
                    fcf = _safe_float(info.get("freeCashflow"))
                    mkt_cap = _safe_float(info.get("marketCap"))
                    fcf_yield = None
                    if fcf and mkt_cap and mkt_cap > 0:
                        fcf_yield = fcf / mkt_cap
                        
                    data = {
                        "roe": roe,
                        "debt_to_equity": debt_to_equity,
                        "eps_growth": eps_growth,
                        "pe_ratio": pe_ratio,
                        "fcf_yield": fcf_yield,
                        "revenue_growth": revenue_growth,
                        "profit_margin": profit_margin
                    }
                    cache.cache_fundamentals(symbol, market, data)
            except Exception as e:
                logger.warning(f"FundamentalScore: could not fetch {yf_ticker}: {e}")
                fs.caveats.append("Yahoo Finance data unavailable")
                
        if data:
            fs.roe = _safe_float(data.get("roe"))
            fs.debt_to_equity = _safe_float(data.get("debt_to_equity"))
            fs.eps_growth = _safe_float(data.get("eps_growth"))
            fs.pe_ratio = _safe_float(data.get("pe_ratio"))
            fs.fcf_yield = _safe_float(data.get("fcf_yield"))
            fs.revenue_growth = _safe_float(data.get("revenue_growth"))
            fs.profit_margin = _safe_float(data.get("profit_margin"))
        else:
            fs.caveats.append("Fundamental data unavailable (check cache / connections)")
            return fs

    # ── Scoring ───────────────────────────────────────────────────────────
    available_fields = 0
    total_possible = 0

    def score_field(value, thresholds, pts_list, field_pts):
        """Return score for a field given thresholds."""
        nonlocal available_fields, total_possible
        total_possible += field_pts
        if value is None:
            return 0
        available_fields += 1
        for threshold, pts in zip(thresholds, pts_list):
            if value >= threshold:
                return pts
        return 0

    # ROE ≥ 15% → +20; ≥ 10% → +12; ≥ 5% → +6
    fs.roe_score = score_field(fs.roe, [0.15, 0.10, 0.05], [20, 12, 6], 20)

    # Debt/Equity: LOW is better. ≤ 0.5 → +15; ≤ 1.0 → +10; ≤ 2.0 → +5
    if fs.debt_to_equity is not None:
        available_fields += 1
        total_possible += 15
        d = fs.debt_to_equity
        if d <= 0.5:
            fs.debt_score = 15
        elif d <= 1.0:
            fs.debt_score = 10
        elif d <= 2.0:
            fs.debt_score = 5
        else:
            fs.debt_score = 0
            fs.caveats.append(f"High leverage (D/E: {d:.1f})")
    else:
        total_possible += 15

    # EPS Growth ≥ 15% → +20; ≥ 10% → +14; ≥ 5% → +7
    fs.eps_growth_score = score_field(fs.eps_growth, [0.15, 0.10, 0.05], [20, 14, 7], 20)

    # P/E: <15 → +15; <25 → +10; <35 → +5; >35 → 0 (overvalued)
    if fs.pe_ratio is not None and fs.pe_ratio > 0:
        available_fields += 1
        total_possible += 15
        pe = fs.pe_ratio
        if pe < 15:
            fs.valuation_score = 15
        elif pe < 25:
            fs.valuation_score = 10
        elif pe < 35:
            fs.valuation_score = 5
        else:
            fs.valuation_score = 0
            fs.caveats.append(f"High P/E ratio ({pe:.0f}x) — may be overvalued")
    else:
        total_possible += 15

    # FCF Yield ≥ 5% → +15; ≥ 3% → +10; ≥ 1% → +5
    fs.fcf_score = score_field(fs.fcf_yield, [0.05, 0.03, 0.01], [15, 10, 5], 15)

    # Revenue Growth ≥ 15% → +10; ≥ 8% → +7; ≥ 3% → +3
    fs.revenue_growth_score = score_field(fs.revenue_growth, [0.15, 0.08, 0.03], [10, 7, 3], 10)

    # Profit Margin ≥ 20% → +5; ≥ 10% → +3; ≥ 5% → +1
    fs.margin_score = score_field(fs.profit_margin, [0.20, 0.10, 0.05], [5, 3, 1], 5)

    # ── Total Score ────────────────────────────────────────────────────────
    raw_total = (fs.roe_score + fs.debt_score + fs.eps_growth_score +
                 fs.valuation_score + fs.fcf_score + fs.revenue_growth_score +
                 fs.margin_score)

    # Scale to 0-100 if not all fields available
    if total_possible > 0 and available_fields > 0:
        # Normalize: score / available_possible × 100
        available_possible = total_possible * (available_fields / 7)
        fs.total_score = min(100, int(raw_total / max(available_possible, 1) * 100))
    else:
        fs.total_score = 0

    # Data coverage
    fs.data_coverage = int(available_fields / 7 * 100)
    if fs.data_coverage < 40:
        fs.caveats.append("Limited fundamental data — score may not be accurate")

    # ── Grade ──────────────────────────────────────────────────────────────
    if fs.total_score >= 85:
        fs.grade, fs.label = "A+", "Exceptional Quality (Buffett-grade)"
    elif fs.total_score >= 70:
        fs.grade, fs.label = "A", "High Quality"
    elif fs.total_score >= 55:
        fs.grade, fs.label = "B", "Good Quality"
    elif fs.total_score >= 40:
        fs.grade, fs.label = "C", "Average Quality"
    elif fs.total_score >= 25:
        fs.grade, fs.label = "D", "Below Average"
    else:
        fs.grade, fs.label = "F", "Low Quality / Speculative"

    logger.debug(f"FundamentalScore [{symbol}]: {fs.total_score}/100 ({fs.grade}) | Coverage: {fs.data_coverage}%")
    return fs


def get_fundamental_dict(symbol: str, market: str) -> dict:
    """Return serializable dict for UI and Senate debate."""
    fs = compute_fundamental_score(symbol, market)
    return {
        "total_score": fs.total_score,
        "grade": fs.grade,
        "label": fs.label,
        "data_coverage": fs.data_coverage,
        "scores": {
            "ROE": {"pts": fs.roe_score, "max": 20, "raw": f"{fs.roe*100:.1f}%" if fs.roe else "N/A"},
            "Debt/Equity": {"pts": fs.debt_score, "max": 15, "raw": f"{fs.debt_to_equity:.1f}" if fs.debt_to_equity else "N/A"},
            "EPS Growth": {"pts": fs.eps_growth_score, "max": 20, "raw": f"{fs.eps_growth*100:.1f}%" if fs.eps_growth else "N/A"},
            "P/E Ratio": {"pts": fs.valuation_score, "max": 15, "raw": f"{fs.pe_ratio:.1f}x" if fs.pe_ratio else "N/A"},
            "FCF Yield": {"pts": fs.fcf_score, "max": 15, "raw": f"{fs.fcf_yield*100:.1f}%" if fs.fcf_yield else "N/A"},
            "Rev Growth": {"pts": fs.revenue_growth_score, "max": 10, "raw": f"{fs.revenue_growth*100:.1f}%" if fs.revenue_growth else "N/A"},
            "Profit Margin": {"pts": fs.margin_score, "max": 5, "raw": f"{fs.profit_margin*100:.1f}%" if fs.profit_margin else "N/A"},
        },
        "caveats": fs.caveats,
    }
