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
        return f"{symbol}.VN"
    return symbol


def _safe_float(val, default=None) -> Optional[float]:
    try:
        f = float(val)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def compute_fundamental_score(symbol: str, market: str) -> FundamentalScore:
    """
    Compute Fundamental Quality Score for a given symbol.
    Returns FundamentalScore dataclass.
    """
    fs = FundamentalScore()
    yf_ticker = _to_yf_ticker(symbol, market)

    try:
        ticker = yf.Ticker(yf_ticker)
        info = ticker.info
    except Exception as e:
        logger.warning(f"FundamentalScore: could not fetch {yf_ticker}: {e}")
        fs.caveats.append("Yahoo Finance data unavailable")
        return fs

    if not info or info.get("symbol") is None:
        fs.caveats.append("No fundamental data returned by Yahoo Finance")
        return fs

    # ── Extract raw values ─────────────────────────────────────────────────
    fs.roe              = _safe_float(info.get("returnOnEquity"))
    fs.debt_to_equity   = _safe_float(info.get("debtToEquity"))
    fs.pe_ratio         = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    fs.profit_margin    = _safe_float(info.get("profitMargins"))
    fs.revenue_growth   = _safe_float(info.get("revenueGrowth"))

    # EPS growth: use earningsGrowth or compute from trailingEps/forwardEps
    fs.eps_growth = _safe_float(info.get("earningsGrowth"))
    if fs.eps_growth is None:
        trailing = _safe_float(info.get("trailingEps"))
        forward  = _safe_float(info.get("forwardEps"))
        if trailing and forward and trailing != 0:
            fs.eps_growth = (forward - trailing) / abs(trailing)

    # FCF Yield = FreeCashFlow / MarketCap
    fcf = _safe_float(info.get("freeCashflow"))
    mkt_cap = _safe_float(info.get("marketCap"))
    if fcf and mkt_cap and mkt_cap > 0:
        fs.fcf_yield = fcf / mkt_cap
    else:
        fs.fcf_yield = None

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
    roe_val = fs.roe if fs.roe is not None else None
    fs.roe_score = score_field(roe_val, [0.15, 0.10, 0.05], [20, 12, 6], 20)

    # Debt/Equity: LOW is better. ≤ 0.5 → +15; ≤ 1.0 → +10; ≤ 2.0 → +5
    if fs.debt_to_equity is not None:
        available_fields += 1
        total_possible += 15
        d = fs.debt_to_equity / 100 if fs.debt_to_equity > 10 else fs.debt_to_equity  # normalize %
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
