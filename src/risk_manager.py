"""
Risk Manager
=============
Professional-grade risk management module with:
- ATR-based trailing stops
- Volatility-adjusted position sizing
- Kelly Criterion optimal sizing
- Drawdown circuit breaker
- Value at Risk (VaR) calculation

Used by both the backtesting engine and the live dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk management parameters."""

    # Trailing stop
    atr_period: int = 14
    atr_multiplier: float = 3.0  # Nới lỏng cho Price Action (Institutional)
    tight_atr_multiplier: float = 1.5 # Vẫn giữ Hard Stop nhưng nới hơn một chút

    # Position sizing
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_pct: float = 0.25  # Max 25% of capital per position
    max_pyramiding: int = 3  # Max additional entries on winners

    # Kelly criterion
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    kelly_lookback: int = 50  # Trades to look back for Kelly calc

    # Drawdown circuit breaker
    max_drawdown_pct: float = 0.15  # Halt at 15% drawdown
    recovery_threshold: float = 0.05  # Resume after 5% recovery from peak

    # Slippage
    slippage_pct: float = 0.0005  # 0.05% per trade

    # VaR
    var_confidence: float = 0.95
    var_lookback: int = 252  # 1 year of trading days


# ─────────────────────────────────────────────
# ATR Calculation
# ─────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range
    of an asset price for a given period.

    Args:
        df: DataFrame with High, Low, Close columns.
        period: Lookback window for the ATR.

    Returns:
        Series containing ATR values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=1).mean()


# ─────────────────────────────────────────────
# RSI Calculation
# ─────────────────────────────────────────────

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    Uses the exponential moving average method (Wilder's smoothing).

    Args:
        series: Price series (typically Close).
        period: RSI lookback period.

    Returns:
        RSI values between 0 and 100.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ─────────────────────────────────────────────
# ADX Calculation
# ─────────────────────────────────────────────

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX).

    Measures trend strength regardless of direction.
    ADX > 20 = trending, ADX > 40 = strong trend.

    Args:
        df: DataFrame with High, Low, Close columns.
        period: ADX lookback period.

    Returns:
        ADX values (0-100 scale).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calculate_atr(df, period)

    # Smoothed directional indicators
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return adx.fillna(0)


# ─────────────────────────────────────────────
# MACD Calculation
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Volume Profile Calculation
# ─────────────────────────────────────────────

def calculate_volume_profile(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Volume Profile features (POC and Value Area) over a rolling window.
    
    Args:
        df: DataFrame with Close and Volume columns.
        window: Rolling lookback window.
        
    Returns:
        DataFrame with 'poc_price', 'poc_dist', and 'inside_va' features.
    """
    result = pd.DataFrame(index=df.index)
    
    close_vals = df["Close"].values
    if "Volume" in df.columns:
        vol_vals = df["Volume"].values
    else:
        vol_vals = np.ones(len(close_vals))

    poc_prices = np.zeros(len(df))
    poc_dists = np.zeros(len(df))
    inside_vas = np.zeros(len(df), dtype=int)
    
    for i in range(len(df)):
        if i < window:
            poc_prices[i] = np.nan
            poc_dists[i] = 0.0
            inside_vas[i] = 0
            continue
            
        start_idx = i - window + 1
        end_idx = i + 1
        
        prices = close_vals[start_idx:end_idx]
        vols = vol_vals[start_idx:end_idx]
        
        # 1. Create Bins
        p_min, p_max = prices.min(), prices.max()
        if p_min == p_max:
            poc_prices[i] = p_min
            poc_dists[i] = 0.0
            inside_vas[i] = 1
            continue
            
        bins = np.linspace(p_min, p_max, 11)
        bin_indices = np.digitize(prices, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        
        # 2. Aggregate Volume
        bin_vols = np.zeros(10)
        np.add.at(bin_vols, bin_indices, vols)
        
        # 3. Find POC
        poc_idx = np.argmax(bin_vols)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # 4. Value Area
        target_va_vol = bin_vols.sum() * 0.70
        va_indices = {poc_idx}
        current_va_vol = bin_vols[poc_idx]
        
        while current_va_vol < target_va_vol and len(va_indices) < 10:
            low_idx = min(va_indices) - 1
            high_idx = max(va_indices) + 1
            
            v_low = bin_vols[low_idx] if low_idx >= 0 else 0
            v_high = bin_vols[high_idx] if high_idx < 10 else 0
            
            if v_low >= v_high and low_idx >= 0:
                va_indices.add(low_idx)
                current_va_vol += v_low
            elif high_idx < 10:
                va_indices.add(high_idx)
                current_va_vol += v_high
            else:
                break
                
        va_low = bins[min(va_indices)]
        va_high = bins[max(va_indices) + 1]
        
        current_price = prices[-1]
        poc_dist = (current_price - poc_price) / poc_price
        poc_prices[i] = poc_price
        poc_dists[i] = poc_dist
        inside_vas[i] = 1 if va_low <= current_price <= va_high else 0

    result["poc_price"] = poc_prices
    result["poc_dist"] = poc_dists
    result["inside_va"] = inside_vas
    
    return result


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal Line, and Histogram.

    Args:
        series: Price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ─────────────────────────────────────────────
# Trailing Stop Engine
# ─────────────────────────────────────────────

class TrailingStopManager:
    """Manages ATR-based trailing stops for open positions.

    The stop ratchets up with price advances but never moves down,
    locking in profits as the trend extends.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.standard_multiplier = config.atr_multiplier
        self.active_multiplier = config.atr_multiplier
        self._stop_price: float = 0.0
        self._is_active: bool = False

    def activate(self, entry_price: float, atr: float, custom_multiplier: float | None = None) -> float:
        """Initialize trailing stop at entry.

        Args:
            entry_price: Trade entry price.
            atr: Current ATR value.
            custom_multiplier: Optional narrower/wider multiplier for this trade.

        Returns:
            Initial stop price.
        """
        self.active_multiplier = custom_multiplier if custom_multiplier is not None else self.standard_multiplier
        self._stop_price = entry_price - (self.active_multiplier * atr)
        self._is_active = True
        return self._stop_price

    def update(self, current_price: float, atr: float) -> float:
        """Update trailing stop — only ratchets upward.

        Args:
            current_price: Current market price.
            atr: Current ATR value.

        Returns:
            Updated stop price.
        """
        if not self._is_active:
            return 0.0

        new_stop = current_price - (self.active_multiplier * atr)
        if new_stop > self._stop_price:
            self._stop_price = new_stop
        return self._stop_price

    def is_triggered(self, current_price: float) -> bool:
        """Check if price has hit the trailing stop."""
        return self._is_active and current_price <= self._stop_price

    def deactivate(self) -> None:
        """Reset after position close."""
        self._stop_price = 0.0
        self._is_active = False

    @property
    def stop_price(self) -> float:
        return self._stop_price


# ─────────────────────────────────────────────
# Position Sizing
# ─────────────────────────────────────────────

class PositionSizer:
    """Calculate optimal position sizes using professional methods."""

    def __init__(self, config: RiskConfig) -> None:
        self.risk_per_trade = config.risk_per_trade
        self.max_position_pct = config.max_position_pct
        self.kelly_fraction = config.kelly_fraction
        self.kelly_lookback = config.kelly_lookback

    def volatility_based_size(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """Position size based on volatility (ATR).

        Risk-per-trade / (ATR × multiplier) = number of shares.
        This ensures each trade risks the same dollar amount.

        Args:
            capital: Current portfolio capital.
            entry_price: Expected entry price.
            atr: Current ATR value.
            atr_multiplier: Stop distance as ATR multiple.

        Returns:
            Number of shares/contracts to trade.
        """
        risk_amount = capital * self.risk_per_trade
        stop_distance = atr * atr_multiplier

        if stop_distance <= 0:
            return 0.0

        shares = risk_amount / stop_distance
        max_shares = (capital * self.max_position_pct) / entry_price
        
        # 🛡️ Crash Shield Override (Zero-Cost Protection)
        from src.strategies.crash_shield import CrashShield
        risk_report = CrashShield.evaluate_risk()
        multiplier = CrashShield.get_action_multiplier(risk_report["level"])
        
        final_shares = min(shares, max_shares) * multiplier
        
        if multiplier < 1.0:
            logger.warning(f"🛡️ Risk Multiplier applied: {multiplier:.1f}x (Level {risk_report['level']})")
            
        return final_shares

    def kelly_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Kelly Criterion position sizing.

        f* = (p × b - q) / b
        where p = win rate, q = loss rate, b = avg_win / avg_loss

        Uses half-Kelly for safety (robust to estimation error).

        Args:
            capital: Current portfolio capital.
            win_rate: Historical win rate (0-1).
            avg_win: Average winning trade PnL.
            avg_loss: Average losing trade PnL (positive number).

        Returns:
            Fraction of capital to risk.
        """
        if avg_loss <= 0 or win_rate <= 0:
            return self.risk_per_trade

        b = avg_win / abs(avg_loss)
        q = 1 - win_rate
        kelly_f = (win_rate * b - q) / b

        # Apply half-Kelly and cap
        kelly_f = max(0, min(kelly_f * self.kelly_fraction, self.max_position_pct))
        return kelly_f


# ─────────────────────────────────────────────
# Drawdown Circuit Breaker
# ─────────────────────────────────────────────

class DrawdownCircuitBreaker:
    """Halts trading when drawdown exceeds threshold.

    Professional risk management: automatic system shutdown
    when losses spiral beyond acceptable levels.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.max_drawdown_pct = config.max_drawdown_pct
        self.recovery_threshold = config.recovery_threshold
        self._peak_equity: float = 0.0
        self._is_halted: bool = False

    def update(self, current_equity: float) -> bool:
        """Update circuit breaker state.

        Args:
            current_equity: Current portfolio equity.

        Returns:
            True if trading is halted.
        """
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity

            if drawdown >= self.max_drawdown_pct:
                if not self._is_halted:
                    logger.warning(
                        f"⚠️ CIRCUIT BREAKER: Drawdown {drawdown:.1%} exceeds "
                        f"{self.max_drawdown_pct:.1%} threshold. Trading halted."
                    )
                self._is_halted = True

            elif self._is_halted and drawdown <= self.recovery_threshold:
                logger.info(
                    f"✅ CIRCUIT BREAKER: Drawdown recovered to {drawdown:.1%}. "
                    f"Trading resumed."
                )
                self._is_halted = False

        return self._is_halted

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    def reset(self, initial_equity: float) -> None:
        """Reset for a new backtest run."""
        self._peak_equity = initial_equity
        self._is_halted = False


# ─────────────────────────────────────────────
# VaR Calculator
# ─────────────────────────────────────────────

def calculate_var(
    equity_curve: list[float] | np.ndarray,
    confidence: float = 0.95,
    lookback: int = 252,
) -> dict[str, float]:
    """Calculate Value at Risk using historical simulation.

    Args:
        equity_curve: List of equity values over time.
        confidence: VaR confidence level (e.g., 0.95 for 95%).
        lookback: Number of periods to use.

    Returns:
        Dictionary with VaR metrics.
    """
    if len(equity_curve) < 10:
        return {"var_pct": 0.0, "var_dollar": 0.0, "cvar_pct": 0.0}

    equity = np.array(equity_curve[-lookback:])
    returns = np.diff(equity) / equity[:-1]

    if len(returns) == 0:
        return {"var_pct": 0.0, "var_dollar": 0.0, "cvar_pct": 0.0}

    # Historical VaR
    var_pct = float(np.percentile(returns, (1 - confidence) * 100))
    current_equity = equity[-1]
    var_dollar = var_pct * current_equity

    # Conditional VaR (Expected Shortfall)
    tail_returns = returns[returns <= var_pct]
    cvar_pct = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_pct

    return {
        "var_pct": abs(var_pct),
        "var_dollar": abs(var_dollar),
        "cvar_pct": abs(cvar_pct),
    }


# ─────────────────────────────────────────────
# Portfolio Optimization (HRP)
# ─────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization engine.
    Implements Hierarchical Risk Parity (HRP) for cross-asset allocation.
    
    HRP is superior to Mean-Variance as it doesn't require matrix inversion
    and is robust to unstable correlation estimates.
    """

    @staticmethod
    def compute_hrp_weights(returns_df: pd.DataFrame) -> pd.Series:
        """
        Compute HRP weights for a given set of asset returns.
        
        Args:
            returns_df: DataFrame where each column is an asset's return series.
            
        Returns:
            Series of weights indexed by asset names.
        """
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import pdist, squareform

        if returns_df.empty or returns_df.shape[1] < 2:
            return pd.Series(1.0, index=returns_df.columns) if not returns_df.empty else pd.Series()

        # 1. Calculate Correlation and Distance matrix
        corr = returns_df.corr().fillna(0)
        # Distance metric: d = sqrt(0.5 * (1 - rho))
        dist = np.sqrt(0.5 * (1 - corr).clip(lower=0))
        
        # 2. Hierarchical Clustering (Single linkage or Ward)
        # We use standard pdist on the correlation matrix columns to find link distances
        link = sch.linkage(pdist(dist), method='single')
        
        # 3. Quasi-Diagonalization (Reordering)
        # Sort assets so similar ones are closer in the list
        def get_quasi_diag(link):
            return sch.to_tree(link, rd=False).pre_order()
        
        sort_ix = get_quasi_diag(link)
        sorted_assets = returns_df.columns[sort_ix].tolist()
        
        # 4. Recursive Bisection
        weights = pd.Series(1.0, index=sorted_assets)
        cluster_list = [sorted_assets]
        
        while len(cluster_list) > 0:
            # Pop the first cluster
            items = cluster_list.pop(0)
            if len(items) <= 1:
                continue
                
            # Split the cluster into two halves (bisect)
            mid = len(items) // 2
            c1 = items[:mid]
            c2 = items[mid:]
            
            # Compute variance for each cluster using Inverse-Variance Parity
            v1 = PortfolioOptimizer._get_cluster_var(returns_df[c1])
            v2 = PortfolioOptimizer._get_cluster_var(returns_df[c2])
            
            # Allocation factor based on variance
            alpha = 1 - v1 / (v1 + v2)
            
            # Allocate weights recursively
            weights[c1] *= alpha
            weights[c2] *= (1 - alpha)
            
            # Appending halves back to process further
            cluster_list.append(c1)
            cluster_list.append(c2)
            
        return weights.reindex(returns_df.columns)

    @staticmethod
    def _get_cluster_var(returns: pd.DataFrame) -> float:
        """Compute the variance of an Inverse-Variance weighted cluster."""
        cov = returns.cov().fillna(0)
        # Use standard np.clip syntax
        ivp = 1.0 / np.clip(np.diag(cov), a_min=1e-8, a_max=None)
        ivp /= ivp.sum()
        
        # Variance = w' * Cov * w
        w = ivp.reshape(-1, 1)
        cluster_var = np.dot(np.dot(w.T, cov.values), w)[0][0]
        return float(cluster_var)

    @staticmethod
    def risk_budget_allocation(
        asset_risks: Dict[str, float], 
        total_risk_budget: float = 1.0
    ) -> Dict[str, float]:
        """
        Simple Risk Budgeting allocation.
        Weight proportional to 1/Risk.
        """
        inv_risk = {k: 1.0 / max(v, 1e-6) for k, v in asset_risks.items()}
        total_inv = sum(inv_risk.values())
        return {k: (v / total_inv) * total_risk_budget for k, v in inv_risk.items()}
