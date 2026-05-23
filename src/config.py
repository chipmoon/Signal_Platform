"""
Centralized Configuration
=========================
All strategy and system parameters as frozen dataclasses.
No magic numbers anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Market Constants
# ─────────────────────────────────────────────

CFTC_CURRENT_URL = "https://www.cftc.gov/dea/newcot/deacmelf.zip"
CFTC_LEGACY_URL_TEMPLATE = (
    "https://www.cftc.gov/files/dea/history/deacmelf{year}.zip"
)

MARKET_CODES: dict[str, str] = {
    "GOLD": "088691",
    "SILVER": "084691",
    "CRUDE_OIL": "067651",
    "NATURAL_GAS": "023651",
    "COPPER": "085692",
    "SP500": "13874A",
    "NASDAQ": "209742",
    "EURUSD": "099741",
    "USDJPY": "097741",
}

# ─────────────────────────────────────────────
# SMA / Trend Constants
# ─────────────────────────────────────────────

DEFAULT_SMA_FAST = 50
DEFAULT_SMA_SLOW = 200
ANNUALIZE_FACTOR = 252


# ─────────────────────────────────────────────
# Strategy Configs
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class COTConfig:
    """Configuration for the COT Monitor strategy."""

    short_ratio_threshold: float = 0.20
    anomaly_multiplier: float = 3.0
    volatility_window: int = 20
    change_lookback: int = 8


@dataclass(frozen=True)
class VolumePriceConfig:
    """Configuration for the Volume-Price Mismatch Detector."""

    volume_spike_threshold: float = 5.0
    price_drop_threshold: float = -0.03
    lookback_window: int = 20
    ml_contamination: float = 0.05
    ml_n_estimators: int = 200


@dataclass(frozen=True)
class VolumeProfileConfig:
    """Configuration for the Volume Profile (Horizontal Volume) analysis."""

    num_bins: int = 50                 # Price levels for the histogram
    value_area_pct: float = 0.70      # Value Area = 70% of total volume
    lookback: int = 40                # Window for "Session" profile


@dataclass(frozen=True)
class BankConfig:
    """Configuration for the Bank Participation Monitor."""

    concentration_threshold: float = 0.50
    usd_strength_window: int = 10
    usd_strength_threshold: float = 0.02


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the backtesting engine."""

    initial_capital: float = 100_000
    position_size: float = 0.10
    stop_loss: float = 0.05
    take_profit: float = 0.10
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005  # 0.05% slippage per trade
    use_trailing_stop: bool = True
    max_pyramiding: int = 3
    # Dynamic weighting
    ai_confidence_weight: bool = True  # Enable dynamic sizing based on AI confidence
    use_kelly_sizing: bool = False  # Adapt size from realized win/loss profile
    kelly_min_trades: int = 20  # Require enough history before Kelly kicks in
    use_var_gate: bool = False  # Block new entries when 1-day VaR is too high
    max_var_pct: float = 0.03  # Max tolerated 1-day VaR (3% of equity)
    # Execution realism
    execute_on_next_bar: bool = True
    execution_price_column: str = "Open"


@dataclass(frozen=True)
class CombinerConfig:
    """Configuration for the signal combiner."""

    # SMA crossover params (Institutional Gold Standard)
    sma_fast: int = 50
    sma_slow: int = 200
    trend_weight: float = 0.70
    shield_weight: float = 0.30
    manipulation_danger_threshold: float = -0.35
    extreme_danger_threshold: float = -0.55
    cot_weight: float = 0.40
    vp_weight: float = 0.35
    bank_weight: float = 0.25
    smc_weight: float = 0.45
    wyckoff_weight: float = 0.40
    debate_weight: float = 0.35
    real_flow_weight: float = 0.50
    qmf_weight: float = 0.25
    rsi_period: int = 14
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0

    # ADX trend strength filter
    adx_period: int = 14
    adx_threshold: float = 20.0  # Only trade when ADX > 20
    # Structural confirmation rules
    structure_min_score: float = 0.50
    buy_score_threshold: float = 0.45
    sell_score_threshold: float = -0.45
    # Fail-closed gating: if True, system blocks entries when real flow is unavailable
    require_real_flow: bool = True



@dataclass(frozen=True)
class SmcConfig:
    """Configuration for the Smart Money Concepts Analyzer."""

    # Order Block detection
    ob_impulse_multiplier: float = 1.5  # Move must be > 1.5x ATR to qualify as impulse
    ob_lookback: int = 3               # Look back N candles before impulse for OB candle
    ob_max_zones: int = 5             # Max unmitigated OBs to track per direction

    # Fair Value Gap detection
    fvg_min_size_pct: float = 0.002   # Min gap size = 0.2% of price
    fvg_fill_threshold: float = 0.90  # Gap is "filled" if price enters 90%+
    fvg_max_zones: int = 5            # Max unfilled FVGs to track

    # Liquidity Pool detection
    liq_tolerance_pct: float = 0.3    # Equal Highs/Lows within 0.3% = same pool
    liq_max_pools: int = 3            # Max pools to track per side

    # Signal generation
    signal_score_threshold: float = 0.25   # Min composite score to trigger signal


@dataclass(frozen=True)
class WyckoffConfig:

    """Configuration for the Wyckoff Analyzer strategy."""

    # Trading Range detection
    tr_lookback: int = 60  # Days to scan for Trading Range
    tr_atr_ratio: float = 0.6  # ATR must be < 60% of long-term ATR for TR
    tr_min_touches: int = 3  # Min times price must touch upper/lower boundary

    # Spring / UTAD detection
    spring_penetration: float = 0.005  # Min % below TR low to qualify as Spring
    spring_max_penetration: float = 0.03  # Max % — beyond this is breakdown, not Spring
    spring_volume_ratio: float = 0.8  # Volume must be < 80% of avg (low vol = Spring)
    spring_recovery_bars: int = 3  # Must recover within N bars

    # SOS / SOW detection
    sos_volume_multiplier: float = 1.5  # Volume > 1.5x avg for Sign of Strength
    sos_spread_percentile: float = 75.0  # Candle spread must be top 25%

    # VSA parameters
    vsa_stopping_vol_multiplier: float = 2.0  # Volume > 2x avg at range bottom
    vsa_stopping_spread_percentile: float = 30.0  # Narrow spread (bottom 30%)
    vsa_no_supply_vol_ratio: float = 0.5  # Volume < 50% avg during pullback
    vsa_shakeout_drop_pct: float = -0.02  # Sharp drop threshold
    vsa_shakeout_recovery_pct: float = 0.01  # Recovery threshold

    # Effort vs Result
    evr_lookback: int = 20  # Rolling window for E/R baseline

    # LPS detection
    lps_pullback_atr_mult: float = 1.0  # Max pullback depth = 1x ATR
    lps_volume_ratio: float = 0.7  # Low volume during LPS pullback

    # General
    ma_long_period: int = 200  # Long-term trend filter
    phase_score_smoothing: int = 5  # Smooth phase scores over N bars


@dataclass(frozen=True)
class StochasticConfig:
    """Configuration for the Stochastic Oscillator momentum filter."""

    k_period: int = 14          # %K lookback window
    d_period: int = 3           # %D smoothing (signal line)
    slowing: int = 3            # %K slowing factor
    overbought: float = 80.0    # Overbought threshold
    oversold: float = 20.0      # Oversold threshold


@dataclass(frozen=True)
class AIConfig:
    """Configuration for the AI Deep Learning Predictor."""

    enabled: bool = True
    model_type: str = "mlp"  # mlp, gradient_boosting
    lookback_days: int = 15
    prediction_horizon: int = 1  # Default for internal signal use
    # Horizons in trading days: 1d, 1m (21d), 2m (42d), 3m (63d), 6m (126d)
    horizons: tuple[int, ...] = (1, 21, 42, 63, 126)
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    max_iter: int = 500
    learning_rate_init: float = 0.001
    walk_forward_window: int = 252  # 1 year rolling training window
    confidence_threshold: float = 0.25 # Veto trades if confidence is below this
    retrain_every: int = 21  # Re-train every 1 month in walk-forward mode


@dataclass(frozen=True)
class SystemConfig:
    """Top-level configuration combining all sub-configs."""

    ticker: str = "GC=F"
    commodity: str = "GOLD"
    years: int = 3
    cot: COTConfig = field(default_factory=COTConfig)
    volume_price: VolumePriceConfig = field(default_factory=VolumePriceConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    combiner: CombinerConfig = field(default_factory=CombinerConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
