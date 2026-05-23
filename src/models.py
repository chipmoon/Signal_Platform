"""
Shared Data Models
==================
Type-safe models for signals, trades, and backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Signal(IntEnum):
    """Trading signal direction."""

    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class TradeRecord:
    """A single completed trade."""

    entry_date: str
    exit_date: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Aggregated backtest output."""

    total_return: float
    total_pnl: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    final_capital: float
    best_trade: float
    worst_trade: float
    hit_rate: float = 0.0
    precision_at_entry: float = 0.0
    expectancy: float = 0.0
    turnover: float = 0.0
    turnover_adjusted_sharpe: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
