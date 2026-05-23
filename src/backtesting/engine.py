"""
Backtesting Engine
==================
Professional-grade event-driven simulator for signal-based strategies.

Features:
    - ATR-based trailing stops (replaces fixed stop-loss)
    - Volatility-adjusted position sizing via RiskManager
    - Pyramiding â€” scale into winning positions (up to 3 units)
    - Slippage simulation for realistic commodity fills
    - Drawdown circuit breaker integration
    - Comprehensive performance metrics as typed ``BacktestResult``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import ANNUALIZE_FACTOR, BacktestConfig
from src.models import BacktestResult, TradeRecord
from src.order_manager import OrderManager
from src.risk_manager import (
    DrawdownCircuitBreaker,
    PositionSizer,
    RiskConfig,
    TrailingStopManager,
    calculate_atr,
    calculate_var,
)


class BacktestEngine:
    """Professional-grade event-driven backtesting engine.

    Accepts a ``BacktestConfig`` dataclass for all parameters.
    Returns a typed ``BacktestResult`` with metrics + trade log.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
        market_id: str = "US"
    ) -> None:
        cfg = config or BacktestConfig()
        self.risk_cfg = risk_config or RiskConfig()
        self.market_id = market_id

        self.initial_capital = cfg.initial_capital
        self.position_size = cfg.position_size
        self.stop_loss = cfg.stop_loss
        self.take_profit = cfg.take_profit
        self.commission_pct = cfg.commission_pct
        self.slippage_pct = cfg.slippage_pct
        self.use_trailing_stop = cfg.use_trailing_stop
        self.max_pyramiding = cfg.max_pyramiding
        self.execute_on_next_bar = cfg.execute_on_next_bar
        self.execution_price_column = cfg.execution_price_column
        self.ai_confidence_weight = cfg.ai_confidence_weight
        self.use_kelly_sizing = cfg.use_kelly_sizing
        self.kelly_min_trades = cfg.kelly_min_trades
        self.use_var_gate = cfg.use_var_gate
        self.max_var_pct = cfg.max_var_pct

        # Risk components
        self.trailing_stop = TrailingStopManager(self.risk_cfg)
        self.position_sizer = PositionSizer(self.risk_cfg)
        self.circuit_breaker = DrawdownCircuitBreaker(self.risk_cfg)
        self.order_manager = OrderManager(cfg)

        # State
        self.capital = self.initial_capital
        self.position: int = 0  # 0=flat, 1+=number of pyramid units
        self.entry_price: float = 0.0
        self.entry_date: str = ""
        self.shares: float = 0.0
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[float] = []
        self._pyramid_count: int = 0
        self._last_run_df: pd.DataFrame | None = None
        self._last_signal_column: str = "combined_signal"

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply realistic slippage to execution price."""
        if is_buy:
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _kelly_weight(self) -> float:
        """Return Kelly-adjusted weight, capped by baseline position size."""
        if not self.use_kelly_sizing or len(self.trades) < self.kelly_min_trades:
            return self.position_size

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if not wins or not losses:
            return self.position_size

        win_rate = len(wins) / len(pnls)
        avg_win = float(np.mean(wins))
        avg_loss = float(abs(np.mean(losses)))
        kelly_f = self.position_sizer.kelly_size(
            capital=self.capital,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )
        return float(np.clip(kelly_f, 0.0, self.position_size))

    def _var_gate_blocked(self) -> bool:
        """Return True when VaR breaches configured threshold."""
        if not self.use_var_gate or len(self.equity_curve) < 20:
            return False
        var_stats = calculate_var(
            self.equity_curve,
            confidence=self.risk_cfg.var_confidence,
            lookback=self.risk_cfg.var_lookback,
        )
        return float(var_stats.get("var_pct", 0.0)) > float(self.max_var_pct)

    def run(
        self,
        data: pd.DataFrame,
        signal_column: str = "combined_signal",
        price_column: str = "Close",
    ) -> pd.DataFrame:
        """Run backtest on signal data.

        Args:
            data: DataFrame with signals, price data, and High/Low for ATR.
            signal_column: Column containing signals (-1, 0, 1).
            price_column: Column containing price data.

        Returns:
            DataFrame with equity curve and trade markers appended.
        """
        df = data.copy()
        df["equity"] = self.initial_capital
        df["trade_marker"] = ""
        df["drawdown_pct"] = 0.0

        # Calculate ATR for trailing stops
        if "High" in df.columns and "Low" in df.columns:
            atr_series = calculate_atr(df, period=self.risk_cfg.atr_period)
        else:
            atr_series = pd.Series(0.0, index=df.index)

        # Reset state
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self._pyramid_count = 0
        self.trailing_stop.deactivate()
        self.circuit_breaker.reset(self.initial_capital)

        prices = df[price_column].values
        raw_signals = df[signal_column].values
        dates = df["Date"].astype(str).values
        atrs = atr_series.values
        exec_prices = df[self.execution_price_column].values if self.execution_price_column in df.columns else prices
        
        # News Sentiment filter if present
        sentiment_biases = df.get("sentiment_bias", pd.Series(0, index=df.index)).values
        
        # AI Predictor signals if present
        ai_biases = df.get("ai_bias", pd.Series(0, index=df.index)).values
        ai_confidences = df.get("ai_confidence", pd.Series(0.0, index=df.index)).values

        if self.execute_on_next_bar:
            signals = np.roll(raw_signals, 1)
            signals[0] = 0
            sentiment_biases = np.roll(sentiment_biases, 1)
            sentiment_biases[0] = 0
            ai_biases = np.roll(ai_biases, 1)
            ai_biases[0] = 0
            ai_confidences = np.roll(ai_confidences, 1)
            ai_confidences[0] = 0.0
        else:
            signals = raw_signals

        peak_equity = self.initial_capital
        ticker = df.attrs.get("ticker", "UNKNOWN")

        for i in range(len(df)):
            price = float(prices[i])
            sig_raw = signals[i]
            signal = int(sig_raw) if pd.notna(sig_raw) else 0
            atr = float(atrs[i]) if atrs[i] > 0 else price * 0.02
            sentiment_bias = int(sentiment_biases[i])

            # Check circuit breaker
            current_equity = self.capital + (self.shares * price if self.position > 0 else 0)
            is_halted = self.circuit_breaker.update(current_equity)

            # Check trailing stop / fixed stop / take-profit if in position
            if self.position > 0:
                pnl_pct = (price - self.entry_price) / self.entry_price

                # Trailing stop check (preferred)
                if self.use_trailing_stop:
                    self.trailing_stop.update(price, atr)
                    if self.trailing_stop.is_triggered(price):
                        exec_price = self._apply_slippage(price, is_buy=False)
                        self._close_position(exec_price, dates[i], "TRAILING_STOP")
                        df.iloc[i, df.columns.get_loc("trade_marker")] = "EXIT(TRAIL)"
                        self.trailing_stop.deactivate()
                else:
                    # Fallback fixed stop-loss
                    if pnl_pct <= -self.stop_loss:
                        exec_price = self._apply_slippage(price, is_buy=False)
                        self._close_position(exec_price, dates[i], "STOP_LOSS")
                        df.iloc[i, df.columns.get_loc("trade_marker")] = "EXIT(SL)"

                # Take profit (still applies with trailing stops as max target)
                if self.position > 0 and pnl_pct >= self.take_profit:
                    exec_price = self._apply_slippage(price, is_buy=False)
                    self._close_position(exec_price, dates[i], "TAKE_PROFIT")
                    df.iloc[i, df.columns.get_loc("trade_marker")] = "EXIT(TP)"
                    self.trailing_stop.deactivate()

            # Process signal (only if circuit breaker allows)
            if not is_halted:
                # â”€â”€ Apply Sentiment Filter (Veto Logic) â”€â”€
                # If signal is BUY but sentiment is BEARISH, block the entry
                effective_signal = signal
                if signal == 1 and sentiment_bias == -1:
                    effective_signal = 0
                    if i % 20 == 0: # Avoid log spam
                        logger.warning(f"Trade Blocked by News Sentiment at {dates[i]}")

                if effective_signal == 1 and self._var_gate_blocked():
                    effective_signal = 0
                    if i % 20 == 0:
                        logger.warning(
                            f"VaR gate blocked entry at {dates[i]} "
                            f"(max_var={self.max_var_pct:.2%})"
                        )

                # â”€â”€ 1. AI Conflict Check (Conflict Sizing Requirement) â”€â”€
                ai_bias = int(ai_biases[i]) if i < len(ai_biases) else 0
                ai_conf = float(ai_confidences[i]) if i < len(ai_confidences) else 0.0
                
                is_ai_conflict = False
                # If AI is confident (>25%) but its bias is OPPOSITE to the combined signal
                if ai_conf > 0.25 and effective_signal != 0:
                    if effective_signal != ai_bias:
                        is_ai_conflict = True
                        if i % 20 == 0:
                            logger.info(f"AI CONFLICT: AI disagrees with signal at {dates[i]}. Scaling down to 20% volume.")

                if effective_signal == 1 and self.position == 0:
                    # â”€â”€ AI Confidence-Driven Risk Management â”€â”€
                    
                    # 2. Dynamic Sizing
                    effective_weight = self.position_size
                    
                    if is_ai_conflict:
                        # Conflict Sizing: Only 20% of standard risk to 'test the waters'
                        effective_weight = self.position_size * 0.2
                    else:
                        effective_weight = self._kelly_weight()
                        # Standard confidence-based scaling
                        if self.ai_confidence_weight:
                            effective_weight *= ai_conf
                    
                    if i % 10 == 0:
                        logger.info(f"Position Sizing: {effective_weight:.1%} weight at {dates[i]}")

                    # INITIAL ENTRY
                    exec_price = self._apply_slippage(float(exec_prices[i]), is_buy=True)

                    # â”€â”€ OrderManager Execution â”€â”€
                    weights = pd.Series({ticker: effective_weight})
                    prices_map = {ticker: exec_price}
                    market_map = {ticker: self.market_id}
                    
                    orders = self.order_manager.generate_orders(
                        weights=weights,
                        prices=prices_map,
                        market_map=market_map,
                        capital=self.capital
                    )

                    if orders:
                        order = orders[0]
                        shares = order.quantity
                        invest = order.value
                        commission = order.commission

                        if shares > 0 and invest + commission <= self.capital:
                            self.shares = shares
                            self.entry_price = exec_price
                            self.entry_date = dates[i]
                            self.position = 1
                            self._pyramid_count = 1
                            self.capital -= (invest + commission)

                            # â”€â”€ 3. Hard Stop Implementation â”€â”€
                            # If Conflict or High Confidence entry, use tight ATR multiplier
                            stop_multiplier = self.risk_cfg.atr_multiplier
                            if is_ai_conflict or ai_conf > 0.25:
                                stop_multiplier = self.risk_cfg.tight_atr_multiplier
                                status_msg = "CONFLICT SIZING" if is_ai_conflict else "HIGH CONFIDENCE"
                                logger.warning(f"ðŸ›¡ï¸ {status_msg} HARD STOP: Using tight ATR multiplier {stop_multiplier} at {dates[i]}")

                            # Activate trailing stop
                            if self.use_trailing_stop:
                                self.trailing_stop.activate(exec_price, atr, custom_multiplier=stop_multiplier)

                            df.iloc[i, df.columns.get_loc("trade_marker")] = "BUY"
                            logger.info(f"Execution: BUY {shares} shares of {ticker} at {exec_price:.2f}")

                elif signal == 1 and self.position > 0 and self._pyramid_count < self.max_pyramiding:
                    # PYRAMID: Add to winning position
                    pnl_pct = (price - self.entry_price) / self.entry_price
                    if pnl_pct > 0.02:  # Only pyramid if already profitable
                        exec_price = self._apply_slippage(float(exec_prices[i]), is_buy=True)
                        add_shares = self.position_sizer.volatility_based_size(
                            capital=self.capital,
                            entry_price=exec_price,
                            atr=atr,
                            atr_multiplier=self.risk_cfg.atr_multiplier,
                        ) * 0.5  # Half-size for pyramid entries

                        if add_shares > 0:
                            add_invest = add_shares * exec_price
                            commission = add_invest * self.commission_pct

                            if add_invest + commission <= self.capital * 0.5:
                                # Update average entry price
                                total_cost = self.shares * self.entry_price + add_shares * exec_price
                                self.shares += add_shares
                                self.entry_price = total_cost / self.shares
                                self._pyramid_count += 1
                                self.capital -= (add_invest + commission)
                                df.iloc[i, df.columns.get_loc("trade_marker")] = f"PYRAMID({self._pyramid_count})"

                elif signal == -1 and self.position > 0:
                    # SELL (close all units)
                    exec_price = self._apply_slippage(float(exec_prices[i]), is_buy=False)
                    self._close_position(exec_price, dates[i], "SIGNAL_EXIT")
                    df.iloc[i, df.columns.get_loc("trade_marker")] = "SELL"
                    self.trailing_stop.deactivate()

            # Update equity
            equity = self.capital
            if self.position > 0:
                equity += self.shares * price
            self.equity_curve.append(equity)
            df.iloc[i, df.columns.get_loc("equity")] = equity

            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            df.iloc[i, df.columns.get_loc("drawdown_pct")] = dd_pct

        # Close any remaining position
        if self.position > 0:
            final_price = float(prices[-1])
            exec_price = self._apply_slippage(final_price, is_buy=False)
            self._close_position(exec_price, dates[-1], "END_OF_DATA")
            self.trailing_stop.deactivate()

        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        self._last_run_df = df.copy()
        self._last_signal_column = signal_column
        logger.success(
            f"Backtest complete: {len(self.trades)} trades, "
            f"final equity: ${final_equity:,.2f}"
        )
        return df

    def _close_position(self, price: float, date: str, reason: str) -> None:
        """Close the current position and record the trade."""
        revenue = self.shares * price
        commission = revenue * self.commission_pct
        net_revenue = revenue - commission
        cost_basis = self.shares * self.entry_price
        pnl = net_revenue - cost_basis
        pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

        self.trades.append(
            TradeRecord(
                entry_date=self.entry_date,
                exit_date=date,
                direction="LONG",
                entry_price=self.entry_price,
                exit_price=price,
                shares=self.shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
            )
        )

        self.capital += net_revenue
        self.position = 0
        self.shares = 0.0
        self.entry_price = 0.0
        self.entry_date = ""
        self._pyramid_count = 0

    def get_result(self) -> BacktestResult:
        """Calculate and return comprehensive performance metrics."""
        if not self.trades:
            return BacktestResult(
                total_return=0.0,
                total_pnl=0.0,
                num_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                final_capital=self.initial_capital,
                best_trade=0.0,
                worst_trade=0.0,
                hit_rate=0.0,
                precision_at_entry=0.0,
                expectancy=0.0,
                turnover=0.0,
                turnover_adjusted_sharpe=0.0,
                trades=self.trades,
                equity_curve=self.equity_curve,
            )

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe Ratio
        if len(self.equity_curve) > 1:
            equity_arr = np.array(self.equity_curve)
            returns = np.diff(equity_arr) / equity_arr[:-1]
            sharpe = (
                float(np.mean(returns) / np.std(returns) * np.sqrt(ANNUALIZE_FACTOR))
                if np.std(returns) > 0
                else 0
            )
            # Sortino Ratio â€” penalizes only downside volatility
            downside = returns[returns < 0]
            downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
            sortino = (
                float(np.mean(returns) / downside_std * np.sqrt(ANNUALIZE_FACTOR))
                if downside_std > 0
                else 0.0
            )
        else:
            sharpe = 0.0
            sortino = 0.0

        # Max Drawdown
        if self.equity_curve:
            equity_arr = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            max_dd = float(np.min(drawdown))
        else:
            max_dd = 0.0

        final = self.equity_curve[-1] if self.equity_curve else self.initial_capital

        # Calmar Ratio â€” annualized return / max drawdown
        total_return_pct = (final - self.initial_capital) / self.initial_capital * 100
        calmar = (
            abs(total_return_pct / (max_dd * 100)) if max_dd != 0 else 0.0
        )

        # Validation metrics (out-of-sample friendly)
        precision_at_entry = win_rate * 100
        expectancy = float(np.mean(pnls)) if pnls else 0.0
        years = max(len(self.equity_curve) / ANNUALIZE_FACTOR, 1e-6)
        turnover = len(self.trades) / years
        turnover_adjusted_sharpe = float(sharpe / (1.0 + turnover))
        hit_rate = 0.0
        if self._last_run_df is not None and self._last_signal_column in self._last_run_df.columns:
            next_ret = self._last_run_df["Close"].pct_change().shift(-1)
            sig = pd.to_numeric(self._last_run_df[self._last_signal_column], errors="coerce").fillna(0)
            active = sig != 0
            if active.any():
                correct = ((sig[active] > 0) & (next_ret[active] > 0)) | (
                    (sig[active] < 0) & (next_ret[active] < 0)
                )
                hit_rate = float(correct.mean() * 100)

        return BacktestResult(
            total_return=total_return_pct,
            total_pnl=total_pnl,
            num_trades=len(self.trades),
            win_rate=win_rate * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd * 100,
            final_capital=final,
            best_trade=max(pnls),
            worst_trade=min(pnls),
            hit_rate=hit_rate,
            precision_at_entry=precision_at_entry,
            expectancy=expectancy,
            turnover=turnover,
            turnover_adjusted_sharpe=turnover_adjusted_sharpe,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

    def print_report(self) -> None:
        """Print a formatted performance report to console."""
        r = self.get_result()
        report = f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘       BACKTEST PERFORMANCE REPORT            â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  Initial Capital:   ${self.initial_capital:>14,.2f}       â•‘
â•‘  Final Capital:     ${r.final_capital:>14,.2f}       â•‘
â•‘  Total Return:      {r.total_return:>14.2f}%      â•‘
â•‘  Total P&L:         ${r.total_pnl:>14,.2f}       â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  Trades:            {r.num_trades:>14d}        â•‘
â•‘  Win Rate:          {r.win_rate:>14.1f}%      â•‘
â•‘  Avg Win:           ${r.avg_win:>14,.2f}       â•‘
â•‘  Avg Loss:          ${r.avg_loss:>14,.2f}       â•‘
â•‘  Best Trade:        ${r.best_trade:>14,.2f}       â•‘
â•‘  Worst Trade:       ${r.worst_trade:>14,.2f}       â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  Profit Factor:     {r.profit_factor:>14.2f}        â•‘
â•‘  Sharpe Ratio:      {r.sharpe_ratio:>14.2f}        â•‘
â•‘  Sortino Ratio:     {r.sortino_ratio:>14.2f}        â•‘
â•‘  Calmar Ratio:      {r.calmar_ratio:>14.2f}        â•‘
â•‘  Max Drawdown:      {r.max_drawdown:>14.2f}%      â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""
        print(report)
