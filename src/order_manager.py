"""
Order Manager
=============
Handles the translation of portfolio weights into executable orders.
Applies market-specific lot size filters and rounding rules.

Markets Supported:
- VN: HOSE/HNX (Lot size 100)
- US: NYSE/NASDAQ (Lot size 1, optional fractional)
- Commodities: Futures (Contract multipliers)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .config import BacktestConfig


@dataclass
class Order:
    """Represents a generated trade instruction."""
    symbol: str
    market: str
    action: str  # BUY, SELL, HOLD
    quantity: float
    price: float
    value: float
    commission: float
    slippage: float
    notes: str = ""


class OrderManager:
    """
    Manages execution logic and lot filtering.
    """

    # Market Configuration
    MARKET_RULES = {
        "VN": {"lot_size": 100, "allow_fractional": False, "currency": "VND"},
        "US": {"lot_size": 1, "allow_fractional": False, "currency": "USD"},
        "COMMODITY": {"lot_size": 1, "allow_fractional": False, "currency": "USD"},
        "MACRO": {"lot_size": 1, "allow_fractional": False, "currency": "USD"},
    }

    # Contract Multipliers for Futures
    MULTIPLIERS = {
        "GC=F": 100,    # Gold: 100oz
        "SI=F": 5000,   # Silver: 5000oz
        "CL=F": 1000,   # Crude: 1000bbl
        "NG=F": 10000,  # Gas: 10000mmBtu
        "HG=F": 25000,  # Copper: 25000lb
    }

    def __init__(self, config: BacktestConfig = BacktestConfig()):
        """
        Initialize Order Manager.
        
        Args:
            config: Backtest configuration for commission/slippage settings.
        """
        self.config = config

    def generate_orders(
        self,
        weights: pd.Series,
        prices: Dict[str, float],
        market_map: Dict[str, str],
        capital: float,
        current_positions: Dict[str, float] = None
    ) -> List[Order]:
        """
        Generate orders from portfolio weights.
        
        Args:
            weights: Series of target weights (0.0 to 1.0)
            prices: Current market prices for each asset
            market_map: Map of symbol -> Market ID (VN, US, etc.)
            capital: Total available capital for allocation
            current_positions: Map of symbol -> current share quantity
            
        Returns:
            List of Order objects
        """
        orders = []
        current_positions = current_positions or {}

        for symbol, target_pct in weights.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                logger.warning(f"Skipping {symbol}: Price unavailable or zero")
                continue

            market = market_map.get(symbol, "US")
            rules = self.MARKET_RULES.get(market, self.MARKET_RULES["US"])
            multiplier = self.MULTIPLIERS.get(symbol, 1.0)

            # 1. Calculate Target Quantity
            # Target Value = Capital * Weight
            # Quantity = Target Value / (Price * Multiplier)
            target_value = capital * target_pct
            raw_target_qty = target_value / (price * multiplier)

            # 2. Apply Lot Filtering & Rounding
            lot_size = rules["lot_size"]
            if rules["allow_fractional"]:
                refined_qty = raw_target_qty
            else:
                # Round down to nearest lot size to avoid over-utilizing capital
                refined_qty = math.floor(raw_target_qty / lot_size) * lot_size

            # 3. Determine Action
            current_qty = current_positions.get(symbol, 0.0)
            diff_qty = refined_qty - current_qty

            if abs(diff_qty) < lot_size:
                # No significant change needed or less than 1 lot
                continue

            action = "BUY" if diff_qty > 0 else "SELL"
            abs_qty = abs(diff_qty)
            order_value = abs_qty * price * multiplier
            
            # 4. Cost Estimation
            commission = order_value * self.config.commission_pct
            slippage = order_value * self.config.slippage_pct

            orders.append(Order(
                symbol=symbol,
                market=market,
                action=action,
                quantity=abs_qty,
                price=price,
                value=order_value,
                commission=commission,
                slippage=slippage,
                notes=f"Weight: {target_pct:.2%}, Lot Size: {lot_size}"
            ))

        return orders

    def validate_capital_requirements(self, orders: List[Order], available_capital: float) -> bool:
        """
        Ensure total order value + costs doesn't exceed available capital.
        """
        total_required = sum(o.value + o.commission + o.slippage for o in orders if o.action == "BUY")
        if total_required > available_capital:
            logger.error(f"Capital Check FAIL: Required ${total_required:.2f}, Available ${available_capital:.2f}")
            return False
        return True
