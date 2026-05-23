"""
Strategy Protocol
=================
Defines the interface that ALL trading strategies must conform to.
Uses structural subtyping (Protocol) — no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TradingStrategy(Protocol):
    """Interface for trading strategy implementations.

    Any class with a ``generate_signals`` method matching this signature
    is considered a valid strategy (duck typing via Protocol).
    """

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Analyze data and append signal columns.

        Args:
            data: DataFrame with at minimum ``Date``, ``Close`` columns.
            **kwargs: Strategy-specific extras (e.g. ``usd_data``).

        Returns:
            DataFrame with the following columns appended:
            - ``{prefix}_signal``: int (-1, 0, 1)
            - ``{prefix}_signal_reason``: str
        """
        ...
