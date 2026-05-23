"""
Volume Profile Analyzer
=======================
Calculates horizontal volume distribution (Price-at-Volume).
Identifies:
- POC (Point of Control): Price level with highest volume.
- VAH (Value Area High): Upper bound containing 70% of volume.
- VAL (Value Area Low): Lower bound containing 70% of volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from loguru import logger
from src.config import VolumeProfileConfig


class VolumeProfile:
    """
    Expert-level Volume Profile (TPO/Volume based).
    Used to detect institutional accumulation and high-liquidity zones.
    """

    def __init__(self, config: VolumeProfileConfig | None = None):
        self.cfg = config or VolumeProfileConfig()

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price/volume data to generate profile metrics.
        
        Args:
            data: Prices (High, Low, Close) and Volume. 
                 Only the last 'lookback' rows are used for the 'Active Session'.
        """
        if data.empty or len(data) < 5:
            return {"status": "INSUFFICIENT_DATA"}

        # Use recent lookback
        df = data.tail(self.cfg.lookback).copy()
        
        # Calculate price range for bins
        price_min = df["Low"].min()
        price_max = df["High"].max()
        
        if price_max == price_min:
            return {"status": "NO_VOLATILITY"}

        # Create bins
        bins = np.linspace(price_min, price_max, self.cfg.num_bins + 1)
        bin_counts = np.zeros(self.cfg.num_bins)
        
        # Distribute volume across bins (Simplified: Volume assigned to Close price bin)
        # Note: A more advanced version would use High-Low range per candle.
        for _, row in df.iterrows():
            price = row["Close"]
            vol = row["Volume"]
            idx = np.digitize(price, bins) - 1
            idx = max(0, min(idx, self.cfg.num_bins - 1))
            bin_counts[idx] += vol

        # 1. Point of Control (POC)
        poc_idx = np.argmax(bin_counts)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # 2. Value Area Calculation (70% total volume around POC)
        total_vol = bin_counts.sum()
        target_vol = total_vol * self.cfg.value_area_pct
        
        # Search outwards from POC
        va_indices = {poc_idx}
        current_vol = bin_counts[poc_idx]
        
        low_ptr = poc_idx - 1
        high_ptr = poc_idx + 1
        
        while current_vol < target_vol and (low_ptr >= 0 or high_ptr < self.cfg.num_bins):
            # Compare volume to the left and right
            v_low = bin_counts[low_ptr] if low_ptr >= 0 else 0
            v_high = bin_counts[high_ptr] if high_ptr < self.cfg.num_bins else 0
            
            if v_low >= v_high and low_ptr >= 0:
                current_vol += v_low
                va_indices.add(low_ptr)
                low_ptr -= 1
            elif high_ptr < self.cfg.num_bins:
                current_vol += v_high
                va_indices.add(high_ptr)
                high_ptr += 1
            else:
                break
        
        va_min_idx = min(va_indices)
        va_max_idx = max(va_indices)
        
        vah = bins[va_max_idx + 1]
        val = bins[va_min_idx]

        # 3. Preparation for Visualization (Histogram)
        histogram = []
        for i in range(self.cfg.num_bins):
            histogram.append({
                "price": round((bins[i] + bins[i+1]) / 2, 4),
                "volume": float(bin_counts[i]),
                "is_poc": i == poc_idx,
                "in_va": i in va_indices
            })

        logger.success(f"Volume Profile complete: POC={poc_price:.2f}, VA={val:.2f}-{vah:.2f}")
        
        return {
            "status": "SUCCESS",
            "poc": poc_price,
            "vah": vah,
            "val": val,
            "histogram": histogram,
            "total_volume": float(total_vol)
        }
