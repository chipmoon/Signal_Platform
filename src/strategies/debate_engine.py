"""
Algorithmic Debate Engine
==========================
Synthesizes signals from 15+ internal strategies to produce a consolidated 
Bull vs Bear score for high-conviction decision making.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

class DebateEngine:
    """Consolidates complex signals into a single actionable score."""

    @staticmethod
    def compute_score_series(df: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic debate score/signal columns for each bar."""
        result = df.copy()

        for col in ("smc_signal", "wyckoff_signal", "bank_signal", "ai_bias", "real_flow_signal"):
            if col not in result.columns:
                result[col] = 0

        score = (
            result["smc_signal"].fillna(0) * 1.20
            + result["wyckoff_signal"].fillna(0) * 1.00
            + result["bank_signal"].fillna(0) * 0.70
            + result["ai_bias"].fillna(0) * 0.60
            + result["real_flow_signal"].fillna(0) * 0.80
        ).astype(float)

        if "SMA_200" in result.columns:
            score += np.where(
                result["Close"] > result["SMA_200"].fillna(result["Close"]),
                0.25,
                -0.25,
            )

        from src.risk_manager import calculate_rsi
        rsi = calculate_rsi(result["Close"], 14).fillna(50.0)
        score += np.where(rsi < 30, 0.30, 0.0)
        score += np.where(rsi > 70, -0.30, 0.0)

        result["debate_score"] = score.clip(-5.0, 5.0)
        result["debate_signal"] = 0
        result.loc[result["debate_score"] >= 1.0, "debate_signal"] = 1
        result.loc[result["debate_score"] <= -1.0, "debate_signal"] = -1
        return result

    @staticmethod
    def debate_trade(df: pd.DataFrame, ticker: str = "") -> dict:
        """
        Interrogate strategies to calculate Bull vs Bear evidence.
        Higher Net Score = Higher Conviction.
        """
        if df.empty:
            return {"net_score": 0.0, "status": "Empty Data"}

        scored = DebateEngine.compute_score_series(df)
        last = scored.iloc[-1]
        bull_score = 0.0
        bear_score = 0.0
        evidence_log = []

        # ── 1. TREND QUALITY (Base) ──
        # Check against Sea Level (SMA 200)
        from src.strategies.regime_detector import RegimeDetector
        regime_data = RegimeDetector.identify(df)
        regime = regime_data.get("regime")
        
        if regime == "BULL":
            bull_score += 1.0
            evidence_log.append("🐂 BULL: Market structure is supportive")
        elif regime == "BEAR":
            bear_score -= 1.0
            evidence_log.append("🐻 BEAR: Trend is hostile")

        # ── 2. INSTITUTIONAL STRUCTURE (SMC) ──
        # If SMC signaled a Buy/Sell, it's high conviction
        if "smc_signal" in last:
            if last["smc_signal"] > 0:
                bull_score += 2.0
                evidence_log.append("🏦 SMC: Institutional demand zone confirmed")
            elif last["smc_signal"] < 0:
                bear_score -= 2.0
                evidence_log.append("🏦 SMC: Supply zone rejection detected")

        # ── 3. MARKET CYCLE (Wyckoff) ──
        if "wyckoff_signal" in last:
            if last["wyckoff_signal"] > 0:
                bull_score += 1.5
                evidence_log.append("📦 WYCKOFF: Accumulation Spring/LPS confirmed")
            elif last["wyckoff_signal"] < 0:
                bear_score -= 1.5
                evidence_log.append("📉 WYCKOFF: Distribution UTAD detected")

        # ── 4. SMART MONEY FLOW (Bank Participation) ──
        if "bank_signal" in last:
            if last["bank_signal"] > 0:
                bull_score += 1.0
                evidence_log.append("💼 BANK: Volumetric accumulation detected")
            elif last["bank_signal"] < 0:
                bear_score -= 1.0
                evidence_log.append("💼 BANK: Institutional unloading noticed")

        # ── 5. AI ML BIAS ──
        if "ai_bias" in last:
            if last["ai_bias"] > 0:
                bull_score += 0.8
                evidence_log.append("🤖 AI: ML Predictor is bullish")
            elif last["ai_bias"] < 0:
                bear_score -= 0.8
                evidence_log.append("🤖 AI: ML Predictor warns of decline")

        # ── 6. EXHAUSTION / QUALITY FILTERS ──
        from src.risk_manager import calculate_rsi
        rsi = calculate_rsi(df["Close"], 14).iloc[-1]
        if rsi > 70:
            bear_score -= 0.5
            evidence_log.append("⚠️ RSI: Approaching overbought conditions")
        elif rsi < 30:
            bull_score += 0.5
            evidence_log.append("💡 RSI: Deep oversold value-buy zone")

        # Final synthesis (blend static evidence with per-bar debate score)
        net_score = round(bull_score + bear_score + float(last.get("debate_score", 0.0)) * 0.25, 2)
        
        # Verdict logic
        if net_score >= 3.0:
            verdict = "🔥 STRONG CONVICTION BUY"
        elif net_score >= 1.0:
            verdict = "🟢 MODERATE BUY"
        elif net_score <= -3.0:
            verdict = "🚫 HIGH RISK: AVOID / EMERGENCY EXIT"
        elif net_score <= -1.0:
            verdict = "🔴 WEAK: BEARISH PRESSURE"
        else:
            verdict = "↔️ NEUTRAL: WAIT FOR SETUP"

        return {
            "net_score": net_score,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "verdict": verdict,
            "evidence": evidence_log,
            "regime": regime,
            "is_overextended": regime_data.get("is_overextended", False),
            "debate_signal": int(last.get("debate_signal", 0)),
            "debate_score": float(last.get("debate_score", 0.0)),
        }
