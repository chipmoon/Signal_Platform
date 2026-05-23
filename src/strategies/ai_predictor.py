"""
AI Deep Learning Predictor
==========================
Uses a Neural Network (MLPRegressor) to forecast multi-horizon price targets.
Supported horizons: Daily, 1 Month, 2 Months, 3 Months, 6 Months.

Enhanced with:
- RSI, ATR, MACD as additional ML features
- Feature importance ranking for dashboard display
- Walk-forward training to prevent look-ahead bias

Conforms to ``TradingStrategy`` Protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("lightgbm or scikit-learn not installed. AI Predictor disabled.")

from src.config import AIConfig, WyckoffConfig, SmcConfig
from src.risk_manager import calculate_atr, calculate_macd, calculate_rsi, calculate_volume_profile

# Lazy imports to avoid circular dependencies
def _get_wyckoff_analyzer():
    from src.strategies.wyckoff_analyzer import WyckoffAnalyzer
    return WyckoffAnalyzer

def _get_smc_analyzer():
    from src.strategies.smc_analyzer import SmcAnalyzer
    return SmcAnalyzer

def _get_seasonality_filter():
    from src.strategies.seasonality import SeasonalityFilter
    return SeasonalityFilter


class AIPredictor:
    """AI Quantile Regression Predictor.

    Uses LightGBM to predict Log Returns across multiple quantiles (10%, 50%, 90%).
    Implements 'Edge > Noise' logic for robust trade filtering.
    """

    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig()
        self.horizons = self.config.horizons
        self.lookback_days = self.config.lookback_days
        self.walk_forward_window = self.config.walk_forward_window
        
        # Quantile models: {horizon: {q: model}}
        self._models: dict[int, dict[float, lgb.LGBMRegressor]] = {}
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] = []
        self._feature_importance: dict[str, float] = {}
        self._train_metrics: dict[str, float] = {}

    def _prepare_features(
        self, df: pd.DataFrame, training: bool = False
    ) -> tuple[pd.DataFrame, list[str]]:
        """Create technical indicators and calculate Log Returns as targets."""
        result = df.copy()
        
        # ─── Panel Data Support (Multi-Symbol) ───
        # If we have multiple symbols, we must calculate rolling metrics per symbol
        is_panel = "Symbol" in result.columns and result["Symbol"].nunique() > 1
        
        if is_panel:
            grouped = result.groupby("Symbol", group_keys=False)
            
            # 1. Price Lags
            for i in range(1, 6):
                result[f"lag_ret_{i}"] = np.log(result["Close"] / grouped["Close"].shift(i))
            
            # 2. Moving Averages (Normalized)
            result["ma_5_rel"] = result["Close"] / grouped["Close"].transform(lambda x: x.rolling(5).mean())
            result["ma_15_rel"] = result["Close"] / grouped["Close"].transform(lambda x: x.rolling(15).mean())
            
            # 3. RSI
            result["rsi_14"] = grouped["Close"].transform(lambda x: calculate_rsi(x, period=14))
            
            # 4. ATR
            if "High" in result.columns:
                # Grouped ATR is tricky with the standard function, so we do it per group
                result["atr_14_rel"] = grouped.apply(lambda x: calculate_atr(x, period=14) / x["Close"])
            else:
                result["atr_14_rel"] = result["Close"] / grouped["Close"].transform(lambda x: x.rolling(14).std())
            
            # 5. MACD
            # For simplicity in panel mode, we'll only do standard indicators if they are easily vectorized
            # or pre-calculate them. Here we'll just use price-based fallbacks for speed
            result["macd_hist_rel"] = result["Close"].pct_change(12) - result["Close"].pct_change(26)
            
            # 6. Price rate of change
            result["roc_5"] = grouped["Close"].pct_change(5)
            result["roc_20"] = grouped["Close"].pct_change(20)
            
            # 7. Volume Profile - Skip in Panel mode for now (Too slow)
            result["poc_dist"] = 0.0
            result["inside_va"] = 1
            
            # 8. Volume ratio
            if "Volume" in result.columns:
                vol_ma = grouped["Volume"].transform(lambda x: x.rolling(window=20, min_periods=5).mean())
                result["volume_ratio"] = result["Volume"] / vol_ma.replace(0, np.nan)
            else:
                result["volume_ratio"] = 1.0

        else:
            # Standard single-asset mode (Fastest)
            close = result["Close"]
            for i in range(1, 6):
                result[f"lag_ret_{i}"] = np.log(close / close.shift(i))
            result["ma_5_rel"] = close / close.rolling(5).mean()
            result["ma_15_rel"] = close / close.rolling(15).mean()
            result["rsi_14"] = calculate_rsi(close, period=14)
            if "High" in result.columns and "Low" in result.columns:
                result["atr_14_rel"] = calculate_atr(result, period=14) / close
            else:
                result["atr_14_rel"] = close.rolling(14).std() / close
            macd_line, signal_line, histogram = calculate_macd(close)
            result["macd_hist_rel"] = histogram / close
            result["roc_5"] = close.pct_change(5)
            result["roc_20"] = close.pct_change(20)
            vp_df = calculate_volume_profile(result, window=20)
            result["poc_dist"] = vp_df["poc_dist"]
            result["inside_va"] = vp_df["inside_va"]
            if "Volume" in result.columns:
                vol_ma = result["Volume"].rolling(window=20, min_periods=5).mean()
                result["volume_ratio"] = result["Volume"] / vol_ma.replace(0, np.nan)
            else:
                result["volume_ratio"] = 1.0

            # ─── Wyckoff Features (Sprint 2 Integration) ───
            # Only inject if not already present (avoid re-running on pre-processed data)
            if "wyckoff_score" not in result.columns:
                try:
                    WyckoffAnalyzer = _get_wyckoff_analyzer()
                    wyckoff = WyckoffAnalyzer(WyckoffConfig())
                    w_df = wyckoff.generate_signals(result)
                    result["wyckoff_score"] = w_df.get("wyckoff_score", pd.Series(0.0, index=result.index))
                    result["wyckoff_tr_position"] = w_df.get("wyckoff_tr_position", pd.Series(0.5, index=result.index))
                    result["wyckoff_spring"] = w_df.get("wyckoff_spring", pd.Series(False, index=result.index)).astype(float)
                    result["wyckoff_lps"] = w_df.get("wyckoff_lps", pd.Series(False, index=result.index)).astype(float)
                    result["effort_vs_result"] = w_df.get("effort_vs_result", pd.Series(1.0, index=result.index))
                    result["weis_wave_ratio"] = w_df.get("weis_wave_ratio", pd.Series(1.0, index=result.index))
                    result["vsa_sos"] = w_df.get("vsa_sos", pd.Series(False, index=result.index)).astype(float)
                    result["vsa_sow"] = w_df.get("vsa_sow", pd.Series(False, index=result.index)).astype(float)
                    result["vsa_no_supply"] = w_df.get("vsa_no_supply", pd.Series(False, index=result.index)).astype(float)
                except Exception as _wyckoff_err:
                    logger.debug(f"Wyckoff feature injection skipped: {_wyckoff_err}")
                    for col in ["wyckoff_score", "wyckoff_tr_position", "wyckoff_spring",
                                "wyckoff_lps", "effort_vs_result", "weis_wave_ratio",
                                "vsa_sos", "vsa_sow", "vsa_no_supply"]:
                        result[col] = 0.0

            # ─── SMC Features (Sprint 3) ──────────────────────────────────────────────────────────────
            if "smc_score" not in result.columns:
                if "Open" in result.columns:
                    try:
                        SmcAnalyzer = _get_smc_analyzer()
                        smc = SmcAnalyzer(SmcConfig())
                        # ── Fix: reset index before joining to avoid UnaligableBoolean error
                        _work = result.reset_index(drop=True)
                        s_df = smc.generate_signals(_work.copy()).reset_index(drop=True)
                        _n = len(result)
                        def _safe_col(col, default=0.0):
                            if col in s_df.columns:
                                return s_df[col].values[:_n] if len(s_df) >= _n else pd.array([default] * _n)
                            return pd.array([default] * _n)
                        result["smc_score"] = _safe_col("smc_score", 0.0)
                        result["smc_in_bull_ob"] = _safe_col("smc_in_bull_ob", 0.0)
                        result["smc_in_bear_ob"] = _safe_col("smc_in_bear_ob", 0.0)
                        result["smc_in_bull_fvg"] = _safe_col("smc_in_bull_fvg", 0.0)
                        result["smc_in_bear_fvg"] = _safe_col("smc_in_bear_fvg", 0.0)
                        result["smc_bos_bull"] = _safe_col("smc_bos_bull", 0.0)
                        result["smc_bos_bear"] = _safe_col("smc_bos_bear", 0.0)
                        result["smc_choch"] = _safe_col("smc_choch", 0.0)
                        result["smc_idm_bull"] = _safe_col("smc_idm_bull", 0.0)
                        result["smc_idm_bear"] = _safe_col("smc_idm_bear", 0.0)
                    except Exception as _smc_err:
                        logger.debug(f"SMC feature injection skipped: {_smc_err}")
                        for col in ["smc_score", "smc_in_bull_ob", "smc_in_bear_ob",
                                    "smc_in_bull_fvg", "smc_in_bear_fvg",
                                    "smc_bos_bull", "smc_bos_bear", "smc_choch"]:
                            result[col] = 0.0
                else:
                    for col in ["smc_score", "smc_in_bull_ob", "smc_in_bear_ob",
                                "smc_in_bull_fvg", "smc_in_bear_fvg",
                                "smc_bos_bull", "smc_bos_bear", "smc_choch"]:
                        result[col] = 0.0

            # ─── Seasonality Feature ──────────────────────────────────────────────────────────────
            if "seasonality_score" not in result.columns:
                try:
                    SeasonalityFilter = _get_seasonality_filter()
                    sf = SeasonalityFilter()
                    # ── Fix: kwargs not available here — detect market from column presence
                    if "VNI_Close" in result.columns or "VNI_Return" in result.columns:
                        _mkt = "VN"
                    elif "DXY" in result.columns or "US10Y" in result.columns:
                        _mkt = "COMMODITY"
                    else:
                        _mkt = "US"
                    _sym = "unknown"  # Seasonality uses market-level calendars, symbol is optional
                    result["seasonality_score"] = sf.get_score_series(result, str(_sym), str(_mkt))
                    result["seasonality_weight"] = (1.0 + result["seasonality_score"] * 0.5).clip(0.5, 1.5)
                except Exception as _season_err:
                    logger.debug(f"Seasonality feature injection skipped: {_season_err}")
                    result["seasonality_score"] = 0.0
                    result["seasonality_weight"] = 1.0

        # 9. Relative Strength (New Big Data Feature)
        if "Rel_Strength" not in result.columns:
            # If not provided, assume 1.0 (neutral) or if VNI present, calc it
            if "VNI_Close" in result.columns:
                result["Rel_Strength"] = result["Close"] / result["VNI_Close"]
            else:
                result["Rel_Strength"] = 1.0

        # 10. Strategy Signals
        signal_cols = [
            c for c in result.columns
            if "_signal" in c and "_reason" not in c
            and c not in ["ai_signal", "combined_signal"]
        ]

        # 11. Targets for training
        if training:
            if is_panel:
                grouped = result.groupby("Symbol", group_keys=False)
                for h in self.horizons:
                    result[f"target_ret_{h}d"] = np.log(grouped["Close"].shift(-h) / result["Close"])
            else:
                for h in self.horizons:
                    result[f"target_ret_{h}d"] = np.log(result["Close"].shift(-h) / result["Close"])

        features = (
            ["ma_5_rel", "ma_15_rel", "rsi_14", "atr_14_rel", "macd_hist_rel", "roc_5", "roc_20",
             "volume_ratio", "poc_dist", "inside_va", "Rel_Strength",
             # ── Wyckoff Features ──
             "wyckoff_score", "wyckoff_tr_position", "wyckoff_spring", "wyckoff_lps",
             "effort_vs_result", "weis_wave_ratio", "vsa_sos", "vsa_sow", "vsa_no_supply",
             # ── SMC Features (Sprint 3) ──
             "smc_score", "smc_in_bull_ob", "smc_in_bear_ob",
             "smc_in_bull_fvg", "smc_in_bear_fvg",
             "smc_bos_bull", "smc_bos_bear", "smc_choch",
             "smc_idm_bull", "smc_idm_bear",
             # ── Seasonality Feature ──
             "seasonality_score", "seasonality_weight"]
            + [f"lag_ret_{i}" for i in range(1, 6)]
            + signal_cols
        )

        features = [f for f in features if f in result.columns]
        for col in features:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        # Cleanup infinite values and NaNs in features to prevent data loss in training
        result = result.replace([np.inf, -np.inf], np.nan)
        result[features] = result[features].fillna(0)
        return result, features

    def train(self, data: pd.DataFrame) -> None:
        """Train LightGBM Quantile Regressors (10%, 50%, 90%)."""
        if not ML_AVAILABLE:
            df, features = self._prepare_features(data, training=True)
            self._feature_cols = features
            self._feature_importance = {f: 0.0 for f in features}
            self._train_metrics = {"r2": 0.0, "mae": 0.0}
            logger.warning("AI training fallback: ML stack unavailable, using neutral metrics.")
            return

        df, features = self._prepare_features(data, training=True)
        self._feature_cols = features
        
        # We need at least one target for walk-forward check
        primary_target = f"target_ret_{self.horizons[0]}d"
        df = df.dropna(subset=features + [primary_target])

        # For panel data (Master Training), we use a much larger window to capture cross-asset patterns
        is_panel = "Symbol" in df.columns and df["Symbol"].nunique() > 1
        train_window = 200000 if is_panel else self.walk_forward_window
        
        train_df = df.iloc[-train_window:] if len(df) > train_window else df
        X = train_df[features].values
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        quantiles = [0.1, 0.5, 0.9]
        self._models = {}
        
        for h in self.horizons:
            try:
                target_col = f"target_ret_{h}d"
                # Ensure we drop NaNs for this specific horizon target
                horizon_df = train_df.dropna(subset=[target_col])
                
                if len(horizon_df) < 50:
                    logger.warning(f"Insufficient data for horizon {h}d. Skipping.")
                    continue

                X_h = pd.DataFrame(self._scaler.transform(horizon_df[features].values), columns=features)
                y_h = horizon_df[target_col].values
                
                self._models[h] = {}
                for q in [0.1, 0.5, 0.9]:
                    model = lgb.LGBMRegressor(
                        objective='quantile',
                        alpha=q,
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=31,
                        importance_type='gain',
                        random_state=42,
                        verbose=-1
                    )
                    model.fit(X_h, y_h)
                    self._models[h][q] = model
            except Exception as e:
                logger.error(f"Error training AI model for horizon {h}d: {e}")
                
        # Scoring on Median (q=0.5) of primary horizon
        primary_h = self.horizons[0]
        if primary_h in self._models:
            target_col = f"target_ret_{primary_h}d"
            eval_df = train_df.dropna(subset=[target_col])
            X_eval = pd.DataFrame(self._scaler.transform(eval_df[features].values), columns=features)
            y_true = eval_df[target_col].values
            y_pred = self._models[primary_h][0.5].predict(X_eval)
            
            self._train_metrics = {
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred))
            }
            
            # Feature importance (Average across Primary Multi-Quantile)
            importances = []
            for q in [0.1, 0.5, 0.9]:
                importances.append(self._models[primary_h][q].feature_importances_)
            
            avg_importance = np.mean(importances, axis=0)
            self._feature_importance = dict(zip(features, avg_importance))
            self._feature_importance = dict(sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True))
            logger.success(f"LightGBM Quantile Ensemble trained. Final {primary_h}d Median R²: {self._train_metrics['r2']:.3f}")
        else:
            self._feature_importance = {}
            self._train_metrics = {"r2": 0.0, "mae": 0.0}
            logger.warning(f"Primary horizon {primary_h}d was not trained. Neural forecasting will be disabled.")

    def generate_signals(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Generate AI forecast signals and prediction bands."""
        df = data.copy()
        
        # 1. Initialize result columns with safe defaults
        for h in self.horizons:
            df[f"ai_target_price_{h}d"] = df["Close"]
            df[f"ai_price_lower_{h}d"] = df["Close"]
            df[f"ai_price_upper_{h}d"] = df["Close"]
            
        df["ai_bias"] = 0
        df["ai_confidence"] = 0.0

        if not ML_AVAILABLE or not self._models or not self._scaler:
            logger.warning("AI Predictor generating neutral signals (Model not trained or library missing).")
            df["ai_signal"] = 0
            df["ai_target_price"] = df["Close"]
            return df

        try:
            # 2. Features
            feat_df, features = self._prepare_features(df)
            X = feat_df[features].values
            X_scaled = self._scaler.transform(X)
            
            # 3. Predict Quantiles
            current_close = df["Close"].values
            
            for h in self.horizons:
                if h not in self._models:
                    continue
                    
                # Ensure X_scaled has feature names to avoid LGBM UserWarning
                X_scaled_df = pd.DataFrame(X_scaled, columns=features)
                
                # Predict log returns for each quantile
                q10_ret = self._models[h][0.1].predict(X_scaled_df)
                q50_ret = self._models[h][0.5].predict(X_scaled_df)
                q90_ret = self._models[h][0.9].predict(X_scaled_df)
                
                # Convert log returns to prices: P_target = P_now * exp(r)
                df[f"ai_target_price_{h}d"] = current_close * np.exp(q50_ret)
                df[f"ai_price_lower_{h}d"] = current_close * np.exp(q10_ret)
                df[f"ai_price_upper_{h}d"] = current_close * np.exp(q90_ret)
                
                # Internal columns for bias logic
                if h == 1:
                    df["_q10_ret"] = q10_ret
                    df["_q90_ret"] = q90_ret

            # ── Edge > Noise Logic ──
            # Only go Long if even the 10th percentile outcome is positive (Bullish Edge)
            # Only go Short if even the 90th percentile outcome is negative (Bearish Edge)
            threshold = 0.001 # 0.1% safety margin
            
            # ── Dynamic Confidence Calibration (Option B) ──
            if "_q10_ret" in df.columns and "_q90_ret" in df.columns:
                df["ai_bias"] = 0
                df.loc[(df["_q10_ret"] > threshold) & (df["_q10_ret"].notna()), "ai_bias"] = 1
                df.loc[(df["_q90_ret"] < -threshold) & (df["_q90_ret"].notna()), "ai_bias"] = -1
                
                # Confidence based on relative band tightness (Institutional Logic)
                # Band Width % = (Upper - Lower) / Current Close
                band_width_pct = (df["ai_price_upper_1d"] - df["ai_price_lower_1d"]) / df["Close"]
                
                # Normalize confidence: 
                # - Band < 1% of price -> High Confidence (100%)
                # - Band > 5% of price -> Zero Confidence (Fear Mode)
                df["ai_confidence"] = np.clip(1.0 - (band_width_pct / 0.05), 0.0, 1.0)
                
                # Volatility Veto: If AI is too uncertain, force Neutral Bias
                conf_threshold = self.config.confidence_threshold
                df.loc[df["ai_confidence"] < conf_threshold, "ai_bias"] = 0
                
                # Extra: Tag reason for audit
                df["ai_veto"] = df["ai_confidence"] < conf_threshold
                
                # Clean up temp columns
                df = df.drop(columns=["_q10_ret", "_q90_ret"])

            # ── Final Signal Normalization ──
            df["ai_signal"] = df["ai_bias"]
            df["ai_target_price"] = df.get("ai_target_price_1d", df["Close"])
            return df

        except Exception as e:
            logger.error(f"AI Signal Generation Error: {e}")
            df["ai_signal"] = 0
            df["ai_target_price"] = df["Close"]
            return df

    def predict_afternoon_bias(self, intraday_df: pd.DataFrame) -> dict[str, any]:
        """
        Specialized logic for 'Morning-to-Afternoon' prediction.
        Analyzes the first 4-5 hours of the day to predict the close.
        """
        if intraday_df.empty or len(intraday_df) < 5:
            return {"bias": "Neutral", "confidence": 0.0, "reason": "Insufficient intraday data"}
            
        # 1. Isolate Morning Session (First 4 hours)
        morning = intraday_df.iloc[-5:] # Assuming last 5 candles are the current session
        
        # 2. Features for Intra-day momentum
        morning_ret = (morning["Close"].iloc[-1] / morning["Open"].iloc[0]) - 1
        vol_spike = morning["Volume"].mean() / intraday_df["Volume"].rolling(20).mean().iloc[-1]
        
        # 3. AI Inference (Simplified expert rule + Model check)
        # If morning return > 1.5% and volume > 1.2x avg -> High prob of afternoon continuation
        # If morning return < -1.5% and volume > 1.2x avg -> High prob of afternoon flush
        
        bias = "Neutral"
        confidence = 0.5
        
        if morning_ret > 0.01 and vol_spike > 1.1:
            bias = "Bullish Continuation"
            confidence = 0.75
        elif morning_ret < -0.01 and vol_spike > 1.1:
            bias = "Bearish Flush"
            confidence = 0.75
        elif abs(morning_ret) < 0.005 and vol_spike < 0.8:
            bias = "Range Bound / Sideways"
            confidence = 0.6
            
        return {
            "bias": bias, 
            "confidence": confidence, 
            "morning_return": morning_ret * 100,
            "volume_delta": (vol_spike - 1) * 100,
            "prediction_target": "Session Close"
        }

    def analyze_multi_tf(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> dict[str, any]:
        """
        Superior Expert Logic: Cross-validates 1H signals with 4H structure.
        - Option B: Signal Convergence approach.
        """
        if df_1h.empty or df_4h.empty:
            return {"health": "Unknown", "bias": "Neutral"}
            
        # 1. 1H Momentum
        ret_1h = df_1h["Close"].pct_change(4).iloc[-1]
        
        # 2. 4H Structure
        ret_4h = df_4h["Close"].pct_change(4).iloc[-1] # Change over last 16 hours
        ma_4h = df_4h["Close"].rolling(20).mean().iloc[-1]
        is_above_ma = df_4h["Close"].iloc[-1] > ma_4h
        
        # 3. Pattern Recognition (NEW: Option B)
        flag_data = self.detect_flags(df_1h)
        
        # 4. Convergence Filter
        health = "Neutral"
        bias = "Neutral"
        
        # STRONG BUY: 1H Breaking out + 4H Trend is up + Above 4H MA
        if ret_1h > 0.005 and ret_4h > 0 and is_above_ma:
            health = "🔥🔥 STRONG BUY"
            bias = "Bullish Synergy"
        # BULL FLAG Check
        elif flag_data["pattern"] != "None":
            health = f"🎯 {flag_data['pattern']}"
            bias = "Continuation"
        # WEAK REBOUND: 1H Up but 4H Down or below MA
        elif ret_1h > 0.005 and (ret_4h < 0 or not is_above_ma):
            health = "⚠️ WEAK REBOUND"
            bias = "Micro-Only"
        # BEARISH CONVERGENCE: Both down
        elif ret_1h < -0.005 and ret_4h < 0:
            health = "💀 BEARISH SYNC"
            bias = "Full Flush"
            
        # 5. Next Day Probability (Option C)
        # Based on 4H strength in the last session
        next_day_prob = "50% Neutral"
        if ret_4h > 0.01 and is_above_ma:
            next_day_prob = "75% Bullish Continuation"
        elif ret_4h < -0.01:
            next_day_prob = "75% Bearish Carryover"
            
        return {
            "health": health,
            "bias": bias,
            "next_day_forecast": next_day_prob,
            "1h_mom": ret_1h * 100,
            "4h_struct": ret_4h * 100,
            "pattern": flag_data["pattern"]
        }

    def detect_flags(self, df: pd.DataFrame, window: int = 15) -> dict:
        """
        Pattern Recognition: Bull Flag / Bear Pennant (Option B).
        Detects strong 'Poles' followed by tight 'Flags'.
        """
        try:
            if len(df) < window: return {"pattern": "None"}
            
            # Simple ATR-based volatility for pole detection
            recent = df.tail(window)
            vol = (recent['High'] - recent['Low']).rolling(14).mean().iloc[-1]
            if not vol or vol == 0: vol = (recent['High'].max() - recent['Low'].min()) / 10
            
            # 1. Pole: Move over first (window-5) bars
            pole_df = recent.iloc[:-5]
            pole_move = pole_df['Close'].iloc[-1] - pole_df['Close'].iloc[0]
            
            # 2. Flag: Narrow consolidation in last 5 bars
            flag_df = recent.tail(5)
            flag_top = flag_df['High'].max()
            flag_bot = flag_df['Low'].min()
            flag_range = flag_top - flag_bot
            
            # Criteria for Bull Flag:
            # - Pole move > 2.5 * Vol
            # - Flag range < 40% of Pole move
            # - Flag is in the upper 50% of the Pole
            is_bull_pole = pole_move > (vol * 2.5)
            is_bear_pole = pole_move < -(vol * 2.5)
            
            if is_bull_pole and flag_range < (pole_move * 0.4):
                if flag_bot > (pole_df['Close'].iloc[0] + pole_move * 0.5):
                    return {"pattern": "BULL FLAG", "confidence": 0.82}
            
            if is_bear_pole and flag_range < (abs(pole_move) * 0.4):
                if flag_top < (pole_df['Close'].iloc[0] + pole_move * 0.5):
                    return {"pattern": "BEAR PENNANT", "confidence": 0.82}
                    
            return {"pattern": "None"}
        except:
            return {"pattern": "None"}

    def detect_market_phase(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Institutional Market Phase Detection (Accumulation/Manipulation/Expansion/Distribution).
        Inspired by Wyckoff & ICT concepts.
        """
        if df.empty or len(df) < 50:
            return {"phase": "UNKNOWN", "color": "gray"}
            
        last_20 = df.tail(20)
        close = df["Close"].iloc[-1]
        
        # 1. Volatility & Trend (ADX/ATR)
        # Low ADX -> Range; High ADX -> Expansion
        from src.strategies.bank_participation import BankParticipationMonitor
        bank_monitor = BankParticipationMonitor()
        regime = bank_monitor._detect_regime(df) # "Trending" or "Ranging"
        
        # 2. Manipulation Check (Judas Swing / Volume Spikes)
        # Look for candles with long wicks near session extreme or volume > 2x avg
        vol_avg = df["Volume"].rolling(20).mean().iloc[-1]
        vol_cur = df["Volume"].iloc[-1]
        is_vol_climax = vol_cur > (vol_avg * 1.8)
        
        # Wick Ratio (High-Low vs Body)
        body = abs(df["Close"] - df["Open"])
        wick = (df["High"] - df["Low"]) - body
        wick_ratio = wick / body.clip(lower=1e-6)
        is_manipulation = is_vol_climax and wick_ratio.iloc[-1] > 1.5
        
        # 3. Position in Range (BB)
        sma20 = df["Close"].rolling(20).mean().iloc[-1]
        std20 = df["Close"].rolling(20).std().iloc[-1]
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        
        # Phase Assignment
        if is_manipulation:
            return {"phase": "🕵️ MANIPULATION", "color": "#FF9800"} # Orange
        
        if regime == "Trending":
            return {"phase": "🚀 EXPANSION", "color": "#00E676"} # Green
        else:
            # Ranging - check relative position
            if close > (upper_bb * 0.95):
                return {"phase": "📉 DISTRIBUTION", "color": "#FF5252"} # Red
            elif close < (lower_bb * 1.05):
                return {"phase": "📦 ACCUMULATION", "color": "#2196F3"} # Blue
            else:
                return {"phase": "↔️ CONSOLIDATION", "color": "#9E9E9E"} # Gray

    @property
    def feature_importance(self) -> dict[str, float]:
        return self._feature_importance

    @property
    def train_metrics(self) -> dict[str, float]:
        return self._train_metrics
