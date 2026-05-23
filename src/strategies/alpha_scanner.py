"""
Alpha Scanner Engine
====================
Tiered scan engine with 1-3 month institutional ranking.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import AIConfig, BankConfig, COTConfig, SmcConfig, StochasticConfig, VolumePriceConfig, WyckoffConfig
from src.plugins import registry
from src.strategies.price_action import PriceActionEngine
from src.strategies.quant_money_flow import QuantMoneyFlowAnalyzer
from src.strategies.smc_analyzer import SmcAnalyzer
from src.strategies.momentum import StochasticOscillator


SCAN_CONFIG = {
    "tier1_max_workers": 10,
    "tier1_max_workers_vn": 4,
    "tier1_max_workers_tw": 6,
    "tier1_min_rs_score": -1.25,
    "tier1_top_n": 90,
    "wyckoff_acceptable_phases": {
        "Phase A", "Phase B", "Phase C", "Phase D", "Re-Accumulation", "ACCUMULATION", "CONSOLIDATION"
    },
    "wyckoff_min_score": -0.2,
    "wyckoff_top_n": 60,
    "tier2_max_workers": 5,
    "tier2_max_workers_vn": 3,
    "tier2_max_workers_tw": 3,
    "tier2_top_n": 60,
    "tier1_timeout_sec": 14,
    "tier2_timeout_sec": 20,
    "scan_total_timeout_sec": 50,
    # Forecast sanity guards for 1-3M scanner outputs
    "max_abs_upside_1m_pct": 45.0,
    "max_abs_upside_3m_pct": 80.0,
    "min_conf_for_buy": 0.08,
    "min_conf_for_strong_buy": 0.25,
}

VN_UNIVERSE_CORE = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]
VN_UNIVERSE_EXTENDED = VN_UNIVERSE_CORE + [
    "AGG", "AGR", "ANV", "ASM", "BSR", "BSI", "BCC", "BMP", "CII", "CMG",
    "DGC", "DIG", "DPM", "DCM", "EVF", "EVG", "GEX", "GMD", "HAG", "HDC",
    "HHV", "HT1", "IDC", "IJC", "ITA", "KBC", "KDH", "KSB", "LCG", "LPB",
    "NAB", "NKG", "NLG", "NT2", "NVL", "PDR", "PHR", "PNJ", "PPC", "PSH",
    "PTB", "PVD", "PVS", "QNS", "REE", "SAF", "SCS", "SHN", "SKG", "SRC",
    "SIP", "TCH", "TDM", "TDH", "TIP", "TLG", "TPH", "VCG", "VCI", "VDS",
    "VGC", "VGS", "VHC", "VID", "VIP", "VIX", "VND", "VOS", "VSG", "VTO",
]
TW_UNIVERSE = [
    "2330", "2317", "2454", "2308", "2303", "2455", "3711", "2382", "3231",
    "2881", "2882", "2891", "2886", "2603", "2609", "1216", "2412", "2409",
    "3008", "2357", "2884", "2379", "8096",
]
US_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "QCOM", "INTC", "MU", "SMCI",
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "LINK-USD", "AVAX-USD", "GLD", "SLV", "QQQ", "SPY", "TLT", "GDX",
]
COMMODITY_UNIVERSE = ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F"]


def _quick_wyckoff_phase(df: pd.DataFrame) -> dict:
    if len(df) < 40 or "Close" not in df.columns:
        return {"phase": "UNKNOWN", "score": 0.0, "in_accumulation": False}
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float) if "High" in df.columns else close
    low = df["Low"].to_numpy(dtype=float) if "Low" in df.columns else close
    vol = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else np.ones(len(close))

    period = min(60, len(df))
    tr_high = np.max(high[-period:])
    tr_low = np.min(low[-period:])
    tr_range_pct = (tr_high - tr_low) / tr_low if tr_low > 0 else 1.0
    tr_pos = (close[-1] - tr_low) / (tr_high - tr_low) if (tr_high - tr_low) > 0 else 0.5

    vol_early = np.mean(vol[-period:-max(period // 2, 1)]) if period >= 4 else 1.0
    vol_late = np.mean(vol[-max(period // 2, 1):]) if period >= 4 else 1.0
    vol_trend = vol_late / vol_early if vol_early > 0 else 1.0

    lows_recent = low[-20:]
    lows_older = low[-40:-20] if len(low) >= 40 else lows_recent
    higher_lows = np.min(lows_recent) > np.min(lows_older)

    score = 0.0
    is_ranging = tr_range_pct < 0.25
    if is_ranging:
        score += 0.2
        if tr_pos < 0.40:
            score += 0.25
        elif tr_pos > 0.60:
            score -= 0.20
    if higher_lows:
        score += 0.20
    if vol_trend > 1.15:
        score += 0.15
    elif vol_trend < 0.80:
        score -= 0.10
    score = float(np.clip(score, -1.0, 1.0))

    if not is_ranging:
        sma20 = float(np.mean(close[-20:]))
        sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else sma20
        if close[-1] > sma20 > sma50:
            phase = "Markup (Uptrend)"
        elif close[-1] < sma20 < sma50:
            phase = "Markdown (Downtrend)"
        else:
            phase = "Transition"
    else:
        if score >= 0.3 and tr_pos < 0.5:
            phase = "Accumulation (Phase C/D)"
        elif score >= 0.1:
            phase = "Re-Accumulation"
        elif score <= -0.2:
            phase = "Distribution"
        else:
            phase = "CONSOLIDATION"

    in_accumulation = phase in SCAN_CONFIG["wyckoff_acceptable_phases"] and score >= SCAN_CONFIG["wyckoff_min_score"]
    return {"phase": phase, "score": score, "in_accumulation": in_accumulation}


class AlphaScannerEngine:
    def __init__(self, extended_universe: bool = False, commodities: bool = False, market_scope: str = "ALL") -> None:
        self.extended_universe = extended_universe
        self.commodities = commodities
        self.market_scope = (market_scope or "ALL").upper()

    def _build_tasks(self) -> list[dict]:
        tasks: list[dict] = []
        if self.market_scope in {"ALL", "VN_TW", "VN"}:
            vn_list = VN_UNIVERSE_EXTENDED if self.extended_universe else VN_UNIVERSE_CORE
            for sym in vn_list:
                tasks.append({"symbol": f"{sym}.VN", "market": "VN", "benchmark": "VNINDEX"})
        if self.market_scope in {"ALL", "VN_TW", "TW"}:
            for sym in TW_UNIVERSE:
                full_sym = f"{sym}.TWO" if sym in ["8096"] else f"{sym}.TW"
                tasks.append({"symbol": full_sym, "market": "TW", "benchmark": "2330.TW"})
        if self.market_scope in {"ALL", "US"}:
            for sym in US_UNIVERSE:
                tasks.append({"symbol": sym, "market": "US", "benchmark": "^GSPC"})
        if self.commodities and self.market_scope in {"ALL", "COMMODITY"}:
            for sym in COMMODITY_UNIVERSE:
                tasks.append({"symbol": sym, "market": "COMMODITY", "benchmark": "GC=F"})
        return tasks

    def scan_universe(self, progress_callback=None) -> list[dict]:
        tasks = self._build_tasks()
        total = len(tasks)
        cfg = SCAN_CONFIG
        scan_started = datetime.now()

        tier1_results: list[dict] = []
        tier1_results = self._run_tier_parallel(
            tasks=tasks,
            worker_fn=self._scan_tier1,
            timeout_sec=cfg["tier1_timeout_sec"],
            max_workers_default=cfg["tier1_max_workers"],
            max_workers_vn=cfg["tier1_max_workers_vn"],
            max_workers_tw=cfg["tier1_max_workers_tw"],
            progress_callback=progress_callback,
            progress_offset=0,
            progress_den=max(total * 3, 1),
            scan_started=scan_started,
            scan_total_timeout_sec=cfg["scan_total_timeout_sec"],
            tier_name="Tier1",
        )

        tier1_pass = [r for r in tier1_results if r.get("rs_score", -999) >= cfg["tier1_min_rs_score"]]
        if len(tier1_pass) < 20:
            # Fallback: keep momentum-ranked candidates even if RS is weak in current regime
            tier1_pass = sorted(tier1_results, key=lambda x: x.get("rs_score", -999), reverse=True)[: max(cfg["tier1_top_n"], 40)]
        tier1_pass.sort(key=lambda x: x.get("rs_score", 0), reverse=True)
        t1_top = tier1_pass[: cfg["tier1_top_n"]]

        wy_candidates: list[dict] = []
        for i, r in enumerate(t1_top):
            rf = self._wyckoff_prefilter(r)
            if rf:
                wy_candidates.append(rf)
            if progress_callback:
                progress_callback((total + i + 1) / max(total * 3, 1))

        wy_candidates.sort(key=lambda x: x.get("wyckoff_prefilter_score", 0), reverse=True)
        wy_top = wy_candidates[: cfg["wyckoff_top_n"]]
        wy_symbols = {r["symbol"] for r in wy_top}
        tier2_tasks = [t for t in tasks if t["symbol"] in wy_symbols]

        tier2_results = self._run_tier_parallel(
            tasks=tier2_tasks,
            worker_fn=self._scan_tier2,
            timeout_sec=cfg["tier2_timeout_sec"],
            max_workers_default=cfg["tier2_max_workers"],
            max_workers_vn=cfg["tier2_max_workers_vn"],
            max_workers_tw=cfg["tier2_max_workers_tw"],
            progress_callback=progress_callback,
            progress_offset=total * 2,
            progress_den=max(total * 3, 1),
            scan_started=scan_started,
            scan_total_timeout_sec=cfg["scan_total_timeout_sec"],
            tier_name="Tier2",
        )

        tier2_results.sort(key=lambda x: x.get("institutional_score", -999), reverse=True)
        return tier2_results[: cfg["tier2_top_n"]]

    @staticmethod
    def _run_tier_parallel(
        tasks: list[dict],
        worker_fn,
        timeout_sec: int,
        max_workers_default: int,
        max_workers_vn: int,
        max_workers_tw: int,
        progress_callback,
        progress_offset: int,
        progress_den: int,
        scan_started: datetime,
        scan_total_timeout_sec: int,
        tier_name: str,
    ) -> list[dict]:
        if not tasks:
            return []
        grouped: dict[str, list[dict]] = {}
        for t in tasks:
            grouped.setdefault(t.get("market", "OTHER"), []).append(t)

        done_count = 0
        out: list[dict] = []
        for market, mk_tasks in grouped.items():
            if (datetime.now() - scan_started).total_seconds() > scan_total_timeout_sec:
                logger.warning(f"{tier_name} global timeout reached, stop remaining markets.")
                break

            if market == "VN":
                workers = max_workers_vn
            elif market == "TW":
                workers = max_workers_tw
            else:
                workers = max_workers_default

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                fut_map = {ex.submit(worker_fn, t): t for t in mk_tasks}
                pending = set(fut_map.keys())
                market_deadline = datetime.now().timestamp() + max(1, int(timeout_sec))
                while pending and datetime.now().timestamp() < market_deadline:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.8,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done:
                        try:
                            r = fut.result()
                            if r:
                                out.append(r)
                        except Exception as exc:
                            logger.debug(f"{tier_name} failed: {exc}")
                        finally:
                            done_count += 1
                            if progress_callback:
                                progress_callback(min((progress_offset + done_count) / progress_den, 1.0))
                    if progress_callback and not done:
                        # keep UI alive during temporary network stalls
                        progress_callback(min((progress_offset + done_count) / progress_den, 1.0))

                if pending:
                    logger.warning(f"{tier_name} market={market} timeout: skipped {len(pending)} slow symbols.")
                    for fut in pending:
                        fut.cancel()
                        done_count += 1
                        if progress_callback:
                            progress_callback(min((progress_offset + done_count) / progress_den, 1.0))
        return out

    @staticmethod
    def _scan_tier1(task: dict) -> Optional[dict]:
        symbol, market, bench_sym = task["symbol"], task["market"], task["benchmark"]
        try:
            provider = registry.get(market)
            if not provider:
                return None
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            df = provider.get_price_data(symbol, start, end)
            if df.empty or len(df) < 50:
                return None
            bench_df = provider.get_price_data(bench_sym, start, end)
            pa = PriceActionEngine.analyze(df, bench_df)
            return {
                "symbol": symbol,
                "market": market,
                "rs_score": float(pa.get("rs_score", 0.0)),
                "structure": pa.get("structure", "Sideways"),
                "volume_status": pa.get("volume_status", "Normal"),
                "last_close": float(df["Close"].iloc[-1]),
                "_df": df,
            }
        except Exception as exc:
            logger.debug(f"Tier1 [{symbol}]: {exc}")
            return None

    @staticmethod
    def _wyckoff_prefilter(r: dict) -> Optional[dict]:
        df: pd.DataFrame = r.get("_df")
        if df is None or len(df) < 40:
            return None
        wy = _quick_wyckoff_phase(df)
        r["wyckoff_quick_phase"] = wy["phase"]
        r["wyckoff_quick_score"] = wy["score"]
        # Soft prefilter: prioritize accumulation, but keep relative opportunities.
        phase = str(wy["phase"]).lower()
        phase_boost = 0.18 if wy["in_accumulation"] else 0.0
        if "markup" in phase:
            phase_boost = max(phase_boost, 0.10)
        elif "transition" in phase:
            phase_boost = max(phase_boost, 0.05)
        elif "markdown" in phase:
            phase_boost = min(phase_boost, -0.18)
        r["wyckoff_prefilter_score"] = float((0.55 * r.get("rs_score", 0.0)) + (0.45 * wy["score"]) + phase_boost)
        return r

    @staticmethod
    def _scan_tier2(task: dict) -> Optional[dict]:
        from src.strategies.ai_predictor import AIPredictor
        from src.strategies.bank_participation import BankParticipationMonitor
        from src.strategies.volume_price import VolumePriceDetector
        from src.strategies.wyckoff_analyzer import WyckoffAnalyzer

        symbol, market, bench_sym = task["symbol"], task["market"], task["benchmark"]
        try:
            provider = registry.get(market)
            if not provider:
                return None
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=220)).strftime("%Y-%m-%d")
            df = provider.get_price_data(symbol, start, end)
            if df.empty or len(df) < 80:
                return None

            bench_df = provider.get_price_data(bench_sym, start, end)
            pa = PriceActionEngine.analyze(df, bench_df)

            ai_bias, ai_conf = "Neutral", 0.0
            ai_target_1d = ai_target_21d = ai_target_63d = float(df["Close"].iloc[-1])
            manipulation = "None"
            df_feat = df.copy()
            try:
                df_feat = VolumePriceDetector(VolumePriceConfig()).generate_signals(df_feat)
                df_feat = BankParticipationMonitor(BankConfig()).generate_signals(df_feat)
                if market == "COMMODITY" and provider.supports_cot_data():
                    df_feat["cot_signal"] = 0
                predictor = AIPredictor(AIConfig(horizons=(1, 21, 63)))
                predictor.train(df_feat)
                ai_df = predictor.generate_signals(df_feat)
                if not ai_df.empty:
                    last = ai_df.iloc[-1]
                    bias_val = int(last.get("ai_bias", 0))
                    ai_bias = "Bullish" if bias_val > 0 else "Bearish" if bias_val < 0 else "Neutral"
                    ai_target_1d = float(last.get("ai_target_price_1d", ai_target_1d))
                    ai_target_21d = float(last.get("ai_target_price_21d", ai_target_21d))
                    ai_target_63d = float(last.get("ai_target_price_63d", ai_target_63d))
                    ai_conf = float(last.get("ai_confidence", 0.0))
                    vp_sig = int(last.get("vp_signal", 0))
                    bank_sig = int(last.get("bank_signal", 0))
                    manipulation = "Accumulation" if (vp_sig > 0 or bank_sig > 0) else ("Distribution" if (vp_sig < 0 or bank_sig < 0) else "None")
            except Exception as ai_err:
                logger.debug(f"AI pipeline [{symbol}]: {ai_err}")

            wy = {"phase": "UNKNOWN", "score": 0.0, "spring": False, "lps": False, "rr": 0.0, "target": 0.0}
            try:
                wy = WyckoffAnalyzer(WyckoffConfig()).analyze_current_state(df_feat)
            except Exception:
                pass

            qmf_score, qmf_signal = 0.0, 0
            try:
                qmf_df = QuantMoneyFlowAnalyzer().generate_signals(df.copy())
                if not qmf_df.empty:
                    qlast = qmf_df.iloc[-1]
                    qmf_score = float(qlast.get("qmf_score", 0.0))
                    qmf_signal = int(qlast.get("qmf_signal", 0))
            except Exception:
                pass

            stoch_k, stoch_d = 50.0, 50.0
            try:
                stoch_df = StochasticOscillator(StochasticConfig()).generate_signals(df.copy())
                if not stoch_df.empty:
                    slast = stoch_df.iloc[-1]
                    stoch_k = float(slast.get("%K", 50.0))
                    stoch_d = float(slast.get("%D", 50.0))
            except Exception:
                pass
            stoch_state = "OVERBOUGHT" if (stoch_k > 85 and stoch_d > 85) else ("OVERSOLD" if (stoch_k < 15 and stoch_d < 15) else "NEUTRAL")

            smc_score = 0.0
            try:
                smc_state = SmcAnalyzer(SmcConfig()).get_current_state(df.copy())
                smc_score = float(smc_state.get("smc_score", 0.0))
            except Exception:
                pass

            curr = float(df["Close"].iloc[-1])
            pred_21d_ret_raw = (ai_target_21d - curr) / curr * 100.0 if curr > 0 else 0.0
            pred_63d_ret_raw = (ai_target_63d - curr) / curr * 100.0 if curr > 0 else 0.0

            # Volatility-aware clipping to prevent outlier forecasts from dominating score
            close_ret = df["Close"].astype(float).pct_change().dropna()
            daily_vol = float(close_ret.rolling(20).std().iloc[-1]) if len(close_ret) >= 20 else float(close_ret.std() if len(close_ret) > 3 else 0.02)
            if not np.isfinite(daily_vol) or daily_vol <= 0:
                daily_vol = 0.02
            vol_cap_1m = float(np.clip(daily_vol * np.sqrt(21) * 100 * 4.0, 10.0, SCAN_CONFIG["max_abs_upside_1m_pct"]))
            vol_cap_3m = float(np.clip(daily_vol * np.sqrt(63) * 100 * 4.5, 20.0, SCAN_CONFIG["max_abs_upside_3m_pct"]))
            pred_21d_ret = float(np.clip(pred_21d_ret_raw, -vol_cap_1m, vol_cap_1m))
            pred_63d_ret = float(np.clip(pred_63d_ret_raw, -vol_cap_3m, vol_cap_3m))
            w_score = float(wy.get("score", 0.0))
            prior = float(np.clip((0.45 * w_score) + (0.35 * smc_score) + (0.20 * qmf_score), -1.0, 1.0))
            confidence_boosted = float(np.clip(ai_conf + (0.10 * prior), 0.0, 1.0))

            score = float(
                np.clip(
                    (0.30 * (pred_21d_ret / 15.0))
                    + (0.35 * (pred_63d_ret / 20.0))
                    + (0.20 * confidence_boosted)
                    + (0.15 * prior),
                    -1.0,
                    1.0,
                )
            )
            veto = False
            veto_reason = ""
            if stoch_state == "OVERBOUGHT" and qmf_signal < 0 and pred_21d_ret > 0:
                veto = True
                veto_reason = "Overbought + outflow contradiction"
                score *= 0.65
            elif stoch_state == "OVERSOLD" and qmf_signal > 0 and pred_21d_ret < 0:
                veto = True
                veto_reason = "Oversold + inflow contradiction"
                score *= 0.65

            # Confidence gate: avoid high recommendations when model confidence is too low
            if confidence_boosted < SCAN_CONFIG["min_conf_for_buy"]:
                recommendation = "WATCH"
                veto = True
                veto_reason = veto_reason or "Low confidence gate"
            elif score >= 0.45 and not veto and confidence_boosted >= SCAN_CONFIG["min_conf_for_strong_buy"]:
                recommendation = "STRONG BUY"
            elif score >= 0.20 and confidence_boosted >= SCAN_CONFIG["min_conf_for_buy"]:
                recommendation = "BUY"
            elif score <= -0.25:
                recommendation = "AVOID"
            else:
                recommendation = "WATCH"

            info = provider.search_assets(symbol.replace(".VN", "").replace(".TW", ""), limit=1)
            name = info[0].name if info else symbol
            sector = info[0].sector if info else "Other"

            return {
                "symbol": symbol,
                "name": name,
                "market": market,
                "sector": sector,
                "last_close": curr,
                "rs_score": float(pa.get("rs_score", 0.0)),
                "structure": pa.get("structure", "Sideways"),
                "volume_status": pa.get("volume_status", "Normal"),
                "ai_bias": ai_bias,
                "ai_target": ai_target_1d,
                "ai_target_21d": ai_target_21d,
                "ai_target_63d": ai_target_63d,
                "pred_21d_ret": pred_21d_ret,
                "pred_63d_ret": pred_63d_ret,
                "pred_21d_ret_raw": float(pred_21d_ret_raw),
                "pred_63d_ret_raw": float(pred_63d_ret_raw),
                "pred_cap_1m": vol_cap_1m,
                "pred_cap_3m": vol_cap_3m,
                "ai_confidence": ai_conf,
                "confidence_boosted": confidence_boosted,
                "manipulation": manipulation,
                "wyckoff_phase": wy.get("phase", "UNKNOWN"),
                "wyckoff_score": w_score,
                "wyckoff_spring": bool(wy.get("spring", False)),
                "wyckoff_lps": bool(wy.get("lps", False)),
                "wyckoff_rr": float(wy.get("rr", 0.0)),
                "wyckoff_target": float(wy.get("target", 0.0)),
                "qmf_score": qmf_score,
                "qmf_signal": qmf_signal,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "stoch_state": stoch_state,
                "smc_score": smc_score,
                "institutional_prior": prior,
                "institutional_score": score,
                "veto": veto,
                "veto_reason": veto_reason,
                "recommendation": recommendation,
                "scan_tier": "DEEP",
            }
        except Exception as exc:
            logger.debug(f"Tier2 deep scan [{symbol}]: {exc}")
            return None

    @staticmethod
    def scan_universe_legacy(progress_callback=None) -> list[dict]:
        engine = AlphaScannerEngine(extended_universe=False, commodities=False)
        return engine.scan_universe(progress_callback=progress_callback)
