"""
Nightly Full-Universe Pre-Scan Pipeline
========================================
Chay moi toi luc 21:00 (Taiwan UTC+8) tren may local:
  1. Fetch full VN universe tu vnstock (khong bi chan IP khi chay local)
  2. Chay AlphaScannerEngine (full Tier1 + Tier2 deep scan)
  3. Output top VN/TW candidates sorted by institutional_score
  4. Push ket qua JSON + OHLCV parquet cache len GitHub
  5. Streamlit Cloud doc JSON -> hien thi ngay, khong can goi API

Usage:
    python scripts/nightly_full_scan.py             # Scan + save, no push
    python scripts/nightly_full_scan.py --push      # Scan + save + git push
    python scripts/nightly_full_scan.py --top 20    # Top N results (default: 20)
    python scripts/nightly_full_scan.py --dry-run   # Test without saving

Setup Task Scheduler: chay setup_nightly_task.ps1 (Admin)
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from src.alerts.telegram import TelegramAlertSender
from src.data.analysis_store import AnalysisStore
from src.plugins import registry
from src.strategies.elliott_wave import ElliottWaveAnalyzer
from src.analytics.fundamental_score import get_fundamental_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCAN_RESULTS_PATH = DATA_DIR / "nightly_scan_results.json"
SCAN_META_PATH = DATA_DIR / "nightly_scan_meta.json"
NIGHTLY_REPORT_PATH = DATA_DIR / "nightly_telegram_snapshot.json"
SMC_ALERT_STATE_PATH = DATA_DIR / "smc_alert_state.json"


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _derive_grade(row: dict) -> str:
    grade = str(row.get("entry_quality_grade", "")).strip().upper()
    if grade in {"A", "B", "C"}:
        return grade

    # Fallback for legacy rows without explicit grade.
    rec = str(row.get("recommendation", "WATCH")).upper()
    score = _safe_float(row.get("institutional_score", 0.0))
    conf = _safe_float(row.get("confidence_boosted", 0.0))
    qmf_signal = int(_safe_float(row.get("qmf_signal", 0)))
    veto = bool(row.get("veto", False))
    if veto:
        return "C"
    if rec in {"STRONG BUY", "BUY"} and score >= 0.35 and conf >= 0.20 and qmf_signal >= 0:
        return "A"
    if score >= 0.12 and conf >= 0.08 and qmf_signal >= 0:
        return "B"
    return "C"


def _fmt_pick_line(i: int, row: dict) -> str:
    sym = str(row.get("symbol", "N/A"))
    rec = str(row.get("recommendation", "WATCH"))
    score = _safe_float(row.get("institutional_score", 0.0))
    up1 = _safe_float(row.get("pred_21d_ret", 0.0))
    up3 = _safe_float(row.get("pred_63d_ret", 0.0))
    return f"{i}. <b>{sym}</b> | {rec} | inst={score:.3f} | 1M={up1:+.1f}% | 3M={up3:+.1f}%"


def _calc_entry_stop_target(row: dict) -> tuple[float, float, float, float]:
    """Derive entry/stop/target with robust fallbacks for nightly alerts."""
    smc_status = str(row.get("smc_entry_status", "")).upper()
    smc_low = _safe_float(row.get("smc_entry_low", 0.0))
    smc_high = _safe_float(row.get("smc_entry_high", 0.0))
    smc_stop = _safe_float(row.get("smc_entry_stop", 0.0))
    smc_target = _safe_float(row.get("smc_entry_target", 0.0))
    smc_rr = _safe_float(row.get("smc_entry_rr", 0.0))
    if smc_status in {"READY", "NEAR"} and smc_low > 0 and smc_high > smc_low and smc_stop > 0 and smc_target > smc_high:
        entry = (smc_low + smc_high) / 2.0
        rr_eff = smc_rr if smc_rr > 0 else (smc_target - smc_high) / max(smc_high - smc_stop, 1e-9)
        return entry, smc_stop, smc_target, rr_eff

    entry = _safe_float(row.get("last_close", 0.0))
    if entry <= 0:
        entry = max(_safe_float(row.get("ai_target_1d", 0.0)), 0.01)

    pred_63d = _safe_float(row.get("pred_63d_ret", 0.0))
    target = _safe_float(row.get("wyckoff_target", 0.0))
    if target <= 0:
        target = entry * (1.0 + (pred_63d / 100.0))
    if target <= 0:
        target = entry

    rr = _safe_float(row.get("wyckoff_rr", 0.0))
    if rr > 0 and target > entry:
        stop = entry - ((target - entry) / rr)
    else:
        conf = _safe_float(row.get("confidence_boosted", 0.0))
        qmf_sig = int(_safe_float(row.get("qmf_signal", 0)))
        stop_pct = 0.06
        if conf >= 0.25:
            stop_pct = 0.05
        elif conf < 0.15:
            stop_pct = 0.07
        if qmf_sig < 0:
            stop_pct = min(stop_pct + 0.01, 0.09)
        stop = entry * (1.0 - stop_pct)

    # Safety clamps to avoid malformed values in notifications
    stop = max(stop, 0.01)
    if stop >= entry:
        stop = entry * 0.95
    if target <= entry:
        target = entry * (1.0 + max(pred_63d, 5.0) / 100.0)

    rr_eff = (target - entry) / max(entry - stop, 1e-9)
    return entry, stop, target, rr_eff


def _fmt_pick_line_mobile(i: int, row: dict) -> str:
    """Compact 2-line format optimized for small phone screens."""
    sym = str(row.get("symbol", "N/A"))
    score = _safe_float(row.get("institutional_score", 0.0))
    up1 = _safe_float(row.get("pred_21d_ret", 0.0))
    entry, stop, target, rr_eff = _calc_entry_stop_target(row)
    line1 = f"<b>{i}) {sym}</b> | S:{score:.2f} | 1M:{up1:+.1f}%"
    line2 = f"E:{entry:.2f}  SL:{stop:.2f}  TP:{target:.2f}  R:{rr_eff:.2f}"
    return f"{line1}\n<code>{line2}</code>"


def _fmt_smc_zone(row: dict) -> str:
    status = str(row.get("smc_entry_status", "NONE")).upper()
    if status not in {"READY", "NEAR", "WATCH"}:
        return "SMC:NONE"
    low = _safe_float(row.get("smc_entry_low", 0.0))
    high = _safe_float(row.get("smc_entry_high", 0.0))
    score = int(_safe_float(row.get("smc_entry_score", 0.0)))
    dist = _safe_float(row.get("smc_entry_distance_pct", 0.0))
    ztype = str(row.get("smc_entry_type", "SMC") or "SMC")
    if low > 0 and high > low:
        return f"SMC:{status} {ztype} {low:.2f}-{high:.2f} ({score}/8, {dist:.1f}%)"
    return f"SMC:{status} {ztype} ({score}/8)"


def _build_nightly_suggestions(ranked: list[dict], grade_a: list[dict], grade_c: list[dict]) -> list[str]:
    tips: list[str] = []
    total = len(ranked)
    if total == 0:
        return ["Không có dữ liệu hợp lệ hôm nay; ưu tiên đứng ngoài và kiểm tra nguồn feed."]

    c_ratio = len(grade_c) / total
    if len(grade_a) <= 2:
        tips.append("Ít Grade A: giảm tần suất vào lệnh mới, ưu tiên bảo toàn vốn.")
    if c_ratio >= 0.50:
        tips.append("Grade C chiếm cao: thị trường nhiễu/phân phối, chỉ giữ setup mạnh.")

    missing_fields = 0
    for r in ranked:
        if _safe_float(r.get("last_close", 0.0)) <= 0 or _safe_float(r.get("institutional_score", 0.0)) == 0.0:
            missing_fields += 1
    if missing_fields > 0:
        tips.append(f"Có {missing_fields} mã thiếu dữ liệu lõi; cần kiểm tra cache/API trước giờ mở cửa.")

    veto_count = sum(1 for r in ranked if bool(r.get("veto", False)))
    if veto_count >= max(3, int(0.25 * total)):
        tips.append("Nhiều mã bị veto: tránh FOMO, chờ xác nhận thêm 1 phiên.")

    if not tips:
        tips.append("Điều kiện dữ liệu ổn định; tập trung Top Grade A, quản trị rủi ro theo SL.")
    return tips[:3]


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj close": "Close",
        "adj_close": "Close",
    }
    out.columns = [str(c) for c in out.columns]
    for c in list(out.columns):
        lc = c.strip().lower()
        if lc in rename_map:
            out = out.rename(columns={c: rename_map[lc]})
    if "Date" not in out.columns:
        out = out.reset_index()
        if "index" in out.columns:
            out = out.rename(columns={"index": "Date"})
    if "Date" not in out.columns:
        return pd.DataFrame()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    need = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        return pd.DataFrame()
    return out[["Date", "Open", "High", "Low", "Close", "Volume"]]


def _weekly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    norm = _normalize_ohlcv_columns(df)
    if norm.empty:
        return pd.DataFrame()
    wk = (
        norm.set_index("Date")
        .resample("W")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
        .reset_index()
    )
    return wk


def _elliott_strategic_tag(symbol: str, market: str) -> tuple[str, str]:
    """
    Return (tag, detail) for weekly Elliott strategic context:
    - OK
    - CAUTION_WAVE5
    - STRATEGIC_ONLY
    - NO_DATA / ERROR
    """
    try:
        provider = registry.get(market)
        if not provider:
            return "NO_DATA", "provider_unavailable"
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=900)).strftime("%Y-%m-%d")
        daily = provider.get_price_data(symbol, start, end, interval="1d")
        if daily is None or daily.empty:
            return "NO_DATA", "no_daily_data"
        wk = _weekly_from_daily(daily)
        if wk.empty:
            return "NO_DATA", "weekly_resample_failed"
        ew = ElliottWaveAnalyzer().get_current_state(wk)
        wave = str(ew.get("current_wave", "?"))
        conf = _safe_float(ew.get("confidence", 0.0))
        if wave == "5" and conf >= 60.0:
            return "CAUTION_WAVE5", f"wave=5 conf={conf:.0f}%"
        if 60.0 <= conf < 75.0:
            return "STRATEGIC_ONLY", f"wave={wave} conf={conf:.0f}%"
        return "OK", f"wave={wave} conf={conf:.0f}%"
    except Exception as exc:
        return "ERROR", str(exc)


def _fundamental_quality(symbol: str, market: str) -> tuple[str, int]:
    """Return (grade, total_score) from AI forecast fundamental module."""
    try:
        f = get_fundamental_dict(symbol=symbol, market=market)
        grade = str(f.get("grade", "N/A")).strip().upper()
        score = int(_safe_float(f.get("total_score", 0.0)))
        if grade == "N/A" or int(_safe_float(f.get("data_coverage", 0))) < 40:
            return "N/A", 0
        return grade if grade else "N/A", score
    except Exception:
        return "N/A", 0


def _save_nightly_snapshot(payload: dict) -> None:
    try:
        NIGHTLY_REPORT_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"Nightly snapshot saved: {NIGHTLY_REPORT_PATH.name}")
    except Exception as exc:
        logger.warning(f"Nightly snapshot save failed: {exc}")


def _load_smc_alert_state() -> dict:
    if not SMC_ALERT_STATE_PATH.exists():
        return {"sent": {}}
    try:
        data = json.loads(SMC_ALERT_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"sent": {}}
    except Exception:
        return {"sent": {}}


def _save_smc_alert_state(state: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SMC_ALERT_STATE_PATH.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"SMC alert state save failed: {exc}")


def _smc_alert_key(row: dict) -> str:
    sym = str(row.get("symbol", "N/A")).upper()
    market = str(row.get("market", "")).upper()
    status = str(row.get("smc_entry_status", "")).upper()
    ztype = str(row.get("smc_entry_type", "SMC")).upper()
    low = round(_safe_float(row.get("smc_entry_low", 0.0)), 2)
    high = round(_safe_float(row.get("smc_entry_high", 0.0)), 2)
    return f"{market}|{sym}|{status}|{ztype}|{low:.2f}-{high:.2f}"


def _select_smc_buy_alerts(results: list[dict], top_n: int = 10) -> list[dict]:
    """Pick high-quality SMC buy alerts from scan rows."""
    rows: list[dict] = []
    for r in results:
        status = str(r.get("smc_entry_status", "")).upper()
        if status not in {"READY", "NEAR"}:
            continue
        if bool(r.get("veto", False)):
            continue
        if str(r.get("market", "")).upper() not in {"VN", "TW"}:
            continue
        smc_score = int(_safe_float(r.get("smc_entry_score", 0.0)))
        rr = _safe_float(r.get("smc_entry_rr", 0.0))
        dist = _safe_float(r.get("smc_entry_distance_pct", 999.0), 999.0)
        inst = _safe_float(r.get("institutional_score", 0.0))
        qmf = int(_safe_float(r.get("qmf_signal", 0.0)))
        stoch = str(r.get("stoch_state", "NEUTRAL")).upper()
        if smc_score < 6:
            continue
        if rr < 1.4:
            continue
        if status == "NEAR" and dist > 1.5:
            continue
        if qmf < 0 or stoch == "OVERBOUGHT":
            continue
        if inst < 0.05:
            continue
        rows.append(r)

    rank = {"READY": 0, "NEAR": 1}
    rows.sort(
        key=lambda x: (
            rank.get(str(x.get("smc_entry_status", "")).upper(), 9),
            -int(_safe_float(x.get("smc_entry_score", 0.0))),
            -_safe_float(x.get("institutional_score", 0.0)),
            _safe_float(x.get("smc_entry_distance_pct", 999.0), 999.0),
        )
    )
    return rows[:top_n]


def _send_new_smc_buy_alerts(results: list[dict], top_n: int = 10, force: bool = False) -> int:
    """Send dedicated Telegram SMC BUY alerts, suppressing duplicates per scan date."""
    sender = TelegramAlertSender(channel="smc")
    if not sender.is_enabled:
        logger.warning("Telegram not configured; skip dedicated SMC buy alerts.")
        return 0

    scan_date = datetime.now().strftime("%Y-%m-%d")
    state = _load_smc_alert_state()
    sent_state = state.setdefault("sent", {})
    candidates = _select_smc_buy_alerts(results, top_n=top_n)
    new_alerts = []
    for row in candidates:
        key = _smc_alert_key(row)
        if not force and sent_state.get(key) == scan_date:
            continue
        new_alerts.append((key, row))

    if not new_alerts:
        logger.info("No new SMC BUY alerts to send.")
        return 0

    lines: list[str] = []
    for i, (key, r) in enumerate(new_alerts, 1):
        sym = html.escape(str(r.get("symbol", "N/A")))
        market = html.escape(str(r.get("market", "")))
        status = html.escape(str(r.get("smc_entry_status", "")))
        ztype = html.escape(str(r.get("smc_entry_type", "SMC") or "SMC"))
        entry, stop, target, rr_eff = _calc_entry_stop_target(r)
        score = int(_safe_float(r.get("smc_entry_score", 0.0)))
        inst = _safe_float(r.get("institutional_score", 0.0))
        dist = _safe_float(r.get("smc_entry_distance_pct", 0.0))
        factors = html.escape(str(r.get("smc_entry_factors", "")))
        lines.append(
            f"{i}) <b>{sym}</b> {market} | <b>{status}</b> {ztype} | SMC:{score}/8 | Inst:{inst:.2f}\n"
            f"<code>Entry:{entry:.2f}  SL:{stop:.2f}  TP:{target:.2f}  R:{rr_eff:.2f}</code>\n"
            f"<i>Distance:{dist:.1f}% | {factors}</i>"
        )

    msg = (
        f"<b>🎯 SMC BUY ALERT — {scan_date}</b>\n"
        "<i>Tactical SMC setup mới, đã lọc trùng trong ngày.</i>\n\n"
        + "\n\n".join(lines)
    )
    if not sender.send_text(msg):
        logger.warning("Dedicated SMC BUY alert send failed.")
        return 0

    for key, _ in new_alerts:
        sent_state[key] = scan_date
    state["updated_at"] = datetime.now().isoformat()
    _save_smc_alert_state(state)
    logger.success(f"Dedicated SMC BUY alerts sent: {len(new_alerts)}")
    return len(new_alerts)


def _send_telegram_nightly_report(results: list[dict], top_n: int, market_scope: str = "VN") -> bool:
    sender = TelegramAlertSender()
    telegram_enabled = sender.is_enabled
    if not telegram_enabled:
        logger.warning("Telegram not configured; generate snapshot without sending messages.")
    if not results:
        _save_nightly_snapshot(
            {
                "generated_at": datetime.now().isoformat(),
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
                "market_scope": market_scope,
                "total_candidates": 0,
                "counts": {"grade_a_raw": 0, "grade_b_raw": 0, "grade_c_raw": 0, "top_a_sent": 0},
                "top_a": [],
                "grade_c_watchout": [],
                "tips": ["Không có dữ liệu hợp lệ hôm nay; đứng ngoài và kiểm tra dữ liệu."],
            }
        )
        if telegram_enabled:
            return sender.send_text("<b>🌙 Nightly Alpha Scan</b>\n\nNo candidates found today.")
        return False

    ranked = sorted(results, key=lambda x: _safe_float(x.get("institutional_score", -999)), reverse=True)
    grade_a = [r for r in ranked if _derive_grade(r) == "A"]
    grade_c = [r for r in ranked if _derive_grade(r) == "C"]

    # Mobile-friendly constraint: keep it short and highly scannable.
    show_a = grade_a[:10]  # Tactical Grade A from scanner.
    show_c = grade_c[:8]
    tactical_a_count = len(show_a)

    strategic_overlay: dict[str, dict] = {}
    for row in show_a:
        sym = str(row.get("symbol", ""))
        mkt = str(row.get("market", "VN"))
        tag, detail = _elliott_strategic_tag(sym, mkt)
        strategic_overlay[sym] = {"tag": tag, "detail": detail}

    date_tag = datetime.now().strftime("%Y-%m-%d")
    grade_b = [r for r in ranked if _derive_grade(r) == "B"]
    caution_wave5 = sum(1 for r in show_a if strategic_overlay.get(str(r.get("symbol", "")), {}).get("tag") == "CAUTION_WAVE5")
    strategic_only = sum(1 for r in show_a if strategic_overlay.get(str(r.get("symbol", "")), {}).get("tag") == "STRATEGIC_ONLY")
    summary = (
        f"<b>🌙 NIGHTLY ALPHA SCAN — {date_tag}</b>\n\n"
        f"Total: <b>{len(ranked)}</b> | A:<b>{len(grade_a)}</b> B:<b>{len(grade_b)}</b> C:<b>{len(grade_c)}</b>\n"
        f"Focus: <b>Top 10 Grade A</b> + Cảnh báo Grade C\n"
        f"Strategic EW check (Top A): Wave5 Caution=<b>{caution_wave5}</b>, StrategicOnly=<b>{strategic_only}</b>"
    )
    if telegram_enabled and not sender.send_text(summary):
        return False

    snapshot_a: list[dict] = []
    if show_a:
        a_lines_list: list[str] = []
        for i, r in enumerate(show_a, 1):
            sym = str(r.get("symbol", "N/A"))
            mkt = str(r.get("market", "VN"))
            overlay = strategic_overlay.get(sym, {"tag": "NO_DATA", "detail": "n/a"})
            tag = str(overlay.get("tag", "NO_DATA"))
            fund_grade, fund_score = _fundamental_quality(sym, mkt)
            entry, stop, target, rr_eff = _calc_entry_stop_target(r)
            score = _safe_float(r.get("institutional_score", 0.0))
            up1 = _safe_float(r.get("pred_21d_ret", 0.0))
            strat = "EW:OK"
            final_grade = "A"
            action_hint = "can add"
            consensus = "BUY"
            if tag == "CAUTION_WAVE5":
                strat = "EW:Wave5⚠"
                final_grade = "B"
                action_hint = "no chase"
                consensus = "WATCH_PULLBACK"
            elif tag == "STRATEGIC_ONLY":
                strat = "EW:Strategic⚠"
                final_grade = "B"
                action_hint = "wait confirm"
                consensus = "WATCH_CONFIRM"
            elif tag in {"NO_DATA", "ERROR"}:
                strat = "EW:N/A"
                action_hint = "small size"
                consensus = "WATCH_RISK"
            if fund_grade in {"D", "F"}:
                consensus = "REDUCE_RISK"
            line1 = f"<b>{i}) {sym}</b> | G:{final_grade} | F:{fund_grade} | S:{score:.2f} | 1M:{up1:+.1f}%"
            line2 = f"E:{entry:.2f}  SL:{stop:.2f}  TP:{target:.2f}  R:{rr_eff:.2f}"
            line3 = f"{strat} | {action_hint} | {consensus} | {_fmt_smc_zone(r)}"
            a_lines_list.append(f"{line1}\n<code>{line2}</code>\n<i>{line3}</i>")
            snapshot_a.append(
                {
                    "rank": i,
                    "symbol": sym,
                    "market": str(r.get("market", "")),
                    "entry_grade_tactical": "A",
                    "entry_grade_final": final_grade,
                    "strategic_tag": tag,
                    "strategic_detail": str(overlay.get("detail", "")),
                    "fundamental_grade": fund_grade,
                    "fundamental_score": fund_score,
                    "consensus_action": consensus,
                    "recommendation": str(r.get("recommendation", "WATCH")),
                    "institutional_score": score,
                    "pred_21d_ret": up1,
                    "pred_63d_ret": _safe_float(r.get("pred_63d_ret", 0.0)),
                    "entry": entry,
                    "stop_loss": stop,
                    "target": target,
                    "rr": rr_eff,
                    "smc_entry_status": str(r.get("smc_entry_status", "NONE")),
                    "smc_entry_type": str(r.get("smc_entry_type", "")),
                    "smc_entry_low": _safe_float(r.get("smc_entry_low", 0.0)),
                    "smc_entry_high": _safe_float(r.get("smc_entry_high", 0.0)),
                    "smc_entry_score": int(_safe_float(r.get("smc_entry_score", 0.0))),
                    "smc_entry_distance_pct": _safe_float(r.get("smc_entry_distance_pct", 0.0)),
                }
            )
        a_lines = "\n\n".join(a_lines_list)
        msg_a = (
            "<b>✅ TOP 10 TACTICAL A — đã check Strategic (Elliott)</b>\n"
            "<i>G = Final entry grade (sau lớp strategic), EW = bối cảnh tuần</i>\n\n"
            f"{a_lines}"
        )
    else:
        msg_a = "<b>✅ TOP 10 TACTICAL A</b>\n\nKhông có Grade A hôm nay."
    if telegram_enabled and not sender.send_text(msg_a):
        return False

    smc_rank = {"READY": 0, "NEAR": 1}
    smc_actionable = [
        r for r in ranked
        if str(r.get("market", "")).upper() in {"VN", "TW"}
        and str(r.get("smc_entry_status", "")).upper() in smc_rank
        and not bool(r.get("veto", False))
    ]
    smc_actionable = sorted(
        smc_actionable,
        key=lambda x: (
            smc_rank.get(str(x.get("smc_entry_status", "")).upper(), 9),
            -int(_safe_float(x.get("smc_entry_score", 0.0))),
            -_safe_float(x.get("institutional_score", 0.0)),
        ),
    )[:10]
    snapshot_smc: list[dict] = []
    if smc_actionable:
        smc_lines: list[str] = []
        for i, r in enumerate(smc_actionable, 1):
            sym = str(r.get("symbol", "N/A"))
            mkt = str(r.get("market", ""))
            entry, stop, target, rr_eff = _calc_entry_stop_target(r)
            status = str(r.get("smc_entry_status", ""))
            score = int(_safe_float(r.get("smc_entry_score", 0.0)))
            factors = str(r.get("smc_entry_factors", ""))
            smc_lines.append(
                f"{i}) <b>{sym}</b> {mkt} | {status} | SMC:{score}/8 | G:{_derive_grade(r)}\n"
                f"<code>E:{entry:.2f}  SL:{stop:.2f}  TP:{target:.2f}  R:{rr_eff:.2f}</code>\n"
                f"<i>{factors}</i>"
            )
            snapshot_smc.append(
                {
                    "rank": i,
                    "symbol": sym,
                    "market": mkt,
                    "entry_grade": _derive_grade(r),
                    "recommendation": str(r.get("recommendation", "WATCH")),
                    "institutional_score": _safe_float(r.get("institutional_score", 0.0)),
                    "smc_entry_status": status,
                    "smc_entry_type": str(r.get("smc_entry_type", "")),
                    "smc_entry_low": _safe_float(r.get("smc_entry_low", 0.0)),
                    "smc_entry_high": _safe_float(r.get("smc_entry_high", 0.0)),
                    "smc_entry_stop": _safe_float(r.get("smc_entry_stop", 0.0)),
                    "smc_entry_target": _safe_float(r.get("smc_entry_target", 0.0)),
                    "smc_entry_rr": _safe_float(r.get("smc_entry_rr", 0.0)),
                    "smc_entry_score": score,
                    "smc_entry_distance_pct": _safe_float(r.get("smc_entry_distance_pct", 0.0)),
                    "smc_entry_factors": factors,
                }
            )
        msg_smc = (
            "<b>🎯 SMC ENTRY RADAR — READY / NEAR</b>\n"
            "<i>READY = trong vùng entry; NEAR = cách vùng tối đa ~2%, chờ chạm/xác nhận.</i>\n\n"
            + "\n\n".join(smc_lines)
        )
    else:
        msg_smc = "<b>🎯 SMC ENTRY RADAR</b>\n\nKhông có mã READY/NEAR hôm nay."
    if telegram_enabled and not sender.send_text(msg_smc):
        return False

    snapshot_c: list[dict] = []
    if show_c:
        c_lines = []
        for i, r in enumerate(show_c, 1):
            sym = str(r.get("symbol", "N/A"))
            score = _safe_float(r.get("institutional_score", 0.0))
            reason = str(r.get("veto_reason", "")).strip()
            if not reason:
                reason = "Weak setup / flow risk"
            c_lines.append(f"{i}) <b>{sym}</b> | S:{score:.2f} | {reason}")
            snapshot_c.append(
                {
                    "rank": i,
                    "symbol": sym,
                    "market": str(r.get("market", "")),
                    "entry_grade": "C",
                    "institutional_score": score,
                    "reason": reason,
                }
            )
        msg_c = "<b>⚠️ GRADE C — Tránh vào mới</b>\n\n" + "\n".join(c_lines)
    else:
        msg_c = "<b>⚠️ GRADE C — Tránh vào mới</b>\n\nKhông có mã Grade C nổi bật."
    if telegram_enabled and not sender.send_text(msg_c):
        return False

    tips = _build_nightly_suggestions(ranked, grade_a, grade_c)
    tips_msg = "<b>💡 Gợi ý sau kiểm tra dữ liệu đêm</b>\n\n" + "\n".join(f"• {t}" for t in tips)
    if telegram_enabled and not sender.send_text(tips_msg):
        return False

    _save_nightly_snapshot(
        {
            "generated_at": datetime.now().isoformat(),
            "scan_date": date_tag,
            "market_scope": market_scope,
            "total_candidates": len(ranked),
            "counts": {
                "grade_a_raw": len(grade_a),
                "grade_b_raw": len(grade_b),
                "grade_c_raw": len(grade_c),
                "top_a_sent": tactical_a_count,
                "wave5_caution_in_top_a": caution_wave5,
                "strategic_only_in_top_a": strategic_only,
                "smc_ready_near": len(snapshot_smc),
            },
            "notes": {
                "grade_definition": "Entry Grade in scanner is tactical (daily 1-3M setup), not fundamental grade.",
                "strategic_definition": "Elliott weekly layer adds caution when Wave 5 / strategic-only context appears.",
                "consensus_definition": "Consensus Action combines tactical entry grade + Elliott strategic tag + fundamental quality.",
            },
            "top_a": snapshot_a,
            "smc_entry_radar": snapshot_smc,
            "grade_c_watchout": snapshot_c,
            "tips": tips,
        }
    )

    # Optional: Taiwan hidden gems section for VN_TW / TW scans.
    if market_scope in {"VN_TW", "TW", "ALL"}:
        tw_rows = [r for r in ranked if str(r.get("market", "")).upper() == "TW"]
        if tw_rows:
            gems = []
            for r in tw_rows:
                grade = _derive_grade(r)
                score = _safe_float(r.get("institutional_score", 0.0))
                conf = _safe_float(r.get("confidence_boosted", 0.0))
                up3 = _safe_float(r.get("pred_63d_ret", 0.0))
                stoch = str(r.get("stoch_state", "NEUTRAL")).upper()
                qmf = int(_safe_float(r.get("qmf_signal", 0)))
                if (
                    grade in {"A", "B"}
                    and score >= 0.12
                    and score <= 0.85
                    and conf >= 0.03
                    and up3 >= 12.0
                    and stoch != "OVERBOUGHT"
                    and qmf >= 0
                    and not bool(r.get("veto", False))
                ):
                    gems.append(r)

            gems = sorted(
                gems,
                key=lambda x: (
                    _safe_float(x.get("pred_63d_ret", 0.0)),
                    _safe_float(x.get("institutional_score", 0.0)),
                ),
                reverse=True,
            )[:5]
            if gems:
                g_lines = []
                for i, r in enumerate(gems, 1):
                    sym = str(r.get("symbol", "N/A"))
                    score = _safe_float(r.get("institutional_score", 0.0))
                    up3 = _safe_float(r.get("pred_63d_ret", 0.0))
                    entry, stop, target, rr_eff = _calc_entry_stop_target(r)
                    g_lines.append(
                        f"{i}) <b>{sym}</b> | S:{score:.2f} | 3M:{up3:+.1f}%\n"
                        f"<code>E:{entry:.2f}  SL:{stop:.2f}  TP:{target:.2f}  R:{rr_eff:.2f}</code>"
                    )
                gems_msg = "<b>💎 Taiwan Hidden Gems (Top 5)</b>\n\n" + "\n\n".join(g_lines)
                if telegram_enabled and not sender.send_text(gems_msg):
                    return False
            else:
                fallback = []
                for r in tw_rows:
                    grade = _derive_grade(r)
                    stoch = str(r.get("stoch_state", "NEUTRAL")).upper()
                    qmf = int(_safe_float(r.get("qmf_signal", 0)))
                    if grade in {"A", "B"} and not bool(r.get("veto", False)) and stoch != "OVERBOUGHT" and qmf >= 0:
                        fallback.append(r)
                fallback = sorted(
                    fallback,
                    key=lambda x: (
                        _safe_float(x.get("confidence_boosted", 0.0)),
                        _safe_float(x.get("institutional_score", 0.0)),
                    ),
                    reverse=True,
                )[:5]
                if fallback:
                    f_lines = []
                    for i, r in enumerate(fallback, 1):
                        sym = str(r.get("symbol", "N/A"))
                        grade = _derive_grade(r)
                        score = _safe_float(r.get("institutional_score", 0.0))
                        conf = _safe_float(r.get("confidence_boosted", 0.0))
                        up3 = _safe_float(r.get("pred_63d_ret", 0.0))
                        entry, stop, target, rr_eff = _calc_entry_stop_target(r)
                        f_lines.append(
                            f"{i}) <b>{sym}</b> | {grade} | S:{score:.2f} | C:{conf:.2f} | 3M:{up3:+.1f}%\n"
                            f"<code>E:{entry:.2f}  SL:{stop:.2f}  TP:{target:.2f}  R:{rr_eff:.2f}</code>"
                        )
                    fallback_msg = (
                        "<b>🧭 Taiwan Setup Watchlist (Fallback)</b>\n"
                        "<i>Không có gem đúng bộ lọc chặt; đây là A/B tốt nhất để theo dõi.</i>\n\n"
                        + "\n\n".join(f_lines)
                    )
                    if telegram_enabled and not sender.send_text(fallback_msg):
                        return False
    return telegram_enabled


def _fetch_full_vn_universe() -> list[str]:
    """
    Fetch the complete VN stock universe from vnstock API (local only).
    Falls back to extended hardcoded list if API fails.
    """
    try:
        from vnstock import Listing
        logger.info("Fetching full VN universe from vnstock...")
        listing = Listing()
        df = listing.symbols_by_exchange(lang='vi')

        # Filter to HOSE + HNX only
        if 'exchange' in df.columns:
            df = df[df['exchange'].isin(['HOSE', 'HNX'])].copy()

        # Filter to STOCK type only
        if 'type' in df.columns:
            for t in ['STOCK', 'Stock', 'stock']:
                if (df['type'] == t).sum() > 0:
                    df = df[df['type'] == t].copy()
                    break

        symbols = sorted(df['symbol'].dropna().unique().tolist())
        logger.success(f"Full VN universe: {len(symbols)} symbols from vnstock API")
        return symbols

    except Exception as e:
        logger.warning(f"vnstock API failed: {e} — using extended hardcoded list")
        # Import the extended list from alpha_scanner as fallback
        from src.strategies.alpha_scanner import VN_UNIVERSE_EXTENDED
        return VN_UNIVERSE_EXTENDED


def _run_full_scan(
    vn_symbols: list[str] | None = None,
    top_n: int = 20,
    market_scope: str = "VN_TW",
) -> list[dict]:
    """
    Run AlphaScannerEngine with nightly-optimized config:
    - Shuffles symbols to avoid alphabetical bias (A-B dominating)
    - Overrides scan_total_timeout_sec 50s -> 1800s (30 min budget)
    - Increases worker counts for faster batch throughput
    """
    import random
    import src.strategies.alpha_scanner as scanner_mod

    original_extended = scanner_mod.VN_UNIVERSE_EXTENDED[:]
    original_config = scanner_mod.SCAN_CONFIG.copy()

    try:
        if vn_symbols:
            # Shuffle to cover all alphabet ranges equally
            shuffled = vn_symbols[:]
            random.shuffle(shuffled)
            scanner_mod.VN_UNIVERSE_EXTENDED = shuffled

        # Override timeouts for nightly batch (no UI waiting)
        scanner_mod.SCAN_CONFIG = {
            **original_config,
            "tier1_max_workers": 12,
            "tier1_max_workers_vn": 8,
            "tier1_timeout_sec": 600,
            "tier2_max_workers": 8,
            "tier2_max_workers_vn": 6,
            "tier2_timeout_sec": 900,
            "scan_total_timeout_sec": 1800,
            "tier1_top_n": 120,
            "wyckoff_top_n": 80,
            "tier2_top_n": top_n * 3,
        }

        if vn_symbols:
            logger.info(
                f"Running nightly scan scope={market_scope}: "
                f"{len(vn_symbols)} VN symbols (shuffled, 30min budget)..."
            )
        else:
            logger.info(f"Running nightly scan scope={market_scope} (30min budget)...")

        from src.strategies.alpha_scanner import AlphaScannerEngine

        start_time = datetime.now()

        def _progress(p: float):
            elapsed = int((datetime.now() - start_time).total_seconds())
            pct = int(p * 100)
            logger.info(f"  Scan progress: {pct}% | {elapsed}s elapsed")

        engine = AlphaScannerEngine(
            extended_universe=True,
            commodities=False,
            market_scope=market_scope,
        )
        results = engine.scan_universe(progress_callback=_progress)

        elapsed = int((datetime.now() - start_time).total_seconds())
        logger.success(f"Scan complete: {len(results)} candidates in {elapsed}s")

        results.sort(key=lambda x: x.get("institutional_score", -999), reverse=True)

        scope = (market_scope or "VN").upper()
        if scope == "VN_TW":
            vn = [r for r in results if str(r.get("market", "")).upper() == "VN"][:top_n]
            tw = [r for r in results if str(r.get("market", "")).upper() == "TW"][:top_n]
            mixed = vn + tw
            mixed.sort(key=lambda x: x.get("institutional_score", -999), reverse=True)
            return mixed
        return results[:top_n]

    finally:
        scanner_mod.VN_UNIVERSE_EXTENDED = original_extended
        scanner_mod.SCAN_CONFIG = original_config

def _save_results(results: list[dict], top_n: int, market_scope: str = "VN") -> None:
    """Save scan results as JSON for Streamlit Cloud to read."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clean results for JSON serialization
    clean = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "_df"}
        # Convert numpy types to native Python
        for k, v in row.items():
            if hasattr(v, 'item'):
                row[k] = v.item()
            elif isinstance(v, float) and (v != v):  # NaN check
                row[k] = None
        clean.append(row)

    SCAN_RESULTS_PATH.write_text(
        json.dumps(clean, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8"
    )

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "generated_at_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_n": top_n,
        "market_scope": market_scope,
        "total_candidates": len(clean),
        "scan_date": datetime.now().strftime("%Y-%m-%d"),
        "symbols": [r.get("symbol", "") for r in clean],
    }
    SCAN_META_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.success(f"Saved {len(clean)} results → {SCAN_RESULTS_PATH.name}")


def _sync_analysis_db() -> None:
    """Sync newly written nightly JSON into persistent query store."""
    try:
        store = AnalysisStore()
        out = store.sync_from_json()
        if out.get("ok"):
            logger.success(
                f"Analysis DB synced: backend={out.get('backend')} "
                f"| inserted={out.get('inserted')} | scan_date={out.get('scan_date')}"
            )
        else:
            logger.warning(f"Analysis DB sync failed: {out}")
    except Exception as exc:
        logger.warning(f"Analysis DB sync exception: {exc}")


def _git_push() -> bool:
    """Commit scan results and OHLCV cache, then push to GitHub."""
    try:
        # Stage results + cache
        files_to_add = [
            str(DATA_DIR),
            str(PROJECT_ROOT / ".cache"),
        ]
        subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        result = subprocess.run(
            ["git", "commit", "-m", f"data: nightly scan + cache update {timestamp}"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8"
        )
        if "nothing to commit" in result.stdout:
            logger.info("No changes to push")
            return True

        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=PROJECT_ROOT, check=True, capture_output=True
        )
        logger.success(f"Pushed to GitHub: {timestamp}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git push failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Nightly multi-market alpha scan")
    parser.add_argument("--push", action="store_true", help="Push results to GitHub")
    parser.add_argument("--top", type=int, default=20, help="Top N results to save (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Run scan but don't save")
    parser.add_argument("--no-fetch", action="store_true", help="Use extended hardcoded list only")
    parser.add_argument("--no-smc-alerts", action="store_true", help="Disable dedicated Telegram SMC BUY alerts")
    parser.add_argument("--force-smc-alerts", action="store_true", help="Send SMC BUY alerts even if already sent today")
    parser.add_argument(
        "--market-scope",
        type=str,
        default="VN_TW",
        choices=["VN", "TW", "VN_TW", "ALL"],
        help="Nightly scan scope. Default VN_TW scans Vietnam and Taiwan.",
    )
    args = parser.parse_args()
    scope = (args.market_scope or "VN").upper()

    logger.info("=" * 60)
    logger.info(f"Nightly Alpha Scan [{scope}] — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    vn_symbols: list[str] | None = None
    includes_vn = scope in {"VN", "VN_TW", "ALL"}

    # Step 1: Get VN universe (only when scan scope includes VN)
    if includes_vn:
        if args.no_fetch:
            from src.strategies.alpha_scanner import VN_UNIVERSE_EXTENDED
            vn_symbols = VN_UNIVERSE_EXTENDED
            logger.info(f"Using hardcoded extended VN list: {len(vn_symbols)} symbols")
        else:
            vn_symbols = _fetch_full_vn_universe()

        # Prefetch VN OHLCV cache (feeds scanner's cache)
        logger.info("Pre-fetching VN OHLCV cache...")
        try:
            import importlib
            cache_script = importlib.util.spec_from_file_location(
                "nightly_vn_cache",
                str(PROJECT_ROOT / "scripts" / "nightly_vn_cache.py")
            )
            cache_mod = importlib.util.module_from_spec(cache_script)
            # Override symbols with full list
            cache_mod_symbols = vn_symbols
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            from scripts.nightly_vn_cache import (
                fetch_with_vnstock, fetch_with_yfinance, save_cache,
                update_stock_list_cache
            )
            cache_dir = PROJECT_ROOT / ".cache"
            update_stock_list_cache(vn_symbols, cache_dir)

            ok, fail = 0, 0
            for sym in vn_symbols:
                df = fetch_with_vnstock(sym, start, end)
                if df is None or df.empty:
                    df = fetch_with_yfinance(sym, start, end)
                if df is not None and not df.empty:
                    save_cache(sym, df, cache_dir)
                    ok += 1
                else:
                    fail += 1
            logger.success(f"VN OHLCV cache: {ok} OK, {fail} failed")
        except Exception as e:
            logger.warning(f"VN OHLCV pre-fetch failed (scan will use existing cache): {e}")
    else:
        logger.info("Scope excludes VN: skip VN universe fetch and VN cache prefetch.")

    # Step 2: Run full alpha scan
    results = _run_full_scan(vn_symbols, top_n=args.top, market_scope=scope)

    if not results:
        logger.error("Scan returned no results!")
        return

    # Step 3: Save
    if not args.dry_run:
        _save_results(results, args.top, scope)
        _sync_analysis_db()
        sent = _send_telegram_nightly_report(results, args.top, scope)
        if sent:
            logger.success("Nightly Telegram report sent.")
        else:
            logger.warning("Nightly Telegram report not sent.")
        if not args.no_smc_alerts:
            _send_new_smc_buy_alerts(results, top_n=min(args.top, 10), force=args.force_smc_alerts)
    else:
        logger.info("[DRY RUN] Would save:")
        for i, r in enumerate(results[:5], 1):
            logger.info(f"  {i}. {r.get('symbol')} — {r.get('recommendation')} | inst={r.get('institutional_score', 0):.3f}")

    # Step 4: Push
    if args.push and not args.dry_run:
        _git_push()

    logger.success("Nightly scan complete!")


if __name__ == "__main__":
    main()
