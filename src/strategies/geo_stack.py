"""
Geo-OSINT -> Causal Scoring -> Execution Gate
=============================================
Deterministic, auditable pipeline for geopolitical risk integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeoStackConfig:
    half_life_hours: float = 36.0
    gate_reduce_threshold: float = 0.45
    gate_pause_threshold: float = 0.70
    max_abs_oil_move_1d_pct: float = 6.0


def _to_utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz=timezone.utc)
    return ts


class GeoSignalPipeline:
    """End-to-end stack: normalize -> score -> causal -> gate."""

    EVENT_SEVERITY_MAP = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.8, "CRITICAL": 1.0}
    EVENT_TYPE_BASE = {
        "SHIPPING_DISRUPTION": 1.0,
        "MILITARY_ESCALATION": 1.0,
        "SANCTIONS_ENFORCEMENT": 0.7,
        "DIPLOMATIC_DEESCALATION": -0.8,
        "INSURANCE_STRESS": 0.6,
        "INFRA_ATTACK": 1.0,
    }
    SOURCE_QUALITY_MAP = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4}

    def __init__(self, config: GeoStackConfig | None = None) -> None:
        self.config = config or GeoStackConfig()

    def normalize_events(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize OSINT events into canonical schema."""
        if raw.empty:
            return pd.DataFrame(
                columns=[
                    "event_ts",
                    "event_type",
                    "severity",
                    "confidence",
                    "source_quality",
                    "region",
                    "headline",
                ]
            )

        df = raw.copy()
        rename_map = {
            "timestamp": "event_ts",
            "time": "event_ts",
            "type": "event_type",
            "sev": "severity",
            "conf": "confidence",
            "source_grade": "source_quality",
            "title": "headline",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        for col in ("event_ts", "event_type", "severity", "confidence", "source_quality", "region", "headline"):
            if col not in df.columns:
                if col == "confidence":
                    df[col] = 0.6
                elif col == "source_quality":
                    df[col] = "B"
                elif col == "severity":
                    df[col] = "MEDIUM"
                elif col == "region":
                    df[col] = "Hormuz"
                else:
                    df[col] = ""

        df["event_ts"] = df["event_ts"].apply(_to_utc_ts)
        df["event_type"] = df["event_type"].astype(str).str.upper().str.strip()
        df["severity"] = df["severity"].astype(str).str.upper().str.strip()
        df["source_quality"] = df["source_quality"].astype(str).str.upper().str.strip()
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.6).clip(0.0, 1.0)
        df["headline"] = df["headline"].astype(str)
        return df.sort_values("event_ts").reset_index(drop=True)

    def score_osint(self, events: pd.DataFrame) -> dict[str, float]:
        """Score recent event stream into geo risk score [0..1]."""
        if events.empty:
            return {"geo_risk_score": 0.0, "support_count": 0.0, "deescalation_count": 0.0}

        now = pd.Timestamp.now(tz=timezone.utc)
        half_life = max(self.config.half_life_hours, 1.0)
        age_hours = (now - events["event_ts"]).dt.total_seconds() / 3600.0
        recency = np.exp(-np.log(2.0) * (age_hours / half_life))

        sev_w = events["severity"].map(self.EVENT_SEVERITY_MAP).fillna(0.5).to_numpy()
        type_w = events["event_type"].map(self.EVENT_TYPE_BASE).fillna(0.25).to_numpy()
        src_w = events["source_quality"].map(self.SOURCE_QUALITY_MAP).fillna(0.7).to_numpy()
        conf_w = events["confidence"].to_numpy()

        raw = type_w * sev_w * src_w * conf_w * recency.to_numpy()
        risk = float(np.clip(np.mean(raw) + np.max(raw) * 0.35, -1.0, 1.0))
        geo_risk_score = float(np.clip((risk + 1.0) / 2.0, 0.0, 1.0))
        support_count = float(np.sum(type_w > 0))
        deescalation_count = float(np.sum(type_w < 0))
        return {
            "geo_risk_score": geo_risk_score,
            "support_count": support_count,
            "deescalation_count": deescalation_count,
        }

    def causal_score(self, osint_score: dict[str, float]) -> dict[str, float]:
        """Map geo risk into directional oil scenarios."""
        geo = float(osint_score.get("geo_risk_score", 0.0))
        # logistic-style probability for upside oil shock
        prob_up = float(1.0 / (1.0 + np.exp(-7.0 * (geo - 0.5))))
        exp_move = float((prob_up - 0.5) * 2.0 * self.config.max_abs_oil_move_1d_pct)
        return {
            "prob_oil_up_1d": prob_up,
            "expected_oil_move_1d_pct": exp_move,
            "tail_risk_flag": float(geo >= self.config.gate_pause_threshold),
        }

    def execution_gate(self, osint_score: dict[str, float], causal: dict[str, float]) -> dict[str, Any]:
        """Risk-first trade gate for downstream execution."""
        geo = float(osint_score.get("geo_risk_score", 0.0))
        exp_move = float(causal.get("expected_oil_move_1d_pct", 0.0))

        if geo >= self.config.gate_pause_threshold:
            return {
                "action": "PAUSE_NEW_LONGS",
                "position_multiplier": 0.0,
                "reason": "Extreme geopolitical stress",
                "geo_risk_score": geo,
                "expected_oil_move_1d_pct": exp_move,
            }
        if geo >= self.config.gate_reduce_threshold:
            return {
                "action": "REDUCE_SIZE",
                "position_multiplier": 0.5,
                "reason": "Elevated geopolitical risk",
                "geo_risk_score": geo,
                "expected_oil_move_1d_pct": exp_move,
            }
        return {
            "action": "NORMAL",
            "position_multiplier": 1.0,
            "reason": "Risk within normal regime",
            "geo_risk_score": geo,
            "expected_oil_move_1d_pct": exp_move,
        }

