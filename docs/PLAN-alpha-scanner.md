# PLAN: Global Top-Down Alpha Scanner 🚀

## Overview
Develop a professional-grade market scanner that identifies institutional "Alpha" by ranking high-value assets across markets (VN, US) using Relative Strength (RS) scoring and AI Daily Bias filtering.

## 🟢 PHASE 1: Core Scanning Engine
**Agent**: `explorer-agent` / `project-planner`
- **File**: `src/strategies/alpha_scanner.py`
- **Logic**:
    - Define high-value subsets:
        - **VN_ALPHA**: VN30 Index members.
        - **US_ALPHA**: Top 20 NASDAQ/S&P tech & growth leaders.
    - **Batch Processing**:
        - Iterate through subsets.
        - Fetch 180 days of historical data for each asset + benchmark.
        - Compute RS Score (Mansfield style) via `PriceActionEngine`.
        - Compute Market Structure (HH/HL) status.
    - **AI Integration**:
        - Parallelize `AIPredictor` inference for each scanned asset to get "Daily Bias" (Bullish/Bearish).

## 🟡 PHASE 2: Intelligence Reporting
**Agent**: `quality-inspector`
- **Grouping Logic**:
    - Aggregate results by **Sector**.
    - Calculate "Sector Strength" (average RS of members).
- **Ranking**:
    - Flat leaderboard of top 20 "Strongest Assets" globally.

## 🔵 PHASE 3: Dashboard & UX
**Agent**: `frontend-specialist`
- **File**: `views/alpha_scanner.py`
- **Components**:
    - **Scanner Status**: Progress bar showing compute progress (Real-time).
    - **Visual Metrics**: Use horizontal "Strength Bars" for RS visibility.
    - **AI Badge**: Confirmation icons (🟢 Bullish / 🔴 Bearish) alongside ticker.
    - **Persistence**: Store results in `st.session_state` to prevent redundant 20s waits.

## 🔴 PHASE 4: Final Audit & Polish
**Agent**: `security-auditor` & `test-engineer`
- Verify data alignment (Naive vs Aware Dates).
- Stress test the "On-Demand" button.
- Ensure efficient memory cleanup after batch scanning.

---

## Agent Assignments
- **Orchestrator**: Manage the integration of `PriceActionEngine` and `AIPredictor`.
- **Specialist**: Build the batch scanning loops in `AlphaScannerEngine`.
- **UI Expert**: Build the premium "Sector Leaderboard" view.

## Verification Checklist
- [ ] Scanner fetches data for all symbols in high-value subsets.
- [ ] RS Scores are calculated relative to correct market benchmarks (VNI vs ^GSPC).
- [ ] AI Bias confirmation correctly filters "Weak" vs "Strong" leaders.
- [ ] Results remain cached after a page switch.
