import json
from pathlib import Path

data = json.loads(Path("data/nightly_scan_results.json").read_text(encoding="utf-8"))
meta = json.loads(Path("data/nightly_scan_meta.json").read_text(encoding="utf-8"))

print("Generated:", meta.get("generated_at_readable"))
print("Total candidates:", meta.get("total_candidates"))
print()
header = f"{'#':<3} {'Symbol':<12} {'Rec':<14} {'Score':<8} {'1M':>8} {'3M':>8}  {'Wyckoff':<28} Veto"
print(header)
print("-" * 100)
for i, r in enumerate(data[:20], 1):
    veto = "VETO" if r.get("veto") else ""
    line = (
        f"{i:<3} {r['symbol']:<12} {r['recommendation']:<14} "
        f"{r['institutional_score']:<8.3f} "
        f"{r['pred_21d_ret']:>+7.1f}%  {r['pred_63d_ret']:>+7.1f}%  "
        f"{str(r.get('wyckoff_phase','')):<28} {veto}"
    )
    print(line)
