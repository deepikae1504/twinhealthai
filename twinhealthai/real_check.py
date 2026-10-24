"""Check how hard the task is on REAL glucose data (open dataset, e.g. CGMacros).
Usage: python real_check.py my_cgm.csv    (columns: timestamp, glucose)
Only the two simple baselines are scored here. Adding the trained model is next work."""
import sys
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

df = pd.read_csv(sys.argv[1], parse_dates=["timestamp"]).set_index("timestamp")
s = df["glucose"].resample("5min").mean().interpolate(limit=3)
future_max = s.shift(-1)[::-1].rolling(24, min_periods=24).max()[::-1]
d = pd.DataFrame({"same": s, "trend": s + 4 * (s - s.shift(6)), "y": future_max > 180}).dropna().iloc[::12]
print(f"Spike rate: {d.y.mean():.0%}, samples: {len(d)}")
for k in ("same", "trend"):
    print(f"{k:6s} AUROC {roc_auc_score(d.y, d[k]):.3f}  PR-AUC {average_precision_score(d.y, d[k]):.3f}")
