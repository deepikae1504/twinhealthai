"""Train the twin's models.   Usage: python train.py --data data

What is different from a plain training script:
  * TRAINS ON MESSY DATA: random CGM gaps, wearable outages and incomplete health records are
    injected (scenarios.py), so the model has practised on the problems real data has.
  * LABEL IS OPERATIONAL: "glucose is <= 180 as far as we can SEE, and goes above 180 in 2 h".
  * FUSION IS TESTED: EHR-only, wearable-only and fused models are scored on unseen patients.
  * ALERT LEVEL IS CHOSEN ON SEPARATE PATIENTS (not the report patients) to limit false alerts per day.
200 patients are never used for training; 100 of them pick the alert level, 100 are used for reports.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

import alerts as AL
from features import EHR_FEATURES, WEARABLE_FEATURES, build_all, ehr_features
from scenarios import random_ehr_mix, random_training_mix

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()
COLS = WEARABLE_FEATURES + EHR_FEATURES
HORIZONS = [6, 12, 18, 24]            # +30, +60, +90, +120 minutes
ROWS = 2016                            # 7 days x 288 readings per patient

ehr = pd.read_csv(f"{a.data}/ehr_data.csv")
if not (ehr.data_type == "synthetic").all():
    raise SystemExit("refusing to train on data not tagged synthetic")
sens = pd.read_csv(f"{a.data}/sensor_data.csv", parse_dates=["timestamp"],
                   dtype={"patient_id": "category", "sleep_stage_simulated": "category", "data_type": "category"})
sens = sens.sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)
if not ((sens.data_type.astype(str) == "synthetic").all() and len(sens) % ROWS == 0):
    raise SystemExit("sensor data must be synthetic and 2016 rows per patient")

pids = np.array(ehr.patient_id.tolist())
np.random.RandomState(0).shuffle(pids)
test_p = list(pids[:200])
val_set, rep_set = set(test_p[0::2]), set(test_p[1::2])      # pick alert level / report results
pidc = sens.patient_id.astype(str)

sm = random_training_mix(sens, seed=1, rows_per_patient=ROWS)            # messy copy used for FEATURES
feat = build_all(sm)
grp_clean = sens.groupby("patient_id", sort=False, observed=True).glucose_mg_dl
grp_seen = sm.groupby("patient_id", sort=False, observed=True).glucose_mg_dl
fut = grp_clean.transform(lambda s: s[::-1].rolling(24, min_periods=24).max()[::-1].shift(-1))
base = grp_seen.ffill()                                                   # last glucose we actually saw
y = (fut > 180).astype(float).where(fut.notna())
tgt = {k: (grp_clean.shift(-k) - base).to_numpy() for k in HORIZONS}

E = random_ehr_mix(ehr_features(ehr).set_index("patient_id"), seed=2)     # incomplete records
static = E.loc[pidc].reset_index(drop=True)
static.index = sens.index
X = pd.concat([feat, static[EHR_FEATURES]], axis=1)[COLS].astype("float32")
del feat, static

okrow = ((base <= 180) & y.notna()).to_numpy()
is_test, is_val, is_rep = (pidc.isin(s).to_numpy() for s in (set(test_p), val_set, rep_set))
tr = np.flatnonzero(~is_test & okrow)[::6]
rep = np.flatnonzero(is_rep & okrow)
yv = y.to_numpy()
bs = base.to_numpy()

# ---------------------------------------------------------------- fusion ablation
sets = {"EHR only": EHR_FEATURES, "Wearable + meal log only": WEARABLE_FEATURES,
        "FUSED (wearable + meal log + EHR)": COLS}
models, abl = {}, {}
for name, cols in sets.items():
    models[name] = HGBC(max_iter=150, learning_rate=0.1, random_state=0).fit(X.iloc[tr][cols], yv[tr])
    p = models[name].predict_proba(X.iloc[rep][cols])[:, 1]
    abl[name] = dict(pr_auc=float(average_precision_score(yv[rep], p)), roc_auc=float(roc_auc_score(yv[rep], p)))
abl["Baseline: glucose stays the same"] = dict(pr_auc=float(average_precision_score(yv[rep], bs[rep])),
                                              roc_auc=float(roc_auc_score(yv[rep], bs[rep])))
clf = models["FUSED (wearable + meal log + EHR)"]
rate = float(yv[rep].mean())

# ---------------------------------------------------------------- forecast curve
regs, curve, persist = {}, {}, {}
for k in HORIZONS:
    ok = tr[~np.isnan(tgt[k][tr])]
    regs[k] = HGBR(max_iter=100, learning_rate=0.1, random_state=0).fit(X.iloc[ok], tgt[k][ok])
    r_ok = rep[~np.isnan(tgt[k][rep])]
    curve[k * 5] = float(mean_absolute_error(tgt[k][r_ok], regs[k].predict(X.iloc[r_ok])))
    persist[k * 5] = float(np.abs(tgt[k][r_ok]).mean())

# ---------------------------------------------------------------- choose the alert level on VALIDATION patients
val_idx = np.flatnonzero(is_val)
risk_val = clf.predict_proba(X.iloc[val_idx])[:, 1]
vp = pidc.to_numpy()[val_idx]
g_seen_v, g_true_v, base_v = (arr[val_idx] for arr in (sm.glucose_mg_dl.to_numpy(), sens.glucose_mg_dl.to_numpy(), bs))
bad = {pid: AL.untrusted(g_seen_v[vp == pid]) for pid in np.unique(vp)}
table = []
for thr in np.round(np.arange(0.20, 0.95, 0.05), 2):
    parts = []
    for pid in np.unique(vp):
        m = vp == pid
        al = AL.run_alerts(risk_val[m], base_v[m], bad[pid], thr)
        parts.append(AL.score(al, AL.spike_events(g_true_v[m]), m.sum() / 288))
    table.append(dict(level=float(thr), **AL.summarise(parts)))
good = [t for t in table if t["false alerts / patient-day"] <= AL.FALSE_ALERTS_PER_DAY]
alert_level = good[0]["level"] if good else table[-1]["level"]

g_obs = sm.glucose_mg_dl.to_numpy()
bundle = dict(clf=clf, regs=regs, cols=COLS, medians=X.iloc[tr].median().to_dict(),
              glucose_range=(float(np.nanpercentile(g_obs, 0.5)), float(np.nanpercentile(g_obs, 99.5))),
              alert_level=alert_level, alert_table=table,
              metrics=dict(risk_pr_auc=abl["FUSED (wearable + meal log + EHR)"]["pr_auc"], risk_base_rate=rate,
                           ablation=abl, curve_mae=curve, curve_mae_persistence=persist))
os.makedirs("models", exist_ok=True)
joblib.dump(bundle, "models/bundle.joblib", compress=3)

demo = test_p[:40]                      # unseen patients, shown in the dashboard
os.makedirs("demo_data", exist_ok=True)
ehr[ehr.patient_id.isin(demo)].to_csv("demo_data/ehr_demo.csv", index=False)
sens[pidc.isin(demo).to_numpy()].to_csv("demo_data/sensor_demo.csv", index=False)

print("fusion ablation (unseen patients, messy data):")
for k, v in abl.items():
    print(f"  {k:36s} PR-AUC {v['pr_auc']:.3f}  ROC-AUC {v['roc_auc']:.3f}")
print(f"  (chance PR-AUC = {rate:.3f})")
print("alert level table (validation patients):")
for t in table[::2]:
    print("  ", t)
print("chosen alert level:", alert_level)
