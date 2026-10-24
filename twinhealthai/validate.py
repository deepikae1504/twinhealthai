#!/usr/bin/env python3
"""Validate the synthetic dataset and run the fusion ablation. Writes validation_report.txt."""
import argparse, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as pr_auc, roc_auc_score as roc, brier_score_loss
from features import (WEARABLE_FEATURES, TIME_FEATURES, EHR_FEATURES, build_all, build_features,
                      ehr_features, spike_onset_label)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="validation_report.txt")
a = ap.parse_args()
L = []
def log(x=""):
    print(x, flush=True); L.append(x)
def chk(ok, msg):
    log(("PASS: " if ok else "FAIL: ") + msg)

ehr = pd.read_csv(f"{a.data}/ehr_data.csv")
sens = pd.read_csv(f"{a.data}/sensor_data.csv", parse_dates=["timestamp"], dtype={"patient_id": "category", "sleep_stage_simulated": "category", "data_type": "category"}).sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)
meals = pd.read_csv(f"{a.data}/meal_log.csv")
nightly = pd.read_csv(f"{a.data}/sleep_nightly.csv")
npat, days = ehr.patient_id.nunique(), 7

log("=== STRUCTURAL CHECKS ===")
chk(len(ehr) == npat and ehr.patient_id.is_unique, f"EHR has {npat} unique patients")
chk(set(sens.patient_id) <= set(ehr.patient_id), "all sensor patient IDs exist in EHR")
chk(len(sens) == npat * days * 288, f"sensor rows = patients x {days} days x 288 ({len(sens):,})")
chk(not sens.duplicated(["patient_id", "timestamp"]).any(), "no duplicate patient/timestamp rows")
step = sens.groupby("patient_id").timestamp.diff().dropna().dt.total_seconds().unique()
chk(len(step) == 1 and step[0] == 300, "every patient is on a regular 5-minute grid")
chk(sens.glucose_mg_dl.between(40, 400).all() and sens.heart_rate_bpm.between(35, 190).all() and sens.hrv_ms.between(5, 140).all(), "glucose / heart rate / HRV within simulation bounds")
chk(((sens.meal_carbs_logged_g > 0) & (sens.sleeping_flag_simulated == 1)).sum() == 0, "no meals logged while the patient is asleep")
chk(set(ehr.columns) >= {"genetic_risk_score_simulated", "tcf7l2_risk_alleles_simulated"}, "simulated genetic markers present in EHR")
chk(set(sens.sleep_stage_simulated.unique()) == {"awake", "light", "deep", "rem"}, "sleep stages present (awake/light/deep/rem)")
chk(ehr.data_type.eq("synthetic").all() and sens.data_type.eq("synthetic").all(), "every row is tagged synthetic")

log("\n=== FEATURES + LABEL ===")
feat = build_all(sens)
sens["y"] = sens.groupby("patient_id", sort=False).glucose_mg_dl.transform(lambda s: spike_onset_label(s))
# causality test: features at T computed from data up to T only must equal the batch value
rng = np.random.default_rng(1)
ok = True
for _ in range(40):
    pid = rng.choice(sens.patient_id.unique()); g = sens[sens.patient_id == pid].reset_index(drop=True)
    T = int(rng.integers(20, len(g) - 1))
    part = build_features(g.iloc[: T + 1]).iloc[-1]
    full = feat.loc[sens.index[sens.patient_id == pid][T]]
    ok &= bool(np.allclose(part.values.astype(float), full.values.astype(float), equal_nan=True, atol=1e-6))
chk(ok, "no future data in features (features from data up to T == features computed with all data, 40 random checks)")
chk(sens.y.isna().groupby(sens.patient_id).sum().eq(24).all(), "last 2h of each patient is unlabelled (24 rows)")

m = sens.merge(ehr[["patient_id", "diabetes_status"]], on="patient_id")
valid = m.y.notna()
log(f"Label: spike ONSET = glucose <= 180 now AND > 180 at some point in the next 2h (rows already > 180 are not labelled positive)")
log(f"Labelled rows: {valid.sum():,}; positive rate overall: {m.y.mean():.2%}")
log(f"Rows already above 180 (excluded from modelling): {(m.glucose_mg_dl > 180).mean():.2%}")
for k, v in m.groupby("diabetes_status").y.mean().items():
    log(f"  positive rate, {k}: {v:.2%}")
pp = m.groupby(["diabetes_status", "patient_id"]).y.mean().reset_index()
for k, v in pp.assign(any=pp.y > 0).groupby("diabetes_status")["any"].mean().items():
    log(f"  patients with at least one positive window, {k}: {v:.0%}")

log("\n=== MEAL-TIME / TIME-OF-DAY CHECK ===")
ml = pd.read_csv(f"{a.data}/meal_log.csv", parse_dates=["timestamp"])
for mt in ["breakfast", "lunch", "dinner"]:
    mins = ml[ml.meal_type == mt].timestamp.dt.hour * 60 + ml[ml.meal_type == mt].timestamp.dt.minute
    log(f"  {mt}: mean {int(mins.mean())//60:02d}:{int(mins.mean())%60:02d}, spread (sd) {mins.std():.0f} min")
log(f"  logged meals per patient-day: {len(ml)/npat/days:.2f}")

# ---------------------------------------------------------------- ablation
log("\n=== FUSION ABLATION (patient-level split, 800 train / 200 test patients) ===")
e = ehr_features(ehr)
import gc
sens = sens[["patient_id", "glucose_mg_dl", "y"]].copy()      # drop heavy string columns to save memory
del m, ml, meals, nightly
static = e.set_index("patient_id").loc[sens.patient_id].reset_index(drop=True).astype("float32")
static.index = sens.index
X = pd.concat([feat.astype("float32"), static], axis=1)
del feat, static; gc.collect()
use = valid & (sens.glucose_mg_dl <= 180)
pids = np.array(ehr.patient_id.tolist()); np.random.RandomState(0).shuffle(pids)
test_p = set(pids[: int(0.2 * len(pids))])
te = use & sens.patient_id.isin(test_p); tr = use & ~sens.patient_id.isin(test_p)
tr_idx = np.flatnonzero(tr.values)[::6]                      # stride 2: neighbouring rows are near-duplicates
yv = sens.y.values
log(f"train rows {len(tr_idx):,} | test rows {te.sum():,} | test positive rate {yv[te.values].mean():.2%} (evaluated only where glucose <= 180)")

def metrics(y, s):
    thr = np.quantile(s[y == 0], 0.9)
    return pr_auc(y, s), roc(y, s), (s[y == 1] > thr).mean()
rows = []
yt = yv[te.values]
for nm, sc in [("Baseline: glucose stays the same", X.loc[te, "glucose_now"].values),
               ("Baseline: linear trend (now + 30m slope)", (X.loc[te, "glucose_now"] + 30 * X.loc[te, "glucose_slope_30m"].fillna(0)).values)]:
    rows.append((nm, *metrics(yt, sc), np.nan))
preds = {}
sets = {"EHR only": EHR_FEATURES, "Wearable + meal log only": WEARABLE_FEATURES,
        "Wearable + meal log + hour of day": WEARABLE_FEATURES + TIME_FEATURES,
        "FUSED (wearable + meal log + EHR)": WEARABLE_FEATURES + EHR_FEATURES,
        "FUSED + hour of day": WEARABLE_FEATURES + EHR_FEATURES + TIME_FEATURES}
for nm, fs in sets.items():
    mdl = HGB(max_iter=100, learning_rate=0.12, random_state=0).fit(X.iloc[tr_idx][fs], yv[tr_idx])
    p = mdl.predict_proba(X.loc[te, fs])[:, 1]; preds[nm] = p
    rows.append((nm, *metrics(yt, p), brier_score_loss(yt, p)))
    print("done", nm, flush=True)
mdl = HGB(max_iter=100, random_state=0).fit(X.iloc[tr_idx][TIME_FEATURES], yv[tr_idx])
p = mdl.predict_proba(X.loc[te, TIME_FEATURES])[:, 1]
rows.append(("Hour of day only", *metrics(yt, p), brier_score_loss(yt, p)))
tab = pd.DataFrame(rows, columns=["model", "PR-AUC", "ROC-AUC", "recall@10%FPR", "Brier"]).round(3)
log(tab.to_string(index=False))
log(f"(positive-rate baseline PR-AUC is about {yt.mean():.3f}; Brier of a constant predictor is {brier_score_loss(yt, np.full(len(yt), yt.mean())):.3f})")

# paired patient-level bootstrap of the fusion gain
tp = sens.loc[te, "patient_id"].values; up = np.unique(tp)
by = {u: np.flatnonzero(tp == u) for u in up}
rb = np.random.default_rng(0)
def boot(n1, n2, B=150):
    d = []
    for _ in range(B):
        idx = np.concatenate([by[u][::3] for u in rb.choice(up, len(up))])
        d.append(pr_auc(yt[idx], preds[n1][idx]) - pr_auc(yt[idx], preds[n2][idx]))
    return np.percentile(d, [2.5, 50, 97.5])
for n1, n2 in [("FUSED (wearable + meal log + EHR)", "Wearable + meal log only"),
               ("FUSED + hour of day", "Wearable + meal log + hour of day")]:
    lo, md, hi = boot(n1, n2)
    log(f"PR-AUC gain from adding EHR, {'without' if 'hour' not in n1 else 'with'} hour of day: {md:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")

p = preds["FUSED (wearable + meal log + EHR)"]
q = pd.qcut(p, 10, duplicates="drop")
cal = pd.DataFrame({"pred": p, "obs": yt}).groupby(q, observed=True).mean().round(3)
log("\nCalibration of the fused model (decile of predicted risk: mean predicted vs observed):")
log(cal.to_string(index=False))

log("\nLIMITATIONS:")
for s in ["All records are simulated; no real patient or wearable data is included.",
          "The simulation rules are hand-written and simplified; every coefficient is an assumption, not a validated physiological value.",
          "The EHR matters in this data BECAUSE the generator makes it matter (insulin resistance, genetics, medication change the glucose response). The ablation shows the model can recover that structure, not that it holds in real patients.",
          "Cohort is type 2 diabetes and prediabetes only; label frequency depends on the chosen 180 mg/dL threshold and simulation settings.",
          "High scores on this data would not prove clinical usefulness. Not for diagnosis, treatment, or real clinical decisions."]:
    log("- " + s)
open(a.out, "w").write("\n".join(L) + "\n")
