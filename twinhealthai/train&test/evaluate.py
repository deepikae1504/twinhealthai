"""Score the saved model on 100 UNSEEN patients under every scenario.
Usage: python evaluate.py --data data        -> writes docs/SCENARIO_RESULTS.md and docs/scenario_results.csv
Run train.py and make_manifest.py first."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import alerts as AL
import security
from features import EHR_FEATURES, build_all, ehr_features
from scenarios import EHR_SCENARIOS, SENSOR_SCENARIOS, apply_ehr, apply_sensor

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()
ROWS = 2016

B = security.load_model()
clf, level, COLS = B["clf"], B["alert_level"], B["cols"]
ehr = pd.read_csv(f"{a.data}/ehr_data.csv")
pids = np.array(ehr.patient_id.tolist())
np.random.RandomState(0).shuffle(pids)
rep = set(list(pids[:200])[1::2])                          # the same 100 report patients train.py used
sens = pd.read_csv(f"{a.data}/sensor_data.csv", parse_dates=["timestamp"],
                   dtype={"patient_id": "category", "sleep_stage_simulated": "category", "data_type": "category"})
sens = sens[sens.patient_id.astype(str).isin(rep)].sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)
pidc = sens.patient_id.astype(str)
E = ehr_features(ehr).set_index("patient_id")
gclean = sens.glucose_mg_dl.to_numpy()
fut = sens.groupby("patient_id", sort=False, observed=True).glucose_mg_dl.transform(
    lambda s: s[::-1].rolling(24, min_periods=24).max()[::-1].shift(-1)).to_numpy()


def run(sd, feat, Etab):
    """Predict for every row, then score rows and alerts. Returns (summary, per-patient scores, row table)."""
    static = Etab.loc[pidc].reset_index(drop=True)
    static.index = sd.index
    X = pd.concat([feat, static[EHR_FEATURES]], axis=1)[COLS].astype("float32")
    risk = clf.predict_proba(X)[:, 1]
    gseen = sd.glucose_mg_dl.to_numpy()
    base = sd.groupby("patient_id", sort=False, observed=True).glucose_mg_dl.ffill().to_numpy()
    ok = (base <= 180) & ~np.isnan(fut)
    y = (fut > 180).astype(int)
    parts, per, untrusted = {}, {}, []
    for i in range(0, len(sd), ROWS):
        s = slice(i, i + ROWS)
        bad = AL.untrusted(gseen[s])
        untrusted.append(bad.mean())
        al = AL.run_alerts(risk[s], base[s], bad, level)
        per[pidc.iloc[i]] = AL.score(al, AL.spike_events(gclean[s]), ROWS / 288)
    rows = pd.DataFrame({"pid": pidc[ok].to_numpy(), "y": y[ok], "p": risk[ok]})
    return per, rows, float(np.mean(untrusted))


def summary(per, rows, unt):
    out = AL.summarise(list(per.values()))
    out = {"PR-AUC": round(average_precision_score(rows.y, rows.p), 2), "chance": round(rows.y.mean(), 2), **out,
           "time system says 'cannot judge'": f"{unt:.0%}"}
    return out


CACHE = "docs/_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
results = []


def save(key, rec):
    cache[key] = rec
    json.dump(cache, open(CACHE, "w"))

clean = sens
feat_clean = build_all(clean)
print("clean features built", flush=True)
per0, rows0, unt0 = run(clean, feat_clean, E)
TODO = [n for n in SENSOR_SCENARIOS if 'S:' + n not in cache]
MAXRUN = 6                                   # scenarios per run, so one run stays short; re-run to continue
if len(TODO) > MAXRUN:
    SENSOR_SCENARIOS = {k: v for k, v in SENSOR_SCENARIOS.items() if k not in TODO[MAXRUN:]}

# ---- sensor problems (health record complete)
for name in SENSOR_SCENARIOS:
    if "S:" + name in cache:
        results.append(cache["S:" + name])
        continue
    if name == "clean":
        per, rows, unt = per0, rows0, unt0
    else:
        sd = apply_sensor(sens, name, seed=3, rows_per_patient=ROWS)
        per, rows, unt = run(sd, build_all(sd), E)
    results.append(dict(kind="Sensor problem", scenario=name, **summary(per, rows, unt)))
    save("S:" + name, results[-1])
    print(results[-1], flush=True)

# ---- health-record problems (sensors clean)
for name in EHR_SCENARIOS:
    if "E:" + name in cache:
        results.append(cache["E:" + name])
        continue
    per, rows, unt = run(clean, feat_clean, apply_ehr(E, name))
    results.append(dict(kind="Health-record problem", scenario=name, **summary(per, rows, unt)))
    save("E:" + name, results[-1])
    print(results[-1], flush=True)

# ---- different kinds of patients (clean data, full record)
info = ehr.set_index("patient_id")
groups = {
    "sex: female": info.sex == "F", "sex: male": info.sex == "M",
    "prediabetes": info.diabetes_status == "prediabetes", "type 2 diabetes": info.diabetes_status == "type2_diabetes",
    "age under 40": info.age_years < 40, "age 40-55": info.age_years.between(40, 54.9), "age 55+": info.age_years >= 55,
    "BMI under 23 (Asian normal)": info.bmi_kg_m2 < 23, "BMI 23-25 (Asian overweight)": info.bmi_kg_m2.between(23, 24.9),
    "BMI 25+ (Asian obese)": info.bmi_kg_m2 >= 25,
    "on metformin": info.medication_status == "metformin", "lifestyle only, no drugs": info.medication_status == "lifestyle_only",
    "other / combination drugs": info.medication_status == "other_or_combination", "no medication recorded": info.medication_status == "none_recorded",
    "high blood pressure history": info.hypertension_diagnosed == 1, "high cholesterol history": info.dyslipidemia_diagnosed == 1,
    "both BP and cholesterol history": (info.hypertension_diagnosed == 1) & (info.dyslipidemia_diagnosed == 1),
    "neither BP nor cholesterol history": (info.hypertension_diagnosed == 0) & (info.dyslipidemia_diagnosed == 0),
    "family history of diabetes": info.family_history_diabetes == 1,
    "high genetic risk (score 75+)": info.genetic_risk_score_simulated >= 75,
}
for name, mask in groups.items():
    ids = set(info.index[mask]) & rep
    sub_per = {k: v for k, v in per0.items() if k in ids}
    sub_rows = rows0[rows0.pid.isin(ids)]
    if len(sub_per) >= 3 and sub_rows.y.sum() > 0:
        results.append(dict(kind="Patient group", scenario=f"{name} ({len(sub_per)} patients)", **summary(sub_per, sub_rows, unt0)))
        results[-1]["time system says 'cannot judge'"] = "-"

df = pd.DataFrame(results)
df.to_csv("docs/scenario_results.csv", index=False)
with open("docs/SCENARIO_RESULTS.md", "w") as f:
    f.write(f"# Scenario results (100 unseen synthetic patients, alert level {level})\n\n"
            "Higher is better for PR-AUC (compare with 'chance'), spikes caught, alerts that were right. "
            "Lower is better for false alerts per patient-day. Synthetic data: shows the pipeline, not clinical accuracy.\n")
    for kind in df.kind.unique():
        part = df[df.kind == kind].drop(columns="kind")
        f.write(f"\n## {kind}\n\n| " + " | ".join(part.columns) + " |\n|" + "---|" * len(part.columns) + "\n")
        for _, r in part.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
print("written docs/SCENARIO_RESULTS.md")
