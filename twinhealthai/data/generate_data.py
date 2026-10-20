#!/usr/bin/env python3
"""
TwinHealth AI - synthetic data generator (v2)

Everything produced here is SIMULATED. The physiology is a simplified,
hand-written model (see docs/data_assumptions.md); it is NOT validated and
every coefficient below is an assumption, not a measured value.

Outputs (in --out):
  ehr_data.csv                  static record, one row per patient
  sensor_data.csv               5-minute wearable stream + logged meals
  meal_log.csv                  one row per LOGGED meal (what a patient/app records)
  sleep_nightly.csv             per-night sleep stage summary
  hidden_simulation_params.csv  hidden "true" parameters (DEBUG ONLY - never a model feature)

Design choices that fix the problems found in v1:
  * meal times are randomised per patient and per day (chronotype + daily jitter,
    weekend shift, skipped meals) so time-of-day is NOT a giveaway
  * a meal log (logged carbs + glycaemic class) is available at the time of eating
  * static record changes the glucose response (insulin resistance, genetics,
    medication) so fusion can matter, and prediabetic patients can occasionally spike
  * simulated genetic markers and per-5-minute sleep stages are included
"""
import argparse
import os
from math import erf, sqrt

import numpy as np
import pandas as pd
from scipy.signal import lfilter

SPD = 288                      # 5-minute steps per day
START = pd.Timestamp("2026-01-05 00:00:00")   # a Monday

# meal_type -> list of (dish, carb_lo_g, carb_hi_g, gi_factor, gi_class(1 low,2 med,3 high), base_weight)
DISHES = {
    "breakfast": [
        ("idli_sambar", 40, 65, 1.10, 2, .28), ("dosa_chutney", 45, 70, 1.10, 2, .22),
        ("poha", 40, 60, 1.15, 3, .12), ("upma", 40, 60, 1.05, 2, .10),
        ("paratha_curd", 50, 80, 1.00, 2, .12), ("ragi_dosa_millet", 35, 55, 0.70, 1, .08),
        ("egg_oats", 20, 35, 0.55, 1, .08)],
    "lunch": [
        ("rice_dal_sabzi", 80, 130, 1.25, 3, .35), ("roti_dal_sabzi", 55, 85, 1.00, 2, .25),
        ("biryani", 90, 140, 1.30, 3, .12), ("curd_rice", 70, 100, 1.10, 2, .10),
        ("millet_meal", 55, 85, 0.70, 1, .10), ("chapati_salad_protein", 35, 55, 0.60, 1, .08)],
    "dinner": [
        ("roti_sabzi", 45, 75, 1.00, 2, .32), ("rice_curry", 65, 110, 1.20, 3, .25),
        ("dosa_chutney", 45, 70, 1.10, 2, .15), ("khichdi", 55, 85, 1.00, 2, .12),
        ("millet_meal", 45, 70, 0.70, 1, .10), ("soup_salad_paneer", 20, 35, 0.50, 1, .06)],
    "snack": [
        ("tea_biscuits", 15, 30, 1.10, 2, .30), ("sweets_mithai", 30, 60, 1.40, 3, .12),
        ("fruit", 15, 30, 0.80, 1, .20), ("samosa_pakora", 25, 45, 1.20, 3, .20),
        ("nuts_sprouts", 5, 12, 0.30, 1, .18)],
}
MED_FACTOR = {"metformin": 0.88, "other_or_combination": 0.78, "none_recorded": 1.0, "lifestyle_only": 1.0}
STAGE_NAMES = {0: "awake", 1: "light", 2: "deep", 3: "rem"}


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


# ----------------------------------------------------------------------------- EHR
def make_ehr(rng, n):
    t2d = rng.random(n) < 0.72
    age = np.where(t2d, rng.normal(50, 11, n), rng.normal(46, 12, n)).clip(20, 80).round().astype(int)
    sex = np.where(rng.random(n) < 0.5, "F", "M")
    bmi = np.where(t2d, rng.normal(27.5, 4.5, n), rng.normal(25.5, 4.2, n)).clip(17, 42).round(1)
    fam = (rng.random(n) < np.where(t2d, 0.45, 0.35)).astype(int)

    pgs_z = rng.normal(0, 1, n) + 0.3 * fam
    alleles = rng.binomial(2, 0.30, n)               # simulated risk-allele count (TCF7L2-style)
    gen = 0.6 * pgs_z + 0.4 * (alleles - 0.6) / 0.66
    gen_score = np.array([100 * phi(z) for z in pgs_z]).round(1)

    a1c = np.where(t2d, rng.normal(7.6, 1.1, n).clip(6.5, 11.5), rng.normal(6.0, 0.2, n).clip(5.7, 6.4)).round(1)
    fasting = np.where(t2d, 105 + 16 * (a1c - 6) + rng.normal(0, 8, n), 100 + 10 * (a1c - 5.7) + rng.normal(0, 5, n))
    fasting = np.where(t2d, fasting.clip(85, 185), fasting.clip(90, 125)).round(1)
    chol = rng.normal(185 + 2 * (bmi - 25), 35, n).clip(110, 330).round(1)
    sbp = (108 + 0.35 * (age - 40) + 0.8 * (bmi - 25) + rng.normal(0, 10, n)).clip(90, 180).round().astype(int)
    htn = (rng.random(n) < np.where(sbp >= 140, 0.7, 0.10)).astype(int)
    dyslip = (rng.random(n) < np.where(chol >= 240, 0.6, 0.08)).astype(int)

    med = np.empty(n, dtype=object)
    for i in range(n):
        if t2d[i]:
            med[i] = rng.choice(["metformin", "other_or_combination", "none_recorded", "lifestyle_only"], p=[.45, .30, .10, .15])
        else:
            med[i] = rng.choice(["lifestyle_only", "metformin", "none_recorded"], p=[.70, .20, .10])

    ehr = pd.DataFrame({
        "patient_id": [f"P{i + 1:04d}" for i in range(n)],
        "age_years": age, "sex": sex, "bmi_kg_m2": bmi,
        "diabetes_status": np.where(t2d, "type2_diabetes", "prediabetes"),
        "family_history_diabetes": fam,
        "hypertension_diagnosed": htn, "dyslipidemia_diagnosed": dyslip,
        "hba1c_percent": a1c, "fasting_glucose_mg_dl": fasting,
        "total_cholesterol_mg_dl": chol, "historical_systolic_bp_mmHg": sbp,
        "medication_status": med,
        "genetic_risk_score_simulated": gen_score,
        "tcf7l2_risk_alleles_simulated": alleles,
        "data_type": "synthetic",
    })
    hidden = pd.DataFrame({"patient_id": ehr.patient_id, "_gen_effect": gen})
    return ehr, hidden


# ----------------------------------------------------------------------------- sleep
def make_stages(rng, L, age):
    """Markov-ish stage sequence: deep early in the night, REM late. 0 wake,1 light,2 deep,3 rem."""
    age_f = float(np.clip(1.35 - 0.012 * (age - 20), 0.6, 1.4))
    st = np.empty(L, dtype=np.int8)
    cur = 1
    for i in range(L):
        f = i / max(L - 1, 1)
        if i == 0:
            st[i] = 1
            cur = 1
            continue
        if rng.random() < 0.82:
            st[i] = cur
            continue
        w = np.array([0.07, 0.95, 0.9 * age_f * (1 - f) ** 2.2, 0.08 + 0.85 * f ** 1.2])
        cur = rng.choice(4, p=w / w.sum())
        st[i] = cur
    return st


# ----------------------------------------------------------------------------- one patient
def sim_patient(rng, p, hid, ndays, s0, spike_scale):
    T = ndays * SPD
    hours_of_day = (np.arange(T) % SPD) / 12.0
    day_idx = np.arange(T) // SPD
    chrono = hid["chrono"]

    # ---- nights -1 .. ndays (night n ends on the morning of day n)
    nights = []
    for n in range(-1, ndays + 1):
        bed_min = 1385 + rng.normal(0, 45) + 0.5 * chrono
        bed_abs = int(round(((n - 1) * 1440 + bed_min) / 5))
        L = int(np.clip(round(rng.normal(hid["sleep_mu"], 0.95) * 12), 60, 108))
        stages = make_stages(rng, L, p["age_years"])
        asleep = (stages != 0).sum()
        nights.append(dict(n=n, bed=bed_abs, wake=bed_abs + L, L=L, stages=stages,
                           sleep_h=asleep / 12.0, deep_min=int((stages == 2).sum() * 5),
                           rem_min=int((stages == 3).sum() * 5), light_min=int((stages == 1).sum() * 5),
                           awake_min=int((stages == 0).sum() * 5), deep_frac=(stages == 2).sum() / L))

    sleeping = np.zeros(T, dtype=np.int8)
    stage_arr = np.zeros(T, dtype=np.int8)
    for nt in nights:
        for j in range(nt["L"]):
            idx = nt["bed"] + j
            if 0 <= idx < T:
                sleeping[idx] = 1
                stage_arr[idx] = nt["stages"][j]
    prev_h = np.full(T, nights[0]["sleep_h"])
    prev_deep = np.full(T, float(nights[0]["deep_min"]))
    prev_rem = np.full(T, float(nights[0]["rem_min"]))
    for nt in nights[1:]:
        w = int(np.clip(nt["wake"], 0, T))
        prev_h[w:] = nt["sleep_h"]
        prev_deep[w:] = nt["deep_min"]
        prev_rem[w:] = nt["rem_min"]

    # ---- daily factors (night n = nights[n+1] drives day n)
    stress = np.clip(rng.normal(0, 1, ndays), -2, 2)
    dayoff = rng.normal(0, 5, ndays)
    sleep_fac = np.ones(ndays)
    poor_sleep = np.zeros(ndays)
    for d in range(ndays):
        nt = nights[d + 1]
        sleep_fac[d] = min(1.4, 1 + 0.07 * max(0, 6.5 - nt["sleep_h"]) + 1.5 * max(0, 0.14 - nt["deep_frac"]))
        poor_sleep[d] = float(nt["sleep_h"] < 6.0)

    # ---- meals
    dish_w = {}
    for mt, lst in DISHES.items():
        w = np.array([x[5] for x in lst]) * np.exp(rng.normal(0, 0.5, len(lst)))
        dish_w[mt] = w / w.sum()
    skip_p = {"breakfast": 0.12, "lunch": 0.05, "dinner": 0.04}

    meal_resp = np.zeros(T)
    meals = []
    walks = []
    medf = MED_FACTOR[p["medication_status"]]
    kmax = 60
    for d in range(ndays):
        weekend = (d % 7) >= 5
        wake_min = (nights[d + 1]["wake"] - d * SPD) * 5
        bed_next = (nights[d + 2]["bed"] - d * SPD) * 5
        hi_d = max(bed_next - 45, 1080)
        plan = [
            ("breakfast", float(np.clip(rng.normal(465 + chrono + (45 if weekend else 0), 55), wake_min + 20, 630))),
            ("lunch", float(np.clip(rng.normal(795 + 0.8 * chrono, 55), 690, 930))),
            ("dinner", float(np.clip(rng.normal(1215 + 0.8 * chrono, 60), min(1110, hi_d), hi_d))),
        ]
        if rng.random() < (0.65 if weekend else 0.55):
            plan.append(("snack", float(np.clip(rng.normal(1005, 50), 930, 1080))))
        for mt, tmin in plan:
            if mt in skip_p and rng.random() < skip_p[mt]:
                continue
            step = int(round((d * 1440 + tmin) / 5))
            if step < 0 or step >= T:
                continue
            k = rng.choice(len(DISHES[mt]), p=dish_w[mt])
            dish, lo, hi, gi, gi_cls, _ = DISHES[mt][k]
            carbs = float(rng.uniform(lo, hi))
            walk = rng.random() < hid["walk_p"]
            walkfac = 0.72 if walk else 1.0
            if walk:
                walks.append((step + int(rng.integers(2, 7)), int(rng.integers(3, 6))))
            amp = carbs * s0 * spike_scale * hid["ir"] * gi * sleep_fac[d] * (1 + 0.04 * max(stress[d], 0)) * medf * walkfac
            tau = max(30.0, 60 - 12 * (gi - 1) + hid["tau_noise"])
            ks = np.arange(1, kmax + 1)
            x = ks * 5.0 / tau
            kern = x * np.exp(1 - x)
            end = min(T, step + kmax + 1)
            meal_resp[step + 1:end] += amp * kern[: end - step - 1]
            logged = rng.random() < 0.85
            meals.append(dict(step=step, meal_type=mt, dish=dish, carbs=carbs, gi_cls=gi_cls,
                              logged=logged, carbs_logged=round(carbs * rng.uniform(0.85, 1.15)),
                              day=d, walk=walk))

    # ---- steps
    awake = sleeping == 0
    steps = rng.poisson(3, T).astype(float) * awake
    prof = (0.02 + 0.06 * np.exp(-((hours_of_day - 8.5) / 1.0) ** 2) + 0.05 * np.exp(-((hours_of_day - 13) / 1.0) ** 2)
            + 0.08 * np.exp(-((hours_of_day - 18.5) / 1.0) ** 2))
    starts = np.flatnonzero((rng.random(T) < prof * hid["activity"]) & awake)
    for s in starts:
        ln = int(rng.integers(1, 5))
        steps[s:s + ln] += rng.integers(80, 240)
    for s, ln in walks:
        steps[s:s + ln] += rng.integers(110, 190)
    steps = (steps * awake).astype(int)
    smooth6 = pd.Series(steps).rolling(6, min_periods=1).sum().values

    # ---- glucose
    dawn = 7 * np.exp(-((hours_of_day - 6.5) / 1.5) ** 2)
    base = p["fasting_glucose_mg_dl"] + dawn - 5 * sleeping + 3 * stress[day_idx] + dayoff[day_idx]
    ar = lfilter([1], [1, -0.85], rng.normal(0, 2.2, T))
    glucose = base + meal_resp - np.clip(0.035 * smooth6, 0, 14) + ar + rng.normal(0, 1.5, T)
    glucose = np.clip(glucose, 45, 380).round(1)

    # ---- heart rate / HRV
    hr = (hid["hr_base"] + 6 * awake - 2 * (1 - awake) + np.clip(0.12 * smooth6 / 6 * 2, 0, 55)
          + np.clip(0.04 * meal_resp, 0, 8) + 2 * stress[day_idx]
          + lfilter([1], [1, -0.8], rng.normal(0, 2.5, T)))
    hr = np.clip(hr, 40, 180).round(1)
    hrv = (hid["hrv_base"] * (1 - 0.06 * stress[day_idx]) * (1 - 0.12 * poor_sleep[day_idx])
           * np.where(sleeping == 1, 1.18, 1.0) * (1 - 0.0015 * np.minimum(steps, 200))
           + lfilter([1], [1, -0.8], rng.normal(0, 3.5, T)))
    hrv = np.clip(hrv, 10, 130).round(1)

    # ---- meal log columns
    carbs_logged = np.zeros(T, dtype=int)
    gi_code = np.zeros(T, dtype=int)
    log_rows = []
    for m in meals:
        if m["logged"]:
            carbs_logged[m["step"]] = m["carbs_logged"]
            gi_code[m["step"]] = m["gi_cls"]
            log_rows.append(dict(patient_id=p["patient_id"], timestamp=START + pd.Timedelta(minutes=5 * m["step"]),
                                 meal_type=m["meal_type"], dish=m["dish"], carbs_logged_g=m["carbs_logged"],
                                 glycemic_class={1: "low", 2: "medium", 3: "high"}[m["gi_cls"]]))

    ts = START + pd.to_timedelta(np.arange(T) * 5, unit="m")
    sensor = pd.DataFrame({
        "patient_id": p["patient_id"], "timestamp": ts, "glucose_mg_dl": glucose,
        "heart_rate_bpm": hr, "hrv_ms": hrv, "steps_5min": steps,
        "sleeping_flag_simulated": sleeping,
        "sleep_stage_simulated": pd.Series(stage_arr).map(STAGE_NAMES).values,
        "sleep_hours_previous_night": prev_h.round(2),
        "deep_sleep_min_previous_night": prev_deep.astype(int),
        "rem_sleep_min_previous_night": prev_rem.astype(int),
        "meal_carbs_logged_g": carbs_logged, "meal_gi_class_code": gi_code,
        "data_type": "synthetic",
    })
    nightly = pd.DataFrame([dict(
        patient_id=p["patient_id"],
        night_ending_date=(START + pd.Timedelta(days=nt["n"])).date(),
        bed_time=START + pd.Timedelta(minutes=5 * nt["bed"]), wake_time=START + pd.Timedelta(minutes=5 * nt["wake"]),
        sleep_hours=round(nt["sleep_h"], 2), deep_min=nt["deep_min"], rem_min=nt["rem_min"],
        light_min=nt["light_min"], awake_in_bed_min=nt["awake_min"], data_type="synthetic")
        for nt in nights if nt["n"] < ndays])
    return sensor, pd.DataFrame(log_rows), nightly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=1000)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s0", type=float, default=0.64, help="mg/dL rise per gram carb at insulin-resistance 1.0")
    ap.add_argument("--spike_scale", type=float, default=1.0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    ehr, hidden = make_ehr(rng, a.patients)
    n = a.patients
    hidden["ir"] = np.exp(0.018 * (ehr.bmi_kg_m2 - 25) + 0.006 * (ehr.age_years - 50) + 0.20 * (ehr.hba1c_percent - 6.5)
                          + 0.12 * ehr.family_history_diabetes - 0.12 * (ehr.diabetes_status == "prediabetes") + 0.12 * hidden["_gen_effect"] + rng.normal(0, 0.17, n))
    hidden["chrono"] = rng.normal(0, 75, n)           # minutes, shifts meal and bed times
    hidden["walk_p"] = rng.uniform(0.10, 0.45, n)
    hidden["activity"] = rng.uniform(0.5, 1.5, n)
    hidden["tau_noise"] = rng.normal(0, 6, n)
    hidden["sleep_mu"] = rng.normal(6.7, 0.6, n)
    hidden["hr_base"] = 62 + 0.12 * (ehr.bmi_kg_m2 - 25) + rng.normal(0, 4, n)
    hidden["hrv_base"] = (58 - 0.35 * (ehr.age_years - 40) + rng.normal(0, 6, n)).clip(20, 90)

    sens, logs, nights = [], [], []
    for i in range(n):
        p = ehr.iloc[i].to_dict()
        h = hidden.iloc[i].to_dict()
        s, l, nt = sim_patient(rng, p, h, a.days, a.s0, a.spike_scale)
        sens.append(s); logs.append(l); nights.append(nt)
        if (i + 1) % 100 == 0:
            print(f"simulated {i + 1}/{n}", flush=True)

    sensor = pd.concat(sens, ignore_index=True)
    meal_log = pd.concat(logs, ignore_index=True)
    nightly = pd.concat(nights, ignore_index=True)
    ehr.to_csv(os.path.join(a.out, "ehr_data.csv"), index=False)
    sensor.to_csv(os.path.join(a.out, "sensor_data.csv"), index=False)
    meal_log.to_csv(os.path.join(a.out, "meal_log.csv"), index=False)
    nightly.to_csv(os.path.join(a.out, "sleep_nightly.csv"), index=False)
    hidden.drop(columns=["_gen_effect"]).round(3).to_csv(os.path.join(a.out, "hidden_simulation_params.csv"), index=False)
    print("done", sensor.shape, meal_log.shape)


if __name__ == "__main__":
    main()
