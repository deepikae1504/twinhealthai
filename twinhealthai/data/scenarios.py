"""
Ways real data goes wrong. One library, used three ways:
  * train.py     - a RANDOM MIX is applied to training data, so the model practises on messy data
  * evaluate.py  - each FIXED scenario is applied to unseen patients and scored
  * tests        - each scenario is checked to run without errors
Sensor scenarios work on the 5-minute table (patients stored one after another, equal length).
Health-record scenarios work on the per-patient numeric table from features.ehr_features().
"""
import numpy as np
import pandas as pd

WEARABLE = ["heart_rate_bpm", "hrv_ms"]
LABS = ["hba1c_percent", "fasting_glucose_mg_dl", "total_cholesterol_mg_dl", "historical_systolic_bp_mmHg"]
GENES = ["genetic_risk_score_simulated", "tcf7l2_risk_alleles_simulated"]
MEDS = ["med_metformin", "med_combo", "med_none_recorded"]
HISTORY = ["family_history_diabetes", "hypertension_diagnosed", "dyslipidemia_diagnosed"]
BASICS = ["age_years", "sex_m", "bmi_kg_m2"]


def _gaps(n, k, length, rng):
    m = np.zeros(n, bool)
    for _ in range(k):
        s = int(rng.integers(0, max(1, n - length)))
        m[s:s + length] = True
    return m


# ---------------------------------------------------------------- sensor scenarios
def _random(rate):
    return lambda a, n, r: a["g"].__setitem__(r.random(n) < rate, np.nan)


def _gap(k, length):
    return lambda a, n, r: a["g"].__setitem__(_gaps(n, k, length, r), np.nan)


def _wearable_off(a, n, r):
    for c in ("hr", "hrv", "steps"):
        a[c][:] = np.nan


def _meal_off(a, n, r):
    a["carbs"][:] = 0
    a["gi"][:] = 0


def _meal_noisy(a, n, r):
    a["carbs"] *= r.uniform(0.6, 1.4, n)
    drop = (r.random(n) < 0.30) & (a["carbs"] > 0)
    a["carbs"][drop] = 0
    a["gi"][drop] = 0


def _frozen(a, n, r):
    for s in r.integers(0, n - 30, 4):
        a["g"][s:s + 24] = a["g"][s]


def _bias(delta):
    return lambda a, n, r: a["g"].__iadd__(delta)


def _noise(a, n, r):
    a["g"] += r.normal(0, 8, n)


def _everything(a, n, r):
    _random(0.20)(a, n, r); _gap(4, 24)(a, n, r); _meal_noisy(a, n, r)
    off = _gaps(n, 3, 48, r)
    for c in ("hr", "hrv", "steps"):
        a[c][off] = np.nan


SENSOR_SCENARIOS = {
    "clean": lambda a, n, r: None,
    "cgm_random_gaps_10pct": _random(0.10),
    "cgm_random_gaps_30pct": _random(0.30),
    "cgm_random_gaps_50pct": _random(0.50),
    "cgm_offline_6x_1h": _gap(6, 12),
    "cgm_offline_3x_3h": _gap(3, 36),
    "wearable_off": _wearable_off,
    "meal_log_off": _meal_off,
    "meal_log_noisy": _meal_noisy,
    "cgm_only (no wearable, no meal log)": lambda a, n, r: (_wearable_off(a, n, r), _meal_off(a, n, r)),
    "cgm_frozen_2h": _frozen,
    "cgm_reads_20_high": _bias(+20),
    "cgm_reads_20_low": _bias(-20),
    "cgm_noisy": _noise,
    "everything_bad_at_once": _everything,
}


def apply_sensor(sens, name, seed=0, rows_per_patient=2016):
    """Return a degraded copy of the sensor table. `sens` must be sorted by patient, equal-length blocks."""
    out = sens.copy()
    fn, r = SENSOR_SCENARIOS[name], np.random.default_rng(seed)
    cols = {"g": "glucose_mg_dl", "hr": "heart_rate_bpm", "hrv": "hrv_ms", "steps": "steps_5min",
            "carbs": "meal_carbs_logged_g", "gi": "meal_gi_class_code"}
    arr = {k: out[c].to_numpy(dtype=float).copy() for k, c in cols.items()}
    for a in range(0, len(out), rows_per_patient):
        view = {k: v[a:a + rows_per_patient] for k, v in arr.items()}
        fn(view, rows_per_patient, r)
    for k, c in cols.items():
        out[c] = arr[k] if k not in ("carbs", "gi") else arr[k].astype(out[c].dtype)
    return out


def random_training_mix(sens, seed=0, rows_per_patient=2016):
    """Each patient gets a random 'quality level' so training sees clean AND messy streams."""
    out, r = sens.copy(), np.random.default_rng(seed)
    g = out["glucose_mg_dl"].to_numpy(dtype=float).copy()
    arr = {c: out[c].to_numpy(dtype=float).copy() for c in ("heart_rate_bpm", "hrv_ms", "steps_5min")}
    for a in range(0, len(out), rows_per_patient):
        n, sl = rows_per_patient, slice(a, a + rows_per_patient)
        rate = r.choice([0.0, 0.03, 0.10, 0.25, 0.40], p=[.25, .25, .2, .2, .1])
        m = (r.random(n) < rate) | _gaps(n, int(r.integers(0, 6)), int(r.integers(6, 37)), r)
        g[sl][m] = np.nan
        if r.random() < 0.3:                                    # wearable offline for stretches
            off = _gaps(n, int(r.integers(1, 5)), int(r.integers(12, 73)), r)
            for c in arr:
                arr[c][sl][off] = np.nan
    out["glucose_mg_dl"] = g
    for c, v in arr.items():
        out[c] = v
    return out


# ---------------------------------------------------------------- health-record scenarios
EHR_SCENARIOS = {
    "ehr_full": [],
    "no_hba1c": ["hba1c_percent"],
    "no_labs": LABS,
    "no_genetics": GENES,
    "no_medication_info": MEDS,
    "no_history_flags": HISTORY,
    "diagnosis_unknown": ["t2d"],
    "ehr_unknown (only age, sex, BMI known)": [c for c in LABS + GENES + MEDS + HISTORY + ["t2d"]],
}


def apply_ehr(e, name):
    out = e.copy()
    for c in EHR_SCENARIOS[name]:
        out[c] = np.nan
    return out


def random_ehr_mix(e, seed=0):
    """Training: ~35% of patients get an incomplete record; ~5% almost no record."""
    out, r = e.copy(), np.random.default_rng(seed)
    for i in range(len(out)):
        u = r.random()
        if u < 0.05:
            out.loc[out.index[i], [c for c in LABS + GENES + MEDS + HISTORY + ["t2d"]]] = np.nan
        elif u < 0.40:
            for grp in (LABS, GENES, MEDS, HISTORY):
                if r.random() < 0.3:
                    out.loc[out.index[i], grp] = np.nan
            if r.random() < 0.1:
                out.loc[out.index[i], "t2d"] = np.nan
    return out


def blank_ehr_row(row, mode):
    """Hide parts of ONE patient's raw health record (used by the dashboard and the tests)."""
    row = dict(row)
    for c in EHR_SCENARIOS[mode]:
        if c in row:
            row[c] = np.nan
        if c.startswith("med_"):
            row["medication_status"] = np.nan
        if c == "t2d":
            row["diabetes_status"] = np.nan
    return row
