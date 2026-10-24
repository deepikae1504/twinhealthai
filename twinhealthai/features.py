"""
Shared, CAUSAL feature builder.

The same function is used (a) to build training data and (b) inside the live
PatientTwin, so training and serving cannot drift apart. Every feature at row t
uses only rows <= t. Rows without enough history get NaN (never a fake 0).

Input: one patient's sensor dataframe sorted by timestamp on a regular 5-minute grid
       (missing readings should appear as NaN glucose/hr/hrv, not as missing rows).
"""
import numpy as np
import pandas as pd

WEARABLE_FEATURES = [
    "glucose_now", "glucose_mean_30m", "glucose_mean_1h", "glucose_std_1h", "glucose_max_1h", "glucose_min_1h",
    "glucose_slope_30m", "glucose_slope_60m", "glucose_accel",
    "hr_mean_1h", "hrv_mean_30m", "hrv_mean_1h", "steps_sum_30m", "steps_sum_1h", "steps_sum_3h",
    "sleeping_flag", "sleep_stage_code", "sleep_hours_prev", "deep_min_prev", "rem_min_prev",
    "min_since_meal", "last_meal_carbs", "last_meal_gi", "carbs_logged_3h",
]
TIME_FEATURES = ["hour_sin", "hour_cos"]
EHR_FEATURES = [
    "age_years", "sex_m", "bmi_kg_m2", "t2d", "family_history_diabetes", "hypertension_diagnosed",
    "dyslipidemia_diagnosed", "hba1c_percent", "fasting_glucose_mg_dl", "total_cholesterol_mg_dl",
    "historical_systolic_bp_mmHg", "med_metformin", "med_combo", "med_none_recorded",
    "genetic_risk_score_simulated", "tcf7l2_risk_alleles_simulated",
]
STAGE_CODE = {"awake": 0, "light": 1, "deep": 2, "rem": 3}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wearable + time features for ONE patient (rows sorted by time)."""
    g, hr, hrv = df["glucose_mg_dl"], df["heart_rate_bpm"], df["hrv_ms"]
    steps = df["steps_5min"]
    out = pd.DataFrame(index=df.index)
    out["glucose_now"] = g
    out["glucose_mean_30m"] = g.rolling(6, min_periods=4).mean()
    out["glucose_mean_1h"] = g.rolling(12, min_periods=8).mean()
    out["glucose_std_1h"] = g.rolling(12, min_periods=8).std()
    out["glucose_max_1h"] = g.rolling(12, min_periods=8).max()
    out["glucose_min_1h"] = g.rolling(12, min_periods=8).min()
    out["glucose_slope_30m"] = (g - g.shift(6)) / 30.0
    out["glucose_slope_60m"] = (g - g.shift(12)) / 60.0
    out["glucose_accel"] = out["glucose_slope_30m"] - (g.shift(6) - g.shift(12)) / 30.0
    out["hr_mean_1h"] = hr.rolling(12, min_periods=8).mean()
    out["hrv_mean_30m"] = hrv.rolling(6, min_periods=4).mean()
    out["hrv_mean_1h"] = hrv.rolling(12, min_periods=8).mean()
    out["steps_sum_30m"] = steps.rolling(6, min_periods=1).sum()
    out["steps_sum_1h"] = steps.rolling(12, min_periods=1).sum()
    out["steps_sum_3h"] = steps.rolling(36, min_periods=1).sum()
    out["sleeping_flag"] = df["sleeping_flag_simulated"]
    out["sleep_stage_code"] = df["sleep_stage_simulated"].map(STAGE_CODE)
    out["sleep_hours_prev"] = df["sleep_hours_previous_night"]
    out["deep_min_prev"] = df["deep_sleep_min_previous_night"]
    out["rem_min_prev"] = df["rem_sleep_min_previous_night"]

    carbs = df["meal_carbs_logged_g"].to_numpy()
    gi = df["meal_gi_class_code"].to_numpy()
    idx = np.arange(len(df))
    last = np.maximum.accumulate(np.where(carbs > 0, idx, -1))
    has = last >= 0
    out["min_since_meal"] = np.where(has, np.minimum((idx - last) * 5, 600), 600)
    out["last_meal_carbs"] = np.where(has, carbs[np.maximum(last, 0)], 0)
    out["last_meal_gi"] = np.where(has, gi[np.maximum(last, 0)], 0)
    out["carbs_logged_3h"] = pd.Series(carbs, index=df.index).rolling(36, min_periods=1).sum()

    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def ehr_features(ehr: pd.DataFrame) -> pd.DataFrame:
    """Static-record features. A missing source value gives NaN (unknown), never a fake 0."""
    e = ehr.copy()
    def flag(col, val):
        return np.where(e[col].isna(), np.nan, (e[col] == val).astype(float))
    e["sex_m"] = flag("sex", "M")
    e["t2d"] = flag("diabetes_status", "type2_diabetes")
    e["med_metformin"] = flag("medication_status", "metformin")
    e["med_combo"] = flag("medication_status", "other_or_combination")
    e["med_none_recorded"] = flag("medication_status", "none_recorded")
    return e[["patient_id"] + EHR_FEATURES]


def build_all(sensor: pd.DataFrame) -> pd.DataFrame:
    """Batch version: features for every patient (sensor sorted by patient, time)."""
    parts = [build_features(g) for _, g in sensor.groupby("patient_id", sort=False)]
    return pd.concat(parts)


def spike_onset_label(glucose: pd.Series, thr: float = 180.0, horizon_steps: int = 24) -> pd.Series:
    """1 if glucose is <= thr now AND exceeds thr at some point in (t, t+2h]. NaN for the last 2h (unknowable)."""
    fut = glucose[::-1].rolling(horizon_steps, min_periods=horizon_steps).max()[::-1].shift(-1)
    lab = ((glucose <= thr) & (fut > thr)).astype(float)
    lab[fut.isna()] = np.nan
    return lab
