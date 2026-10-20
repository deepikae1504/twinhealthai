"""
Security helpers for TwinHealth AI.

What they do (and do NOT do) is written in SECURITY.md. Short version:
  1. verify()           - a file is loaded only if its SHA-256 hash matches SHA256SUMS
                          (stops a swapped model file, which could run code when loaded).
  2. require_synthetic() - refuses any table that is not tagged data_type == "synthetic"
                          (stops real patient data from being loaded by mistake).
  3. sanitise()         - cleans every incoming sensor reading (wrong types, huge values, junk text).
  4. password_ok()      - optional password gate for the dashboard (set TWINHEALTH_PASSWORD).
"""
import hashlib
import hmac
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS"
MAX_FILE_MB = 200


class SecurityError(Exception):
    pass


# ------------------------------------------------------------ file integrity
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest():
    out = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                out[name.strip().lstrip("*")] = digest
    return out


def verify(path):
    """Return the resolved path only if it is inside the project AND its hash matches SHA256SUMS."""
    p = Path(path)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    try:
        rel = p.relative_to(ROOT).as_posix()
    except ValueError:
        raise SecurityError(f"Refusing to load a file outside the project: {p}")
    if not p.is_file() or p.stat().st_size > MAX_FILE_MB * 1024 * 1024:
        raise SecurityError(f"{rel} is missing or too large")
    expected = _manifest().get(rel)
    if expected is None:
        raise SecurityError(f"{rel} is not listed in SHA256SUMS")
    if not hmac.compare_digest(sha256(p), expected):
        raise SecurityError(f"{rel} has been changed (hash mismatch). Refusing to load it.")
    return p


def load_model(path="models/bundle.joblib"):
    import joblib
    return joblib.load(verify(path))


def require_synthetic(df, name="table"):
    if "data_type" not in df.columns or not (df["data_type"].astype(str) == "synthetic").all():
        raise SecurityError(f"{name}: every row must be tagged data_type='synthetic'. "
                            "This prototype must never load real patient data.")
    return df


def read_csv_safe(path, **kw):
    df = pd.read_csv(verify(path), **kw)
    return require_synthetic(df, Path(path).name)


# ------------------------------------------------------------ input cleaning
NUMERIC_LIMITS = {          # very wide "could a sensor ever send this?" limits; twin.py applies tighter ones
    "glucose_mg_dl": (0, 1000), "heart_rate_bpm": (0, 400), "hrv_ms": (0, 500),
    "steps_5min": (0, 3000), "sleep_hours_previous_night": (0, 24),
    "deep_sleep_min_previous_night": (0, 1440), "rem_sleep_min_previous_night": (0, 1440),
    "meal_carbs_logged_g": (0, 500),
}
ALLOWED_STAGES = {"awake", "light", "deep", "rem"}
ALLOWED_FLAGS = {"sleeping_flag_simulated": {0, 1}, "meal_gi_class_code": {0, 1, 2, 3}}


def sanitise(reading):
    """Return (clean_reading, issues). clean_reading is None if the timestamp is unusable.
    Never raises on bad input; bad values become NaN and are listed in `issues`."""
    issues = []
    if not isinstance(reading, dict):
        return None, ["reading is not a dictionary"]
    try:
        ts = pd.Timestamp(reading.get("timestamp"))
        if pd.isna(ts):
            raise ValueError
    except Exception:
        return None, ["unusable timestamp"]
    clean = {"timestamp": ts}
    for k, (lo, hi) in NUMERIC_LIMITS.items():
        v = reading.get(k, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
            if reading.get(k) not in (None, ""):
                issues.append(f"{k}: not a number")
        if np.isfinite(v) and not (lo <= v <= hi):
            issues.append(f"{k}: {v:g} outside possible range")
            v = np.nan
        clean[k] = v if np.isfinite(v) else np.nan
    for k, ok in ALLOWED_FLAGS.items():
        v = reading.get(k, 0)
        try:
            v = int(float(v))
        except (TypeError, ValueError):
            v = 0
        clean[k] = v if v in ok else 0
    stage = str(reading.get("sleep_stage_simulated", "awake")).lower()
    clean["sleep_stage_simulated"] = stage if stage in ALLOWED_STAGES else "awake"
    return clean, issues


# ------------------------------------------------------------ optional password gate
def password_ok(st):
    """If TWINHEALTH_PASSWORD is set, ask for it (5 wrong tries lock the session). Otherwise open."""
    pw = os.environ.get("TWINHEALTH_PASSWORD")
    if not pw or st.session_state.get("auth"):
        return True
    fails = st.session_state.get("fails", 0)
    if fails >= 5:
        st.error("Too many wrong attempts. Close this tab and try again later.")
        return False
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state.auth = True
            st.rerun()
        st.session_state.fails = fails + 1
        st.error("Wrong password.")
    return False


# ------------------------------------------------------------ health-record cleaning
EHR_NUMERIC = {"age_years": (18, 100), "bmi_kg_m2": (12, 70), "hba1c_percent": (3, 20),
               "fasting_glucose_mg_dl": (40, 500), "total_cholesterol_mg_dl": (50, 500),
               "historical_systolic_bp_mmHg": (60, 260), "genetic_risk_score_simulated": (0, 100),
               "tcf7l2_risk_alleles_simulated": (0, 2)}
EHR_FLAGS = ["family_history_diabetes", "hypertension_diagnosed", "dyslipidemia_diagnosed"]
EHR_CATEGORIES = {"sex": {"F", "M"}, "diabetes_status": {"type2_diabetes", "prediabetes"},
                  "medication_status": {"metformin", "other_or_combination", "none_recorded", "lifestyle_only"}}


def sanitise_ehr(rec):
    """Return (clean_record, issues). Missing, wrong-type or impossible values become NaN
    ("unknown"), never a made-up number. The model was trained with such unknowns."""
    issues, clean = [], {"patient_id": str(rec.get("patient_id", "unknown"))[:32], "data_type": "synthetic"}
    for k, (lo, hi) in EHR_NUMERIC.items():
        try:
            v = float(rec.get(k, np.nan))
        except (TypeError, ValueError):
            v = np.nan
        if np.isfinite(v) and not (lo <= v <= hi):
            issues.append(f"{k}: {v:g} is impossible, treated as unknown")
            v = np.nan
        clean[k] = v if np.isfinite(v) else np.nan
    for k in EHR_FLAGS:
        try:
            v = float(rec.get(k, np.nan))
        except (TypeError, ValueError):
            v = np.nan
        clean[k] = v if v in (0.0, 1.0) else np.nan
    for k, ok in EHR_CATEGORIES.items():
        v = rec.get(k)
        clean[k] = v if isinstance(v, str) and v in ok else np.nan
    return clean, issues
