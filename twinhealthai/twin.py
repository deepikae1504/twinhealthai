"""PatientTwin - a virtual patient that keeps state and updates with every new reading.

Static record (EHR) + live wearable stream -> current risk of a glucose spike in the next 2h,
a 2-hour glucose curve, reasons, data-quality notes, and a what-if copy of itself.
Uses the SAME feature code as training (features.py), so serving cannot drift from training.
"""
import copy
import numpy as np
import pandas as pd
import alerts as AL
import security
from features import build_features, ehr_features, EHR_FEATURES, WEARABLE_FEATURES

STEP = pd.Timedelta(minutes=5)
BUFFER_STEPS = 288                                  # keep the last 24 hours
SENSOR_COLS = ["glucose_mg_dl", "heart_rate_bpm", "hrv_ms", "steps_5min", "sleeping_flag_simulated",
               "sleep_stage_simulated", "sleep_hours_previous_night", "deep_sleep_min_previous_night",
               "rem_sleep_min_previous_night", "meal_carbs_logged_g", "meal_gi_class_code"]
PLAUSIBLE = {"glucose_mg_dl": (40, 400), "heart_rate_bpm": (30, 220), "hrv_ms": (5, 150)}
GROUPS = {   # feature groups used for "reasons"
    "Current glucose & trend": [f for f in WEARABLE_FEATURES if f.startswith("glucose")],
    "Recent meal": ["min_since_meal", "last_meal_carbs", "last_meal_gi", "carbs_logged_3h"],
    "Activity": ["steps_sum_30m", "steps_sum_1h", "steps_sum_3h"],
    "Heart rate & stress (HRV)": ["hr_mean_1h", "hrv_mean_30m", "hrv_mean_1h"],
    "Sleep": ["sleeping_flag", "sleep_stage_code", "sleep_hours_prev", "deep_min_prev", "rem_min_prev"],
    "Health record (EHR)": EHR_FEATURES,
}


class PatientTwin:
    def __init__(self, ehr_row, bundle):
        self.ehr_raw = dict(ehr_row)
        self.ehr, self.ehr_issues = security.sanitise_ehr(self.ehr_raw)       # impossible/missing -> unknown (NaN)
        self.b = bundle
        self.ehr_x = ehr_features(pd.DataFrame([self.ehr])).iloc[0][EHR_FEATURES].astype(float)
        self.ehr_unknown = [c for c in EHR_FEATURES if pd.isna(self.ehr_x[c])]
        self.alert_level = float(bundle.get("alert_level", 0.5))
        self._run = 0                                                          # readings in a row above the alert level
        self.buf = pd.DataFrame(columns=SENSOR_COLS)
        self.buf.index = pd.DatetimeIndex([], name="timestamp")
        self.notes, self.timeline, self.risk_history = [], [], []
        self.state, self._last_alert = None, None

    # ------------------------------------------------------------------ ingest
    def update(self, reading):
        """Add one reading (dict with 'timestamp' + any SENSOR_COLS) and re-predict. Never raises on bad data."""
        clean, issues = security.sanitise(reading)
        if clean is None:
            self._note(pd.Timestamp.now().floor("5min"), "unusable reading rejected: " + "; ".join(issues))
            return self.state
        ts = clean["timestamp"].floor("5min")
        for msg in issues:
            self._note(ts, "reading cleaned: " + msg)
        row = {c: clean.get(c, np.nan) for c in SENSOR_COLS}
        for c, (lo, hi) in PLAUSIBLE.items():
            v = row[c]
            if v is not None and not pd.isna(v) and not (lo <= v <= hi):
                self._note(ts, f"implausible {c.split('_')[0]} value {v:g} ignored")
                row[c] = np.nan
        if len(self.buf) and ts < self.buf.index[0]:
            self._note(ts, "reading older than the 24h window dropped")
            return self.state
        if ts in self.buf.index:
            old = self.buf.loc[ts]
            if pd.isna(old["glucose_mg_dl"]) and not pd.isna(row["glucose_mg_dl"]):
                self._note(ts, "late reading merged into its slot")
                self.buf.loc[ts] = pd.Series(row)
            else:
                self._note(ts, "duplicate reading ignored")
        else:
            if len(self.buf) and ts < self.buf.index[-1]:
                self._note(ts, "late reading inserted")
            self.buf.loc[ts] = pd.Series(row)
        self._regularise()
        self.state = self._compute(self.buf)
        self._log_events()
        return self.state

    def _regularise(self):
        b = self.buf.sort_index()
        full = pd.date_range(b.index[0], b.index[-1], freq=STEP, name="timestamp")
        if len(full) != len(b):
            b = b.reindex(full)
            for c in ["sleeping_flag_simulated", "sleep_stage_simulated", "sleep_hours_previous_night",
                      "deep_sleep_min_previous_night", "rem_sleep_min_previous_night"]:
                b[c] = b[c].ffill()
            for c in ["meal_carbs_logged_g", "meal_gi_class_code"]:
                b[c] = b[c].fillna(0)
        self.buf = b.iloc[-BUFFER_STEPS:]

    # ------------------------------------------------------------------ predict
    def _row(self, buf):
        d = buf.copy(); d["timestamp"] = d.index
        f = build_features(d.reset_index(drop=True)).iloc[[-1]].reset_index(drop=True)
        f = f.assign(**self.ehr_x.to_dict())
        return f[self.b["cols"]].astype("float32")

    def _compute(self, buf):
        x = self._row(buf)
        g_now = buf["glucose_mg_dl"].iloc[-1]
        last_valid = buf["glucose_mg_dl"].dropna()
        base = g_now if not pd.isna(g_now) else (last_valid.iloc[-1] if len(last_valid) else np.nan)
        risk = float(self.b["clf"].predict_proba(x)[0, 1])
        curve = {0: base}
        for k, m in self.b["regs"].items():
            curve[k * 5] = float(base + m.predict(x)[0]) if not pd.isna(base) else np.nan
        recent = buf["glucose_mg_dl"].iloc[-12:]
        why = []
        if len(buf) < 12: why.append("warming up: less than 1 hour of history")
        if pd.isna(g_now): why.append("latest glucose reading missing")
        if recent.isna().mean() > 0.25: why.append(f"{recent.isna().mean():.0%} of the last hour's glucose missing")
        lo, hi = self.b["glucose_range"]
        if not pd.isna(g_now) and not (lo <= g_now <= hi): why.append("glucose outside the range seen in training")
        last12 = buf["glucose_mg_dl"].iloc[-12:]
        if len(last12) == 12 and last12.notna().all() and last12.max() == last12.min():
            why.append("glucose readings look frozen (sensor stuck?)")
        already_high = (not pd.isna(base)) and base > 180
        return dict(time=buf.index[-1], glucose=base, risk=None if already_high else risk, raw_risk=risk,
                    curve=curve, ehr_unknown=len(self.ehr_unknown), already_high=already_high, uncertain=bool(why), uncertain_why=why)

    def explain(self, buf=None):
        """Which input groups push the risk up/down? (replace a group with typical values, see risk change)"""
        buf = self.buf if buf is None else buf
        x = self._row(buf); base = float(self.b["clf"].predict_proba(x)[0, 1]); out = {}
        for name, cols in GROUPS.items():
            x2 = x.copy()
            for c in cols:
                x2[c] = self.b["medians"][c]
            out[name] = base - float(self.b["clf"].predict_proba(x2)[0, 1])
        return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))

    # ------------------------------------------------------------------ what-if
    def what_if(self, scenario):
        """Change ONE input on a copy of the recent data and re-predict (the real twin is untouched). Shows how the MODEL reacts - not proof of treatment."""
        b = self.buf.copy(); msg = None
        recent = b.index[-36:]
        if scenario == "millet":
            meal = b.loc[recent, "meal_carbs_logged_g"] > 0
            if not meal.any():
                msg = "No meal logged in the last 3 hours, so there is nothing to swap."
            b.loc[recent[meal.values], "meal_carbs_logged_g"] *= 0.75
            b.loc[recent[meal.values], "meal_gi_class_code"] = 1
        elif scenario == "portion":
            meal = b.loc[recent, "meal_carbs_logged_g"] > 0
            if not meal.any():
                msg = "No meal logged in the last 3 hours, so there is nothing to shrink."
            b.loc[recent[meal.values], "meal_carbs_logged_g"] *= 0.7
        s = self._compute(b)
        return dict(state=s, note=msg, reasons=self.explain(b))

    # ------------------------------------------------------------------ bookkeeping
    def _note(self, ts, text):
        self.notes.append((ts, text)); self.notes = self.notes[-50:]

    def _log_events(self):
        """Alert rule is in alerts.py (same rule is used when scoring): level + persistence + trusted data + cool-down."""
        s = self.state; t = s["time"]
        high = s["raw_risk"] >= self.alert_level and not s["already_high"]
        self._run = self._run + 1 if (high and not s["uncertain"]) else 0
        cooled = self._last_alert is None or (t - self._last_alert) >= pd.Timedelta(minutes=5 * AL.COOLDOWN)
        if s["already_high"]:
            if not self.timeline or self.timeline[-1][1] != "above 180":
                self.timeline.append((t, "above 180", f"Glucose is {s['glucose']:.0f} mg/dL (above 180)"))
        elif self._run >= AL.PERSIST and cooled:
            self.timeline.append((t, "alert", f"Spike likely within 2h (risk {s['raw_risk']:.0%})"))
            self._last_alert = t
        elif high and s["uncertain"] and (not self.timeline or self.timeline[-1][1] != "cannot judge"):
            self.timeline.append((t, "cannot judge", "Risk may be high but the data is unreliable: " + "; ".join(s["uncertain_why"])))
        self.risk_history.append((t, s["raw_risk"], s["glucose"]))
        self.risk_history = self.risk_history[-600:]

    def fork(self):
        return copy.deepcopy(self)
