"""
When does the twin actually ALERT, and how do we score alerts like a doctor would?

An alert fires only if ALL of these hold (this is how false alarms are reduced):
  1. risk >= the alert level chosen on separate validation patients
  2. the risk stays at or above that level for K readings in a row (not a one-reading blip)
  3. glucose is not already above 180 (then the spike has happened, a different message is shown)
  4. the data is trustworthy right now (no big gaps, no frozen sensor, enough history)
  5. no other alert in the last 60 minutes
The SAME rule is used by the live twin (twin.py) and by the scoring code below.
No rule can promise zero false alarms or zero missed spikes: we measure both.
"""
import numpy as np
import pandas as pd

LIMIT = 180.0
HORIZON = 24            # 24 readings = 2 hours
PERSIST = 3             # risk must stay high for 3 readings (15 minutes)
COOLDOWN = 12           # 12 readings = 60 minutes between alerts
MAX_MISSING = 0.25      # more than 25% of the last hour missing = not trustworthy
FALSE_ALERTS_PER_DAY = 0.5   # target used to pick the alert level


def untrusted(g_obs):
    """True where the glucose data cannot be trusted (same logic as twin.py)."""
    s = pd.Series(np.asarray(g_obs, dtype=float))
    miss = s.isna().astype(float)
    frac = miss.rolling(12, min_periods=1).mean().to_numpy()
    frozen = (s.rolling(12).max() == s.rolling(12).min()).to_numpy()
    young = np.arange(len(s)) < 11
    return (frac > MAX_MISSING) | s.isna().to_numpy() | frozen | young


def run_alerts(risk, g_base, bad, thr):
    ok = (np.nan_to_num(risk, nan=0.0) >= thr) & (np.nan_to_num(g_base, nan=999.0) <= LIMIT) & ~bad
    out, last, run = [], -10 ** 9, 0
    for t in range(len(ok)):
        run = run + 1 if ok[t] else 0
        if run >= PERSIST and t - last >= COOLDOWN:
            out.append(t)
            last = t
    return np.array(out, dtype=int)


def spike_events(g_true):
    """Moments glucose newly rises above 180 (crossings within 2 h of an earlier one are merged)."""
    over = np.asarray(g_true) > LIMIT
    ev = []
    for t in np.flatnonzero(over & ~np.r_[False, over[:-1]]):
        if not ev or t - ev[-1] > HORIZON:
            ev.append(int(t))
    return np.array(ev, dtype=int)


def score(alerts, events, days):
    """Doctor-style scoring: false alerts per day, share of alerts that were right, spikes caught, warning time."""
    A, E = np.asarray(alerts), np.asarray(events)
    right = 0
    for a in A:
        i = np.searchsorted(E, a, "right")
        right += int(i < len(E) and E[i] - a <= HORIZON)
    caught, lead = 0, []
    for e in E:
        i = np.searchsorted(A, e - HORIZON, "left")
        if i < len(A) and A[i] < e:
            caught += 1
            lead.append((e - A[i]) * 5)
    return dict(alerts=len(A), right=right, events=len(E), caught=caught, lead=lead, days=days)


def summarise(parts):
    """Combine per-patient score dicts."""
    tot = {k: sum(p[k] for p in parts) for k in ("alerts", "right", "events", "caught", "days")}
    lead = [x for p in parts for x in p["lead"]]
    return {
        "false alerts / patient-day": round(float((tot["alerts"] - tot["right"]) / max(tot["days"], 1e-9)), 2),
        "alerts that were right": round(tot["right"] / tot["alerts"], 2) if tot["alerts"] else float("nan"),
        "spikes caught": round(tot["caught"] / tot["events"], 2) if tot["events"] else float("nan"),
        "median warning (min)": int(np.median(lead)) if lead else 0,
    }
