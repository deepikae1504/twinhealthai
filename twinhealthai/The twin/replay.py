"""Replay a patient's day as a stream of readings, optionally with realistic sensor problems."""
import numpy as np
import pandas as pd


def replay(sensor_df, messy=False, seed=0):
    """Yield readings (dicts) in arrival order. messy=True: ~8% dropped, ~4% delivered late,
    ~5% of glucose values missing while other sensors arrive, ~0.5% absurd glucose spikes."""
    df = sensor_df.sort_values("timestamp").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = np.arange(len(df)).astype(float)
    keep = np.ones(len(df), bool)
    if messy:
        keep = rng.random(len(df)) > 0.08
        late = rng.random(len(df)) < 0.04
        order = order + np.where(late, rng.integers(2, 5, len(df)), 0) + 0.1
    for i in np.argsort(order, kind="stable"):
        if not keep[i]:
            continue
        r = df.iloc[i].to_dict()
        r = {k: r[k] for k in r if k not in ("patient_id", "data_type", "y")}
        if messy:
            u = rng.random()
            if u < 0.05: r["glucose_mg_dl"] = np.nan
            elif u < 0.055: r["glucose_mg_dl"] = float(rng.choice([8, 650]))
        yield r
