"""Scenario tests: every kind of data problem in scenarios.py must run safely."""
import numpy as np
import pandas as pd
import pytest

from conftest import patient
from features import build_features
from replay import replay
from scenarios import EHR_SCENARIOS, SENSOR_SCENARIOS, apply_sensor, blank_ehr_row
from twin import PatientTwin

N_WARM, N_RUN = 150, 40           # readings of history, then readings watched (5 minutes each)


def run_twin(bundle, ehr_row, sens, n=N_WARM + N_RUN):
    tw = PatientTwin(ehr_row, bundle)
    for r in list(replay(sens.iloc[:n], messy=False))[:n]:
        tw.update(r)
    return tw


def test_features_never_use_future_data(demo):
    for k in range(3):
        _, P = patient(demo, k)
        full = build_features(P)
        for T in (400, 900, 1500):
            cut = build_features(P.iloc[:T + 1])
            np.testing.assert_allclose(full.iloc[T].to_numpy(float), cut.iloc[T].to_numpy(float), equal_nan=True, rtol=1e-6)


@pytest.mark.parametrize("name", list(SENSOR_SCENARIOS))
def test_every_sensor_problem_runs_safely(bundle, demo, name):
    e, P = patient(demo, 1)
    W = P.iloc[:N_WARM + N_RUN].reset_index(drop=True)          # the scenario acts INSIDE the window the twin reads
    tw = run_twin(bundle, e, apply_sensor(W, name, seed=3, rows_per_patient=len(W)))
    s = tw.state
    assert 0.0 <= s["raw_risk"] <= 1.0
    if s["uncertain"]:
        assert s["uncertain_why"]


@pytest.mark.parametrize("name", list(EHR_SCENARIOS))
def test_every_health_record_problem_runs_safely(bundle, demo, name):
    e, P = patient(demo, 2)
    tw = run_twin(bundle, blank_ehr_row(e, name), P)
    assert 0.0 <= tw.state["raw_risk"] <= 1.0
    assert tw.state["ehr_unknown"] >= (0 if name == "ehr_full" else 1)


def test_missing_health_record_entirely(bundle, demo):
    e, P = patient(demo, 0)
    bare = {"patient_id": "new", "age_years": 50, "sex": "F", "bmi_kg_m2": 27}
    tw = run_twin(bundle, bare, P)
    assert tw.state["ehr_unknown"] >= 10 and 0.0 <= tw.state["raw_risk"] <= 1.0


def test_garbage_health_record_values(bundle, demo):
    e, P = patient(demo, 0)
    e = dict(e, age_years=999, hba1c_percent="high", sex="??", bmi_kg_m2=-5)
    tw = run_twin(bundle, e, P)
    assert tw.ehr_issues and 0.0 <= tw.state["raw_risk"] <= 1.0


def test_sensor_dead_means_cannot_judge_and_no_alert(bundle, demo):
    e, P = patient(demo, 1)
    P = P.copy()
    P.loc[N_WARM - 60:, ["glucose_mg_dl", "heart_rate_bpm", "hrv_ms"]] = np.nan     # everything goes silent
    tw = run_twin(bundle, e, P, n=N_WARM + 60)
    assert tw.state["uncertain"]
    after = [k for t, k, _ in tw.timeline if t > P.timestamp.iloc[N_WARM]]
    assert "alert" not in after


def test_absurd_values_are_ignored(bundle, demo):
    e, P = patient(demo, 1)
    tw = run_twin(bundle, e, P, n=N_WARM)
    r = {"timestamp": P.timestamp.iloc[N_WARM], "glucose_mg_dl": 650, "heart_rate_bpm": -20}
    s = tw.update(r)
    assert s["glucose"] < 400 and any("implausible" in n[1] or "cleaned" in n[1] for n in tw.notes)


def test_text_and_none_in_readings_do_not_crash(bundle, demo):
    e, P = patient(demo, 1)
    tw = run_twin(bundle, e, P, n=N_WARM)
    ts = P.timestamp.iloc[N_WARM]
    for bad in ({"timestamp": ts, "glucose_mg_dl": "oops"}, {"timestamp": "not a time"}, {"timestamp": ts, "hrv_ms": None},
                {"glucose_mg_dl": 100}, "string", None, {"timestamp": ts, "glucose_mg_dl": float("inf")}):
        assert tw.update(bad) is not None


def test_old_duplicate_and_late_readings(bundle, demo):
    e, P = patient(demo, 1)
    tw = run_twin(bundle, e, P, n=N_WARM)
    old = {"timestamp": P.timestamp.iloc[5], "glucose_mg_dl": 100}
    dup = {"timestamp": P.timestamp.iloc[N_WARM - 1], "glucose_mg_dl": 123}
    n0 = len(tw.buf)
    tw.update(old); tw.update(dup)
    assert len(tw.buf) == n0 and tw.state is not None


def test_what_if_never_changes_the_real_twin(bundle, demo):
    e, P = patient(demo, 1)
    tw = run_twin(bundle, e, P, n=N_WARM)
    before = (tw.state["raw_risk"], len(tw.buf))
    f = tw.fork()
    for r in list(replay(P.iloc[N_WARM:N_WARM + 20], messy=False)):
        f.update(r)
    assert (tw.state["raw_risk"], len(tw.buf)) == before


def test_different_patients_get_different_risks(bundle, demo):
    risks = {round(run_twin(bundle, *patient(demo, k), n=80).state["raw_risk"], 3) for k in range(6)}
    assert len(risks) > 3
