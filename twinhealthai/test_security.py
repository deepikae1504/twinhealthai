import numpy as np
import pandas as pd
import pytest

import security


def test_tampered_file_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "model.bin"
    f.write_bytes(b"original")
    (tmp_path / "SHA256SUMS").write_text(f"{security.sha256(f)}  model.bin\n")
    monkeypatch.setattr(security, "ROOT", tmp_path)
    monkeypatch.setattr(security, "MANIFEST", tmp_path / "SHA256SUMS")
    assert security.verify("model.bin") == f.resolve()
    f.write_bytes(b"swapped by an attacker")
    with pytest.raises(security.SecurityError, match="hash mismatch"):
        security.verify("model.bin")


def test_unlisted_and_outside_files_are_refused(tmp_path):
    with pytest.raises(security.SecurityError):
        security.verify("/etc/passwd")
    with pytest.raises(security.SecurityError):
        security.verify("../outside.csv")


def test_non_synthetic_data_is_refused():
    with pytest.raises(security.SecurityError):
        security.require_synthetic(pd.DataFrame({"x": [1], "data_type": ["real"]}))
    with pytest.raises(security.SecurityError):
        security.require_synthetic(pd.DataFrame({"x": [1]}))


def test_garbage_readings_are_cleaned_not_crashing():
    clean, issues = security.sanitise({"timestamp": "2026-01-05 10:00", "glucose_mg_dl": "abc", "hrv_ms": 1e9,
                                       "steps_5min": [1, 2], "sleep_stage_simulated": "<script>", "meal_gi_class_code": 99})
    assert np.isnan(clean["glucose_mg_dl"]) and np.isnan(clean["hrv_ms"]) and np.isnan(clean["steps_5min"])
    assert clean["sleep_stage_simulated"] == "awake" and clean["meal_gi_class_code"] == 0 and issues
    assert security.sanitise("not a dict")[0] is None
    assert security.sanitise({"timestamp": "garbage"})[0] is None


def test_impossible_health_record_values_become_unknown():
    clean, issues = security.sanitise_ehr({"patient_id": "x" * 100, "age_years": 250, "hba1c_percent": "n/a",
                                           "sex": "X", "hypertension_diagnosed": 7, "bmi_kg_m2": 27})
    assert np.isnan(clean["age_years"]) and np.isnan(clean["hba1c_percent"]) and np.isnan(clean["sex"])
    assert np.isnan(clean["hypertension_diagnosed"]) and clean["bmi_kg_m2"] == 27 and len(clean["patient_id"]) == 32
    assert issues


def test_shipped_files_match_manifest():
    for f in ["models/bundle.joblib", "demo_data/ehr_demo.csv", "demo_data/sensor_demo.csv"]:
        security.verify(f)
