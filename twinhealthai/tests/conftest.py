import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import security  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return security.load_model()


@pytest.fixture(scope="session")
def demo():
    return (security.read_csv_safe("demo_data/ehr_demo.csv"),
            security.read_csv_safe("demo_data/sensor_demo.csv", parse_dates=["timestamp"]))


def patient(demo, k):
    ehr, sens = demo
    pid = ehr.patient_id.iloc[k]
    return ehr[ehr.patient_id == pid].iloc[0].to_dict(), sens[sens.patient_id == pid].reset_index(drop=True)
