import numpy as np

import alerts as AL


def _run(risk, bad=None, base=None, thr=0.5):
    n = len(risk)
    return AL.run_alerts(np.array(risk, float), np.full(n, 120.0) if base is None else base,
                         np.zeros(n, bool) if bad is None else np.array(bad), thr)


def test_one_high_reading_is_not_an_alert():
    assert len(_run([0.1, 0.9, 0.1, 0.1, 0.1])) == 0


def test_three_high_readings_in_a_row_alert_once_then_cool_down():
    r = [0.1] * 3 + [0.9] * 10
    assert list(_run(r)) == [5]                       # fires at the 3rd high reading, then 60-min cool-down
    assert list(_run([0.1] * 3 + [0.9] * 20)) == [5, 17]   # a second alert only after the cool-down


def test_untrusted_data_never_alerts():
    assert len(_run([0.9] * 10, bad=[True] * 10)) == 0


def test_already_high_glucose_is_not_a_new_spike_alert():
    assert len(_run([0.9] * 10, base=np.full(10, 200.0))) == 0


def test_scoring_counts_false_alerts_and_caught_spikes():
    g = np.full(400, 120.0)
    g[200:] = 200.0                                    # one spike at index 200
    ev = AL.spike_events(g)
    assert list(ev) == [200]
    s = AL.score(np.array([150, 300]), ev, days=1)     # 150 is too early (over 2 h before), 300 is after the spike
    assert s["right"] == 0 and s["caught"] == 0
    s = AL.score(np.array([190]), ev, days=1)
    assert s["right"] == 1 and s["caught"] == 1 and s["lead"] == [50]
