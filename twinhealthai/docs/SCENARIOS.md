# Scenario catalogue: what can go wrong, and what the twin does

Measured on 100 unseen synthetic patients at the default "Low false alarms" level (full table: `SCENARIO_RESULTS.md`,
re-create with `python evaluate.py`). Reference, clean data: 69% of spikes caught, 65% of alerts right, 0.69 false alerts per patient-day, median warning 35 minutes.

How the twin protects itself, in order: **(1) clean every reading, (2) decide if the data is trustworthy, (3) predict from what is known (the model was trained with gaps), (4) alert only if the risk is high for 15 minutes, (5) otherwise say "cannot judge".**

## A. Sensor problems
| What goes wrong | What the twin does | Test | Measured result |
|---|---|---|---|
| Random missing glucose readings (10 / 30 / 50%) | Carries the last reading, marks data "less reliable" once over 25% of the last hour is missing | `test_every_sensor_problem_runs_safely` | 10%: 62% caught. 30%: 30% caught, says "cannot judge" 60% of the time. 50%: 3% caught, "cannot judge" 94%. **It stops alerting rather than guess.** |
| CGM offline for an hour or three hours | Same as above, and alerts resume when data returns | same | 65 to 66% caught |
| Wearable (heart rate, HRV, steps) off | Model uses the other inputs | same | 70% caught, PR-AUC 0.62 |
| No meal log | Model works without it, but is weaker | same | PR-AUC 0.55, 44% caught |
| Wrong meal log (carbs off by up to 40%, 30% of meals missing) | Trained with a noisy log | same | 62% caught |
| CGM only (no wearable, no meal log) | Works, much weaker | same | PR-AUC 0.53, 34% caught |
| Frozen sensor (same value for 2 hours) | Detected, marked "cannot judge" | `test_every_sensor_problem_runs_safely[cgm_frozen_2h]` | 67% caught |
| Absurd values (glucose 8 or 650, negative heart rate) | Rejected and logged, never used | `test_absurd_values_are_ignored` | n/a |
| Text, empty, wrong type, bad timestamp | Cleaned or rejected, never crashes | `test_text_and_none_in_readings_do_not_crash`, `test_garbage_readings_are_cleaned_not_crashing` | n/a |
| Late, duplicate or very old readings | Merged, ignored or dropped, and logged | `test_old_duplicate_and_late_readings` | n/a |
| Sensor reads 20 mg/dL too high | **Not handled well** | `evaluate.py` | PR-AUC falls to 0.43, only half of alerts right. Needs calibration checks |
| Sensor reads 20 mg/dL too low | Not handled well | same | 0.89 false alerts per day |
| Everything bad at once | Abstains about 38% of the time | `evaluate.py` | 42% caught, 0.43 false alerts per day |
| Sensor silent for hours | "Cannot judge", no alert | `test_sensor_dead_means_cannot_judge_and_no_alert` | n/a |

## B. Health-record problems
| What goes wrong | What the twin does | Measured result |
|---|---|---|
| No HbA1c | Unknown field, model uses the rest | PR-AUC 0.63 (full record 0.64) |
| No labs at all | Same | **PR-AUC 0.59, warning time drops from 35 to 20 minutes** |
| No genetics / medicine info / history flags / diagnosis | Same | PR-AUC 0.64 each |
| Only age, sex and BMI known | Same | PR-AUC 0.60, 64% caught, warning 20 minutes |
| Impossible values (age 999, BMI -5, sex "??") | Treated as unknown, logged | `test_garbage_health_record_values` |
| Record entirely missing | Works from sensors alone | `test_missing_health_record_entirely` |

**What this shows about fusion:** removing the health record lowers accuracy and shortens the warning, so both streams contribute. Unseen patients, messy data: health record only 0.35, wearable and meal log only 0.59, fused 0.62 (chance 0.12, "glucose stays the same" 0.35). The gain is modest, and the simulator was built to make the record matter.

## C. Different kinds of patients
Clean data, full record. Full table in `SCENARIO_RESULTS.md`.
| Group | Result | Comment |
|---|---|---|
| Women / men | PR-AUC 0.65 / 0.63 | Similar |
| Type 2 diabetes | 0.66, about 1 false alert per day | |
| **Prediabetes** | **PR-AUC 0.29 (chance 0.02), only 50% caught, warning 10 minutes** | Rare spikes, weakest group |
| **BMI under 23** | PR-AUC 0.50 (chance 0.05), 78% caught | Lower precision |
| Age under 40 / 40 to 55 / 55+ | 0.59 / 0.64 / 0.66 | |
| BP history, cholesterol history, both, neither | 0.47 (15 patients), 0.64, see table, 0.66 | Small groups, wide uncertainty |
| Lifestyle only, metformin, combination drugs | 0.61 / 0.66 / 0.65 | |

Groups have 14 to 70 patients, so treat differences below about 0.05 as noise.

## D. Not covered (be honest in the video)
Kidney disease, insulin therapy, steroids, pregnancy, type 1 diabetes, illness, alcohol, fasting and festival eating, time-zone travel, real sensor drift, and any real patient. The simulator does not contain them, so the twin cannot be said to handle them.
