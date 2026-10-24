# TwinHealth AI: a Digital Twin for glucose spikes in type 2 diabetes (India)

| | |
|---|---|
| **Team** | `<cosmic bloom>` |
| **College ** | `<GRT Institute of Engineering and Technology>` |
| **Video (20+ min)** | `<https://drive.google.com/file/d/1VHtPL3s8PtOyE2GuDWsatUtUcjRj09rQ/view?usp=sharing>` |
| **License** | MIT (see `LICENSE`) |
| **Architecture** | `docs/architecture.pdf` |
| **Presentation** | `<docs/TwinHealth_AI.pdf>` |

## Problem and use case
Type 2 diabetes is very common in India, and doctors see patients only in short visits. TwinHealth AI keeps a **virtual copy of one patient** that joins the **health record** (static) with **wearable and glucose-sensor data** (live, every 5 minutes). It warns when glucose is likely to **rise above 180 mg/dL within the next 2 hours**, shows why, and lets the doctor test "what if" changes (sleep, a walk, a lighter meal). It is decision support for a doctor, not a diagnosis tool.

## What the brief asks, and where it is
| Requirement | Where |
|---|---|
| Static data (history, labs, genetics) | `data/ehr_data.csv` from `generate_data.py` |
| Live data (glucose, HRV, sleep, steps) | `data/sensor_data.csv` |
| Fusion of both | `train.py` (fusion test), `twin.py` |
| Predict an adverse event 2 h ahead | spike risk and 2-hour curve, `twin.py` |
| Doctor dashboard | `app.py` |
| Synthetic or open data only | enforced in code, see `SECURITY.md` |

## How it works
1. `generate_data.py` simulates 1,000 patients for 7 days (Indian meals, genetics, sleep stages, meal log).
2. `scenarios.py` adds real-world trouble: missing readings, offline sensors, wrong meal logs, incomplete health records.
3. `train.py` trains gradient boosting on that **messy** data and tests on 200 unseen patients. It compares health-record-only, wearable-only and fused models, and chooses the alert level on separate validation patients.
4. `twin.py` is the virtual patient: it keeps 24 hours of state, updates with every reading, and cleans bad input (`security.py`).
5. `alerts.py` decides when to alert: risk above the level for 15 minutes, trustworthy data, 60-minute cool-down. Otherwise it says **"cannot judge"** instead of guessing.
6. `app.py` is the doctor dashboard, with a simulator for every sensor and record problem.

## Results (unseen synthetic patients)
- **Fusion:** PR-AUC, health record only **0.35**, wearable and meal log only **0.59**, fused **0.62** (chance 0.12, "glucose stays the same" 0.35).
- **Alerts at the default "Low false alarms" level, clean data:** 69% of spikes caught, 65% of alerts right, **0.69 false alerts per patient-day**, median warning 35 minutes.
- **Trade-off:** fewer false alarms means more missed spikes. The dashboard has an alert-style slider, and the full table is in the "Model facts" tab.
- **Messy data:** with 30% missing glucose the twin catches 30% of spikes and says "cannot judge" 60% of the time; at 50% missing it almost stops alerting. See `docs/SCENARIOS.md` and `docs/SCENARIO_RESULTS.md` for 23 scenarios and 20 patient groups.
- **Weak spots (reported on purpose):** prediabetes patients (50% caught), sensors reading 20 mg/dL too high, thin patients (BMI under 23) get lower precision.

## Run it
```bash
pip install -r requirements.txt                       # Python 3.11+
python generate_data.py --out data                    # about 1,000 patients (large files, not committed)
python train.py --data data                           # trains and saves models/ and demo_data/
python make_manifest.py                               # records file hashes in SHA256SUMS
python evaluate.py --data data                        # re-creates docs/SCENARIO_RESULTS.md (run until it finishes)
streamlit run app.py                                  # dashboard at http://localhost:8501
pytest -q                                             # 43+ tests (takes a few minutes)
```
The repository already contains a trained model, a 40-patient demo set and `SHA256SUMS`, so you can run `streamlit run app.py` straight away.

## Honest limits
- **All data is simulated by rules we wrote.** Scores show the pipeline works. They do **not** show clinical accuracy. A model trained on one simulator fell to near chance on a different one.
- **The health record helps in this data partly because the simulator makes it help.**
- **No setting gives zero false alarms and catches every spike.** We report both numbers.
- Not simulated: kidney disease, insulin use, steroids, pregnancy, type 1 diabetes, festival fasting. The twin makes no claim about them.
- Real-data testing (an open CGM dataset, for example via `real_check.py`) is still to do.
- Not a medical device. Not for diagnosis or treatment.

## Security and privacy
No real patient data, hash-checked model and data files, cleaned inputs, local-only dashboard with an optional password, pinned libraries, CI scans. Details and limits: `SECURITY.md`.

## Files
`app.py` dashboard | `twin.py` virtual patient | `alerts.py` alert rule | `features.py` features | `train.py` training | `evaluate.py` scenario report | `scenarios.py` data problems | `security.py` protections | `generate_data.py`, `validate.py` data | `tests/` | `docs/`
