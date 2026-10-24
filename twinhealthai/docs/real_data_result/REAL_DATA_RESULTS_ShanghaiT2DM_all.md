# Real-data test: ShanghaiT2DM (all)

**Read first.** The model was trained on synthetic data only and is not tuned on this data. Used 109 patients, 271,968 moments, 40,202 spike moments. Input readings every 15 minutes (filled to a 5-minute grid by straight-line interpolation between neighbouring readings, which is an approximation). Columns missing from this dataset are treated as UNKNOWN. Profile used: **cgm_only** (heart rate: no, meal log: no, health record: yes). Check that this dataset's population (country, age, diabetes type) matches what the model was trained for before drawing any conclusion.

## Can the twin predict spikes better than simple rules? (PR-AUC, higher is better)

| Method | PR-AUC |
|---|---|
| Twin model (profile: cgm_only) | 0.347 |
| Baseline: glucose stays the same | 0.334 |
| Baseline: straight-line trend | 0.409 |
| Chance (spike rate) | 0.148 |

## Alerts at the default level (0.5)

- Spikes caught: 56%
- Alerts that were right: 47%
- False alerts per patient-day: 1.14
- Median warning: 10.0 minutes

## Trade-off at other alert levels

| level | false alerts / patient-day | alerts right | spikes caught | median warning (min) |
|---|---|---|---|---|
| 0.2 | 5.2 | 0.28 | 0.87 | 60.0 |
| 0.3 | 3.27 | 0.34 | 0.8 | 35.0 |
| 0.4 | 2.07 | 0.39 | 0.7 | 20.0 |
| 0.5 | 1.14 | 0.47 | 0.56 | 10.0 |
| 0.6 | 0.55 | 0.55 | 0.41 | 10.0 |
| 0.7 | 0.19 | 0.67 | 0.24 | 5.0 |
