# Real-data test: ShanghaiT2DM (no_insulin)

**Read first.** The model was trained on synthetic data only and is not tuned on this data. Used 53 patients, 156,038 moments, 19,243 spike moments. Input readings every 15 minutes (filled to a 5-minute grid by straight-line interpolation between neighbouring readings, which is an approximation). Columns missing from this dataset are treated as UNKNOWN. Profile used: **cgm_only** (heart rate: no, meal log: no, health record: yes). Check that this dataset's population (country, age, diabetes type) matches what the model was trained for before drawing any conclusion.

## Can the twin predict spikes better than simple rules? (PR-AUC, higher is better)

| Method | PR-AUC |
|---|---|
| Twin model (profile: cgm_only) | 0.305 |
| Baseline: glucose stays the same | 0.289 |
| Baseline: straight-line trend | 0.364 |
| Chance (spike rate) | 0.123 |

## Alerts at the default level (0.5)

- Spikes caught: 54%
- Alerts that were right: 47%
- False alerts per patient-day: 0.91
- Median warning: 10.0 minutes

## Trade-off at other alert levels

| level | false alerts / patient-day | alerts right | spikes caught | median warning (min) |
|---|---|---|---|---|
| 0.2 | 4.43 | 0.27 | 0.86 | 40.0 |
| 0.3 | 2.61 | 0.34 | 0.79 | 20.0 |
| 0.4 | 1.64 | 0.4 | 0.67 | 15.0 |
| 0.5 | 0.91 | 0.47 | 0.54 | 10.0 |
| 0.6 | 0.43 | 0.57 | 0.4 | 10.0 |
| 0.7 | 0.16 | 0.66 | 0.23 | 5.0 |
