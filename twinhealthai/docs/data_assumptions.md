# Data assumptions (read this before trusting any number)

All data is **simulated** by `generate_data.py` (seed 42, 1,000 patients, 7 days, one reading every 5 minutes).
No real person's data is used or needed. Every row is tagged `data_type = synthetic`, and the code refuses to load
anything that is not.

**The simulator is hand-written and NOT clinically validated. Every coefficient is an assumption, not a measurement.**
The Indian touches (dishes, BMI cut-offs, family history, TCF7L2-style risk alleles) are design choices, not national statistics.

## Health record (EHR), one row per patient
| Field | How it is made |
|---|---|
| Diagnosis | 72% type 2 diabetes, 28% prediabetes (a design choice, not a prevalence estimate) |
| Age, BMI | Normal distributions (mean about 50 years, BMI 25 to 28), clipped to plausible ranges |
| Family history | About 35 to 45% (higher if type 2 diabetes) |
| Genetics | A simulated polygenic score (0 to 100) and a simulated risk-allele count (0 to 2, TCF7L2-style, allele chance 30%) |
| Labs, BP, medicine, BP/cholesterol history | Drawn from distributions in `make_ehr` in `generate_data.py` (read the code for exact values) |
| Medicine effect | Metformin lowers the simulated glucose response by 12%, combination drugs by 22% |

## Wearable stream, every 5 minutes
- **Glucose (CGM):** baseline + meal responses + slow random drift. Meal response size depends on carbs, dish GI factor, the patient's simulated insulin resistance (from the health record and genetics) and medicine. Sleep and other lifestyle effects are also simulated (see the generator for exact rules).
- **Heart rate, HRV, steps, sleep stages (awake/light/deep/REM):** simulated with day/night rhythm and links to sleep and glucose (see the generator for exact rules).
- **Meals:** Indian dishes (idli, dosa, poha, rice and dal, roti, biryani, millet meals, sweets and others) with carb ranges and GI classes. Meal times differ per patient and per day, weekends shift, meals are sometimes skipped (validation report: breakfast mean 08:08, lunch 13:22, dinner 20:19, each with a spread of 70 to 90 minutes).
- **Meal log:** a patient logs about 85% of meals, and logged carbs are off by up to 15%. The model only sees the log, never the true carbs.
- **Hidden parameters** (`hidden_simulation_params.csv`) are for debugging only and are never a model input.

## What the synthetic data cannot tell you
1. The model partly learns the simulator's own rules. Scores here show the pipeline works; they do **not** show clinical accuracy.
2. The health record matters in this data **because the simulator makes it matter**.
3. We tested two earlier, different simulators against each other: a model trained on one fell to near chance on the other. So results depend on simulator assumptions, and a real-data test is needed before any clinical claim.
4. Cohort is type 2 diabetes and prediabetes only. Other conditions (kidney disease, insulin use, steroids, pregnancy) are **not simulated**, so the twin makes no claims about them.
5. The sensors in the simulator have no missing data. Missing data, frozen sensors and bad readings are added afterwards by `scenarios.py`.
