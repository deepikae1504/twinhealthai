"""Convert the ShanghaiT2DM open dataset (figshare) into the two simple files real_eval.py reads.
Usage: python convert_shanghai.py --src <folder with Shanghai_T2DM/ and Shanghai_T2DM_Summary.xlsx> --out <folder>
Writes cgm_all.csv, cgm_no_insulin.csv and ehr.csv. Tested on the real files (109 recordings).
Needs: pip install openpyxl xlrd"""
import argparse
import os
import re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default=".")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

S = pd.read_excel(f"{a.src}/Shanghai_T2DM_Summary.xlsx")
num = lambda c: pd.to_numeric(S[c], errors="coerce")        # text cells become unknown (NaN)
S["pid"] = S["Patient Number"].astype(str).str.strip()
agents = S["Hypoglycemic Agents"].fillna("none").astype(str).str.lower()


def medication(t):
    parts = [p.strip() for p in re.split(r"[,;]", t) if p.strip() and p.strip() != "none"]
    return "lifestyle_only" if not parts else "metformin" if parts == ["metformin"] else "other_or_combination"


ehr = pd.DataFrame({
    "patient_id": S.pid, "age_years": num("Age (years)"), "sex": S["Gender (Female=1, Male=2)"].map({1: "F", 2: "M"}),
    "bmi_kg_m2": num("BMI (kg/m2)"), "hba1c_percent": num("HbA1c (mmol/mol)") * 0.09148 + 2.152,        # mmol/mol -> %
    "fasting_glucose_mg_dl": num("Fasting Plasma Glucose (mg/dl)"),
    "total_cholesterol_mg_dl": num("Total Cholesterol (mmol/L)") * 38.67,                            # mmol/L -> mg/dL
    "diabetes_status": "type2_diabetes", "medication_status": agents.map(medication),
    "hypertension_diagnosed": S["Comorbidities"].fillna("").astype(str).str.lower().str.contains("hypertension").astype(int),
})
ehr.to_csv(f"{a.out}/ehr.csv", index=False)
on_insulin = set(S.pid[agents.str.contains("insulin|novolin|humulin|gansulin|lantus|aspart|glargine|detemir|lispro")])

rows = []
for f in sorted(os.listdir(f"{a.src}/Shanghai_T2DM")):
    if not f.lower().endswith((".xls", ".xlsx")):
        continue
    d = pd.read_excel(f"{a.src}/Shanghai_T2DM/{f}")
    pid = os.path.splitext(f)[0]
    col = [c for c in d.columns if "cgm" in str(c).lower()][0]                  # two files call it "CGM " (still mg/dL)
    g = pd.to_numeric(d[col], errors="coerce")
    if not 40 <= g.median() <= 400:
        raise SystemExit(f"{f}: median glucose {g.median():.1f} does not look like mg/dL, check the unit")
    rows.append(pd.DataFrame({"patient_id": pid, "timestamp": pd.to_datetime(d["Date"], errors="coerce"), "glucose": g}).dropna())
cgm = pd.concat(rows)
cgm.to_csv(f"{a.out}/cgm_all.csv", index=False)
cgm[~cgm.patient_id.isin(on_insulin)].to_csv(f"{a.out}/cgm_no_insulin.csv", index=False)
print(f"{cgm.patient_id.nunique()} recordings, {len(cgm):,} glucose readings; {len(on_insulin)} recordings on insulin; "
      f"{cgm[~cgm.patient_id.isin(on_insulin)].patient_id.nunique()} not on insulin")
missing = set(cgm.patient_id) - set(ehr.patient_id)
print("recordings without a summary row:", len(missing))
