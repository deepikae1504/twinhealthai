"""
Applies the demo and login fixes to YOUR redesigned app.py without touching your design.

    python patch_app.py            (patches app.py in this folder; a backup app_backup.py is saved first)
    python patch_app.py other.py

What it changes:
  1. Sensor problems now hit the part of the replay you are about to watch (stuck / offline sensors start 30 min in).
  2. The warning reads "Cannot judge reliably right now" and is shown on every page when the data is unreliable.
  3. The Data Quality page names the problem you chose (it only looked at the old checkbox).
  4. Login: constant-time password check, 5 wrong tries lock the session, and a clear message if no accounts exist.
Each change is skipped, with a message, if its text is not found in your file.
"""
import ast
import pathlib
import shutil
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
src = path.read_text(encoding="utf-8")
backup = path.with_name(path.stem + "_backup.py")
if not backup.exists():                                   # never overwrite your first, original backup
    shutil.copy(path, backup)

PATCHES = [
    ("import hmac", "import base64\n", "import base64\nimport hmac\n"),
    ("problem hits the replay window",
     "    P = apply_sensor(P, scn, seed=3, rows_per_patient=len(P))\n",
     '''    a0, a1 = START_IDX, START_IDX + 300                      # the problem hits the next ~25 hours of replay, so you SEE it
    W = apply_sensor(P.iloc[a0:a1].reset_index(drop=True), scn, seed=3, rows_per_patient=a1 - a0)
    if scn.startswith("cgm_frozen"):                           # stuck sensor starts 30 minutes in
        W.loc[6:36, "glucose_mg_dl"] = W.loc[6, "glucose_mg_dl"]
    elif scn.startswith("cgm_offline"):                        # sensor goes offline 30 minutes in
        W.loc[6:6 + (36 if "3h" in scn else 12), "glucose_mg_dl"] = float("nan")
    P = pd.concat([P.iloc[:a0], W, P.iloc[a1:]], ignore_index=True)    # history before the start stays clean
'''),
    ("warning wording on the dashboard",
     '            st.warning("Prediction uncertainty: " + "; ".join(S["uncertain_why"]))\n',
     '''            st.warning("Cannot judge reliably right now: " + "; ".join(S["uncertain_why"]))
        else:
            st.caption("Data quality: good, the twin can judge.")
'''),
    ("banner on every page",
     'st.info("SIMULATED data. Research prototype only; not validated for real patients or clinical decisions.")\n',
     '''st.info("SIMULATED data. Research prototype only; not validated for real patients or clinical decisions.")
if scn != "clean" or ehr_mode != "ehr_full":
    st.caption(f"Simulating: sensor problem = {scn}; health-record problem = {ehr_mode}")
if S["uncertain"] and ss.page != "dashboard":
    st.warning("Cannot judge reliably right now: " + "; ".join(S["uncertain_why"]))
'''),
    ("data quality page names the problem",
     '        st.warning("Bad sensor simulation enabled" if messy else "Standard sensor simulation selected")\n',
     '''        _sim = ([scn] if scn != "clean" else []) + (["late / absurd readings"] if messy else [])
        if _sim:
            st.warning("Simulated sensor problem: " + ", ".join(_sim))
        else:
            st.success("Clean sensor stream (no problem simulated)")
'''),
    ("login check",
     "        if expected is not None and doctor_password and str(expected) == doctor_password:\n",
     '''        if ss.get("login_fails", 0) >= 5:
            st.error("Too many failed attempts. Close this tab and try again later.")
        elif not accounts:
            st.error("No doctor accounts are set up. Create .streamlit/secrets.toml (see secrets.toml.example).")
        elif expected is not None and doctor_password and hmac.compare_digest(str(expected).encode(), doctor_password.encode()):
'''),
    ("count failed logins",
     '        else:\n            st.error("Access denied. Check your Doctor ID and password.")\n',
     '        else:\n            ss.login_fails = ss.get("login_fails", 0) + 1\n            st.error("Access denied. Check your Doctor ID and password.")\n'),
]

for name, old, new in PATCHES:
    if new in src:
        print("already :", name)
    elif src.count(old) == 1:
        src = src.replace(old, new)
        print("applied :", name)
    elif src.count(old) == 0:
        print("SKIPPED :", name, "(text not found, maybe already patched or edited)")
    else:
        print("SKIPPED :", name, "(text found more than once)")

try:
    ast.parse(src)
except SyntaxError as e:
    sys.exit(f"The patched file would have a syntax error ({e}). Nothing was changed. Your backup is app_backup.py.")
path.write_text(src, encoding="utf-8")
print(f"done. Backup saved as {path.stem}_backup.py")
