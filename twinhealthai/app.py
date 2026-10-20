import base64
"""TwinHealth AI - doctor dashboard (simulated data, research prototype). Run: streamlit run app.py"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, streamlit as st
from twin import PatientTwin
from replay import replay
import security
from scenarios import EHR_SCENARIOS, SENSOR_SCENARIOS, apply_sensor, blank_ehr_row as blank_ehr

st.set_page_config(
    page_title="TwinHealth AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Visual system ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
.stApp {
    background:
        radial-gradient(circle at 82% 8%, rgba(0,229,255,.10), transparent 26%),
        radial-gradient(circle at 10% 85%, rgba(83,61,255,.12), transparent 28%),
        #070B24;
    color: #EAFBFF;
}
[data-testid="stHeader"] { background: rgba(7,11,36,.82); }
.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem; }
h1,h2,h3,h4 { font-family: "Manrope", sans-serif !important; color: #F4FDFF !important; }
p, label, .stCaption, [data-testid="stMarkdownContainer"] { color: #AFC2D6; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1033 0%, #080C27 100%);
    border-right: 1px solid rgba(0,229,255,.14);
}
[data-testid="stSidebar"] * { color: #EAFBFF; }
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: #101A46; color: #EAFBFF; border: 1px solid #20356B;
    border-radius: 10px; font-weight: 700;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    border-color: #00E5FF; box-shadow: 0 0 18px rgba(0,229,255,.15);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(15,24,60,.94), rgba(9,16,43,.94));
    border: 1px solid rgba(0,229,255,.16);
    border-radius: 16px;
    padding: 14px;
}
[data-testid="stMetricLabel"] { color: #8FA9C2 !important; }
[data-testid="stMetricValue"] { color: #F3FDFF !important; }


.login-logo-wrap {
    display:flex; justify-content:center; align-items:center;
    padding: 1rem 0 1.4rem; perspective: 900px;
}
.login-logo {
    width: min(290px, 70vw); height:auto; object-fit:contain;
    animation: loginLogoFloat 4.8s ease-in-out infinite;
    filter: drop-shadow(0 0 22px rgba(0,229,255,.24));
    transform-origin:center;
}
/* Landing-page logo only. Doctor-login animation is intentionally unchanged. */
.landing-logo-wrap {
    display:flex; justify-content:center; align-items:center;
    width:100%; padding:1.5rem 0 1.2rem;
}
.landing-logo {
    display:block; width:260px; max-width:70vw; height:auto; object-fit:contain;
    animation: landingLogoFloat 6s ease-in-out infinite;
    filter: drop-shadow(0 8px 18px rgba(0,229,255,.16));
    transform-origin:center;
}
@keyframes landingLogoFloat {
    0%,100% { transform:translateY(0); }
    50% { transform:translateY(-7px); }
}

@keyframes loginLogoFloat {
    0%,100% { transform: translateY(0) rotateY(-3deg); }
    50% { transform: translateY(-13px) rotateY(3deg); }
}

.hero-shell {
    position: relative; overflow: hidden; min-height: 560px; border-radius: 32px;
    border: 1px solid rgba(0,229,255,.22);
    background:
      radial-gradient(circle at 73% 43%, rgba(0,229,255,.14), transparent 23%),
      radial-gradient(circle at 84% 72%, rgba(83,61,255,.20), transparent 30%),
      linear-gradient(135deg, #0B1035 0%, #090D2A 48%, #10174A 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 30px 80px rgba(0,0,0,.32);
}
.hero-grid {
    position:absolute; inset:0; opacity:.25;
    background-image: linear-gradient(rgba(0,229,255,.09) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(0,229,255,.09) 1px, transparent 1px);
    background-size: 38px 38px;
    mask-image: linear-gradient(to right, transparent 0%, black 40%, black 100%);
}
.hero-glow {
    position:absolute; width:390px; height:390px; right:8%; top:12%;
    border-radius:50%;
    background: radial-gradient(circle, rgba(0,229,255,.30) 0%, rgba(0,229,255,.08) 38%, transparent 70%);
    filter: blur(5px);
}
.patient {
    position:absolute; right:15%; top:13%; width:250px; height:400px;
    animation: patientFloat 5s ease-in-out infinite;
    filter: drop-shadow(0 0 25px rgba(0,229,255,.22));
}
.patient .head {
    position:absolute; left:74px; top:8px; width:102px; height:112px;
    border-radius:50% 50% 46% 46%;
    background:linear-gradient(145deg,#C9D7E8,#8FA6C3 60%,#5E7595);
    box-shadow: inset -12px -8px 18px rgba(19,28,60,.35), 0 0 18px rgba(0,229,255,.10);
}
.patient .neck {
    position:absolute; left:105px; top:103px; width:40px; height:48px;
    background:linear-gradient(90deg,#8298B2,#B8C9DC); border-radius:12px;
}
.patient .body {
    position:absolute; left:34px; top:130px; width:182px; height:188px;
    border-radius:78px 78px 42px 42px;
    background:linear-gradient(145deg,#16285D,#123E68 52%,#087D9A);
    border:1px solid rgba(0,229,255,.45);
    box-shadow: inset 12px 0 25px rgba(0,229,255,.08), 0 0 45px rgba(0,229,255,.10);
}
.patient .core {
    position:absolute; left:85px; top:175px; width:80px; height:80px;
    border-radius:50%; border:1px solid rgba(0,229,255,.65);
    box-shadow: 0 0 22px rgba(0,229,255,.45), inset 0 0 25px rgba(0,229,255,.15);
}
.patient .core:after {
    content:""; position:absolute; left:24px; top:24px; width:30px; height:30px;
    border-radius:50%; background:#00E5FF; box-shadow:0 0 30px #00E5FF;
}
.patient .line {
    position:absolute; left:122px; top:150px; width:5px; height:138px;
    border-radius:10px; background:#00E5FF; box-shadow:0 0 18px #00E5FF;
}
.patient .arm {
    position:absolute; top:155px; width:45px; height:150px; border-radius:30px;
    background:linear-gradient(90deg,#152A5B,#1A7190);
    border:1px solid rgba(0,229,255,.24);
}
.patient .arm.left { left:8px; transform:rotate(10deg); }
.patient .arm.right { right:3px; transform:rotate(-10deg); }
.patient .leg {
    position:absolute; top:292px; width:62px; height:104px; border-radius:25px 25px 18px 18px;
    background:linear-gradient(90deg,#10214D,#1C4774);
}
.patient .leg.left { left:67px; }
.patient .leg.right { left:127px; }

.data-orbit {
    position:absolute; right:5%; top:20%; width:480px; height:290px;
    border:1px solid rgba(0,229,255,.35); border-radius:50%; transform:rotate(-18deg);
}
.data-orbit.two {
    right:2%; top:27%; width:500px; height:230px;
    border-color:rgba(110,92,255,.42); transform:rotate(28deg);
}
.float-card {
    position:absolute; z-index:3; min-width:150px; padding:13px 16px;
    border-radius:15px; background:rgba(10,17,48,.84);
    border:1px solid rgba(0,229,255,.25);
    box-shadow:0 15px 35px rgba(0,0,0,.28), 0 0 18px rgba(0,229,255,.07);
    backdrop-filter:blur(12px);
}
.float-label { font-size:.66rem; letter-spacing:1px; color:#7191AD; font-weight:800; }
.float-value { font-size:1rem; color:#F4FDFF; font-weight:800; margin-top:4px; }
.card-a { left:5%; top:15%; } .card-b { right:4%; bottom:15%; } .card-c { left:10%; bottom:10%; }

.hero-eyebrow {
    display:inline-block; padding:7px 12px; border-radius:999px;
    background:rgba(0,229,255,.08); border:1px solid rgba(0,229,255,.24);
    color:#00E5FF; font-size:.72rem; font-weight:800; letter-spacing:1.5px;
}
.hero-title {
    font-family:Manrope,sans-serif; font-size:clamp(2.7rem,5vw,5.5rem);
    line-height:1.0; letter-spacing:-3px; font-weight:800; color:#F4FDFF;
    margin:18px 0 14px;
}
.hero-title span { color:#00E5FF; text-shadow:0 0 25px rgba(0,229,255,.28); }
.hero-copy { font-size:1.05rem; line-height:1.75; color:#AFC2D6; max-width:650px; }
.launch-wrap .stButton button {
    background:linear-gradient(135deg,#00E5FF,#0098FF) !important;
    color:#041225 !important; border:0 !important; border-radius:13px !important;
    font-weight:900 !important; padding:0.8rem 1.4rem !important;
    box-shadow:0 0 25px rgba(0,229,255,.28);
}
.launch-wrap .stButton button:hover { box-shadow:0 0 40px rgba(0,229,255,.45); transform:translateY(-1px); }

.section-card {
    background:linear-gradient(145deg,rgba(14,22,55,.96),rgba(9,15,39,.96));
    border:1px solid rgba(0,229,255,.13); border-radius:18px;
    padding:20px; margin-bottom:14px;
    box-shadow:0 14px 35px rgba(0,0,0,.18);
}
.eyebrow { font-size:.7rem; letter-spacing:1.5px; text-transform:uppercase; color:#00E5FF; font-weight:800; }
.profile-row { display:flex; justify-content:space-between; gap:12px; padding:11px 0; border-bottom:1px solid rgba(160,190,215,.10); font-size:.88rem; }
.profile-row:last-child { border-bottom:0; }
.status-pill { display:inline-block; border-radius:999px; padding:7px 12px; font-size:.74rem; font-weight:800; }
.status-high { background:rgba(255,75,120,.13); color:#FF6B92; border:1px solid rgba(255,75,120,.22); }
.status-low { background:rgba(0,229,255,.08); color:#5BEFFF; border:1px solid rgba(0,229,255,.18); }
.footer { border-top:1px solid rgba(160,190,215,.12); margin-top:28px; padding-top:15px; color:#70869E; font-size:.77rem; }

@keyframes patientFloat { 0%,100% { transform:translateY(0) rotate(-1deg); } 50% { transform:translateY(-10px) rotate(1deg); } }
[data-testid="stMetric"] { min-height: 120px; }
.section-card { height: 100%; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 7px 10px; border-radius: 10px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(0,229,255,.08); }
@media (max-width: 900px) {
  .hero-shell { min-height: 680px; }
  .patient { transform: scale(.78); transform-origin: top right; right: 3%; top: 45%; }
  .hero-title { letter-spacing: -1.5px; }
  .float-card { transform: scale(.85); }
}
</style>
""", unsafe_allow_html=True)

START_IDX = 96                                              # begin the day at 08:00 with 8h of history


@st.cache_resource
def load():
    b = security.load_model("models/bundle.joblib")                 # refuses a file whose hash changed
    e = security.read_csv_safe("demo_data/ehr_demo.csv")             # refuses data not tagged synthetic
    s = security.read_csv_safe("demo_data/sensor_demo.csv", parse_dates=["timestamp"])
    return b, e, s


def fmt(v, f="{}"):
    return "unknown" if pd.isna(v) else f.format(v)



ss = st.session_state
ss.setdefault("page", "landing")

# ------------------------------------------------------------------ landing page
if ss.page == "landing":
    st.markdown('<div class="hero-shell" style="padding:2rem;min-height:unset;"><div class="hero-grid"></div><div class="hero-title">Your health.<br><span>Our digital twin.</span></div><div class="hero-copy">Explore simulated health records, glucose forecasts, and early-warning insights.</div></div>', unsafe_allow_html=True)
    # Center the logo at its existing 260px width and animate it gently.
    with open("twinhealth_logo.png", "rb") as logo_file:
        landing_logo_b64 = base64.b64encode(logo_file.read()).decode("ascii")
    st.markdown(
        '<div class="landing-logo-wrap"><img class="landing-logo" '
        'src="data:image/png;base64,' + landing_logo_b64 +
        '" alt="TwinHealth AI logo"></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-card"><h3>Organized patient context</h3><p>Review health-record details, glucose forecasts, model signals, and data quality.</p></div>', unsafe_allow_html=True)
    if st.button("Enter Doctor Portal  →", type="primary"):
        ss.page = "login"
        st.rerun()
    st.caption("Synthetic research prototype only · Not for clinical use")
    st.stop()

# ------------------------------------------------------------------ doctor access
if ss.page == "login":
    logo_path = "twinhealth_logo.png"
    with open(logo_path, "rb") as logo_file:
        logo_b64 = base64.b64encode(logo_file.read()).decode("ascii")
    st.markdown(
        '<div class="login-logo-wrap"><img class="login-logo" src="data:image/png;base64,' +
        logo_b64 + '" alt="TwinHealth AI logo"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-card"><div class="eyebrow">AUTHORIZED DOCTOR ACCESS</div>'
        '<h1>Doctor sign in</h1><p>Sign in with your assigned Doctor ID and password '
        'to open the clinical digital twin workspace.</p></div>',
        unsafe_allow_html=True,
    )
    with st.form("doctor_login_form"):
        doctor_id = st.text_input("Doctor ID", placeholder="Enter your assigned ID")
        doctor_password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        try:
            accounts = dict(st.secrets.get("doctor_accounts", {}))
        except Exception:
            accounts = {}
        expected = accounts.get(doctor_id.strip())
        if expected is not None and doctor_password and str(expected) == doctor_password:
            ss["authenticated_doctor"] = doctor_id.strip()
            ss.page = "dashboard"
            st.rerun()
        else:
            st.error("Access denied. Check your Doctor ID and password.")
    if st.button("← Back to landing page"):
        ss.page = "landing"
        st.rerun()
    st.stop()

if ss.page not in ("landing", "login") and not security.password_ok(st):
    st.stop()

B, EHR, SENS = load()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("TwinHealth AI")
st.sidebar.caption("DOCTOR PORTAL · SYNTHETIC DEMO")
page_options = ["Dashboard", "Why This Risk", "What-if Lab", "Data Quality", "Model Facts"]
page_map = {"Dashboard": "dashboard", "Why This Risk": "why", "What-if Lab": "whatif",
            "Data Quality": "data_quality", "Model Facts": "model_facts"}
current_label = next((label for label, key in page_map.items() if key == ss.page), "Dashboard")
ss.page = page_map[st.sidebar.radio("Workspace", page_options, index=page_options.index(current_label))]
st.sidebar.divider()
pid = st.sidebar.selectbox("Virtual patient", EHR.patient_id.tolist())
messy = st.sidebar.checkbox("Simulate bad sensor data (gaps, late, absurd readings)", value=False)
scn = st.sidebar.selectbox("Sensor problem to simulate", list(SENSOR_SCENARIOS))
ehr_mode = st.sidebar.selectbox("Health-record problem to simulate", list(EHR_SCENARIOS))
per_tick = st.sidebar.slider("Replay speed (readings per tick, 1 reading = 5 min)", 1, 12, 3)
style = st.sidebar.select_slider("Alert style", ["Low false alarms", "Balanced", "Catch more spikes"], value="Low false alarms",
                                 help="Fewer false alarms means more missed spikes. See the trade-off table in 'Model facts'.")
LEVELS = {"Low false alarms": B["alert_level"], "Balanced": 0.35, "Catch more spikes": 0.25}
restart = st.sidebar.button("Restart")
if ss.get("pid") != pid or ss.get("messy") != messy or ss.get("scn") != scn or ss.get("ehr_mode") != ehr_mode or restart:
    P = SENS[SENS.patient_id == pid].reset_index(drop=True)
    P = apply_sensor(P, scn, seed=3, rows_per_patient=len(P))
    ss.stream = list(replay(P, messy=messy, seed=1))
    ss.twin = PatientTwin(blank_ehr(EHR[EHR.patient_id == pid].iloc[0].to_dict(), ehr_mode), B)
    with st.spinner("Loading the patient's history..."):
        for r in ss.stream[:START_IDX]:
            ss.twin.update(r)
    ss.i, ss.running, ss.pid, ss.messy, ss.scn, ss.ehr_mode = START_IDX, False, pid, messy, scn, ehr_mode
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Play"): ss.running = True
if c2.button("Pause"): ss.running = False
if c3.button("+1 h"):
    ss.running = False
    for r in ss.stream[ss.i: ss.i + 12]: ss.twin.update(r)
    ss.i += 12
if st.sidebar.button("← Back to landing page", use_container_width=True):
    ss.page = "landing"
    st.rerun()
if st.sidebar.button("Sign out", use_container_width=True):
    ss.pop("authenticated_doctor", None)
    ss.page = "landing"
    st.rerun()
st.sidebar.caption("Simulated patient data. Research prototype - not for diagnosis, treatment, or any real clinical decision.")

# ------------------------------------------------------------------ workspace header
st.markdown(
    '<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
    '<div style="width:38px;height:38px;border-radius:12px;background:rgba(0,229,255,.10);'
    'border:1px solid rgba(0,229,255,.25);display:flex;align-items:center;justify-content:center;'
    'color:#00E5FF;font-weight:900;">✦</div>'
    '<div><div style="font-family:Manrope;font-weight:800;font-size:1.2rem;color:#F4FDFF;">TwinHealth AI</div>'
    '<div style="font-size:.72rem;letter-spacing:1.2px;color:#6F8BA5;">CLINICAL DIGITAL TWIN</div></div>'
    '<div style="margin-left:auto;"><span class="status-pill status-low">● SYNTHETIC DEMO</span></div></div>',
    unsafe_allow_html=True,
)
tw: PatientTwin = ss.twin
tw.alert_level = LEVELS[style]
S = tw.state
e = tw.ehr
page_titles = {
    "dashboard": ("Doctor Dashboard", "Patient status, glucose trajectory and early-warning signals."),
    "why": ("Why This Risk", "Explore which input groups influence the current model estimate."),
    "whatif": ("What-if Lab", "Compare model output under a hypothetical input change."),
    "data_quality": ("Data Quality & Timeline", "Review simulated sensor and health-record quality."),
    "model_facts": ("Model Facts", "Validation metrics, alert trade-offs and forecast errors."),
}
page_title, page_subtitle = page_titles.get(ss.page, page_titles["dashboard"])
st.markdown(
    f'<div class="eyebrow">HEALTHCARE INTELLIGENCE · VIRTUAL PATIENT</div>'
    f'<h1 style="margin:6px 0 0 0;">{page_title}</h1>'
    f'<p style="color:#8FA9C2;margin-top:3px;">{page_subtitle}</p>',
    unsafe_allow_html=True,
)
st.info("SIMULATED data. Research prototype only; not validated for real patients or clinical decisions.")

# ------------------------------------------------------------------ alert popup
def show_glucose_alert(kind, message):
    # Streamlit dialog is visual-only; it does not alter the model/replay state.
    if hasattr(st, "dialog"):
        @st.dialog("Glucose early-warning")
        def _dialog():
            if kind == "observed":
                st.error("🔴 OBSERVED HIGH READING")
            else:
                st.warning("⚡ PREDICTED GLUCOSE SPIKE")
            st.markdown(message)
            st.caption("Synthetic research prototype. This alert is not a clinical decision or diagnosis.")
            if st.button("Acknowledge", type="primary", use_container_width=True):
                ss.alert_open = False
                st.rerun()
        _dialog()
    else:
        st.warning(message)

# Trigger only when the alert state changes, avoiding repeated pop-ups on reruns.
alert_kind = None
alert_message = None
if S.get("already_high"):
    alert_kind = "observed"
    alert_message = f"The latest simulated glucose reading is {S['glucose']:.0f} mg/dL, above the configured 180 mg/dL threshold."
elif tw.timeline and tw.timeline[-1][1] == "alert":
    alert_kind = "predicted"
    alert_message = tw.timeline[-1][2]

if alert_kind:
    alert_key = f"{pid}|{ss.i}|{alert_kind}|{alert_message}"
    if ss.get("last_alert_key") != alert_key and not ss.get("alert_open", False):
        ss.last_alert_key = alert_key
        ss.alert_open = True

if ss.get("alert_open") and alert_kind:
    show_glucose_alert(alert_kind, alert_message)

# ------------------------------------------------------------------ dashboard overview
if ss.page == "dashboard":
    patient_col, risk_col, peak_col = st.columns([1.15, 1, 1])
    with patient_col:
        st.markdown('<div class="section-card"><div class="eyebrow">PATIENT NOTEPAD · SIMULATED RECORD</div>', unsafe_allow_html=True)
        st.subheader(f"Patient {pid}")
        st.markdown(f'''
        <div style="background:#fffdf7;color:#202b3b;border:1px solid #e9e2d4;border-radius:12px;padding:18px 20px;">
          <div style="font-weight:700;letter-spacing:1px;color:#687386;margin-bottom:12px;">PATIENT SUMMARY</div>
          <div style="display:grid;grid-template-columns:120px 1fr;gap:10px 14px;">
            <div>Age / sex</div><b>{fmt(e['age_years'], '{:.0f}')} years · {fmt(e['sex'])}</b>
            <div>Condition</div><b>{fmt(e['diabetes_status']).replace('_', ' ')}</b>
            <div>HbA1c</div><b>{fmt(e['hba1c_percent'], '{}%')}</b>
            <div>BMI</div><b>{fmt(e['bmi_kg_m2'])}</b>
            <div>Medication</div><b>{fmt(e['medication_status']).replace('_', ' ')}</b>
          </div><hr>
          <div style="font-weight:700;letter-spacing:1px;color:#687386;margin:12px 0 6px;">ADDITIONAL CONTEXT</div>
          <div>Family history: <b>{fmt(e['family_history_diabetes'], '{:.0f}').replace('1', 'Yes').replace('0', 'No')}</b><br>
          Hypertension: <b>{fmt(e['hypertension_diagnosed'], '{:.0f}').replace('1', 'Yes').replace('0', 'No')}</b><br>
          Dyslipidemia: <b>{fmt(e['dyslipidemia_diagnosed'], '{:.0f}').replace('1', 'Yes').replace('0', 'No')}</b><br>
          Simulated genetic risk: <b>{fmt(e['genetic_risk_score_simulated'], '{:.0f}')}/100</b></div>
        </div>''', unsafe_allow_html=True)
        if S["ehr_unknown"]:
            st.warning(f"{S['ehr_unknown']} health-record fields are unknown.")
        st.markdown('</div>', unsafe_allow_html=True)

    with risk_col:
        st.markdown('<div class="section-card"><div class="eyebrow">SPIKE RISK · NEXT 2 HOURS</div>', unsafe_allow_html=True)
        if S["already_high"]:
            st.metric("Glucose now", f"{S['glucose']:.0f} mg/dL")
            st.error("Observed reading is above 180 mg/dL.")
        else:
            risk = S["raw_risk"]
            st.metric("Chance of exceeding 180", f"{risk:.0%}")
            st.progress(min(max(risk, 0.0), 1.0))
            threshold = tw.alert_level
            level = "HIGH" if risk >= threshold else "MODERATE" if risk >= threshold / 2 else "LOW"
            st.markdown(f"**Risk level: {level}**")
            st.caption(f"Alert threshold: {threshold:.0%}; sustained for 15 minutes with trustworthy data.")
        st.caption(f"Current glucose: {S['glucose']:.0f} mg/dL" if not pd.isna(S["glucose"]) else "No glucose reading available.")
        if S["uncertain"]:
            st.warning("Prediction uncertainty: " + "; ".join(S["uncertain_why"]))
        st.markdown('</div>', unsafe_allow_html=True)

    with peak_col:
        st.markdown('<div class="section-card"><div class="eyebrow">FORECAST PEAK</div>', unsafe_allow_html=True)
        curve = S["curve"]
        if curve:
            peak_minute, peak_value = max(curve.items(), key=lambda item: item[1])
            delta = f"{peak_value - S['glucose']:+.0f} mg/dL" if not pd.isna(S["glucose"]) else None
            st.metric("Predicted peak", f"{peak_value:.0f} mg/dL", delta=delta)
            st.caption(f"Forecast horizon: +{peak_minute} minutes")
            st.warning("Forecast may cross 180 mg/dL." if peak_value >= 180 else "Forecast remains below 180 mg/dL.")
        else:
            st.write("Forecast unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ main chart
def draw(ax, buf, curve, color="C1", label="Forecast (model)"):
    g = buf["glucose_mg_dl"].iloc[-72:]
    ax.plot(g.index, g.values, color="#00E5FF", lw=2.2, label="Glucose (sensor)")
    t0 = buf.index[-1]
    xs = [t0 + pd.Timedelta(minutes=m) for m in curve]
    ax.plot(xs, list(curve.values()), color=color, ls="--", marker="o", ms=4, label=label)
    ax.axhline(180, color="#FF4B78", lw=1.4, ls="--", alpha=.85); ax.text(g.index[0], 183, "180 mg/dL", color="#FF6B92", fontsize=8)
    for t, v in buf["meal_carbs_logged_g"].iloc[-72:].items():
        if v > 0:
            ax.axvline(t, color="#7185A5", alpha=.35); ax.text(t, ax.get_ylim()[0] + 2, f" meal {int(v)}g", fontsize=7, color="#8FA9C2", rotation=90)
    ax.set_ylabel("mg/dL"); ax.legend(loc="upper left", fontsize=8); ax.margins(x=0.01)


if ss.page == "dashboard":
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#0A1033")
    draw(ax, tw.buf, S["curve"])
    ax.set_facecolor("#080D2B")
    ax.tick_params(colors="#8FA9C2")
    ax.xaxis.label.set_color("#AFC2D6"); ax.yaxis.label.set_color("#AFC2D6")
    for spine in ax.spines.values(): spine.set_color("#20356B")
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Simulated clock: {S['time']:%a %d %b %H:%M}. Solid = sensor, dashed = 2-hour forecast. "
               f"Replay position {ss.i}/{len(ss.stream)}.")


# Dashboard navigation cards
if ss.page == "dashboard":
    st.markdown("### Quick analysis")
    nav_cols = st.columns(4)
    nav_items = [
        ("Why This Risk", "Understand which inputs influence risk.", "why"),
        ("What-if Lab", "Compare a hypothetical scenario.", "whatif"),
        ("Data Quality", "Inspect sensor and EHR notes.", "data_quality"),
        ("Model Facts", "Review validation and forecast errors.", "model_facts"),
    ]
    for col, (label, description, target) in zip(nav_cols, nav_items):
        with col:
            st.markdown(
                f'<div class="section-card"><div class="eyebrow">{label.upper()}</div>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label} →", key=f"nav_{target}", use_container_width=True):
                ss.page = target
                st.rerun()

# ------------------------------------------------------------------ dedicated analysis workspaces
if ss.page == "why":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What is influencing the prediction?")
    why = tw.explain()
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0A1033")
    labels, values = list(why)[::-1], list(why.values())[::-1]
    ax.barh(labels, values, color=["#00C9D8" if value <= 0 else "#7C6CFF" for value in values])
    ax.axvline(0, color="#AFC2D6", lw=.8)
    ax.set_xlabel("Change in model risk", color="#AFC2D6")
    ax.set_facecolor("#080D2B")
    ax.tick_params(colors="#AFC2D6")
    for spine in ax.spines.values():
        spine.set_color("#20356B")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("The model replaces one input group with typical values and measures how its risk estimate changes. This is an explanation of model behavior, not proof of causation.")
    st.markdown('</div>', unsafe_allow_html=True)

elif ss.page == "whatif":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Scenario comparison")
    scenario_options = {"millet": "Swap the last meal for a millet (low-GI) meal",
                        "portion": "Eat 30% less of the last meal"}
    scenario = st.selectbox("Choose a hypothetical scenario", list(scenario_options),
                            format_func=lambda value: scenario_options[value])
    result = tw.what_if(scenario)
    if result["note"]:
        st.info(result["note"])
    else:
        left, right = st.columns(2)
        for col, title, state, line_color in [
            (left, "CURRENT", S, "#00E5FF"),
            (right, "WHAT-IF", result["state"], "#8A7CFF")
        ]:
            with col:
                st.markdown(f"**{title}**")
                st.metric("Risk", "n/a (already high)" if state["already_high"] else f"{state['raw_risk']:.0%}")
                fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#0A1033")
                draw(ax, tw.buf, state["curve"], color=line_color, label=title.title())
                ax.set_facecolor("#080D2B")
                ax.tick_params(colors="#AFC2D6")
                for spine in ax.spines.values():
                    spine.set_color("#20356B")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        if not S["already_high"] and not result["state"]["already_high"]:
            delta = (result["state"]["raw_risk"] - S["raw_risk"]) * 100
            st.metric("Change in model risk", f"{delta:+.1f} percentage points")
    st.warning("This shows how the MODEL reacts to changed inputs. It is not proof that a change would help a real patient. Discuss diet or medication changes with a qualified clinician.")
    st.markdown('</div>', unsafe_allow_html=True)

elif ss.page == "data_quality":
    dq_col, timeline_col = st.columns([1, 1.4])
    with dq_col:
        st.markdown('<div class="section-card"><div class="eyebrow">SYSTEM DATA HEALTH</div>', unsafe_allow_html=True)
        st.subheader("Data quality")
        st.markdown("**Sensor stream**")
        st.warning("Bad sensor simulation enabled" if messy else "Standard sensor simulation selected")
        st.markdown("**Health record**")
        st.warning(f"Partial · {S['ehr_unknown']} unknown fields" if S["ehr_unknown"] else "Synthetic health-record fields populated")
        st.markdown("**Prediction reliability**")
        st.warning("; ".join(S["uncertain_why"]) if S["uncertain"] else "No active uncertainty flags")
        st.markdown('</div>', unsafe_allow_html=True)
    with timeline_col:
        st.markdown('<div class="section-card"><div class="eyebrow">RECENT EVENTS</div>', unsafe_allow_html=True)
        st.subheader("Timeline")
        if tw.timeline:
            for timestamp, kind, message in tw.timeline[-12:][::-1]:
                if kind == "alert":
                    st.error(f"⚡ {timestamp:%H:%M} · {message}")
                else:
                    st.info(f"{timestamp:%H:%M} · {message}")
        else:
            st.caption("No timeline events recorded yet.")
        st.subheader("Data-quality notes")
        if tw.notes:
            for timestamp, message in tw.notes[-10:][::-1]:
                st.write(f"**{timestamp:%H:%M}** · {message}")
        else:
            st.success("No data-quality notes so far. Use the sidebar simulation controls to explore data issues.")
        st.markdown('</div>', unsafe_allow_html=True)

elif ss.page == "model_facts":
    metrics = B["metrics"]
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Validation metrics")
    st.caption(f"PR-AUC baseline / chance level: {metrics['risk_base_rate']:.2f}")
    st.dataframe(pd.DataFrame({"PR-AUC": {key: round(value["pr_auc"], 2)
                                          for key, value in metrics["ablation"].items()}}),
                 use_container_width=True)
    st.subheader("Alert-level trade-off")
    st.caption("A lower threshold can catch more spikes while increasing false alerts.")
    st.dataframe(pd.DataFrame(B["alert_table"]).rename(columns={"level": "Alert level"}),
                 hide_index=True, use_container_width=True)
    st.subheader("Forecast error")
    st.dataframe(pd.DataFrame({
        "Forecast error (mg/dL)": {f"+{key} min": round(value, 1) for key, value in metrics["curve_mae"].items()},
        "'Stays the same' baseline": {f"+{key} min": round(value, 1) for key, value in metrics["curve_mae_persistence"].items()},
    }), use_container_width=True)
    st.caption("All data is simulated by hand-written rules. These scores demonstrate pipeline behavior, not clinical accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ advance replay
if ss.running and ss.i < len(ss.stream):
    for r in ss.stream[ss.i: ss.i + per_tick]:
        tw.update(r)
    ss.i += per_tick
    time.sleep(0.3)
    st.rerun()
elif ss.running:
    ss.running = False
    st.success("Replay finished.")

st.markdown(
    '<div class="footer"><b>TwinHealth AI</b> · Healthcare Digital Twin proof of concept · '
    'Synthetic data only · Not for diagnosis, treatment, or real-time clinical decisions</div>',
    unsafe_allow_html=True,
)
