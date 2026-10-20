# 🩺 TwinHealth AI
### Your Health. Our Digital Twin.

**Team Cosmic Bloom | 4 Members**

> An AI-powered Healthcare Digital Twin prototype for Type 2 Diabetes monitoring and early warning of potential glucose spikes.

---

## 🌸 Team Cosmic Bloom

**Team Name:** Cosmic Bloom  

---

## 📌 About the Project

**TwinHealth AI** is a healthcare technology proof of concept that demonstrates how a **Digital Twin** can combine simulated Electronic Health Record (EHR) information with timestamped sensor data to maintain a continuously updated virtual patient.

The system estimates the risk of a **high-glucose event within the next two hours** and presents the results through an interactive doctor-facing dashboard.

The project is designed as a technical and educational prototype for demonstrating:

- Digital Twin concepts
- Healthcare AI/ML
- Time-series sensor processing
- Glucose spike-risk prediction
- Early-warning alerts
- Data-quality monitoring
- Model explainability
- What-if analysis
- Secure demo access

> ⚠️ **Research Prototype Only**  
> TwinHealth AI uses synthetic patient data and simulated sensor streams. It is **not clinically validated** and must not be used for diagnosis, treatment, medication decisions, or real-patient monitoring.

---

# 🎯 Problem Statement

Type 2 diabetes requires continuous monitoring because glucose levels can change throughout the day based on factors such as meals, physical activity, sleep, medication, and other contextual information.

Traditional clinical measurements often provide only snapshots of a patient's condition.

TwinHealth AI explores the following challenge:

> **Can a simulated patient Digital Twin combine historical health information and changing sensor readings to identify a potential high-glucose event up to two hours in advance?**

The system brings historical patient information and changing sensor observations together into a single virtual patient representation.

---

# 💡 Proposed Solution

TwinHealth AI creates a simulated Digital Twin of a patient.

The system combines two primary information sources.

### 🧬 1. Historical Health Information

The simulated EHR can contain:

- Age
- Sex
- Diabetes status
- HbA1c
- BMI
- Medication status
- Family history
- Hypertension status
- Dyslipidemia status
- Simulated genetic risk score

### 📡 2. Dynamic Sensor & Contextual Data

The simulated sensor stream can include:

- Glucose
- Heart rate
- Heart-rate variability (HRV)
- Steps
- Activity
- Sleep-related measurements
- Meal-related contextual information

The Digital Twin continuously updates its virtual state as new readings are replayed.

The resulting state is used for:

**Patient Context → Digital Twin → Risk Prediction → Alert Assessment → Doctor Dashboard**

---

# 🏥 Healthcare Use Case

**Domain:** Digital Health & Healthcare Analytics

**Condition:** Type 2 Diabetes

**Primary Use Case:** Early warning of potential glucose spikes

**Intended User:** Doctor / Healthcare Professional

### Example Workflow

1. Doctor selects a virtual patient.
2. Synthetic EHR information is loaded.
3. Simulated sensor history is loaded.
4. The Digital Twin processes the patient timeline.
5. New sensor readings update the virtual patient state.
6. The model estimates the risk of glucose exceeding **180 mg/dL within two hours**.
7. Forecast information is displayed when available.
8. Alert conditions are evaluated.
9. The doctor can investigate:
   - Why This Risk
   - What-if Lab
   - Data Quality & Time
   - Model Facts

---

# ✨ Key Features

## 🧬 Virtual Patient

- Synthetic patient representation
- Historical EHR context
- Dynamic sensor readings
- Continuously updated Digital Twin state
- Multiple virtual patients

## 📈 Glucose Monitoring

- Current glucose monitoring
- Configurable glucose threshold
- Recent glucose history
- Predicted glucose trajectory when available
- Forecast horizon visualization

## ⚠️ Early-Warning Alerts

TwinHealth AI distinguishes between different alert situations.

### 🔴 Observed High Glucose

Triggered when the current observed glucose reading crosses the configured threshold.

### 🟠 Predicted Glucose Spike

Triggered when the model estimates an elevated risk of crossing the glucose threshold within the next two hours.

The default threshold used by the project is:

**180 mg/dL**

The dashboard provides a clear warning interface and acknowledgement action.

---

# 🔬 Why This Risk

The system provides a dedicated model-explanation interface.

It uses feature-group perturbation to examine how changes to selected input groups affect the model's estimated risk.

This allows users to explore model behavior rather than simply receiving a prediction.

> Model explanations describe model behavior. They should not be interpreted as causal medical explanations.

---

# 🧪 What-if Lab

The What-if Lab allows users to explore predefined hypothetical meal scenarios.

The system compares:

**Current Scenario**

vs.

**Hypothetical Scenario**

and displays differences in the model-generated risk when available.

These simulations are intended to demonstrate model behavior.

> What-if results are model simulations and do not establish that a real-world dietary, treatment, or medical intervention will produce the same outcome.

---

# 🕒 Sensor Replay

TwinHealth AI includes a replayable simulated sensor stream.

The dashboard supports:

- ▶️ Play
- ⏸️ Pause
- 🔄 Restart
- ⏩ Hour-based advancement
- Adjustable readings processed per update

This allows the Digital Twin to be observed as if new sensor readings were arriving over time.

---

# 🛡️ Data Quality & Uncertainty

The project demonstrates how prediction reliability can be affected by problematic data.

Supported simulated problems include:

- Missing readings
- Sensor gaps
- Late readings
- Implausible values
- Health-record issues
- Other simulated data-quality problems

The dashboard can display data-quality warnings and uncertainty information when raised by the system.

---

# 📊 Model Facts

The Model Facts section presents available model and evaluation information.

Depending on the trained model bundle, this can include:

- PR-AUC
- Precision
- Recall
- False-alert rate
- Forecast MAE
- Model comparisons
- Alert-threshold comparisons
- Forecast errors

Only measurements generated by the actual evaluation pipeline should be presented as final results.

---

# 🏗️ System Architecture


<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/9615e4bb-4d96-4296-846b-246ce9d01d7d" />


---

# 🔄 Digital Twin Workflow

```text
Patient Selection
       ↓
Synthetic EHR Loading
       ↓
Sensor History Loading
       ↓
Digital Twin Initialization
       ↓
Sensor Replay
       ↓
Twin State Update
       ↓
Risk Estimation
       ↓
Glucose Forecast
       ↓
Alert Evaluation
       ↓
Doctor Dashboard
       ↓
Explanation / What-if / Data Quality / Model Facts
```

### Step-by-Step

1. **Initialize**  
   Select a virtual patient and load their synthetic health record.

2. **Load History**  
   Process an initial portion of the simulated sensor stream.

3. **Update**  
   Add new readings to the Digital Twin state.

4. **Estimate Risk**  
   Calculate the model-generated future-event risk.

5. **Forecast**  
   Retrieve the glucose trajectory when available.

6. **Check Alerts**  
   Evaluate predicted-risk and observed-glucose conditions.

7. **Explain**  
   Explore model behavior through feature-group perturbations.

8. **Explore**  
   Compare predefined hypothetical scenarios.

9. **Inspect Quality**  
   Review missing, late, or implausible readings.

---

# 🤖 AI / ML Approach

TwinHealth AI uses a modular machine-learning workflow.

## 1. Data Fusion

Historical EHR information provides patient context.

Recent sensor readings provide information about the changing patient state.

These inputs are processed together to support the risk estimate.

---

## 2. Glucose Spike-Risk Prediction

The risk component estimates the likelihood of a future high-glucose event within the configured two-hour horizon.

### Event Threshold

**Glucose ≥ 180 mg/dL**

The exact model algorithm and features are defined by the training pipeline and saved model bundle.

---

## 3. Glucose Forecasting

The project can display a future glucose trajectory when a forecasting component is available.

It is important to distinguish:

**Risk Prediction**

> Estimates the probability of a future event.

**Glucose Forecast**

> Predicts future glucose values.

These are separate outputs and should be evaluated separately.

---

## 4. Explainability

The explanation interface uses feature-group perturbation to examine how the model responds to changes in selected inputs.

This provides insight into model behavior without claiming causal relationships.

---

# 🧪 Evaluation

The project includes an automated evaluation pipeline and scenario-based testing.

Evaluation can include:

| Metric | Purpose |
|---|---|
| PR-AUC | Measures precision-recall discrimination |
| Precision | Measures proportion of predicted events that were actual events |
| Recall | Measures proportion of actual events detected |
| False-alert rate | Measures incorrect alert frequency using a defined denominator |
| MAE | Measures glucose forecasting error when forecasting is available |

### Evaluation Principles

The project aims to:

- Split data by patient
- Avoid future-data leakage
- Define the future two-hour target correctly
- Exclude incomplete future windows
- Fit preprocessing using training data only
- Compare against baseline approaches
- Evaluate class imbalance
- Examine model performance across patients
- Evaluate forecasting separately from classification

> Only measured results from the actual evaluation pipeline should be reported as final model performance.

---

# 📁 Project Structure

```text
twinhealth-final/
│
├── README.md
├── LICENSE
├── SECURITY.md
├── requirements.txt
├── requirements-dev.txt
├── SHA256SUMS
├── .gitignore
│
├── ── THE TWIN ──
├── app.py
├── twin.py
├── alerts.py
├── features.py
├── replay.py
├── security.py
│
├── ── DATA ──
├── generate_data.py
├── validate.py
├── scenarios.py
│
├── ── TRAINING & TESTING ──
├── train.py
├── evaluate.py
├── make_manifest.py
├── real_eval.py
├── convert_shanghai.py
│
├── models/
│   └── bundle.joblib
│
├── demo_data/
│   ├── ehr_demo.csv
│   └── sensor_demo.csv
│
├── tests/
│   ├── conftest.py
│   ├── test_security.py
│   ├── test_alerts.py
│   └── test_twin.py
│
├── docs/
│   ├── TwinHealth_AI_Project_Presentation.pptx
│   ├── TwinHealth_AI_Project_Presentation.pdf
│   ├── architecture.pdf
│   ├── architecture.png
│   ├── make_architecture.py
│   ├── data_assumptions.md
│   ├── SCENARIOS.md
│   ├── SCENARIO_RESULTS.md
│   ├── scenario_results.csv
│   ├── REAL_DATA.md
│   ├── REAL_DATA_RESULTS_ShanghaiT2DM_all.md
│   └── REAL_DATA_RESULTS_ShanghaiT2DM_no_insulin.md
│
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       └── ci.yml
│
└── .streamlit/
    └── config.toml
```

---

# 🧩 Main Components

| File | Purpose |
|---|---|
| `app.py` | Streamlit doctor dashboard |
| `twin.py` | Digital Twin state and model operations |
| `alerts.py` | Alert rules and scoring |
| `features.py` | Converts readings into model features |
| `replay.py` | Replays simulated sensor data |
| `security.py` | File validation, input cleaning and authentication safeguards |
| `generate_data.py` | Generates synthetic patient data |
| `validate.py` | Validates generated data |
| `scenarios.py` | Defines simulated data-quality scenarios |
| `train.py` | Model training and model selection |
| `evaluate.py` | Model and scenario evaluation |
| `real_eval.py` | Evaluation using an open real-world dataset |
| `make_manifest.py` | Generates SHA256 file fingerprints |
| `convert_shanghai.py` | Converts ShanghaiT2DM data into project tables |

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **Pandas** | Data processing |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Machine learning |
| **Streamlit** | Interactive doctor dashboard |
| **Matplotlib** | Visualization |
| **Joblib** | Model serialization/loading |
| **Pytest** | Automated testing |
| **Git** | Version control |
| **GitHub** | Source code and collaboration |

---

# 🔐 Security

TwinHealth AI includes security-oriented components for the prototype environment.

These include:

- Input cleaning
- File validation
- Model/data integrity checks
- SHA256 file fingerprints
- Authentication gate
- Secret separation
- Automated security tests
- GitHub dependency monitoring

### Important

Local credentials must **never be committed to GitHub**.

Use Streamlit secrets.

Example:

```toml
[doctor_accounts]
demo_doctor = "REPLACE_WITH_LOCAL_PASSWORD"
```

Add the secrets file to `.gitignore`.

> The demonstration login is not a production healthcare identity-management system.

---

# 🧪 Testing

The repository contains automated tests covering:

- Security
- File tampering
- Invalid data
- Alert behavior
- Digital Twin scenarios
- Future-data leakage prevention
- What-if safety

Run the test suite with:

```bash
pytest
```

The project structure currently includes **43 automatic tests**.

---

# ⚙️ Installation

## Prerequisites

- Python 3.10+
- Git
- Visual Studio Code or another code editor

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/twinhealth-final.git
cd twinhealth-final
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn joblib
```

---

## 4. Verify Required Files

Make sure these files exist:

```text
app.py
twin.py
alerts.py
features.py
replay.py
security.py

models/bundle.joblib

demo_data/ehr_demo.csv
demo_data/sensor_demo.csv
```

---

# ▶️ Running TwinHealth AI

From the project root:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal.

Open that URL in your browser.

---

# 🖥️ Dashboard Experience

## 🌌 Landing Page

The competition-facing interface introduces:

**TwinHealth AI**

with the project branding:

> **Your Health. Our Digital Twin.**

---

## 🔐 Doctor Portal

The application provides a doctor-facing authentication gate before entering the simulated healthcare workspace.

The authentication is intended for the demonstration environment.

---

## 🩺 Dashboard

The main dashboard provides an organized view of:

- Selected patient
- Patient context
- Current glucose
- Risk estimate
- Alert state
- Glucose timeline
- Forecast information
- Sensor replay controls

---

## 🔬 Analysis Pages

### Dashboard
Main patient monitoring workspace.

### Why This Risk
Explore model behavior and feature-group effects.

### What-if Lab
Compare predefined hypothetical scenarios.

### Data Quality & Time
Inspect sensor timeline, data problems and uncertainty.

### Model Facts
Review available model metrics and evaluation information.

---

# 📊 Demo Data

The repository contains synthetic demonstration data for dashboard testing.

```text
demo_data/
├── ehr_demo.csv
└── sensor_demo.csv
```

The demonstration dataset is designed to allow the Digital Twin and dashboard workflow to be explored without using identifiable patient information.

---

# 🌍 Real-Data Evaluation

The project also includes tooling for evaluating the Digital Twin workflow against an open dataset.

Relevant files include:

```text
real_eval.py
convert_shanghai.py
docs/REAL_DATA.md
```

and the corresponding evaluation documentation.

Real-world evaluation does **not** turn the prototype into a clinically validated healthcare system.

---

# ⚠️ Limitations

TwinHealth AI is a research and educational prototype.

### 1. Synthetic Data

Generated patient patterns may not represent the full diversity and complexity of real patients.

### 2. No Clinical Validation

The model has not been established as safe or effective for clinical decision-making.

### 3. Simulated Sensors

The current system replays simulated sensor streams rather than demonstrating a verified live medical-device integration.

### 4. Prediction Uncertainty

Missing, delayed, or implausible readings may affect prediction reliability.

### 5. Risk vs Forecast

A predicted event probability and a predicted glucose trajectory are separate outputs.

### 6. Explainability

Feature perturbation demonstrates model behavior and should not be interpreted as causal evidence.

### 7. What-if Analysis

Hypothetical scenarios are model simulations rather than clinical recommendations.

### 8. Authentication

The demonstration login is not production-grade healthcare identity and access management.

### 9. Regulatory Compliance

The prototype has not been independently certified or assessed for healthcare regulatory compliance.

---

# 🔮 Future Enhancements

Possible future improvements include:

- Real-time authorized wearable-data integration
- Improved temporal model validation
- Additional forecasting models
- More robust missing-data handling
- Uncertainty calibration
- Model monitoring
- Data-drift detection
- Advanced explainability
- Role-based access control
- Audit logging
- Patient-subgroup evaluation
- Independent clinical review
- Privacy and security assessment

---

# 🔒 Data Privacy

TwinHealth AI is designed around synthetic demonstration data.

The project does not require real patient medical records for its intended demonstration.

### Never commit:

```text
Passwords
API keys
Access tokens
Private credentials
Real patient records
Identifiable health information
```

Use synthetic data when publishing or demonstrating the project.

---

# 🏆 Competition Value

TwinHealth AI demonstrates how multiple AI and software components can work together as a single healthcare Digital Twin workflow.

### Our key innovation areas

**🧬 Digital Twin**

A continuously updated virtual representation of a simulated patient.

**🤖 AI/ML Risk Prediction**

Estimates potential future high-glucose events.

**📡 Time-Series Replay**

Simulates changing patient data over time.

**⚠️ Early Warning**

Separates observed high-glucose events from predicted future risks.

**🔍 Explainability**

Allows users to explore why model risk changes.

**🧪 What-if Analysis**

Enables controlled hypothetical scenario comparison.

**🛡️ Data Quality**

Shows how problematic input data can affect the system.

**🔐 Security**

Includes authentication, validation, integrity checks and automated tests.

**📊 Doctor-Centric Dashboard**

Presents complex model outputs in an organized healthcare interface.

---

# 📈 Project Philosophy

TwinHealth AI is built around a simple idea:

> **Don't just show the data. Build a living representation of the patient and show how that representation changes over time.**

The Digital Twin connects:

```text
Historical Health Information
            +
Changing Sensor Data
            ↓
     Virtual Patient
            ↓
      AI/ML Analysis
            ↓
   Risk + Forecast + Alerts
            ↓
     Doctor Dashboard
```

---

# 👩‍💻 Team Cosmic Bloom

### 🌸 Cosmic Bloom — 4 Members

We are a four-member team exploring how **Artificial Intelligence, Machine Learning, Digital Twins, and healthcare analytics** can be combined to create meaningful technology solutions.

Our goal with TwinHealth AI is to demonstrate a technical prototype that connects data, machine learning, simulation, explainability, security, and visualization into one healthcare-focused system.

---

# 📄 License

This project is intended for educational and research purposes.

If this repository is released under the MIT License, the standard MIT License should be included in the `LICENSE` file with the appropriate team/author information.

---

# ⚕️ Disclaimer

**TwinHealth AI is a research and educational prototype.**

It uses synthetic patient records and simulated sensor streams.

It is **not a medical device**, has not been clinically validated, and must not be used to:

- Diagnose a medical condition
- Recommend treatment
- Make medication decisions
- Replace professional medical judgment
- Monitor real patients

Any future real-world deployment would require appropriate clinical validation, security assessment, privacy review, regulatory evaluation, and authorized healthcare integration.

---

# 🌸 Cosmic Bloom × TwinHealth AI

### **Your Health. Our Digital Twin.**

*Exploring how historical health information and changing sensor data can work together to create a more informative virtual patient representation.*
