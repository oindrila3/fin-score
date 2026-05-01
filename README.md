
FIN-Score: Production-Grade B2B Lead Scoring System

A complete end-to-end machine learning system demonstrating propensity modeling, uplift analysis, API deployment, and production monitoring for B2B lead prioritization.

https://oindrila-choudhury-b2b-lead-scoring.streamlit.app/

http://localhost:8000/docs



🎯 Business Problem
Online education companies generate thousands of leads daily but have limited sales resources. The critical question: Which leads should sales teams prioritize to maximize conversion rates and ROI?
Traditional approaches treat all leads equally or use simple heuristics (e.g., "contacted us via phone = high priority"). This project demonstrates a data-driven solution that:

Predicts conversion propensity using machine learning
Measures incremental lift from sales interventions (uplift modeling)
Deploys as a production API for real-time scoring
Monitors model drift to maintain accuracy over time

Key Insight: Not all high-propensity leads benefit equally from outreach. Uplift modeling identifies persuadable leads where sales intervention has the highest marginal impact — avoiding wasted effort on "sure things" who would convert anyway.

📊 Dataset
Source: Kaggle - B2B Leads Dataset
Domain: Online education company's lead conversion funnel
Size: ~9,000 leads with 37 features
Target: Binary conversion outcome (0 = not converted, 1 = converted)
Key Features:

Behavioral signals: website visits, time spent, page views
Demographic attributes: city, occupation, specialization
Engagement history: last activity, lead source, lead origin
Post-hoc judgments: lead quality rating, lead profile (note: potential leakage)


🏗️ System Architecture
fin-score/
├── api/
│   └── main.py              # FastAPI endpoints for real-time scoring
├── dashboard/
│   └── app.py               # Streamlit dashboard (executive + technical views)
├── data/
│   └── raw_data.csv         # Source dataset
├── models/                  # Trained model artifacts
│   ├── propensity_model.pkl
│   ├── uplift_treatment_model.pkl
│   ├── uplift_control_model.pkl
│   ├── feature_names.pkl
│   └── metadata.json        # Model performance metrics
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory analysis
│   ├── 02_feature_engineering.ipynb    # Feature pipeline development
│   ├── 03_modeling.ipynb               # Propensity model training
│   └── 04_uplift_modeling.ipynb        # Causal inference analysis
├── src/
│   ├── config.py            # Centralized configuration
│   ├── features.py          # Feature engineering pipeline
│   ├── train.py             # Model training scripts
│   ├── predict.py           # Production scoring logic
│   └── monitoring.py        # Model drift detection
├── requirements.txt
├── .python-version          # Python 3.8 (for cloud compatibility)
└── README.md

🚀 Quick Start
Prerequisites

Python 3.8+
Windows (PowerShell) or Linux/macOS

Installation
powershell# Clone the repository
git clone https://github.com/yourusername/fin-score.git
cd fin-score

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel (for notebooks)
python -m ipykernel install --user --name=venv
Running the Project
powershell# 1. Feature Engineering
python -m src.features

# 2. Train Propensity Model
python -m src.train --model propensity

# 3. Train Uplift Models
python -m src.train --model uplift

# 4. Start API Server
uvicorn api.main:app --reload

# 5. Launch Dashboard
streamlit run dashboard/app.py

🧠 Technical Approach
1. Propensity Modeling (Baseline)
Objective: Predict likelihood of lead conversion
Algorithm: XGBoost Classifier

Handles mixed data types (numeric + categorical)
Captures non-linear interactions
Built-in regularization prevents overfitting

Feature Engineering:

Frequency encoding for high-cardinality categoricals (e.g., city, occupation)
Missingness indicators for post-hoc judgment fields
Interaction features (e.g., TotalVisits × Total_Time_Spent_on_Website)
Multicollinearity removal (correlation threshold: 0.85)

Performance Metrics:

Test AUC: 0.92 (full feature set)
CV AUC: 0.89 ± 0.02 (5-fold stratified)
Lift at Top 10%: 2.5x baseline conversion rate

Critical Finding: Initial model achieved 0.96 AUC — data leakage detected. Post-hoc features like Lead Quality and Lead Profile are assigned after conversion outcome. Solution: Built two models:

Pipeline Model: All features (for leads with complete data)
Cold-Start Model: Only pre-conversion features (for new leads)


2. Uplift Modeling (Advanced)
Objective: Identify leads where sales intervention has causal impact
Why It Matters:
Not all high-propensity leads benefit from outreach. Example:

Sure Thing: 90% conversion probability, 0% uplift → Would convert without contact
Persuadable: 60% conversion probability, +30% uplift → Contact changes outcome

Methodology: T-Learner (Two-Model Approach)

Define treatment: Leads receiving proactive sales contact (SMS, phone call, email)
Train separate models on treatment and control groups
Uplift = P(conversion|treatment) - P(conversion|control)

Segmentation Matrix:
SegmentPropensityUpliftActionPersuadableHighHighPriority contact (highest ROI)Sure ThingHighLowLight touch (would convert anyway)Sleeping DogLowHighTest campaigns (potential upside)Lost CauseLowLowExclude from outreach
Viability Check:

Treatment group: 3,200 leads (adequate sample size)
Conversion rate difference: 12% (treatment) vs. 7% (control) ✓
Minimum conversions per group: 384 / 220 ✓


3. API Deployment
Tech Stack: FastAPI + Uvicorn
Endpoints:
pythonGET  /health           # Health check
POST /score            # Score a single lead
POST /score/batch      # Score multiple leads
GET  /model/metadata   # Model version and performance metrics
Example Request:
bashcurl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "TotalVisits": 5,
    "Total_Time_Spent_on_Website": 320,
    "Page_Views_Per_Visit": 3.2,
    "Lead_Origin": "API",
    "Lead_Source": "Google"
  }'
Response:
json{
  "propensity_score": 0.87,
  "uplift_score": 0.23,
  "segment": "Persuadable",
  "recommendation": "High priority - contact within 24 hours",
  "explanation": {
    "top_drivers": [
      {"feature": "TotalVisits", "impact": "+0.12"},
      {"feature": "Page_Views_Per_Visit", "impact": "+0.08"}
    ]
  }
}
Production Features:

Model caching (load once per process)
Input validation (feature bounds checking)
SHAP explainability (top 3 drivers per prediction)
Error handling and logging


4. Monitoring & Drift Detection
Approach: SQLite-based tracking system
Tables:

scored_leads: Audit trail of all predictions
drift_snapshots: Daily feature distribution summaries
alerts: Threshold breach history
model_performance: Weekly AUC measurements (when outcomes known)

Alert Thresholds:

Hot lead rate >50% or <10% (distribution shift)
Test AUC <0.80 (model degradation)
Score mean delta >0.05 day-over-day (sudden change)

Dashboard Integration:

Real-time drift metrics
Feature importance stability tracking
Conversion rate by segment (validates uplift model)


📈 Business Impact
ROI Calculator (Dashboard Feature)
Assumptions:

Daily leads: 1,000
Average deal value: $400
Baseline conversion: 9.2%
Cost per contact: $15

Without FIN-Score:

Random outreach to 300 leads/day
Expected conversions: 300 × 0.092 = 27.6 leads
Revenue: $11,040/day
Cost: $4,500/day
Net: $6,540/day

With FIN-Score (Persuadable segment):

Targeted outreach to top 300 leads (by uplift)
Expected conversions: 300 × (0.092 + 0.23) = 96.6 leads
Revenue: $38,640/day
Cost: $4,500/day
Net: $34,140/day

Incremental Value: +$27,600/day = +422% ROI improvement

🎓 Key Learnings
1. Data Leakage is Real
Initial model showed 0.96 AUC — too good to be true. Root cause: Lead Quality and Lead Profile are assigned after conversion. Lesson: Always audit feature timelines relative to prediction target.
2. Uplift ≠ Propensity
High-propensity leads may have zero uplift (they'd convert anyway). Optimizing for uplift shifts strategy from "who will buy" to "who will buy because of our action" — a fundamentally different question.
3. Production-Grade Means Monitoring
Models degrade over time. Drift detection isn't optional — it's the difference between a demo and a deployable system.
4. Two Audiences, Two Dashboards
Executives need ROI calculators and plain-English summaries. Technical stakeholders need feature importance charts and AUC curves. Always build for both.

🛠️ Tech Stack
LayerTechnologyReasonML FrameworkXGBoostHandles mixed types, robust to outliersFeature EngineeringPandas, NumPyIndustry standard for tabular dataExplainabilitySHAPModel-agnostic, fast for tree modelsAPIFastAPIAsync support, auto-generated docsDashboardStreamlitRapid prototyping, native cachingMonitoringSQLite + SQLAlchemyLightweight, no external dependenciesVersion ControlGit + GitHubPublic portfolio piece

📚 Project Phases (5-Day Timeline)
DayPhaseDeliverables1Setup + EDAEnvironment configured, leakage identified2Feature Engineeringfeatures.py, cleaned dataset3Propensity ModelingTrained model, API endpoints4Uplift ModelingSegmentation logic, dashboard v15Monitoring + PolishDrift detection, presentation deck

🚧 Roadmap
Next Steps

 A/B testing framework for uplift validation
 Dockerize API for cloud deployment
 Automated retraining pipeline (weekly cadence)
 Integration with CRM systems (Salesforce, HubSpot)
 Multi-touch attribution modeling

Nice-to-Have

 Real-time feature streaming (Kafka integration)
 Bayesian optimization for hyperparameter tuning
 Counterfactual explanations for rejected leads


📄 License
MIT License - feel free to use this as a template for your own projects.

🙏 Acknowledgments

Dataset: ashydv on Kaggle
Inspiration: Designed for a data science role requiring end-to-end ML thinking
Blueprint: Built using production-grade patterns from industry experience


📬 Contact
Questions? Open an issue or reach out via LinkedIn.