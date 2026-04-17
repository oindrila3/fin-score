# ============================================================
# config.py — Central configuration for FIN-Score pipeline
# All parameters, paths, and constants live here
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Override config with environment variables if present
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))

# ── Project Paths ────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

# ── Data Files ───────────────────────────────────────────────
RAW_DATA_PATH = DATA_DIR / "leads_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "leads_processed.csv"

# ── Target Variable ──────────────────────────────────────────
TARGET_COLUMN = "Converted"

# ── Model Parameters ─────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "auc"
}

# Class weight — optimized via grid search
# Mathematical formula gives 1.59, grid search confirms 2.0 is optimal
SCALE_POS_WEIGHT = 2.0

# ── Train/Test Split ─────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Model Variants ───────────────────────────────────────────

# Features available AFTER sales rep interaction
# Used for leads already in CRM with history
PIPELINE_FEATURES_TO_EXCLUDE = []  # Uses all features

# Features NOT available at lead creation time
# These are post-hoc sales rep judgment columns
COLDSTART_FEATURES_TO_EXCLUDE = [
    'Lead Quality_freq_encoded',
    'Lead Profile_freq_encoded', 
    'Tags_freq_encoded',
    'was_lead_quality_assessed',
    'was_lead_profile_assessed',
    'was_tags_assessed',
]

# Model file names
PIPELINE_MODEL_NAME = 'pipeline_model.pkl'
COLDSTART_MODEL_NAME = 'coldstart_model.pkl'
PIPELINE_FEATURES_NAME = 'pipeline_feature_names.pkl'
COLDSTART_FEATURES_NAME = 'coldstart_feature_names.pkl'

# ── Scoring Thresholds ───────────────────────────────────────
HIGH_PRIORITY_THRESHOLD = 0.7
MEDIUM_PRIORITY_THRESHOLD = 0.4

# ── Lead Priority Labels ─────────────────────────────────────
PRIORITY_LABELS = {
    "high": "Hot Lead",
    "medium": "Warm Lead",
    "low": "Cold Lead"
}

# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Root Directory:           {ROOT_DIR}")
    print(f"Data Directory:           {DATA_DIR}")
    print(f"Model Directory:          {MODEL_DIR}")
    print(f"Raw Data Path:            {RAW_DATA_PATH}")
    print(f"Processed Data Path:      {PROCESSED_DATA_PATH}")
    print(f"Target Column:            {TARGET_COLUMN}")
    print(f"High Priority Threshold:  {HIGH_PRIORITY_THRESHOLD}")
    print(f"Medium Priority Threshold:{MEDIUM_PRIORITY_THRESHOLD}")
    print(f"Test Size:                {TEST_SIZE}")
    print(f"Scale Pos Weight:         {SCALE_POS_WEIGHT}")
    print(f"Environment:              {ENVIRONMENT}")
    print(f"API Host:                 {API_HOST}")
    print(f"API Port:                 {API_PORT}")
    print(f"\nModel Files:")
    print(f"  Pipeline:   {PIPELINE_MODEL_NAME}")
    print(f"  Coldstart:  {COLDSTART_MODEL_NAME}")
    print(f"\nColdstart Excluded Features:")
    for f in COLDSTART_FEATURES_TO_EXCLUDE:
        print(f"  - {f}")
    print(f"\n✅ Config loaded successfully!")
