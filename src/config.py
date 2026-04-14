# ============================================================
# config.py — Central configuration for FIN-Score pipeline
# All parameters, paths, and constants live here
# ============================================================

import os
from pathlib import Path

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

# ── Scoring Thresholds ───────────────────────────────────────
HIGH_PRIORITY_THRESHOLD = 0.7
MEDIUM_PRIORITY_THRESHOLD = 0.4

# ── Lead Priority Labels ─────────────────────────────────────
PRIORITY_LABELS = {
    "high": "Hot Lead",
    "medium": "Warm Lead",
    "low": "Cold Lead"
}
