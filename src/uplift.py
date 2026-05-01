# ============================================================
# src/uplift.py -- Uplift Modeling for FIN-Score
#
# T-Learner approach:
#   - Train one model on treatment group (sales contacted)
#   - Train one model on control group (no sales contact)
#   - Uplift = P(convert|treated) - P(convert|control)
#
# This answers: "Which leads NEED a sales call to convert?"
# vs "Which leads would convert anyway?"
# ============================================================

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR, PROCESSED_DATA_PATH

# ── Treatment definition ──────────────────────────────────
# These activities indicate sales proactively reached out
TREATMENT_ACTIVITIES = [
    'SMS Sent',
    'Email Bounced',        # sales tried, email bounced
    'Had a Phone Conversation',
    'Approached upfront',
    'Email Received'
]

# Uplift segments
SEGMENT_LABELS = {
    'persuadable':  'Persuadable',   # high uplift, call them
    'sure_thing':   'Sure Thing',    # converts anyway
    'sleeping_dog': 'Sleeping Dog',  # low propensity, worth nudge
    'lost_cause':   'Lost Cause'     # don't bother
}

# Thresholds
PROPENSITY_THRESHOLD = 0.50   # above = high propensity
UPLIFT_THRESHOLD     = 0.10   # above = meaningful uplift

def load_processed_data() -> pd.DataFrame:
    """ Load the processed feature engineering dataset. """
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}."
            "Run src/feature.py instead first."
        )
    return pd.read_csv(PROCESSED_DATA_PATH)

def create_treatment_flag(df : pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary treatment flag .
    Treatment = 1: sales proactively contacted this lead
    Treatment = 0: Lead engaged organically , no sales push 
    """

    df = df.copy()
    #We need to reload raw data to get Last Activity
    raw_path = os.path.join(
        os.path.dirname(PROCESSED_DATA_PATH),'leads_raw.csv'
    )

    raw = pd.read_csv(raw_path)
    #Align index
    raw = raw.reset_index(drop = True)
    df = df.reset_index(drop=True)

    # Create treatment flag from raw Last Activity
    df['treatment'] = raw['Last Activity'].isin(
        TREATMENT_ACTIVITIES
    ).astype(int)

    print(f"\nTreatment group size: "
          f"{df['treatment'].sum():,} "
          f"({df['treatment'].mean():.1%})")
    print(f"Control group size:   "
          f"{(df['treatment'] == 0).sum():,} "
          f"({(df['treatment'] == 0).mean():.1%})")
    print(f"Conversion rate in treatment: "
          f"{df[df['treatment']==1]['Converted'].mean():.1%}")
    print(f"Conversion rate in control:   "
          f"{df[df['treatment']==0]['Converted'].mean():.1%}")

    return df

def train_uplift_models(df: pd.DataFrame) -> dict:
    """
      T-Learner: train separate models on treatment and control.
    Why T-Learner?
    Simple to explain: two separate models
    Interpretable: each model tells a clear story
     Good baseline: industry standard starting point
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    # Load feature names from existing coldstart model
    features_path = os.path.join(
        MODEL_DIR, 'coldstart_feature_names.pkl'
    )
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    # Keep only features that exist in our data
    available_features = [
        f for f in feature_names
        if f in df.columns
    ]

    print(f"\nUsing {len(available_features)} features "
          f"for uplift models")

    X = df[available_features].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)
    y = df['Converted']
    t = df['treatment']

    # Split into treatment and control
    X_treatment = X[t == 1]
    y_treatment = y[t == 1]
    X_control   = X[t == 0]
    y_control   = y[t == 0]

    print(f"\nTraining treatment model on "
          f"{len(X_treatment):,} leads...")
    print(f"Training control model on "
          f"{len(X_control):,} leads...")

    # XGBoost params — same family as propensity model
    params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }

    # Treatment model — what happens when sales calls
    treatment_model = XGBClassifier(**params)
    treatment_model.fit(X_treatment, y_treatment)

    # Control model -- what happens without sales call
    control_model = XGBClassifier(**params)
    control_model.fit(X_control, y_control)

    # Cross validation on full dataset
    cv_model = XGBClassifier(**params)
    cv_scores = cross_val_score(
        cv_model, X, y, cv=5, scoring='roc_auc'
    )
    print(f"\nCross-validation AUC: "
          f"{cv_scores.mean():.4f} "
          f"+/- {cv_scores.std():.4f}")

    return {
        'treatment_model': treatment_model,
        'control_model':   control_model,
        'feature_names':   available_features,
        'cv_auc':          float(cv_scores.mean()),
        'cv_std':          float(cv_scores.std()),
        'n_treatment':     int(len(X_treatment)),
        'n_control':       int(len(X_control))
    }


def score_uplift(df: pd.DataFrame,
                 treatment_model,
                 control_model,
                 feature_names: list) -> pd.DataFrame:
    """
    Compute uplift scores for all leads.

    uplift_score = P(convert|treated) - P(convert|control)

    Positive uplift: calling helps
    Near zero uplift: calling makes no difference
    Negative uplift: calling hurts (do not disturb!)
    """
    X = df[feature_names].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)

    # P(convert | sales called)
    p_treatment = treatment_model.predict_proba(X)[:, 1]

    # P(convert | no sales call)
    p_control = control_model.predict_proba(X)[:, 1]

    # Uplift = incremental effect of calling
    uplift = p_treatment - p_control

    df = df.copy()
    df['p_treatment'] = p_treatment
    df['p_control']   = p_control
    df['uplift_score'] = uplift

    # Assign segments based on propensity and uplift
    df['uplift_segment'] = 'Lost Cause'

    # High propensity + high uplift = Persuadable
    mask_persuadable = (
        (p_treatment >= PROPENSITY_THRESHOLD) &
        (uplift >= UPLIFT_THRESHOLD)
    )
    # High propensity + low uplift = Sure Thing
    mask_sure_thing = (
        (p_treatment >= PROPENSITY_THRESHOLD) &
        (uplift < UPLIFT_THRESHOLD)
    )
    # Low propensity + high uplift = Sleeping Dog
    mask_sleeping_dog = (
        (p_treatment < PROPENSITY_THRESHOLD) &
        (uplift >= UPLIFT_THRESHOLD)
    )

    df.loc[mask_persuadable,  'uplift_segment'] = 'Persuadable'
    df.loc[mask_sure_thing,   'uplift_segment'] = 'Sure Thing'
    df.loc[mask_sleeping_dog, 'uplift_segment'] = 'Sleeping Dog'

    print("\nUplift segment distribution:")
    print(df['uplift_segment'].value_counts())
    print(f"\nMean uplift score: {uplift.mean():.4f}")
    print(f"Leads with positive uplift: "
          f"{(uplift > 0).sum():,} "
          f"({(uplift > 0).mean():.1%})")
    print(f"\nConversion by segment:")
    print(df.groupby('uplift_segment')['Converted'].mean()
            .sort_values(ascending=False)
            .apply(lambda x: f"{x:.1%}"))

    return df


def save_uplift_models(results: dict,
                       df_scored: pd.DataFrame):
    """Save models and metadata to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save models
    with open(os.path.join(
        MODEL_DIR, 'uplift_treatment_model.pkl'
    ), 'wb') as f:
        pickle.dump(results['treatment_model'], f)

    with open(os.path.join(
        MODEL_DIR, 'uplift_control_model.pkl'
    ), 'wb') as f:
        pickle.dump(results['control_model'], f)

    # Save feature names
    with open(os.path.join(
        MODEL_DIR, 'uplift_feature_names.pkl'
    ), 'wb') as f:
        pickle.dump(results['feature_names'], f)

    # Save segment distribution
    segment_dist = df_scored['uplift_segment']\
        .value_counts().to_dict()

    metadata = {
        'model_type':       'T-Learner Uplift',
        'treatment_definition': TREATMENT_ACTIVITIES,
        'n_treatment':      results['n_treatment'],
        'n_control':        results['n_control'],
        'cv_auc':           results['cv_auc'],
        'cv_std':           results['cv_std'],
        'propensity_threshold': PROPENSITY_THRESHOLD,
        'uplift_threshold': UPLIFT_THRESHOLD,
        'segment_distribution': segment_dist,
        'mean_uplift':      float(
            df_scored['uplift_score'].mean()
        ),
        'pct_positive_uplift': float(
            (df_scored['uplift_score'] > 0).mean()
        ),
        'trained_at': datetime.now(
            timezone.utc
        ).isoformat()
    }

    with open(os.path.join(
        MODEL_DIR, 'uplift_metadata.json'
    ), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save scored leads sample for dashboard
    df_scored[[
        'uplift_score', 'uplift_segment',
        'p_treatment', 'p_control', 'Converted'
    ]].to_csv(
        os.path.join(MODEL_DIR, 'uplift_scores.csv'),
        index=False
    )

    print(f"\nModels saved to {MODEL_DIR}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")


def run_uplift_pipeline():
    """Full uplift modeling pipeline."""
    print("=" * 60)
    print("FIN-Score Uplift Modeling Pipeline")
    print("=" * 60)

    print("\n1. Loading processed data...")
    df = load_processed_data()
    print(f"   Loaded {len(df):,} leads")

    print("\n2. Creating treatment flag...")
    df = create_treatment_flag(df)

    print("\n3. Training T-Learner models...")
    results = train_uplift_models(df)

    print("\n4. Scoring all leads for uplift...")
    df_scored = score_uplift(
        df,
        results['treatment_model'],
        results['control_model'],
        results['feature_names']
    )

    print("\n5. Saving models and metadata...")
    save_uplift_models(results, df_scored)

    print("\n" + "=" * 60)
    print("Uplift pipeline complete!")
    print("=" * 60)

    return df_scored


if __name__ == "__main__":
    df_scored = run_uplift_pipeline()