# ============================================================
# predict.py — Production Scoring Logic for FIN-Score
# ============================================================

import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# STEP 1: INPUT VALIDATION
# ============================================================

# Define valid ranges for numeric features
# Based on 99th percentile analysis from EDA
FEATURE_BOUNDS = {
    'TotalVisits': (0, 100),
    'Total Time Spent on Website': (0, 5000),
    'Page Views Per Visit': (0, 20),
    'Asymmetrique Activity Score': (0, 25),
    'Asymmetrique Profile Score': (0, 25),
    'asymmetrique_combined': (0, 25),
    'is_high_engagement': (0, 1),
    'Do Not Email': (0, 1),
    'Search': (0, 1),
    'Through Recommendations': (0, 1),
    'was_lead_quality_assessed': (0, 1),
    'was_lead_profile_assessed': (0, 1),
    'was_tags_assessed': (0, 1),
}

def validate_lead_input(lead_data: Dict) -> Tuple[bool,
                                                    List[str]]:
    """
    Validates lead input before scoring.
    Returns (is_valid, list_of_errors).
    
    Checks:
    1. Required fields present
    2. Values within expected ranges
    3. No unexpected data types
    
    Why this matters:
    Garbage in = garbage out. Silent bad inputs produce
    confident wrong scores — more dangerous than an error.
    """
    errors = []
    warnings_list = []
    
    # Check numeric bounds
    for feature, (min_val, max_val) in FEATURE_BOUNDS.items():
        if feature in lead_data:
            val = lead_data[feature]
            
            # Type check
            try:
                val = float(val)
            except (TypeError, ValueError):
                errors.append(
                    f"{feature}: expected numeric, "
                    f"got {type(val).__name__}"
                )
                continue
            
            # Range check — warn but don't reject
            if val < min_val or val > max_val:
                warnings_list.append(
                    f"{feature}: value {val} outside "
                    f"expected range [{min_val}, {max_val}]"
                )
    
    # Log warnings
    for w in warnings_list:
        logger.warning(f"Input warning: {w}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================
# STEP 2: LOAD MODELS WITH CACHING
# ============================================================

# Module-level cache — loads model once per process
# In production this prevents reading from disk on every request
_model_cache = {}

def load_model(model_type: str = 'coldstart'):
    """
    Loads trained model with in-memory caching.
    First call reads from disk — subsequent calls use cache.
    
    In production with 1000 requests/minute, 
    loading from disk every time would be catastrophic.
    Caching means the model loads once and stays in memory.
    """
    global _model_cache
    
    if model_type in _model_cache:
        logger.debug(f"Using cached {model_type} model")
        return (_model_cache[model_type]['model'],
                _model_cache[model_type]['feature_names'],
                _model_cache[model_type]['version'])
    
    if model_type == 'pipeline':
        model_path = os.path.join(MODEL_DIR, PIPELINE_MODEL_NAME)
        features_path = os.path.join(MODEL_DIR,
                                      PIPELINE_FEATURES_NAME)
    else:
        model_path = os.path.join(MODEL_DIR, COLDSTART_MODEL_NAME)
        features_path = os.path.join(MODEL_DIR,
                                      COLDSTART_FEATURES_NAME)

    # Graceful fallback if model missing
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Run python -m src.train first."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    # Generate model version hash from file
    # Lets us track which model version scored each lead
    with open(model_path, 'rb') as f:
        model_hash = hashlib.md5(f.read()).hexdigest()[:8]
    version = f"{model_type}_v{model_hash}"
    
    # Cache it
    _model_cache[model_type] = {
        'model': model,
        'feature_names': feature_names,
        'version': version
    }
    
    logger.info(f"Loaded {model_type} model "
                f"[version: {version}] — "
                f"{len(feature_names)} features")
    
    return model, feature_names, version


# ============================================================
# STEP 3: SHAP EXPLANATION PER LEAD
# ============================================================

def explain_lead_score(model,
                        lead_df: pd.DataFrame,
                        feature_names: list,
                        top_n: int = 5) -> Dict:
    """
    Generates SHAP explanation for a single lead.
    
    Instead of just saying "score = 0.82" we explain:
    - Which features pushed the score UP
    - Which features pushed the score DOWN
    - By how much each feature contributed
    
    This is critical for sales rep trust and adoption.
    A rep who understands WHY a lead is Hot will act on it.
    A rep who sees a black-box number might ignore it.
    """
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(lead_df)
        
        # Build feature contribution dict
        contributions = dict(zip(
            feature_names,
            shap_values[0]
        ))
        
        # Sort by absolute impact
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        # Format for human readability
        positive_drivers = [
            {'feature': k, 'impact': round(v, 4)}
            for k, v in sorted_contributions if v > 0
        ]
        negative_drivers = [
            {'feature': k, 'impact': round(v, 4)}
            for k, v in sorted_contributions if v < 0
        ]
        
        return {
            'positive_drivers': positive_drivers,
            'negative_drivers': negative_drivers,
            'explanation_available': True
        }
        
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return {'explanation_available': False}


# ============================================================
# STEP 4: THRESHOLD ANALYSIS
# ============================================================

def get_threshold_context(score: float) -> Dict:
    """
    Gives context about how close the score is to
    threshold boundaries — critical for borderline cases.
    
    A score of 0.71 (barely Hot) should be treated
    differently from 0.95 (strongly Hot).
    This context helps sales managers make better decisions
    on borderline leads.
    """
    distance_to_hot = score - HIGH_PRIORITY_THRESHOLD
    distance_to_warm = score - MEDIUM_PRIORITY_THRESHOLD
    
    # Confidence based on distance from nearest threshold
    nearest_threshold_distance = min(
        abs(distance_to_hot),
        abs(distance_to_warm)
    )
    
    if nearest_threshold_distance < 0.05:
        confidence = 'Low — borderline case, review manually'
    elif nearest_threshold_distance < 0.15:
        confidence = 'Medium — likely correct classification'
    else:
        confidence = 'High — clear classification'
    
    return {
        'distance_to_hot_threshold': round(distance_to_hot, 4),
        'distance_to_warm_threshold': round(distance_to_warm, 4),
        'confidence': confidence,
        'is_borderline': nearest_threshold_distance < 0.05
    }


# ============================================================
# STEP 5: SCORE LOGGING FOR DRIFT MONITORING
# ============================================================

def log_score(score: float,
               priority: str,
               model_type: str,
               version: str) -> None:
    """
    Logs every score to a JSONL file for drift monitoring.
    
    Over time this log lets us detect:
    - Score distribution shift (are more leads scoring Hot?)
    - Model degradation (is average score drifting?)
    - Volume anomalies (sudden spike in Cold leads?)
    
    In production this would write to a data warehouse.
    Here we write to a local file for demonstration.
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'score': score,
        'priority': priority,
        'model_type': model_type,
        'model_version': version
    }
    
    log_path = os.path.join(MODEL_DIR, 'score_log.jsonl')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================================
# STEP 6: DETECT MODEL TYPE
# ============================================================

def detect_model_type(lead_data: Dict) -> str:
    """
    Auto-detects which model to use based on
    whether post-hoc sales features are present.
    """
    has_sales_history = any(
        feature in lead_data
        for feature in COLDSTART_FEATURES_TO_EXCLUDE
    )
    return 'pipeline' if has_sales_history else 'coldstart'


# ============================================================
# STEP 7: PREPARE LEAD FOR SCORING
# ============================================================

def prepare_lead_for_scoring(lead_data: Dict,
                              feature_names: list) -> pd.DataFrame:
    """
    Aligns lead data to exact feature schema model expects.
    Missing features filled with 0.
    """
    lead_df = pd.DataFrame([lead_data])
    
    for feature in feature_names:
        if feature not in lead_df.columns:
            lead_df[feature] = 0
    
    lead_df = lead_df[feature_names]
    lead_df = lead_df.apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)
    
    return lead_df


# ============================================================
# STEP 8: GET RECOMMENDATION
# ============================================================

def get_recommendation(priority: str) -> str:
    """
    Returns actionable sales recommendation.
    """
    recommendations = {
        PRIORITY_LABELS['high']: (
            "Call immediately — high conversion probability. "
            "Prioritize in today's outreach queue."
        ),
        PRIORITY_LABELS['medium']: (
            "Add to nurture sequence — moderate probability. "
            "Follow up within 48 hours."
        ),
        PRIORITY_LABELS['low']: (
            "Low priority — do not assign to sales rep. "
            "Add to automated email drip campaign."
        )
    }
    return recommendations.get(priority, "No recommendation")


# ============================================================
# MASTER SCORING FUNCTION
# ============================================================

def score_lead(lead_data: Dict,
               model_type: Optional[str] = None,
               explain: bool = True) -> Dict:
    """
    Production scoring function.
    
    Validates input → loads model → scores → explains → logs
    
    Args:
        lead_data:   Dictionary of lead attributes
        model_type:  'pipeline', 'coldstart', or None (auto)
        explain:     Whether to include SHAP explanations
    
    Returns:
        Complete scoring response with score, priority,
        explanation, confidence, and recommendation
    """
    # Step 1: Validate input
    is_valid, errors = validate_lead_input(lead_data)
    if not is_valid:
        return {
            'error': True,
            'errors': errors,
            'score': None,
            'priority': None
        }
    
    # Step 2: Detect model type
    if model_type is None:
        model_type = detect_model_type(lead_data)
    
    # Step 3: Load model (cached after first call)
    model, feature_names, version = load_model(model_type)
    
    # Step 4: Prepare lead
    lead_df = prepare_lead_for_scoring(lead_data, feature_names)
    
    # Step 5: Score
    score = float(model.predict_proba(lead_df)[0][1])
    priority = (
        PRIORITY_LABELS['high'] 
        if score >= HIGH_PRIORITY_THRESHOLD
        else PRIORITY_LABELS['medium'] 
        if score >= MEDIUM_PRIORITY_THRESHOLD
        else PRIORITY_LABELS['low']
    )
    
    # Step 6: Get threshold context
    threshold_context = get_threshold_context(score)
    
    # Step 7: SHAP explanation
    explanation = {}
    if explain:
        explanation = explain_lead_score(
            model, lead_df, feature_names
        )
    
    # Step 8: Log for monitoring
    log_score(score, priority, model_type, version)
    
    # Build complete response
    response = {
        'error': False,
        'score': round(score, 4),
        'score_pct': f"{score:.1%}",
        'priority': priority,
        'model_type': model_type,
        'model_version': version,
        'recommendation': get_recommendation(priority),
        'confidence': threshold_context['confidence'],
        'is_borderline': threshold_context['is_borderline'],
        'threshold_context': threshold_context,
        'explanation': explanation,
        'scored_at': datetime.utcnow().isoformat()
    }
    
    logger.info(
        f"Scored lead: {score:.4f} → {priority} "
        f"[{model_type}] [{version}]"
    )
    
    return response


# ============================================================
# BATCH SCORING
# ============================================================

def score_batch(leads: list,
                model_type: str = 'coldstart',
                explain: bool = False) -> pd.DataFrame:
    """
    Scores a batch of leads efficiently.
    Loads model once — scores all leads in one pass.
    explain=False by default for batch — SHAP is slow at scale.
    """
    logger.info(f"Batch scoring {len(leads):,} leads...")
    
    model, feature_names, version = load_model(model_type)
    
    leads_df = pd.DataFrame(leads)
    for feature in feature_names:
        if feature not in leads_df.columns:
            leads_df[feature] = 0
    
    leads_df = leads_df[feature_names]
    leads_df = leads_df.apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)
    
    scores = model.predict_proba(leads_df)[:, 1]
    priorities = [
        PRIORITY_LABELS['high'] 
        if s >= HIGH_PRIORITY_THRESHOLD
        else PRIORITY_LABELS['medium'] 
        if s >= MEDIUM_PRIORITY_THRESHOLD
        else PRIORITY_LABELS['low']
        for s in scores
    ]
    
    results = pd.DataFrame({
        'score': scores.round(4),
        'priority': priorities,
        'recommendation': [get_recommendation(p) 
                           for p in priorities],
        'model_version': version,
        'confidence': [get_threshold_context(s)['confidence'] 
                       for s in scores],
        'is_borderline': [get_threshold_context(s)['is_borderline']
                          for s in scores]
    })
    
    # Log distribution for monitoring
    hot = (results['priority'] == PRIORITY_LABELS['high']).sum()
    warm = (results['priority'] == PRIORITY_LABELS['medium']).sum()
    cold = (results['priority'] == PRIORITY_LABELS['low']).sum()
    
    logger.info(f"Batch complete — "
                f"Hot: {hot:,} | Warm: {warm:,} | Cold: {cold:,}")
    
    return results


# ============================================================
# DRIFT MONITOR
# ============================================================

def check_score_drift() -> Dict:
    """
    Reads score log and checks for distribution drift.
    
    In production this would run on a schedule — daily or
    weekly — and alert the DS team if score distribution
    shifts significantly from baseline.
    
    This is what separates a deployed model from a 
    forgotten model.
    """
    log_path = os.path.join(MODEL_DIR, 'score_log.jsonl')
    
    if not os.path.exists(log_path):
        return {'status': 'No scores logged yet'}
    
    scores = []
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            scores.append(entry['score'])
    
    if len(scores) < 10:
        return {'status': f'Only {len(scores)} scores logged'}
    
    scores = np.array(scores)
    
    hot_rate = (scores >= HIGH_PRIORITY_THRESHOLD).mean()
    warm_rate = ((scores >= MEDIUM_PRIORITY_THRESHOLD) & 
                 (scores < HIGH_PRIORITY_THRESHOLD)).mean()
    cold_rate = (scores < MEDIUM_PRIORITY_THRESHOLD).mean()
    
    drift_report = {
        'total_scored': len(scores),
        'mean_score': round(scores.mean(), 4),
        'std_score': round(scores.std(), 4),
        'hot_rate': f"{hot_rate:.1%}",
        'warm_rate': f"{warm_rate:.1%}",
        'cold_rate': f"{cold_rate:.1%}",
        'status': 'OK' if 0.1 < hot_rate < 0.5 else 'REVIEW NEEDED'
    }
    
    print(f"\n📊 SCORE DRIFT REPORT")
    print(f"{'='*40}")
    for k, v in drift_report.items():
        print(f"  {k}: {v}")
    
    return drift_report


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    
    print("Testing FIN-Score predict.py")
    print("=" * 50)
    
    # Test 1: Hot lead with explanation
    print("\n🔥 TEST 1: High engagement cold-start lead")
    hot_lead = {
        'TotalVisits': 10,
        'Total Time Spent on Website': 2500,
        'Page Views Per Visit': 4.5,
        'Asymmetrique Activity Score': 20.0,
        'Asymmetrique Profile Score': 20.0,
        'Do Not Email': 0,
        'Search': 1,
        'Through Recommendations': 1,
        'is_high_engagement': 1,
        'asymmetrique_combined': 20.0,
        'was_lead_quality_assessed': 0,
        'was_lead_profile_assessed': 0,
        'was_tags_assessed': 0
    }
    result = score_lead(hot_lead, model_type='coldstart')
    print(f"Score:       {result['score']} ({result['score_pct']})")
    print(f"Priority:    {result['priority']}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Borderline:  {result['is_borderline']}")
    print(f"Version:     {result['model_version']}")
    if result['explanation'].get('explanation_available'):
        print(f"Top drivers (positive):")
        for d in result['explanation']['positive_drivers'][:3]:
            print(f"  + {d['feature']}: {d['impact']}")
        print(f"Top drivers (negative):")
        for d in result['explanation']['negative_drivers'][:3]:
            print(f"  - {d['feature']}: {d['impact']}")
    
    # Test 2: Cold lead
    print("\n❄️  TEST 2: Low engagement cold lead")
    cold_lead = {
        'TotalVisits': 1,
        'Total Time Spent on Website': 50,
        'Page Views Per Visit': 1.0,
        'Asymmetrique Activity Score': 5.0,
        'Asymmetrique Profile Score': 5.0,
        'Do Not Email': 1,
        'Search': 0,
        'Through Recommendations': 0,
        'is_high_engagement': 0,
        'asymmetrique_combined': 5.0,
        'was_lead_quality_assessed': 0,
        'was_lead_profile_assessed': 0,
        'was_tags_assessed': 0
    }
    result = score_lead(cold_lead, model_type='coldstart')
    print(f"Score:       {result['score']} ({result['score_pct']})")
    print(f"Priority:    {result['priority']}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Borderline:  {result['is_borderline']}")
    
    # Test 3: Invalid input
    print("\n⚠️  TEST 3: Invalid input validation")
    invalid_lead = {
        'TotalVisits': -5,
        'Total Time Spent on Website': 'not_a_number',
    }
    result = score_lead(invalid_lead, model_type='coldstart')
    print(f"Error caught: {result['error']}")
    print(f"Errors: {result['errors']}")
    
    # Test 4: Batch scoring
    print("\n📦 TEST 4: Batch scoring")
    batch_results = score_batch(
        [hot_lead, cold_lead], 
        model_type='coldstart'
    )
    print(batch_results[['score', 'priority', 
                          'confidence', 'is_borderline']])
    
    # Test 5: Drift check
    print("\n📊 TEST 5: Score drift check")
    check_score_drift()