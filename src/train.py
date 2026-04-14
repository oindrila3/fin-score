# ============================================================
# train.py — Model Training Pipeline for FIN-Score
# Trains XGBoost propensity model on processed leads data
# ============================================================

import pandas as pd
import numpy as np
import logging
import os
import sys
import pickle
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# STEP 1: LOAD PROCESSED DATA
# ============================================================

def load_processed_data() -> pd.DataFrame:
    """
    Loads the feature-engineered dataset produced by features.py.
    Validates that target column exists before proceeding.
    """
    logger.info("Loading processed data...")

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            f"Run features.py first."
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    logger.info(f"Conversion rate: {df[TARGET_COLUMN].mean():.1%}")
    return df


# ============================================================
# STEP 2: PREPARE FEATURES AND TARGET
# ============================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                  pd.Series,
                                                  list]:
    """
    Separates features from target and ensures all
    columns are numeric — XGBoost requires numeric input.

    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature column names
    """
    logger.info("Preparing features and target...")

    # Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Handle any remaining non-numeric columns
    # These shouldn't exist after features.py but we check anyway
    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        logger.warning(f"Found non-numeric columns: {non_numeric}")
        logger.warning("Label encoding them as fallback...")
        le = LabelEncoder()
        for col in non_numeric:
            X[col] = le.fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()
    logger.info(f"Features: {len(feature_names)} columns")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, feature_names


# ============================================================
# STEP 3: TRAIN TEST SPLIT
# ============================================================

def split_data(X: pd.DataFrame,
               y: pd.Series) -> Tuple:
    """
    Stratified train/test split — preserves conversion rate
    in both splits. Critical for imbalanced datasets.

    Stratified means if overall conversion is 38%,
    both train and test will also be ~38%.
    """
    logger.info(f"Splitting data: {1-TEST_SIZE:.0%} train / "
                f"{TEST_SIZE:.0%} test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Preserves class balance in both splits
    )

    logger.info(f"Train: {X_train.shape[0]:,} rows | "
                f"Conversion: {y_train.mean():.1%}")
    logger.info(f"Test:  {X_test.shape[0]:,} rows | "
                f"Conversion: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 4: CALCULATE CLASS WEIGHT
# ============================================================

def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculates scale_pos_weight for XGBoost to handle
    class imbalance. This tells XGBoost to penalize
    missing a conversion more than missing a non-conversion.

    Formula: count(negative) / count(positive)
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    logger.info(f"  Non-converted: {n_negative:,}")
    logger.info(f"  Converted:     {n_positive:,}")

    return scale_pos_weight


# ============================================================
# STEP 5: TRAIN XGBOOST MODEL
# ============================================================

def train_xgboost(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   scale_pos_weight: float) -> xgb.XGBClassifier:
    """
    Trains XGBoost classifier with parameters from config.py.
    Uses scale_pos_weight to handle class imbalance.

    XGBoost builds an ensemble of decision trees where each
    tree corrects the errors of the previous one.
    """
    logger.info("Training XGBoost model...")

    # Build params from config + dynamic class weight
    params = {**MODEL_PARAMS, 'scale_pos_weight': scale_pos_weight}

    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )

    logger.info("Model training complete!")
    return model


# ============================================================
# STEP 6: CROSS VALIDATION
# ============================================================

def run_cross_validation(model: xgb.XGBClassifier,
                          X: pd.DataFrame,
                          y: pd.Series) -> Dict:
    """
    Runs 5-fold stratified cross validation.
    This proves the model generalizes — not just memorizing
    training data. Critical for credibility with stakeholders.

    Each fold trains on 80% and tests on 20%, rotating 5 times.
    """
    logger.info("Running 5-fold cross validation...")

    cv = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=RANDOM_STATE)

    cv_scores = cross_val_score(model, X, y,
                                cv=cv,
                                scoring='roc_auc',
                                n_jobs=-1)

    results = {
        'mean_auc': cv_scores.mean(),
        'std_auc': cv_scores.std(),
        'all_scores': cv_scores
    }

    logger.info(f"CV AUC: {results['mean_auc']:.4f} "
                f"(+/- {results['std_auc']:.4f})")

    return results


# ============================================================
# STEP 7: EVALUATE MODEL
# ============================================================

def evaluate_model(model: xgb.XGBClassifier,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    feature_names: list) -> Dict:
    """
    Comprehensive model evaluation:
    - AUC-ROC: Overall discrimination ability
    - Precision/Recall: Quality of positive predictions
    - Confusion Matrix: Breakdown of errors
    - Lift: How much better than random
    - SHAP: Feature importance explanations
    """
    logger.info("Evaluating model on test set...")

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= HIGH_PRIORITY_THRESHOLD).astype(int)

    # Core metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Test AUC-ROC: {auc:.4f}")

    # Classification report
    print("\n📊 CLASSIFICATION REPORT:")
    print("=" * 50)
    print(classification_report(y_test, y_pred,
                                 target_names=['Not Converted',
                                               'Converted']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 CONFUSION MATRIX:")
    print(f"  True Negatives:  {cm[0][0]:,} (correctly identified cold leads)")
    print(f"  False Positives: {cm[0][1]:,} (cold leads wrongly flagged hot)")
    print(f"  False Negatives: {cm[1][0]:,} (hot leads we missed)")
    print(f"  True Positives:  {cm[1][1]:,} (correctly identified hot leads)")

    # Business lift calculation
    # How much better are we vs randomly calling leads?
    baseline_rate = y_test.mean()
    top_decile = pd.DataFrame({
        'proba': y_pred_proba,
        'actual': y_test
    }).sort_values('proba', ascending=False)

    top_10pct = top_decile.head(int(len(top_decile) * 0.1))
    top_decile_conversion = top_10pct['actual'].mean()
    lift = top_decile_conversion / baseline_rate

    print(f"\n📈 BUSINESS LIFT ANALYSIS:")
    print(f"  Baseline conversion rate: {baseline_rate:.1%}")
    print(f"  Top 10% leads conversion: {top_decile_conversion:.1%}")
    print(f"  Lift: {lift:.1f}x better than random")
    print(f"  Translation: By calling only top 10% of leads,")
    print(f"  we get {lift:.1f}x more conversions per call made")

    return {
        'auc': auc,
        'lift': lift,
        'y_pred_proba': y_pred_proba,
        'baseline_rate': baseline_rate
    }


# ============================================================
# STEP 8: PLOT RESULTS
# ============================================================

def plot_model_results(model: xgb.XGBClassifier,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        y_pred_proba: np.ndarray,
                        feature_names: list) -> None:
    """
    Creates four key visualizations:
    1. ROC Curve — model discrimination
    2. Score Distribution — how scores separate converters
    3. SHAP Feature Importance — why the model decides
    4. Lift Curve — business value at each threshold
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Plot 1: ROC Curve ─────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, color='#2ecc71', lw=2,
                    label=f'AUC = {auc:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1,
                    label='Random baseline')
    axes[0, 0].fill_between(fpr, tpr, alpha=0.1, color='#2ecc71')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0, 0].legend()

    # ── Plot 2: Score Distribution ────────────────────────────
    converted_scores = y_pred_proba[y_test == 1]
    not_converted_scores = y_pred_proba[y_test == 0]
    axes[0, 1].hist(not_converted_scores, bins=50, alpha=0.6,
                    color='#e74c3c', label='Not Converted', density=True)
    axes[0, 1].hist(converted_scores, bins=50, alpha=0.6,
                    color='#2ecc71', label='Converted', density=True)
    axes[0, 1].axvline(x=HIGH_PRIORITY_THRESHOLD, color='black',
                       linestyle='--', label=f'Hot threshold '
                       f'({HIGH_PRIORITY_THRESHOLD})')
    axes[0, 1].axvline(x=MEDIUM_PRIORITY_THRESHOLD, color='gray',
                       linestyle='--', label=f'Warm threshold '
                       f'({MEDIUM_PRIORITY_THRESHOLD})')
    axes[0, 1].set_xlabel('Predicted Conversion Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Score Distribution by Outcome',
                          fontsize=13, fontweight='bold')
    axes[0, 1].legend()

    # ── Plot 3: Feature Importance ────────────────────────────
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True).tail(15)

    axes[1, 0].barh(importance.index, importance.values,
                    color='steelblue')
    axes[1, 0].set_title('Top 15 Feature Importances',
                          fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Importance Score')

    # ── Plot 4: Lift Curve ────────────────────────────────────
    df_lift = pd.DataFrame({
        'proba': y_pred_proba,
        'actual': y_test.values
    }).sort_values('proba', ascending=False)

    baseline = y_test.mean()
    cumulative_lift = []
    percentiles = range(1, 101)

    for p in percentiles:
        n = int(len(df_lift) * p / 100)
        lift = df_lift.head(n)['actual'].mean() / baseline
        cumulative_lift.append(lift)

    axes[1, 1].plot(percentiles, cumulative_lift,
                    color='#2ecc71', lw=2, label='FIN-Score Lift')
    axes[1, 1].axhline(y=1, color='red', linestyle='--',
                        label='Random baseline')
    axes[1, 1].fill_between(percentiles, cumulative_lift, 1,
                             where=[l > 1 for l in cumulative_lift],
                             alpha=0.1, color='#2ecc71')
    axes[1, 1].set_xlabel('% of Leads Called')
    axes[1, 1].set_ylabel('Lift over Random')
    axes[1, 1].set_title('Cumulative Lift Curve',
                          fontsize=13, fontweight='bold')
    axes[1, 1].legend()

    plt.suptitle('FIN-Score Model Performance',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/model_performance.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved model performance plots")


# ============================================================
# STEP 9: SAVE MODEL
# ============================================================

def save_model(model: xgb.XGBClassifier,
               feature_names: list) -> None:
    """
    Serializes trained model and feature names to disk.
    The API loads these files to score new leads in real time.
    pickle stores the exact trained model object.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, 'fin_score_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    # Save feature names — API needs exact same columns
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"Feature names saved to {features_path}")


# ============================================================
# MASTER TRAINING PIPELINE
# ============================================================

def run_training_pipeline() -> Dict:
    """
    Runs the complete model training pipeline.
    Called by notebooks and directly from terminal.

    Returns:
        Dictionary with model, metrics, and feature names
    """
    logger.info("=" * 50)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 50)

    # Load and prepare
    df = load_processed_data()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    model = train_xgboost(X_train, y_train, scale_pos_weight)

    # Evaluate
    cv_results = run_cross_validation(model, X, y)
    eval_results = evaluate_model(model, X_test,
                                   y_test, feature_names)

    # Visualize
    plot_model_results(model, X_test, y_test,
                       eval_results['y_pred_proba'],
                       feature_names)

    # Save
    save_model(model, feature_names)

    logger.info("=" * 50)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Final AUC:  {eval_results['auc']:.4f}")
    logger.info(f"Lift:       {eval_results['lift']:.1f}x")
    logger.info("=" * 50)

    return {
        'model': model,
        'feature_names': feature_names,
        'auc': eval_results['auc'],
        'lift': eval_results['lift'],
        'cv_results': cv_results
    }


# ============================================================
# TEST — Run directly to train model
# ============================================================
if __name__ == "__main__":
    results = run_training_pipeline()
    print(f"\n🎯 FIN-Score Model Ready!")
    print(f"   AUC:  {results['auc']:.4f}")
    print(f"   Lift: {results['lift']:.1f}x")
    print(f"   CV:   {results['cv_results']['mean_auc']:.4f} "
          f"(+/- {results['cv_results']['std_auc']:.4f})")