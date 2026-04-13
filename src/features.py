# ============================================================
# features.py — Feature Engineering Pipeline for FIN-Score
# All data cleaning and transformation logic lives here
# Called by notebooks AND the production scoring API
# ============================================================

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

import sys
sys.path.append('..')
from src.config import *

# ── Logging Setup ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# STEP 1: COLUMNS TO DROP
# ============================================================

COLUMNS_TO_DROP = [
    'Prospect ID',
    'Lead Number',
    'Do Not Call',
    'Receive More Updates About Our Courses',
    'Update me on Supply Chain Content',
    'Get updates on DM Content',
    'I agree to pay the amount through cheque',
    'How did you hear about X Education',
    'What matters most to you in choosing a course',
]

BINARY_YES_NO_COLUMNS = [
    'Do Not Email',
    'Search',
    'Magazine',
    'Newspaper Article',
    'X Education Forums',
    'Newspaper',
    'Digital Advertisement',
    'Through Recommendations',
    'A free copy of Mastering The Interview',
]

ASYMMETRIQUE_INDEX_COLUMNS = [
    'Asymmetrique Activity Index',
    'Asymmetrique Profile Index',
]

CATEGORICAL_COLUMNS = [
    'Lead Origin',
    'Lead Source',
    'Last Activity',
    'Last Notable Activity',
    'Specialization',
    'What is your current occupation',
    'City',
    'Tags',
    'Lead Quality',
    'Lead Profile',
    'Country',
]

NUMERIC_COLUMNS = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Asymmetrique Activity Score',
    'Asymmetrique Profile Score',
]

# ============================================================
# STEP 2: REPLACE 'Select' WITH NaN
# ============================================================

def replace_select_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'Select' placeholder values with NaN. 
    'Select' means the user did not fill in those field.
    """ 
    logger.info("Replacing 'Select' placeholder with NaN..")
    df = df.replace('Select', np.nan)
    logger.info(f"Done. Total nulls after replacement:"
                f"{df.isnull().sum().sum():,}")
    return df

# ============================================================
# STEP 3: DROP IRRELEVANT COLUMNS
# ============================================================

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that carry no predictive signal:
    - ID columns
    - Columns with near-zero variance
    - Columns with >80% missing after Select replacement
    """

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    logger.info(f"Dropping {len(cols_to_drop)} irrelevant columns...")
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Remaining columns: {df.shape[1]}")
    return df


# ============================================================
# STEP 4: ENCODE BINARY YES/NO COLUMNS
# ============================================================

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Yes/No columns to binary 1/0.
    Makes these features usable by XGBoost directly.
    """
    logger.info("Encoding binary Yes/No columns...")
    for col in BINARY_YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df


# ============================================================
# STEP 5: ENCODE ASYMMETRIQUE INDEX (ORDINAL)
# ============================================================

def encode_asymmetrique_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts ordinal index columns to numeric.
    '01.High' > '02.Medium' > '03.Low'
    Reversed so higher number = higher value.
    """
    logger.info("Encoding Asymmetrique index columns...")
    ordinal_map = {
        '01.High': 3,
        '02.Medium': 2,
        '03.Low': 1
    }
    for col in ASYMMETRIQUE_INDEX_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(ordinal_map)
    return df


# ============================================================
# STEP 6: IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values by column type:
    - Numeric columns: median imputation (robust to outliers)
    - Categorical columns: mode imputation
    """
    logger.info("Imputing missing values...")

    # Numeric — median
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: filled with median={median_val:.2f}")

    # Asymmetrique index — median after encoding
    for col in ASYMMETRIQUE_INDEX_COLUMNS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical — mode
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: filled with mode='{mode_val}'")

    return df


# ============================================================
# STEP 7: ENCODE CATEGORICAL COLUMNS
# ============================================================

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies frequency encoding to high-cardinality categoricals.
    Frequency encoding replaces categories with their occurrence
    rate — preserving signal without exploding dimensionality.
    """
    logger.info("Frequency encoding categorical columns...")
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col + '_freq_encoded'] = df[col].map(freq_map)
            df = df.drop(columns=[col])
            logger.info(f"  {col}: frequency encoded")
    return df


# ============================================================
# STEP 8: ENGINEER NEW FEATURES
# ============================================================

def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from existing ones.
    These derived signals often outperform raw features.
    """
    logger.info("Engineering new features...")

    # Engagement score — composite behavioral signal
    if all(c in df.columns for c in ['TotalVisits',
                                      'Total Time Spent on Website',
                                      'Page Views Per Visit']):
        df['engagement_score'] = (
            df['TotalVisits'] * 0.3 +
            df['Total Time Spent on Website'] * 0.5 +
            df['Page Views Per Visit'] * 0.2
        )
        logger.info("  Created: engagement_score")

    # Asymmetrique composite score
    if all(c in df.columns for c in ['Asymmetrique Activity Score',
                                      'Asymmetrique Profile Score']):
        df['asymmetrique_combined'] = (
            df['Asymmetrique Activity Score'] +
            df['Asymmetrique Profile Score']
        ) / 2
        logger.info("  Created: asymmetrique_combined")

    # High engagement flag
    if 'Total Time Spent on Website' in df.columns:
        threshold = df['Total Time Spent on Website'].median()
        df['is_high_engagement'] = (
            df['Total Time Spent on Website'] > threshold
        ).astype(int)
        logger.info("  Created: is_high_engagement")

    return df


# ============================================================
# STEP 9: REMOVE OUTLIERS
# ============================================================

def remove_outliers(df: pd.DataFrame,
                    cols: List[str] = None) -> pd.DataFrame:
    """
    Caps outliers at 99th percentile (Winsorization).
    Prevents extreme values from distorting the model
    without losing the rows entirely.
    """
    if cols is None:
        cols = NUMERIC_COLUMNS

    logger.info("Capping outliers at 99th percentile...")
    for col in cols:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            original_max = df[col].max()
            df[col] = df[col].clip(upper=cap)
            if original_max > cap:
                logger.info(f"  {col}: capped {original_max:.1f} → {cap:.1f}")
    return df


# ============================================================
# MASTER PIPELINE FUNCTION
# ============================================================

def run_feature_pipeline(df: pd.DataFrame,
                          save: bool = True) -> pd.DataFrame:
    """
    Runs the complete feature engineering pipeline in order.
    This is the single function called by train.py and the API.

    Args:
        df: Raw leads DataFrame
        save: Whether to save processed data to disk

    Returns:
        Fully processed DataFrame ready for modeling
    """
    logger.info("=" * 50)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 50)

    df = df.copy()
    df = replace_select_with_nan(df)
    df = drop_irrelevant_columns(df)
    df = encode_binary_columns(df)
    df = encode_asymmetrique_index(df)
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = engineer_new_features(df)
    df = encode_categorical_columns(df)

    logger.info("=" * 50)
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Final columns: {df.shape[1]}")
    logger.info("=" * 50)

    if save:
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved to {PROCESSED_DATA_PATH}")

    return df

# ============================================================
# TEST — Run this file directly to validate pipeline
# ============================================================
if __name__ == "__main__":

    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw shape: {df.shape}")

    print("\nRunning pipeline...")
    df_processed = run_feature_pipeline(df, save=True)

    print(f"\n✅ SUCCESS")
    print(f"Raw shape:       {df.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print(f"\nFinal columns:")
    for col in df_processed.columns:
        print(f"  - {col}")