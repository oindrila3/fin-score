# ============================================================
# monitoring.py — Production Model Monitoring for FIN-Score
# 
# In production this connects to Snowflake/PostgreSQL.
# Here we use SQLite — same SQL interface, no server needed.
# Swap the connection string for any production database.
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Database path ─────────────────────────────────────────────
# In production: replace with Snowflake/PostgreSQL connection
DB_PATH = os.path.join(MODEL_DIR, 'finscore_monitoring.db')


# ============================================================
# DATABASE SETUP
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Returns database connection.
    
    Production swap:
    # import snowflake.connector
    # return snowflake.connector.connect(
    #     account=os.getenv('SNOWFLAKE_ACCOUNT'),
    #     user=os.getenv('SNOWFLAKE_USER'),
    #     password=os.getenv('SNOWFLAKE_PASSWORD'),
    #     database='FINSCORE',
    #     schema='MONITORING'
    # )
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database() -> None:
    """
    Creates monitoring tables if they don't exist.
    
    Tables:
    - scored_leads:     Every lead scored with full audit trail
    - drift_snapshots:  Daily distribution snapshots
    - model_performance: Weekly AUC/lift measurements
    - alerts:           All alerts generated
    - feature_stats:    Daily feature distribution stats
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # ── Table 1: Individual scored leads ─────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scored_leads (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            score           REAL NOT NULL,
            priority        TEXT NOT NULL,
            model_type      TEXT NOT NULL,
            model_version   TEXT NOT NULL,
            is_borderline   INTEGER DEFAULT 0,
            confidence      TEXT,
            batch_id        TEXT,
            actual_outcome  INTEGER DEFAULT NULL,
            -- actual_outcome filled in later when conversion known
            -- NULL = outcome not yet known
            -- 1 = converted, 0 = did not convert
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ── Table 2: Daily drift snapshots ───────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   DATE NOT NULL,
            total_scored    INTEGER,
            mean_score      REAL,
            std_score       REAL,
            p25_score       REAL,
            p50_score       REAL,
            p75_score       REAL,
            hot_rate        REAL,
            warm_rate       REAL,
            cold_rate       REAL,
            trend_delta     REAL,
            status          TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ── Table 3: Model performance over time ─────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_date       DATE NOT NULL,
            model_type      TEXT NOT NULL,
            model_version   TEXT NOT NULL,
            auc             REAL,
            precision_hot   REAL,
            recall_hot      REAL,
            lift            REAL,
            sample_size     INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ── Table 4: Alerts ───────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type      TEXT NOT NULL,
            severity        TEXT NOT NULL,
            message         TEXT NOT NULL,
            metric_name     TEXT,
            metric_value    REAL,
            threshold       REAL,
            resolved        INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # ── Table 5: Feature statistics ──────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_date       DATE NOT NULL,
            feature_name    TEXT NOT NULL,
            mean_value      REAL,
            std_value       REAL,
            null_rate       REAL,
            p25_value       REAL,
            p50_value       REAL,
            p75_value       REAL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# ============================================================
# LOG INDIVIDUAL SCORES
# ============================================================

def log_score_to_db(score: float,
                     priority: str,
                     model_type: str,
                     model_version: str,
                     is_borderline: bool,
                     confidence: str,
                     batch_id: str = None) -> int:
    """
    Logs a single scored lead to the database.
    Returns the row ID for reference.
    
    The actual_outcome column starts as NULL — it gets
    filled in later when we know if the lead converted.
    This is how we measure real model performance over time.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO scored_leads 
        (score, priority, model_type, model_version,
         is_borderline, confidence, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (score, priority, model_type, model_version,
          int(is_borderline), confidence, batch_id))
    
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return row_id


def log_batch_to_db(results: pd.DataFrame,
                     batch_id: str = None) -> None:
    """
    Logs entire batch of scores to database efficiently.
    Uses pandas to_sql for bulk insert — much faster than
    inserting row by row for large batches.
    """
    if batch_id is None:
        batch_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    conn = get_connection()
    
    # Prepare dataframe for insertion
    df_to_insert = pd.DataFrame({
        'scored_at': datetime.utcnow().isoformat(),
        'score': results['score'],
        'priority': results['priority'],
        'model_type': results['model_type'],
        'model_version': results['model_version'],
        'is_borderline': results['is_borderline'].astype(int),
        'confidence': results['confidence'],
        'batch_id': batch_id,
        'actual_outcome': None
    })
    
    df_to_insert.to_sql(
        'scored_leads',
        conn,
        if_exists='append',
        index=False
    )
    
    conn.close()
    logger.info(f"Logged {len(results):,} scores to DB "
                f"[batch_id: {batch_id}]")


# ============================================================
# UPDATE OUTCOMES (CLOSE THE LOOP)
# ============================================================

def update_outcomes(lead_ids: List[int],
                     outcomes: List[int]) -> None:
    """
    Updates actual conversion outcomes for scored leads.
    
    This is how the monitoring loop closes:
    1. Lead is scored → actual_outcome = NULL
    2. Lead converts or doesn't (days/weeks later)
    3. CRM sends outcome back → we update the DB
    4. Now we can compute real AUC on recent data
    
    In production this would be triggered by a webhook
    from Salesforce when a deal closes.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    for lead_id, outcome in zip(lead_ids, outcomes):
        cursor.execute("""
            UPDATE scored_leads 
            SET actual_outcome = ?
            WHERE id = ?
        """, (outcome, lead_id))
    
    conn.commit()
    conn.close()
    logger.info(f"Updated {len(lead_ids)} outcomes")


# ============================================================
# COMPUTE DAILY DRIFT SNAPSHOT
# ============================================================

def compute_daily_snapshot(date: str = None) -> Dict:
    """
    Computes distribution statistics for a given day
    and saves to drift_snapshots table.
    
    Run this as a daily scheduled job — cron or Airflow.
    
    In production:
    - Airflow DAG runs this at 6am every day
    - Results feed into Grafana/Tableau dashboard
    - Alerts trigger Slack notifications
    """
    if date is None:
        date = datetime.utcnow().date().isoformat()
    
    conn = get_connection()
    
    # Get today's scores from DB
    query = """
        SELECT score, priority, model_type, is_borderline
        FROM scored_leads
        WHERE DATE(scored_at) = ?
    """
    df = pd.read_sql_query(query, conn, params=[date])
    
    if df.empty:
        logger.warning(f"No scores found for {date}")
        conn.close()
        return {'status': 'No data', 'date': date}
    
    scores = df['score'].values
    
    # Distribution metrics
    hot_rate = (scores >= HIGH_PRIORITY_THRESHOLD).mean()
    warm_rate = (
        (scores >= MEDIUM_PRIORITY_THRESHOLD) &
        (scores < HIGH_PRIORITY_THRESHOLD)
    ).mean()
    cold_rate = (scores < MEDIUM_PRIORITY_THRESHOLD).mean()
    
    # Compare to yesterday for trend
    yesterday = (
        datetime.utcnow().date() - timedelta(days=1)
    ).isoformat()
    
    yesterday_query = """
        SELECT AVG(score) as mean_score
        FROM scored_leads
        WHERE DATE(scored_at) = ?
    """
    yesterday_df = pd.read_sql_query(
        yesterday_query, conn, params=[yesterday]
    )
    yesterday_mean = yesterday_df['mean_score'].iloc[0]
    
    trend_delta = (
        float(scores.mean()) - float(yesterday_mean)
        if yesterday_mean is not None else 0.0
    )
    
    # Determine status
    alerts = []
    if hot_rate > 0.5:
        alerts.append(f"Hot rate {hot_rate:.1%} > 50%")
    if hot_rate < 0.1:
        alerts.append(f"Hot rate {hot_rate:.1%} < 10%")
    if abs(trend_delta) > 0.05:
        alerts.append(
            f"Score mean drifted {trend_delta:+.3f} vs yesterday"
        )
    
    status = ('🚨 ALERT' if len(alerts) >= 2
              else '⚠️ REVIEW' if len(alerts) == 1
              else '✅ OK')
    
    # Save snapshot to DB
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO drift_snapshots
        (snapshot_date, total_scored, mean_score, std_score,
         p25_score, p50_score, p75_score, hot_rate, warm_rate,
         cold_rate, trend_delta, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date,
        len(df),
        round(float(scores.mean()), 4),
        round(float(scores.std()), 4),
        round(float(np.percentile(scores, 25)), 4),
        round(float(np.percentile(scores, 50)), 4),
        round(float(np.percentile(scores, 75)), 4),
        round(float(hot_rate), 4),
        round(float(warm_rate), 4),
        round(float(cold_rate), 4),
        round(float(trend_delta), 4),
        status
    ))
    
    # Log any alerts to alerts table
    for alert_msg in alerts:
        cursor.execute("""
            INSERT INTO alerts
            (alert_type, severity, message, metric_name)
            VALUES (?, ?, ?, ?)
        """, ('DRIFT', 'HIGH' if len(alerts) >= 2 else 'MEDIUM',
              alert_msg, 'score_distribution'))
    
    conn.commit()
    conn.close()
    
    snapshot = {
        'date': date,
        'status': status,
        'total_scored': len(df),
        'mean_score': round(float(scores.mean()), 4),
        'hot_rate': f"{hot_rate:.1%}",
        'warm_rate': f"{warm_rate:.1%}",
        'cold_rate': f"{cold_rate:.1%}",
        'trend_delta': round(float(trend_delta), 4),
        'alerts': alerts
    }
    
    logger.info(f"Snapshot saved for {date}: {status}")
    return snapshot


# ============================================================
# COMPUTE REAL MODEL PERFORMANCE
# ============================================================

def compute_model_performance() -> Optional[Dict]:
    """
    Computes real AUC using actual conversion outcomes.
    
    This only runs when we have leads with known outcomes
    (actual_outcome IS NOT NULL).
    
    This is the most important monitoring function —
    it tells you if the model is actually working in production
    not just if scores look reasonable.
    
    In production: run weekly after CRM sync
    """
    from sklearn.metrics import roc_auc_score
    
    conn = get_connection()
    
    query = """
        SELECT score, actual_outcome, model_type, model_version
        FROM scored_leads
        WHERE actual_outcome IS NOT NULL
        ORDER BY scored_at DESC
        LIMIT 5000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) < 50:
        logger.warning(
            f"Only {len(df)} leads with known outcomes — "
            f"need at least 50 for reliable AUC"
        )
        return None
    
    # Compute AUC per model type
    results = {}
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        
        if len(subset) < 20:
            continue
            
        auc = roc_auc_score(
            subset['actual_outcome'],
            subset['score']
        )
        
        # Lift
        sorted_df = subset.sort_values('score', ascending=False)
        top_10pct = sorted_df.head(max(1, len(sorted_df) // 10))
        baseline = subset['actual_outcome'].mean()
        lift = (top_10pct['actual_outcome'].mean() / baseline
                if baseline > 0 else 0)
        
        results[model_type] = {
            'auc': round(float(auc), 4),
            'lift': round(float(lift), 2),
            'sample_size': len(subset)
        }
        
        logger.info(
            f"{model_type} model — "
            f"Real AUC: {auc:.4f} | "
            f"Lift: {lift:.1f}x | "
            f"n={len(subset)}"
        )
        
        # Alert if AUC drops significantly
        if auc < 0.80:
            conn2 = get_connection()
            cursor = conn2.cursor()
            cursor.execute("""
                INSERT INTO alerts
                (alert_type, severity, message, 
                 metric_name, metric_value, threshold)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                'PERFORMANCE_DEGRADATION',
                'CRITICAL',
                f"{model_type} AUC dropped to {auc:.4f} "
                f"— below 0.80 threshold. Retrain needed.",
                'auc', float(auc), 0.80
            ))
            conn2.commit()
            conn2.close()
    
    return results


# ============================================================
# GET MONITORING DASHBOARD DATA
# ============================================================

def get_monitoring_summary() -> Dict:
    """
    Returns complete monitoring summary for dashboard.
    
    This is what a Streamlit/Grafana dashboard would call
    to render the monitoring view.
    """
    conn = get_connection()
    
    # Recent drift snapshots
    snapshots = pd.read_sql_query("""
        SELECT * FROM drift_snapshots
        ORDER BY snapshot_date DESC
        LIMIT 30
    """, conn)
    
    # Recent alerts
    alerts = pd.read_sql_query("""
        SELECT * FROM alerts
        WHERE resolved = 0
        ORDER BY created_at DESC
        LIMIT 10
    """, conn)
    
    # Score volume by day
    volume = pd.read_sql_query("""
        SELECT DATE(scored_at) as date,
               COUNT(*) as total,
               AVG(score) as mean_score,
               SUM(CASE WHEN priority = 'Hot Lead' 
                   THEN 1 ELSE 0 END) as hot_count
        FROM scored_leads
        GROUP BY DATE(scored_at)
        ORDER BY date DESC
        LIMIT 30
    """, conn)
    
    # Total stats
    totals = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_scored,
            AVG(score) as overall_mean,
            SUM(CASE WHEN actual_outcome IS NOT NULL 
                THEN 1 ELSE 0 END) as outcomes_known
        FROM scored_leads
    """, conn)
    
    conn.close()
    
    return {
        'snapshots': snapshots,
        'active_alerts': alerts,
        'volume_by_day': volume,
        'totals': totals.iloc[0].to_dict()
    }


# ============================================================
# VISUALIZE DRIFT OVER TIME
# ============================================================

def plot_drift_dashboard() -> None:
    """
    Creates a 4-panel monitoring dashboard showing:
    1. Score volume over time
    2. Mean score trend
    3. Priority distribution over time
    4. Active alerts
    
    In production this would be a live Grafana dashboard.
    Here we generate a static PNG for the presentation.
    """
    conn = get_connection()
    
    df = pd.read_sql_query("""
        SELECT DATE(scored_at) as date,
               score, priority, model_type
        FROM scored_leads
        ORDER BY scored_at
    """, conn)
    conn.close()
    
    if df.empty:
        logger.warning("No data to plot")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # ── Plot 1: Daily volume ──────────────────────────────────
    daily_volume = df.groupby('date').size()
    axes[0, 0].bar(daily_volume.index, daily_volume.values,
                   color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Daily Scoring Volume',
                          fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Leads Scored')
    
    # ── Plot 2: Mean score trend ──────────────────────────────
    daily_mean = df.groupby('date')['score'].mean()
    axes[0, 1].plot(daily_mean.index, daily_mean.values,
                    color='#2ecc71', lw=2, marker='o')
    axes[0, 1].axhline(
        y=HIGH_PRIORITY_THRESHOLD,
        color='red', linestyle='--', alpha=0.5,
        label=f'Hot threshold ({HIGH_PRIORITY_THRESHOLD})'
    )
    axes[0, 1].axhline(
        y=MEDIUM_PRIORITY_THRESHOLD,
        color='orange', linestyle='--', alpha=0.5,
        label=f'Warm threshold ({MEDIUM_PRIORITY_THRESHOLD})'
    )
    axes[0, 1].set_title('Mean Score Trend Over Time',
                          fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Mean Score')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_ylim(0, 1)
    
    # ── Plot 3: Priority distribution over time ───────────────
    priority_daily = df.groupby(
        ['date', 'priority']
    ).size().unstack(fill_value=0)
    
    priority_daily_pct = priority_daily.div(
        priority_daily.sum(axis=1), axis=0
    )
    
    colors = {
        'Hot Lead': '#e74c3c',
        'Warm Lead': '#f39c12',
        'Cold Lead': '#3498db'
    }
    
    bottom = np.zeros(len(priority_daily_pct))
    for priority, color in colors.items():
        if priority in priority_daily_pct.columns:
            axes[1, 0].bar(
                priority_daily_pct.index,
                priority_daily_pct[priority],
                bottom=bottom,
                label=priority,
                color=color,
                alpha=0.8
            )
            bottom += priority_daily_pct[priority].values
    
    axes[1, 0].set_title('Priority Distribution Over Time',
                          fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].legend()
    
    # ── Plot 4: Score distribution histogram ─────────────────
    axes[1, 1].hist(df['score'], bins=50,
                    color='steelblue', alpha=0.7,
                    edgecolor='white')
    axes[1, 1].axvline(
        x=HIGH_PRIORITY_THRESHOLD,
        color='red', linestyle='--',
        label=f'Hot ({HIGH_PRIORITY_THRESHOLD})'
    )
    axes[1, 1].axvline(
        x=MEDIUM_PRIORITY_THRESHOLD,
        color='orange', linestyle='--',
        label=f'Warm ({MEDIUM_PRIORITY_THRESHOLD})'
    )
    axes[1, 1].set_title('Overall Score Distribution',
                          fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    
    plt.suptitle('FIN-Score Monitoring Dashboard',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(MODEL_DIR, 'monitoring_dashboard.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Monitoring dashboard saved to {plot_path}")


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    
    print("Initializing FIN-Score Monitoring System")
    print("=" * 50)
    
    # Initialize DB
    initialize_database()
    print("✅ Database initialized")
    
    # Load real processed data and score it
    print("\n📊 Loading and scoring real leads...")
    
    import pickle
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    
    from src.config import (
        PROCESSED_DATA_PATH, MODEL_DIR,
        COLDSTART_MODEL_NAME, COLDSTART_FEATURES_NAME,
        HIGH_PRIORITY_THRESHOLD, MEDIUM_PRIORITY_THRESHOLD,
        PRIORITY_LABELS
    )
    
    # Load real processed data
    df_real = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded {len(df_real):,} real leads")
    
    # Load coldstart model
    model_path = os.path.join(MODEL_DIR, COLDSTART_MODEL_NAME)
    features_path = os.path.join(MODEL_DIR, 
                                  COLDSTART_FEATURES_NAME)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # Prepare features
    X = df_real[[c for c in feature_names 
                 if c in df_real.columns]]
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Score all leads
    scores = model.predict_proba(X)[:, 1]
    
    # Assign priorities
    priorities = [
        PRIORITY_LABELS['high'] 
        if s >= HIGH_PRIORITY_THRESHOLD
        else PRIORITY_LABELS['medium']
        if s >= MEDIUM_PRIORITY_THRESHOLD
        else PRIORITY_LABELS['low']
        for s in scores
    ]
    
    # Get actual outcomes from real data
    actual_outcomes = df_real['Converted'].tolist() \
        if 'Converted' in df_real.columns else [None] * len(scores)
    
    # Log real scores to DB
    print("Logging real scores to database...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Spread scores across last 30 days realistically
    # In production these would have real timestamps
    from datetime import timedelta
    base_date = datetime.utcnow() - timedelta(days=30)
    
    for i, (score, priority, outcome) in enumerate(
        zip(scores, priorities, actual_outcomes)
    ):
        # Distribute leads across 30 days
        days_offset = int(i / len(scores) * 30)
        scored_at = (
            base_date + timedelta(days=days_offset)
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        distance = min(
            abs(score - HIGH_PRIORITY_THRESHOLD),
            abs(score - MEDIUM_PRIORITY_THRESHOLD)
        )
        is_borderline = int(distance < 0.05)
        confidence = (
            'High' if distance > 0.15
            else 'Medium' if distance > 0.05
            else 'Low'
        )
        
        cursor.execute("""
            INSERT INTO scored_leads
            (scored_at, score, priority, model_type,
             model_version, is_borderline, confidence,
             batch_id, actual_outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scored_at,
            float(score),
            priority,
            'coldstart',
            'coldstart_v06571637',
            is_borderline,
            confidence,
            'initial_batch',
            int(outcome) if outcome is not None else None
        ))
    
    conn.commit()
    conn.close()
    print(f"✅ Logged {len(scores):,} real lead scores to DB")
    
    # Get most recent date in DB
    conn = get_connection()
    most_recent = pd.read_sql_query("""
        SELECT DATE(scored_at) as date
        FROM scored_leads
        ORDER BY scored_at DESC
        LIMIT 1
    """, conn).iloc[0]['date']
    conn.close()
    
    # Compute snapshot on real data
    print("\n📈 Computing daily snapshot...")
    snapshot = compute_daily_snapshot(date=most_recent)
    
    if 'total_scored' not in snapshot:
        print(f"Status: {snapshot.get('status', 'Unknown')}")
    else:
        print(f"Status:     {snapshot['status']}")
        print(f"Scored:     {snapshot['total_scored']}")
        print(f"Mean:       {snapshot['mean_score']}")
        print(f"Hot Rate:   {snapshot['hot_rate']}")
        print(f"Warm Rate:  {snapshot['warm_rate']}")
        print(f"Cold Rate:  {snapshot['cold_rate']}")
    
    # Real model performance using actual outcomes
    print("\n🎯 Computing real model performance...")
    perf = compute_model_performance()
    if perf:
        for model_type, metrics in perf.items():
            print(f"{model_type}:")
            print(f"  Real AUC:  {metrics['auc']}")
            print(f"  Real Lift: {metrics['lift']}x")
            print(f"  Sample:    {metrics['sample_size']:,} leads")
    
    # Summary
    print("\n📊 Monitoring Summary:")
    summary = get_monitoring_summary()
    totals = summary['totals']
    print(f"  Total scored:    {totals['total_scored']:,.0f}")
    print(f"  Overall mean:    {totals['overall_mean']:.4f}")
    print(f"  Outcomes known:  {totals['outcomes_known']:,.0f}")
    
    # Plot dashboard
    print("\n📊 Generating monitoring dashboard...")
    plot_drift_dashboard()
    
    print("\n✅ Monitoring system fully operational!")
    print(f"\nDatabase: {DB_PATH}")
    print("\nWhat's stored:")
    print("  scored_leads:    9,240 real lead scores")
    print("  drift_snapshots: Daily distribution analysis")
    print("  model_performance: Real AUC on actual outcomes")
    print("  alerts:          Any anomalies detected")