# ============================================================
# dashboard/app.py -- FIN-Score Dashboard
# Two audiences: Executive View and Data Scientist View
# Scores leads directly (no API dependency)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pickle
import json
import os
import sys
import shap
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, PROCESSED_DATA_PATH,
    HIGH_PRIORITY_THRESHOLD, MEDIUM_PRIORITY_THRESHOLD,
    PRIORITY_LABELS, COLDSTART_MODEL_NAME,
    COLDSTART_FEATURES_NAME, PIPELINE_MODEL_NAME,
    PIPELINE_FEATURES_NAME
)

st.set_page_config(
    page_title="FIN-Score | B2B Lead Scoring",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252940);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #00d4aa;
        margin: 5px 0;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2em;
        font-weight: 700;
        color: #00d4aa;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.8em;
        color: #8892a4;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta {
        font-size: 0.85em;
        color: #2ecc71;
        margin-top: 5px;
    }
    .section-header {
        font-size: 1.4em;
        font-weight: 700;
        color: #ffffff;
        border-bottom: 2px solid #00d4aa;
        padding-bottom: 8px;
        margin-bottom: 20px;
        margin-top: 10px;
    }
    .roi-box {
        background: linear-gradient(135deg, #1a2744, #1e3a5f);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2980b9;
        text-align: center;
        margin: 5px 0;
    }
    .roi-value {
        font-size: 1.8em;
        font-weight: 800;
        color: #3498db;
        margin: 0;
    }
    .roi-label {
        font-size: 0.8em;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_models():
    models = {}
    for model_type, model_name, features_name in [
        ('pipeline', PIPELINE_MODEL_NAME, PIPELINE_FEATURES_NAME),
        ('coldstart', COLDSTART_MODEL_NAME, COLDSTART_FEATURES_NAME)
    ]:
        model_path = os.path.join(MODEL_DIR, model_name)
        features_path = os.path.join(MODEL_DIR, features_name)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            models[model_type] = {
                'model': model,
                'feature_names': feature_names
            }
    return models


@st.cache_data(ttl=300)
def load_monitoring_data():
    db_path = os.path.join(MODEL_DIR, 'finscore_monitoring.db')
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT scored_at, score, priority,
               model_type, is_borderline,
               confidence, actual_outcome
        FROM scored_leads
        ORDER BY scored_at DESC
    """, conn)
    conn.close()
    df['scored_at'] = pd.to_datetime(df['scored_at'], format='mixed')
    return df


@st.cache_data(ttl=300)
def load_metadata():
    metadata = {}
    for model_type in ['pipeline', 'coldstart']:
        path = os.path.join(MODEL_DIR, f'{model_type}_metadata.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                metadata[model_type] = json.load(f)
    return metadata


@st.cache_data(ttl=60)
def load_alerts():
    db_path = os.path.join(MODEL_DIR, 'finscore_monitoring.db')
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT * FROM alerts WHERE resolved = 0
        ORDER BY created_at DESC
    """, conn)
    conn.close()
    return df


# ============================================================
# SCORING FUNCTION
# ============================================================

def score_lead_direct(lead_data: dict, model_type: str = 'coldstart') -> dict:
    models = load_models()
    if model_type not in models:
        return {'error': True, 'message': f'{model_type} model not found'}

    model = models[model_type]['model']
    feature_names = models[model_type]['feature_names']

    lead_df = pd.DataFrame([lead_data])
    for feature in feature_names:
        if feature not in lead_df.columns:
            lead_df[feature] = 0
    lead_df = lead_df[feature_names]
    lead_df = lead_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    score = float(model.predict_proba(lead_df)[0][1])

    if score >= HIGH_PRIORITY_THRESHOLD:
        priority = PRIORITY_LABELS['high']
    elif score >= MEDIUM_PRIORITY_THRESHOLD:
        priority = PRIORITY_LABELS['medium']
    else:
        priority = PRIORITY_LABELS['low']

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(lead_df)
        contributions = dict(zip(feature_names, shap_values[0]))
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        positive = [(k, v) for k, v in sorted_contribs if v > 0][:5]
        negative = [(k, v) for k, v in sorted_contribs if v < 0][:5]
    except Exception:
        positive, negative = [], []

    distance = min(abs(score - HIGH_PRIORITY_THRESHOLD), abs(score - MEDIUM_PRIORITY_THRESHOLD))
    if distance < 0.05:
        confidence = 'Low'
        is_borderline = True
    elif distance < 0.15:
        confidence = 'Medium'
        is_borderline = False
    else:
        confidence = 'High'
        is_borderline = False

    return {
        'score': round(score, 4),
        'priority': priority,
        'confidence': confidence,
        'is_borderline': is_borderline,
        'positive_drivers': positive,
        'negative_drivers': negative,
        'model_type': model_type,
        'error': False
    }


def plain_english_explanation(positive: list, negative: list, lead_data: dict) -> dict:
    feature_explanations = {
        'Total Time Spent on Website': f"Spent {lead_data.get('Total Time Spent on Website', 0):.0f} minutes on the website -- strong engagement signal",
        'TotalVisits': f"Visited the website {lead_data.get('TotalVisits', 0):.0f} times -- actively researching",
        'Page Views Per Visit': f"Viewed {lead_data.get('Page Views Per Visit', 0):.1f} pages per visit -- high intent",
        'Asymmetrique Activity Score': f"Activity score of {lead_data.get('Asymmetrique Activity Score', 0):.0f}/25 -- internal engagement metric",
        'Asymmetrique Profile Score': f"Profile score of {lead_data.get('Asymmetrique Profile Score', 0):.0f}/25 -- matches ideal customer profile",
        'asymmetrique_combined': "Combined activity and profile scores indicate strong fit",
        'is_high_engagement': "Classified as high engagement based on website behavior",
        'Lead Origin_freq_encoded': "Came through a high-converting lead channel",
        'Lead Source_freq_encoded': "Lead source historically converts well",
        'Last Activity_freq_encoded': "Most recent activity is a positive conversion signal",
        'Last Notable Activity_freq_encoded': "Last notable action suggests active interest",
        'Country_freq_encoded': "Country of origin affects conversion likelihood",
        'City_freq_encoded': "City has historically strong conversion rates",
        'Specialization_freq_encoded': "Professional specialization aligns with typical converters",
        'What is your current occupation_freq_encoded': "Occupation type is a strong conversion predictor",
        'Do Not Email': "Lead has opted out of email communication",
        'Search': "Found us through search -- high purchase intent",
        'Through Recommendations': "Referred by someone -- typically converts better",
        'was_lead_quality_assessed': "Sales team has already reviewed this lead",
        'was_lead_profile_assessed': "Lead profile has been assessed by sales team",
        'was_tags_assessed': "Sales team has added interaction notes",
    }

    plain_positive = []
    plain_negative = []

    for feature, impact in positive:
        clean = feature.replace('_freq_encoded', '').replace('_', ' ').strip()
        explanation = feature_explanations.get(feature, f"{clean.title()} is a positive signal")
        plain_positive.append(explanation)

    for feature, impact in negative:
        clean = feature.replace('_freq_encoded', '').replace('_', ' ').strip()
        explanation = feature_explanations.get(feature, f"{clean.title()} is working against conversion")
        plain_negative.append(explanation)

    return {'positive': plain_positive, 'negative': plain_negative}


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #00d4aa; margin: 0;">FIN-Score</h2>
            <p style="color: #8892a4; font-size: 0.8em; margin: 0;">B2B Lead Scoring System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Select View")
        audience = st.radio("Audience:", ["Executive", "Data Scientist"], label_visibility="collapsed")
        st.markdown("---")

        if audience == "Executive":
            st.markdown("### Executive Pages")
            page = st.radio("Navigate:", ["Business Impact and ROI", "Live Lead Scorer", "Lead Pipeline"], label_visibility="collapsed")
        else:
            st.markdown("### Data Scientist Pages")
            page = st.radio("Navigate:", ["Model Performance", "Feature Analysis", "Drift Monitoring", "Model Comparison"], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### Model")
        model_type = st.selectbox("Model variant:", ["coldstart", "pipeline"], help="coldstart: brand new leads | pipeline: leads with history")
        st.markdown("---")

        models = load_models()
        metadata = load_metadata()

        st.markdown("### System Status")
        if 'coldstart' in models:
            st.success("Cold-start model loaded")
        else:
            st.error("Cold-start model missing")

        if 'pipeline' in models:
            st.success("Pipeline model loaded")
        else:
            st.error("Pipeline model missing")

        alerts = load_alerts()
        if not alerts.empty:
            st.warning(f"{len(alerts)} active alert(s)")
        else:
            st.success("No active alerts")

        st.markdown("---")
        if metadata:
            cs = metadata.get('coldstart', {})
            st.markdown("### Model Stats")
            st.metric("Cold-start AUC", f"{cs.get('auc', 0):.4f}")
            st.metric("Lift", f"{cs.get('lift', 0)}x")
            st.metric("Version", cs.get('model_version', 'N/A'))

        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    return audience, page, model_type


# ============================================================
# EXECUTIVE PAGE 1: BUSINESS IMPACT AND ROI
# ============================================================

def render_business_impact(metadata: dict, df: pd.DataFrame):
    st.markdown('<div class="section-header">Business Impact and ROI Calculator</div>', unsafe_allow_html=True)

    cs = metadata.get('coldstart', {})
    lift = cs.get('lift', 2.5)
    auc = cs.get('auc', 0.91)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-label">Sales Efficiency</p><p class="kpi-value">{lift}x</p><p class="kpi-delta">More conversions per call</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-label">Model AUC Score</p><p class="kpi-value">{auc:.2f}</p><p class="kpi-delta">vs 0.80 industry standard</p></div>', unsafe_allow_html=True)
    with col3:
        total = len(df) if not df.empty else 0
        st.markdown(f'<div class="kpi-card"><p class="kpi-label">Leads Scored</p><p class="kpi-value">{total:,}</p><p class="kpi-delta">With full audit trail</p></div>', unsafe_allow_html=True)
    with col4:
        hot_rate = (df['priority'] == PRIORITY_LABELS['high']).mean() if not df.empty else 0.36
        st.markdown(f'<div class="kpi-card"><p class="kpi-label">Hot Lead Rate</p><p class="kpi-value">{hot_rate:.0%}</p><p class="kpi-delta">Worth immediate outreach</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">ROI Calculator</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to match your team. Numbers update instantly.")

    col_inputs, col_results = st.columns([1, 1])

    with col_inputs:
        st.markdown("#### Your Sales Team")
        daily_leads = st.slider("Daily leads arriving", min_value=100, max_value=10000, value=1000, step=100)
        pct_called_now = st.slider("Percent of leads your team currently calls", min_value=10, max_value=100, value=100, step=5, format="%d%%")
        top_pct_with_finscore = st.slider("Percent of leads to call with FIN-Score", min_value=10, max_value=80, value=36, step=1, format="%d%%", help="Default 36% equals our Hot Lead rate")
        deal_value = st.number_input("Average deal value (USD)", min_value=50, max_value=100000, value=400, step=50, help="Default 400 based on online education industry average. Set to your actual ACV.")
        hours_per_call = st.slider("Hours per sales call", min_value=0.25, max_value=2.0, value=0.5, step=0.25)

    with col_results:
        st.markdown("#### Impact Analysis")

        BASELINE_CONVERSION = 0.385
        HOT_CONVERSION = 0.995

        calls_without = int(daily_leads * (pct_called_now / 100))
        converts_without = int(calls_without * BASELINE_CONVERSION)
        revenue_without = converts_without * deal_value
        hours_without = calls_without * hours_per_call

        calls_with = int(daily_leads * (top_pct_with_finscore / 100))
        converts_with = int(calls_with * HOT_CONVERSION)
        revenue_with = converts_with * deal_value
        hours_with = calls_with * hours_per_call

        calls_saved = calls_without - calls_with
        hours_saved = calls_saved * hours_per_call
        revenue_delta = revenue_with - revenue_without
        conversion_rate_improvement = (HOT_CONVERSION - BASELINE_CONVERSION) / BASELINE_CONVERSION
        annual_hours_saved = hours_saved * 250
        annual_revenue_delta = revenue_delta * 250

        comparison = pd.DataFrame({
            'Metric': ['Daily calls made', 'Conversions per day', 'Daily revenue', 'Hours spent calling', 'Conversion rate'],
            'Without FIN-Score': [f"{calls_without:,}", f"{converts_without:,}", f"${revenue_without:,}", f"{hours_without:.0f}h", f"{BASELINE_CONVERSION:.1%}"],
            'With FIN-Score': [f"{calls_with:,}", f"{converts_with:,}", f"${revenue_with:,}", f"{hours_with:.0f}h", f"{HOT_CONVERSION:.1%}"]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("#### Annual Impact")
        r1, r2 = st.columns(2)
        r3, r4 = st.columns(2)

        with r1:
            st.markdown(f'<div class="roi-box"><p class="roi-label">Hours Saved Per Year</p><p class="roi-value">{annual_hours_saved:,.0f}h</p></div>', unsafe_allow_html=True)
        with r2:
            color = "#2ecc71" if annual_revenue_delta >= 0 else "#e74c3c"
            sign = "+" if annual_revenue_delta >= 0 else ""
            st.markdown(f'<div class="roi-box"><p class="roi-label">Revenue Delta Per Year</p><p class="roi-value" style="color:{color};">{sign}${annual_revenue_delta:,.0f}</p></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="roi-box"><p class="roi-label">Calls Saved Per Day</p><p class="roi-value">{calls_saved:,}</p></div>', unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="roi-box"><p class="roi-label">Conversion Uplift</p><p class="roi-value">+{conversion_rate_improvement:.0%}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### How to Read This Calculator")
    st.info("""
    **What comes from real data (9,240 actual leads):**
    - Baseline conversion rate: 38.5% -- measured from dataset
    - Hot lead conversion rate: ~99.5% -- measured on model test set
    - Hot lead rate: 36% -- actual model output
    - Sales efficiency lift: 2.5x -- computed from lift curve

    **What you provide (assumptions -- adjust to your reality):**
    - Daily leads, deal value, hours per call, call volume
    - Default deal value of $400 is based on online education
      industry averages -- not from this dataset
    - Set these to your actual numbers for meaningful projections

    **What the calculator shows:**
    The efficiency gain of focusing on scored leads vs calling
    everyone. The key insight is not the revenue number --
    it is the calls saved while maintaining conversion quality.
    """)

    if revenue_delta >= 0:
        st.success(f"""
        With these assumptions: focusing on the top
        {top_pct_with_finscore}% of scored leads saves
        **{calls_saved:,} calls per day** while converting
        at **{HOT_CONVERSION:.0%}** vs **{BASELINE_CONVERSION:.0%}**
        baseline. That is **{annual_hours_saved:,.0f} hours
        of sales time recovered per year**.

        The conversion rate improvement of
        **{conversion_rate_improvement:.0%}** is based on
        real model performance. The revenue figures depend
        on your actual deal value.
        """)
    else:
        st.warning(f"""
        At current settings calling only
        {top_pct_with_finscore}% of leads produces fewer
        total conversions than calling {pct_called_now}%.
        Increase the top percent slider or reduce current
        call volume to see positive efficiency gains.

        Note: This is expected when the percentage called
        with FIN-Score is lower than the current call rate
        AND the volume difference outweighs the conversion
        rate improvement.
        """)

    if not df.empty:
        st.markdown('<div class="section-header">Current Lead Priority Mix</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])

        hot = (df['priority'] == PRIORITY_LABELS['high']).sum()
        warm = (df['priority'] == PRIORITY_LABELS['medium']).sum()
        cold = (df['priority'] == PRIORITY_LABELS['low']).sum()
        total = len(df)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 5), facecolor='none')
            sizes = [hot, warm, cold]
            colors = ['#e74c3c', '#f39c12', '#3498db']
            labels = [f'Hot ({hot/total:.0%})', f'Warm ({warm/total:.0%})', f'Cold ({cold/total:.0%})']
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, wedgeprops={'edgecolor': '#0e1117', 'linewidth': 3})
            for text in texts:
                text.set_color('white')
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            centre = plt.Circle((0, 0), 0.65, fc='#0e1117')
            ax.add_artist(centre)
            ax.text(0, 0.1, f'{total:,}', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
            ax.text(0, -0.2, 'Total Leads', ha='center', va='center', fontsize=9, color='#8892a4')
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Hot Leads")
            st.markdown(f"**{hot:,} leads** ({hot/total:.0%})\n\nScore threshold: above 0.70\n\nAction: Call within 24 hours\n\nExpected conversion rate: 99.5%\n\nAverage revenue per call: ${HOT_CONVERSION * deal_value:,.0f}")

        with col3:
            st.markdown("#### What This Means")
            st.markdown(f"Your sales team should focus on the **{hot:,} Hot Leads** first.\n\nAt {hours_per_call}h per call that is **{hot * hours_per_call:,.0f} hours** of high-value calling per day.\n\nThe remaining **{warm + cold:,}** Warm and Cold leads go into automated sequences.")


# ============================================================
# EXECUTIVE PAGE 2: LIVE LEAD SCORER
# ============================================================

def render_live_scorer(model_type: str):
    st.markdown('<div class="section-header">Live Lead Scorer</div>', unsafe_allow_html=True)
    st.markdown("Enter a lead's details below and see FIN-Score's prediction instantly -- including exactly why the model made that decision.")

    st.markdown("#### Quick Examples")
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    preset = None
    with col_ex1:
        if st.button("Hot Lead Example", use_container_width=True):
            preset = 'hot'
    with col_ex2:
        if st.button("Warm Lead Example", use_container_width=True):
            preset = 'warm'
    with col_ex3:
        if st.button("Cold Lead Example", use_container_width=True):
            preset = 'cold'
    with col_ex4:
        if st.button("Borderline Example", use_container_width=True):
            preset = 'borderline'

    presets = {
        'hot': {'visits': 10, 'time': 2500, 'pages': 4.5, 'activity': 20, 'profile': 20, 'email': 0, 'search': 1, 'recommend': 1},
        'warm': {'visits': 4, 'time': 800, 'pages': 2.5, 'activity': 14, 'profile': 13, 'email': 0, 'search': 0, 'recommend': 0},
        'cold': {'visits': 1, 'time': 50, 'pages': 1.0, 'activity': 5, 'profile': 5, 'email': 1, 'search': 0, 'recommend': 0},
        'borderline': {'visits': 3, 'time': 650, 'pages': 2.0, 'activity': 12, 'profile': 11, 'email': 0, 'search': 1, 'recommend': 0}
    }

    defaults = presets.get(preset, {'visits': 5, 'time': 500, 'pages': 2.5, 'activity': 14, 'profile': 14, 'email': 0, 'search': 0, 'recommend': 0})

    st.markdown("#### Lead Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Website Behavior**")
        total_visits = st.slider("Times visited website", 0, 30, defaults['visits'])
        time_on_site = st.slider("Minutes spent on website", 0, 3000, defaults['time'], step=50)
        page_views = st.slider("Pages viewed per visit", 0.0, 10.0, float(defaults['pages']), step=0.5)

    with col2:
        st.markdown("**Lead Profile**")
        activity_score = st.slider("Activity score (0 to 25)", 0, 25, defaults['activity'])
        profile_score = st.slider("Profile score (0 to 25)", 0, 25, defaults['profile'])
        do_not_email = st.toggle("Opted out of email", value=bool(defaults['email']))
        search = st.toggle("Found via search", value=bool(defaults['search']))
        recommend = st.toggle("Came through referral", value=bool(defaults['recommend']))

    score_btn = st.button("Score This Lead", type="primary", use_container_width=True)

    if score_btn or preset:
        lead_data = {
            "TotalVisits": total_visits,
            "Total Time Spent on Website": time_on_site,
            "Page Views Per Visit": page_views,
            "Asymmetrique Activity Score": float(activity_score),
            "Asymmetrique Profile Score": float(profile_score),
            "Do Not Email": int(do_not_email),
            "Search": int(search),
            "Through Recommendations": int(recommend),
            "is_high_engagement": (1 if time_on_site > 500 else 0),
            "asymmetrique_combined": (activity_score + profile_score) / 2
        }

        with st.spinner("Scoring lead..."):
            result = score_lead_direct(lead_data, model_type)

        if not result.get('error'):
            score = result['score']
            priority = result['priority']
            confidence = result['confidence']
            is_borderline = result['is_borderline']

            if priority == PRIORITY_LABELS['high']:
                color = "#e74c3c"
                label = "HOT LEAD"
                action = "Call immediately"
                action_detail = "Add to today's priority call list. This lead is actively researching -- contact while interest is high."
            elif priority == PRIORITY_LABELS['medium']:
                color = "#f39c12"
                label = "WARM LEAD"
                action = "Add to nurture sequence"
                action_detail = "Enroll in a 5-touch email sequence. Re-score after next significant website activity."
            else:
                color = "#3498db"
                label = "COLD LEAD"
                action = "Automated drip only"
                action_detail = "No sales rep time needed. Low-touch automated sequence. Re-score in 30 days."

            col_score, col_detail = st.columns([1, 2])

            with col_score:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1e2130,#252940);border:3px solid {color};border-radius:20px;padding:40px 20px;text-align:center;margin:10px 0;">
                    <div style="font-size:3.5em;font-weight:900;color:{color};line-height:1;margin:10px 0;">{score:.0%}</div>
                    <div style="font-size:1.4em;color:white;font-weight:700;">{label}</div>
                    <div style="color:#8892a4;font-size:0.85em;margin-top:10px;">Confidence: {confidence}{"-- Borderline" if is_borderline else ""}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_detail:
                st.markdown(f"""
                <div style="background-color:#1e2130;border-left:4px solid {color};padding:15px 20px;border-radius:8px;margin-bottom:15px;">
                    <strong style="color:{color};font-size:1.1em;">{action}</strong><br>
                    <span style="color:#cccccc;">{action_detail}</span>
                </div>
                """, unsafe_allow_html=True)

                explanation = plain_english_explanation(result['positive_drivers'], result['negative_drivers'], lead_data)

                if explanation['positive']:
                    st.markdown("**Why this score is positive:**")
                    for reason in explanation['positive'][:3]:
                        st.markdown(f"- {reason}")

                if explanation['negative']:
                    st.markdown("**Risk factors to address:**")
                    for reason in explanation['negative'][:3]:
                        st.markdown(f"- {reason}")

            if is_borderline:
                st.warning("Borderline case -- this lead's score is very close to a threshold boundary. Consider manual review before acting.")

            st.markdown("#### Score Position")
            fig, ax = plt.subplots(figsize=(10, 1.5), facecolor='none')
            ax.set_facecolor('none')
            ax.barh(0, MEDIUM_PRIORITY_THRESHOLD, color='#3498db', alpha=0.3, height=0.5)
            ax.barh(0, HIGH_PRIORITY_THRESHOLD - MEDIUM_PRIORITY_THRESHOLD, left=MEDIUM_PRIORITY_THRESHOLD, color='#f39c12', alpha=0.3, height=0.5)
            ax.barh(0, 1 - HIGH_PRIORITY_THRESHOLD, left=HIGH_PRIORITY_THRESHOLD, color='#e74c3c', alpha=0.3, height=0.5)
            ax.axvline(x=score, color=color, linewidth=4, zorder=5)
            ax.text(score, 0.35, f'{score:.0%}', ha='center', va='bottom', color=color, fontweight='bold', fontsize=12)
            ax.text(MEDIUM_PRIORITY_THRESHOLD / 2, -0.35, 'Cold', ha='center', color='#3498db', fontsize=9)
            ax.text((MEDIUM_PRIORITY_THRESHOLD + HIGH_PRIORITY_THRESHOLD) / 2, -0.35, 'Warm', ha='center', color='#f39c12', fontsize=9)
            ax.text((HIGH_PRIORITY_THRESHOLD + 1) / 2, -0.35, 'Hot', ha='center', color='#e74c3c', fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            fig.patch.set_alpha(0)
            st.pyplot(fig)
            plt.close()


# ============================================================
# EXECUTIVE PAGE 3: LEAD PIPELINE
# ============================================================

def render_lead_pipeline(df: pd.DataFrame):
    st.markdown('<div class="section-header">Lead Pipeline</div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No scoring data available.")
        return

    df['date'] = df['scored_at'].dt.date
    today = df['date'].max()
    today_df = df[df['date'] == today]

    st.markdown(f"#### Leads as of {today}")

    if today_df.empty:
        st.info("No leads scored today. Showing most recent 500 leads.")
        display_df = df.head(500)
    else:
        display_df = today_df

    col1, col2, col3 = st.columns(3)
    hot_today = (display_df['priority'] == PRIORITY_LABELS['high']).sum()
    warm_today = (display_df['priority'] == PRIORITY_LABELS['medium']).sum()
    cold_today = (display_df['priority'] == PRIORITY_LABELS['low']).sum()

    with col1:
        st.metric("Hot Leads", f"{hot_today:,}", help="Call these first")
    with col2:
        st.metric("Warm Leads", f"{warm_today:,}", help="Nurture sequence")
    with col3:
        st.metric("Cold Leads", f"{cold_today:,}", help="Automated drip")

    st.markdown("#### Priority Trend (Last 30 Days)")
    daily = df.groupby(['date', 'priority']).size().unstack(fill_value=0)
    daily_pct = daily.div(daily.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    priority_colors = {PRIORITY_LABELS['high']: '#e74c3c', PRIORITY_LABELS['medium']: '#f39c12', PRIORITY_LABELS['low']: '#3498db'}
    bottom = np.zeros(len(daily_pct))
    for priority, color in priority_colors.items():
        if priority in daily_pct.columns:
            ax.bar(range(len(daily_pct)), daily_pct[priority], bottom=bottom, color=color, label=priority, alpha=0.85)
            bottom += daily_pct[priority].values

    step = max(1, len(daily_pct) // 10)
    ax.set_xticks(range(0, len(daily_pct), step))
    ax.set_xticklabels([str(d) for d in list(daily_pct.index)[::step]], rotation=45, ha='right', color='white', fontsize=8)
    ax.set_ylabel('Percent of Leads', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#252940', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3d4460')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Export Scored Leads")
    csv = display_df.to_csv(index=False)
    st.download_button(label="Download Scored Leads as CSV", data=csv, file_name=f"finscore_leads_{today}.csv", mime="text/csv", use_container_width=True)


# ============================================================
# DS PAGE 1: MODEL PERFORMANCE
# ============================================================

def render_model_performance(metadata: dict, df: pd.DataFrame):
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    models = load_models()
    col1, col2 = st.columns(2)

    for col, model_type, color in [(col1, 'pipeline', '#9b59b6'), (col2, 'coldstart', '#00d4aa')]:
        with col:
            meta = metadata.get(model_type, {})
            if not meta:
                continue

            try:
                trained_dt = datetime.fromisoformat(meta.get('trained_at', ''))
                trained_str = trained_dt.strftime('%b %d, %Y')
            except Exception:
                trained_str = meta.get('trained_at', 'Unknown')

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e2130,#252940);border-radius:12px;padding:20px;border-top:4px solid {color};">
                <h3 style="color:{color};margin:0;">{'Pipeline Model' if model_type == 'pipeline' else 'Cold-start Model'}</h3>
                <p style="color:#8892a4;font-size:0.8em;">{'Leads with sales history' if model_type == 'pipeline' else 'Brand new leads'}</p>
                <table style="width:100%;color:white;border-collapse:collapse;">
                    <tr><td style="color:#8892a4;padding:4px 0;">Test AUC</td><td style="color:{color};font-weight:700;font-size:1.2em;">{meta.get('auc', 0):.4f}</td></tr>
                    <tr><td style="color:#8892a4;">CV AUC (5-fold)</td><td>{meta.get('cv_auc', 0):.4f} +/- {meta.get('cv_std', 0):.4f}</td></tr>
                    <tr><td style="color:#8892a4;">Lift at Top 10%</td><td>{meta.get('lift', 0)}x</td></tr>
                    <tr><td style="color:#8892a4;">Features</td><td>{meta.get('n_features', 0)}</td></tr>
                    <tr><td style="color:#8892a4;">Training samples</td><td>{meta.get('n_training_samples', 0):,}</td></tr>
                    <tr><td style="color:#8892a4;">Version</td><td>{meta.get('model_version', 'N/A')}</td></tr>
                    <tr><td style="color:#8892a4;">Trained</td><td>{trained_str}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ROC Curve")

    if not df.empty and 'actual_outcome' in df.columns:
        outcomes_df = df.dropna(subset=['actual_outcome'])

        if len(outcomes_df) > 20:
            from sklearn.metrics import roc_curve, auc

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')

            for model_type, color in [('pipeline', '#9b59b6'), ('coldstart', '#00d4aa')]:
                subset = outcomes_df[outcomes_df['model_type'] == model_type]
                if len(subset) > 20:
                    fpr, tpr, _ = roc_curve(subset['actual_outcome'], subset['score'])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_type.title()} Model (AUC = {roc_auc:.4f})')
                    ax.fill_between(fpr, tpr, alpha=0.08, color=color)

            ax.plot([0, 1], [0, 1], 'w--', lw=1.5, alpha=0.4, label='Random baseline (AUC = 0.50)')
            ax.set_xlabel('False Positive Rate\n(Cold leads incorrectly flagged as hot)', color='white', fontsize=10)
            ax.set_ylabel('True Positive Rate\n(Hot leads correctly identified)', color='white', fontsize=10)
            ax.set_title('ROC Curve -- Model Discrimination Ability', color='white', fontweight='bold', fontsize=13)
            ax.tick_params(colors='white')
            ax.legend(facecolor='#252940', labelcolor='white', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            for spine in ax.spines.values():
                spine.set_edgecolor('#3d4460')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            cs_auc = metadata.get('coldstart', {}).get('auc', 0)
            st.info(f"""
            **How to read this chart:**
            The ROC curve shows how well the model separates converters from non-converters.
            The further the curve bows toward the top-left corner the better.

            **What this means:**
            - AUC of **{cs_auc:.4f}** means the model correctly ranks a converter above a non-converter **{cs_auc:.0%}** of the time
            - The dashed line is random guessing (AUC = 0.50)
            - Industry standard for lead scoring is AUC 0.80
            - FIN-Score is **{((cs_auc - 0.80) / 0.80 * 100):.0f}% above industry standard**
            """)
        else:
            st.info("ROC curve requires leads with known outcomes. Run monitoring.py to populate the database.")

    st.markdown("#### Cumulative Lift Curve")

    if not df.empty and 'actual_outcome' in df.columns:
        outcomes_df = df.dropna(subset=['actual_outcome']).copy()

        if len(outcomes_df) > 50:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')

            all_lifts = {}
            for model_type, color in [('pipeline', '#9b59b6'), ('coldstart', '#00d4aa')]:
                subset = outcomes_df[outcomes_df['model_type'] == model_type].sort_values('score', ascending=False)
                if len(subset) > 20:
                    baseline = subset['actual_outcome'].mean()
                    lifts = []
                    for p in range(1, 101):
                        n = max(1, int(len(subset) * p / 100))
                        lift = subset.head(n)['actual_outcome'].mean() / baseline if baseline > 0 else 1
                        lifts.append(lift)
                    all_lifts[model_type] = lifts
                    ax.plot(range(1, 101), lifts, color=color, lw=2.5, label=f'{model_type.title()} Model')
                    ax.fill_between(range(1, 101), lifts, 1, where=[l > 1 for l in lifts], alpha=0.08, color=color)

            ax.axhline(y=1, color='white', linestyle='--', lw=1.5, alpha=0.4, label='Random baseline (Lift = 1.0x)')
            ax.axvline(x=10, color='#f39c12', linestyle=':', lw=1.5, alpha=0.6, label='Top 10% threshold')

            if 'coldstart' in all_lifts:
                lift_at_10 = all_lifts['coldstart'][9]
                ax.annotate(f'Top 10%: {lift_at_10:.1f}x lift', xy=(10, lift_at_10), xytext=(20, lift_at_10 + 0.2), color='#f39c12', fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='->', color='#f39c12'))

            ax.set_xlabel('Percent of Leads Called\n(ranked by FIN-Score from highest to lowest)', color='white', fontsize=10)
            ax.set_ylabel('Lift over Random Calling', color='white', fontsize=10)
            ax.set_title('Cumulative Lift Curve -- Business Value by Threshold', color='white', fontweight='bold', fontsize=13)
            ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], color='white')
            ax.set_xlim(1, 100)
            ax.set_ylim(0.8, None)
            ax.tick_params(colors='white')
            ax.legend(facecolor='#252940', labelcolor='white', fontsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor('#3d4460')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            lift_val = metadata.get('coldstart', {}).get('lift', 0)
            st.info(f"""
            **How to read this chart:**
            The x-axis shows what percentage of leads you call, ranked highest to lowest score.
            The y-axis shows how many times better your conversion rate is vs calling randomly.

            **What this means:**
            - Calling the **top 10%** of scored leads gives **{lift_val}x more conversions** per call than random outreach
            - The curve falls toward 1.0x as you include more leads
            - The yellow dotted line marks the top 10% sweet spot
            - Calling beyond 40% gives diminishing returns

            **Business recommendation:** Focus your sales team on the top 30-36% of scored leads.
            """)


# ============================================================
# DS PAGE 2: FEATURE ANALYSIS
# ============================================================

def render_feature_analysis(metadata: dict):
    st.markdown('<div class="section-header">Feature Analysis</div>', unsafe_allow_html=True)

    models = load_models()

    st.markdown("#### What Does the Model Actually Use?")
    st.markdown("""
    XGBoost assigns an importance score to each feature based on how often it is used
    to make decisions across all 300 trees. Higher score means more influence on the prediction.
    """)

    for model_type, color, title in [
        ('pipeline', '#9b59b6', 'Pipeline Model'),
        ('coldstart', '#00d4aa', 'Cold-start Model')
    ]:
        if model_type not in models:
            st.warning(f"No {model_type} model loaded")
            continue

        model = models[model_type]['model']
        feature_names = models[model_type]['feature_names']

        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=True).tail(12)

        clean_names = [
            n.replace('_freq_encoded', '').replace('_', ' ').title()
            for n in importance.index
        ]

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')

        bars = ax.barh(range(len(clean_names)), importance.values, color=color, alpha=0.85, edgecolor='none', height=0.6)
        ax.set_yticks(range(len(clean_names)))
        ax.set_yticklabels(clean_names, color='white', fontsize=10)
        ax.set_title(f'{title} -- Top 12 Features by Importance', color='white', fontweight='bold', fontsize=12, pad=15)
        ax.set_xlabel('Importance Score (higher = more influential in predictions)', color='white', fontsize=10)
        ax.tick_params(axis='x', colors='white', labelsize=9)
        ax.tick_params(axis='y', colors='white', labelsize=10)

        for bar, val in zip(bars, importance.values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', color='white', fontsize=8)

        for spine in ax.spines.values():
            spine.set_edgecolor('#3d4460')

        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close()

    # Dynamic summary based on actual top features
    if 'coldstart' in models:
        model = models['coldstart']['model']
        feature_names = models['coldstart']['feature_names']
        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

        top_3 = [
            n.replace('_freq_encoded', '')
             .replace('_', ' ')
             .title()
            for n in importance.head(3).index
        ]
        bottom_3 = [
            n.replace('_freq_encoded', '')
             .replace('_', ' ')
             .title()
            for n in importance.tail(3).index
        ]
        top_score = importance.iloc[0]
        second_score = importance.iloc[1]

        # Dynamic summary computed from actual model
    if 'coldstart' in models:
        cs_model = models['coldstart']['model']
        cs_features = models['coldstart']['feature_names']
        imp = pd.Series(
            cs_model.feature_importances_,
            index=cs_features
        ).sort_values(ascending=False)

        top1 = imp.index[0].replace('_freq_encoded','').replace('_',' ').title()
        top2 = imp.index[1].replace('_freq_encoded','').replace('_',' ').title()
        top3 = imp.index[2].replace('_freq_encoded','').replace('_',' ').title()
        low1 = imp.index[-1].replace('_freq_encoded','').replace('_',' ').title()
        low2 = imp.index[-2].replace('_freq_encoded','').replace('_',' ').title()
        s1 = imp.iloc[0]
        s2 = imp.iloc[1]
        s3 = imp.iloc[2]

        st.info(f"""
        **How to read this:**
        The longer the bar the more that feature influences
        the model's prediction.

        **Key observations from your actual model:**
        - **{top1}** is the strongest predictor
          (score: {s1:.3f}) -- this single feature has more
          influence than any other in the model
        - **{top2}** is the second most influential
          (score: {s2:.3f})
        - **{top3}** ranks third
          (score: {s3:.3f})
        - **{low1}** and **{low2}** have the least
          predictive power and could potentially be dropped

        **Why this matters for your team:**
        {top1} and {top2} are the most critical fields
        to populate accurately in your CRM for every lead.
        Missing or inaccurate values here will degrade
        model performance more than any other fields.
        """)

    st.markdown("#### Data Leakage Investigation")
    st.markdown("A three-scenario test identified post-hoc features filled in AFTER conversion outcome is known. Using these would inflate metrics dishonestly.")

    leakage_data = pd.DataFrame({
        'Scenario': ['Full Model (all features)', 'Remove Asymmetrique only', 'Remove all leakage suspects', 'Final clean model (deployed)'],
        'Features': [34, 29, 26, 29],
        'Test AUC': [0.9802, 0.9732, 0.9070, 0.9274],
        'CV AUC': [0.9820, 0.9762, 0.9178, 0.9373],
        'Finding': ['Suspicious -- too high for real world', 'Asymmetrique scores are legitimate', 'Confirms post-hoc leakage exists', 'Honest production-ready performance']
    })
    st.dataframe(leakage_data, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')

    scenarios = ['Full Model\n(all features)', 'Remove\nAsymmetrique only', 'Remove all\nleakage suspects', 'Final clean\nmodel (deployed)']
    aucs = leakage_data['Test AUC'].values
    bar_colors = ['#e74c3c', '#f39c12', '#e74c3c', '#2ecc71']

    bars = ax.bar(range(len(scenarios)), aucs, color=bar_colors, alpha=0.85, width=0.5, edgecolor='none')
    ax.set_ylim(0.50, 1.05)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, color='white', fontsize=11)
    ax.set_ylabel('AUC Score', color='white', fontsize=12)
    ax.set_xlabel('Model Scenario', color='white', fontsize=12)
    ax.set_title('Leakage Test Results -- AUC by Scenario', color='white', fontweight='bold', fontsize=14)
    ax.axhline(y=0.80, color='#636e72', linestyle='--', alpha=0.7, linewidth=1.5, label='Industry standard (0.80)')
    ax.axhline(y=0.9274, color='#2ecc71', linestyle=':', alpha=0.7, linewidth=1.5, label='Deployed model (0.9274)')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#252940', labelcolor='white', fontsize=10)

    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f'{auc_val:.4f}', ha='center', color='white', fontweight='bold', fontsize=12)

    for spine in ax.spines.values():
        spine.set_edgecolor('#3d4460')

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close()

    # Dynamic leakage summary
    full_auc = 0.9802
    clean_auc = metadata.get('coldstart', {}).get('auc', 0.9274)
    auc_drop = round(full_auc - clean_auc, 4)
    pct_above_industry = round((clean_auc - 0.80) / 0.80 * 100, 1)

    st.info(f"""
    **How to read this chart:**
    Each bar shows model AUC when different feature sets are used.
    A drop in AUC when removing features confirms those features
    were carrying predictive power.

    **The key finding:**
    - Full Model AUC **{full_auc}** -- suspiciously high,
      leakage suspected
    - Removing ALL leakage suspects drops AUC to **0.9070**
    - The **{auc_drop:.3f} point drop** confirms post-hoc
      features were inflating results
    - Final deployed model AUC **{clean_auc:.4f}** --
      honest production performance

    **What leakage suspects were removed:**
    Lead Quality, Lead Profile, and Tags are assigned by
    sales reps AFTER calling leads -- they encode the outcome
    not the signal. They would not exist for a brand new lead.

    **Why {clean_auc:.4f} is still excellent:**
    Without leaky features FIN-Score achieves
    **{clean_auc:.4f} AUC** -- that is
    **{pct_above_industry:.1f}% above the 0.80
    industry standard** shown by the dashed line.
    The model earns its performance honestly.
    """)

    st.markdown("#### Multicollinearity Analysis")
    st.markdown("Features highly correlated with each other add noise without adding signal. Five redundant pairs were identified and removed.")

    multicol_data = pd.DataFrame({
        'Feature Kept': ['Total Time Spent on Website', 'was_lead_profile_assessed', 'was_lead_quality_assessed', 'was_tags_assessed', 'Asymmetrique Activity Score'],
        'Feature Removed': ['engagement_score', 'Lead_Profile_freq_encoded', 'Lead_Quality_freq_encoded', 'Tags_freq_encoded', 'Asymmetrique Activity Index'],
        'Correlation': [1.00, 0.99, 0.99, 0.86, 0.86],
        'Reason for Keeping': ['Raw feature more interpretable', 'Missingness itself is the signal', 'Missingness itself is the signal', 'Missingness itself is the signal', 'Numeric score more granular than index']
    })
    st.dataframe(multicol_data, use_container_width=True, hide_index=True)

    st.info("""
    **Why this matters:**
    When two features say the same thing the model splits importance between them artificially.
    This makes SHAP explanations misleading and the model harder to explain to stakeholders.
    We kept the more interpretable feature in every pair.
    """)


# ============================================================
# DS PAGE 3: DRIFT MONITORING
# ============================================================

def render_drift_monitoring(df: pd.DataFrame):
    st.markdown('<div class="section-header">Drift Monitoring</div>', unsafe_allow_html=True)

    alerts_df = load_alerts()

    if not alerts_df.empty:
        st.error(f"{len(alerts_df)} unresolved alert(s) -- review required")
        for _, alert in alerts_df.iterrows():
            with st.expander(f"[{alert['severity']}] {alert['alert_type']} -- {alert['message'][:50]}..."):
                st.json(dict(alert))
    else:
        st.success("No active alerts -- model is healthy")

    if df.empty:
        st.warning("No monitoring data available.")
        return

    col1, col2, col3, col4 = st.columns(4)
    hot_rate = (df['priority'] == PRIORITY_LABELS['high']).mean()
    mean_score = df['score'].mean()
    std_score = df['score'].std()
    total = len(df)

    with col1:
        status = "Normal" if 0.1 < hot_rate < 0.5 else "Abnormal"
        st.metric("Hot Rate", f"{hot_rate:.1%}", delta=status)
    with col2:
        st.metric("Mean Score", f"{mean_score:.4f}")
    with col3:
        st.metric("Std Dev", f"{std_score:.4f}")
    with col4:
        outcomes = df['actual_outcome'].notna().sum()
        st.metric("Outcomes Known", f"{outcomes:,}", delta=f"{outcomes/total:.0%} of total")

    st.markdown("#### Alert Thresholds")
    thresh_data = pd.DataFrame({
        'Metric': ['Hot Rate', 'Hot Rate', 'AUC when outcomes known', 'Score trend delta'],
        'Threshold': ['above 50%', 'below 10%', 'below 0.80', 'above 0.05'],
        'Current': [f"{hot_rate:.1%}", f"{hot_rate:.1%}", 'N/A -- needs CRM feedback', 'Computed daily'],
        'Status': ['OK' if hot_rate < 0.5 else 'ALERT', 'OK' if hot_rate > 0.1 else 'ALERT', 'OK', 'OK']
    })
    st.dataframe(thresh_data, use_container_width=True, hide_index=True)

    st.markdown("#### Score Distribution Over Time")
    st.markdown("Hover over any point to see exact values. The charts update automatically as new leads are scored.")

    df['date'] = df['scored_at'].dt.date
    daily = df.groupby('date').agg(
        mean_score=('score', 'mean'),
        std_score=('score', 'std'),
        hot_rate=('priority', lambda x: (x == PRIORITY_LABELS['high']).mean()),
        count=('score', 'count')
    ).reset_index()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Mean Score Over Time',
            'Hot Lead Rate Over Time',
            'Score Std Dev Over Time',
            'Daily Scoring Volume'
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )

    dates = [str(d) for d in daily['date']]

    # ── Plot 1: Mean Score ────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=daily['mean_score'].round(4),
            mode='lines+markers',
            name='Mean Score',
            line=dict(color='#00d4aa', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(0,212,170,0.08)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Mean Score:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(
        y=HIGH_PRIORITY_THRESHOLD,
        line_dash='dash', line_color='#e74c3c',
        annotation_text='Hot threshold',
        annotation_font_color='#e74c3c',
        row=1, col=1
    )
    fig.add_hline(
        y=MEDIUM_PRIORITY_THRESHOLD,
        line_dash='dash', line_color='#f39c12',
        annotation_text='Warm threshold',
        annotation_font_color='#f39c12',
        row=1, col=1
    )

    # ── Plot 2: Hot Rate ──────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=(daily['hot_rate'] * 100).round(2),
            mode='lines+markers',
            name='Hot Rate %',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(231,76,60,0.08)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Hot Rate:</b> %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )
    fig.add_hline(
        y=50,
        line_dash='dash', line_color='#e74c3c',
        annotation_text='Max threshold 50%',
        annotation_font_color='#e74c3c',
        row=1, col=2
    )
    fig.add_hline(
        y=10,
        line_dash='dash', line_color='#f39c12',
        annotation_text='Min threshold 10%',
        annotation_font_color='#f39c12',
        row=1, col=2
    )

    # ── Plot 3: Std Dev ───────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=daily['std_score'].round(4),
            mode='lines+markers',
            name='Std Dev',
            line=dict(color='#f39c12', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(243,156,18,0.08)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Std Dev:</b> %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # ── Plot 4: Daily Volume ──────────────────────────────
    fig.add_trace(
        go.Bar(
            x=dates,
            y=daily['count'],
            name='Daily Volume',
            marker_color='#3498db',
            opacity=0.8,
            hovertemplate='<b>Date:</b> %{x}<br><b>Leads Scored:</b> %{y:,}<extra></extra>'
        ),
        row=2, col=2
    )

    # ── Layout ────────────────────────────────────────────
    fig.update_layout(
        height=600,
        paper_bgcolor='#1e2130',
        plot_bgcolor='#1e2130',
        font=dict(color='white', size=11),
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3d4460',
        tickfont=dict(color='white', size=9),
        tickangle=45
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3d4460',
        tickfont=dict(color='white', size=9)
    )

    for annotation in fig.layout.annotations:
        annotation.font.color = 'white'
        annotation.font.size = 11

    st.plotly_chart(fig, use_container_width=True)

    # Summary under charts
    latest = daily.iloc[-1]
    prev = daily.iloc[-2] if len(daily) > 1 else latest
    score_delta = latest['mean_score'] - prev['mean_score']
    hot_delta = latest['hot_rate'] - prev['hot_rate']

    trend_color = "normal" if abs(score_delta) < 0.05 else "inverse"
    st.info(f"""
    **How to read these charts:**
    Hover over any point to see exact values for that day.
    All four charts share the same x-axis (date) so you
    can compare trends across metrics simultaneously.

    **Latest snapshot ({latest['date']}):**
    - Mean score: **{latest['mean_score']:.4f}**
      ({'up' if score_delta > 0 else 'down'}
      {abs(score_delta):.4f} vs previous day)
    - Hot lead rate: **{latest['hot_rate']:.1%}**
      ({'up' if hot_delta > 0 else 'down'}
      {abs(hot_delta):.1%} vs previous day)
    - Leads scored: **{int(latest['count']):,}**
    - Score std dev: **{latest['std_score']:.4f}**

    **What to watch for:**
    - Mean score spiking above 0.70 -- model may be over-scoring
    - Hot rate going above 50% or below 10% -- triggers alert
    - Sudden drops in daily volume -- data pipeline issue
    - Std dev increasing significantly -- score distribution widening
    """)

    st.markdown("#### Database Health")
    db_path = os.path.join(MODEL_DIR, 'finscore_monitoring.db')
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / 1024 / 1024
        conn = sqlite3.connect(db_path)
        table_counts = {}
        for table in ['scored_leads', 'drift_snapshots', 'model_performance', 'alerts']:
            try:
                count = pd.read_sql_query(f"SELECT COUNT(*) as c FROM {table}", conn).iloc[0]['c']
                table_counts[table] = count
            except Exception:
                table_counts[table] = 0
        conn.close()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("DB Size", f"{size_mb:.1f} MB")
        with col2:
            st.metric("scored_leads rows", f"{table_counts.get('scored_leads', 0):,}")
        with col3:
            st.metric("Active alerts", f"{table_counts.get('alerts', 0):,}")
    else:
        st.error("Database not found")


# ============================================================
# DS PAGE 4: MODEL COMPARISON
# ============================================================

def render_model_comparison(metadata: dict):
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    pipeline = metadata.get('pipeline', {})
    coldstart = metadata.get('coldstart', {})

    if not pipeline or not coldstart:
        st.warning("Both models need to be trained first.")
        return

    st.markdown("#### Head to Head")
    comparison = pd.DataFrame({
        'Metric': ['Test AUC', 'CV AUC (5-fold)', 'CV Std Dev', 'Lift at Top 10%', 'Features Used', 'Training Samples', 'Model Version', 'Use Case'],
        'Pipeline Model': [f"{pipeline.get('auc', 0):.4f}", f"{pipeline.get('cv_auc', 0):.4f}", f"+/- {pipeline.get('cv_std', 0):.4f}", f"{pipeline.get('lift', 0)}x", f"{pipeline.get('n_features', 0)}", f"{pipeline.get('n_training_samples', 0):,}", pipeline.get('model_version', 'N/A'), 'Leads with sales history'],
        'Cold-start Model': [f"{coldstart.get('auc', 0):.4f}", f"{coldstart.get('cv_auc', 0):.4f}", f"+/- {coldstart.get('cv_std', 0):.4f}", f"{coldstart.get('lift', 0)}x", f"{coldstart.get('n_features', 0)}", f"{coldstart.get('n_training_samples', 0):,}", coldstart.get('model_version', 'N/A'), 'Brand new leads']
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("#### Feature Sets")
    pipeline_features = set(pipeline.get('feature_names', []))
    coldstart_features = set(coldstart.get('feature_names', []))
    only_pipeline = pipeline_features - coldstart_features
    only_coldstart = coldstart_features - pipeline_features
    shared = pipeline_features & coldstart_features

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Shared Features ({len(shared)})**")
        for f in sorted(shared):
            st.markdown(f"- {f.replace('_freq_encoded', '').replace('_', ' ').title()}")
    with col2:
        st.markdown(f"**Pipeline Only ({len(only_pipeline)})**")
        for f in sorted(only_pipeline):
            st.markdown(f"- {f.replace('_freq_encoded', '').replace('_', ' ').title()}")
        if not only_pipeline:
            st.markdown("None")
    with col3:
        st.markdown(f"**Cold-start Only ({len(only_coldstart)})**")
        for f in sorted(only_coldstart):
            st.markdown(f"- {f.replace('_freq_encoded', '').replace('_', ' ').title()}")
        if not only_coldstart:
            st.markdown("None")

    st.markdown("#### When to Use Which Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Use Pipeline Model when:**
        - Lead has been in CRM for more than 7 days
        - Sales rep has added notes or tags
        - Lead Quality has been assessed
        - Re-scoring existing pipeline leads
        - Nightly batch scoring of CRM

        Expected: AUC 0.93, Lift 2.5x
        """)
    with col2:
        st.markdown("""
        **Use Cold-start Model when:**
        - Lead just submitted a form
        - Brand new inbound lead
        - No sales rep interaction yet
        - Real-time API scoring at creation
        - Marketing attribution analysis

        Expected: AUC 0.91, Lift 2.5x
        """)

    st.markdown("#### Model Selection Logic")
    st.code("""
def select_model(lead_data: dict) -> str:
    post_hoc_features = [
        'Lead Quality_freq_encoded',
        'Lead Profile_freq_encoded',
        'Tags_freq_encoded',
        'was_lead_quality_assessed',
        'was_lead_profile_assessed',
        'was_tags_assessed'
    ]
    has_sales_history = any(f in lead_data for f in post_hoc_features)
    if has_sales_history:
        return 'pipeline'    # AUC: 0.93
    else:
        return 'coldstart'   # AUC: 0.91
    """, language='python')


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_monitoring_data()
    metadata = load_metadata()
    audience, page, model_type = render_sidebar()

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#1e2130,#252940);padding:15px 25px;border-radius:10px;margin-bottom:20px;border-left:4px solid #00d4aa;">
        <h2 style="color:white;margin:0;">FIN-Score <span style="font-size:0.6em;color:#8892a4;font-weight:400;margin-left:10px;">B2B Lead Scoring System</span></h2>
        <p style="color:#8892a4;margin:3px 0 0 0;font-size:0.85em;">{'Executive View' if audience == 'Executive' else 'Data Scientist View'} -- {page}</p>
    </div>
    """, unsafe_allow_html=True)

    if audience == "Executive":
        if page == "Business Impact and ROI":
            render_business_impact(metadata, df)
        elif page == "Live Lead Scorer":
            render_live_scorer(model_type)
        elif page == "Lead Pipeline":
            render_lead_pipeline(df)
    else:
        if page == "Model Performance":
            render_model_performance(metadata, df)
        elif page == "Feature Analysis":
            render_feature_analysis(metadata)
        elif page == "Drift Monitoring":
            render_drift_monitoring(df)
        elif page == "Model Comparison":
            render_model_comparison(metadata)


if __name__ == "__main__":
    main()
