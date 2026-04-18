# ============================================================
# api/main.py — FIN-Score FastAPI Endpoint
#
# Exposes the lead scoring model as a REST API.
# In production this sits behind an API gateway and
# integrates directly with Salesforce/Marketo via webhook.
#
# Endpoints:
#   GET  /health        — liveness check
#   GET  /model/info    — model metadata and performance
#   POST /score         — score a single lead
#   POST /score/batch   — score multiple leads
#   GET  /monitoring    — current score distribution
# ============================================================

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.config import *
from src.predict import score_lead, score_batch

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """
    Converts numpy types to Python native types
    so FastAPI can serialize them to JSON.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="FIN-Score API",
    description="""
    B2B Lead Scoring System
    
    FIN-Score predicts the probability of a B2B lead converting
    to a paying customer using behavioral and firmographic signals.
    
    Model Variants
    - **Pipeline Model** (AUC: 0.93) — for leads with sales history
    - **Cold-start Model** (AUC: 0.91) — for brand new leads
    
    Priority Labels
    🔥 Hot Lead  — score ≥ 0.70 — call immediately
    🟡 Warm Lead — score ≥ 0.40 — nurture sequence
    ❄️ Cold Lead — score < 0.40 — automated drip
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS Middleware ───────────────────────────────────────────
# Allows Salesforce/Marketo to call our API from their servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST AND RESPONSE MODELS
# Pydantic models define exactly what the API accepts
# and returns — with automatic validation
# ============================================================

class LeadScoringRequest(BaseModel):
    """
    Input model for scoring a single lead.
    All fields are optional — missing fields default to 0.
    The model handles missing values gracefully.
    """
    # Behavioral signals
    TotalVisits: Optional[float] = Field(
        default=0,
        ge=0,
        le=100,
        description="Number of website visits"
    )
    Total_Time_Spent_on_Website: Optional[float] = Field(
        default=0,
        alias="Total Time Spent on Website",
        ge=0,
        description="Total minutes spent on website"
    )
    Page_Views_Per_Visit: Optional[float] = Field(
        default=0,
        alias="Page Views Per Visit",
        ge=0,
        description="Average pages viewed per visit"
    )

    # Internal scores
    Asymmetrique_Activity_Score: Optional[float] = Field(
        default=0,
        alias="Asymmetrique Activity Score",
        ge=0,
        le=25,
        description="Internal activity score"
    )
    Asymmetrique_Profile_Score: Optional[float] = Field(
        default=0,
        alias="Asymmetrique Profile Score",
        ge=0,
        le=25,
        description="Internal profile score"
    )

    # Model selection
    model_type: Optional[str] = Field(
        default=None,
        description="'pipeline' or 'coldstart' — auto-detected if None"
    )

    # Include SHAP explanation
    explain: Optional[bool] = Field(
        default=True,
        description="Include SHAP feature explanations"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "TotalVisits": 5,
                "Total Time Spent on Website": 1500,
                "Page Views Per Visit": 3.5,
                "Asymmetrique Activity Score": 15.0,
                "Asymmetrique Profile Score": 17.0,
                "model_type": "coldstart",
                "explain": True
            }
        }


class BatchScoringRequest(BaseModel):
    """
    Input model for scoring multiple leads at once.
    """
    leads: List[Dict[str, Any]] = Field(
        description="List of lead dictionaries to score"
    )
    model_type: Optional[str] = Field(
        default='coldstart',
        description="Model variant to use for all leads"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "leads": [
                    {
                        "TotalVisits": 10,
                        "Total Time Spent on Website": 2500,
                        "Asymmetrique Activity Score": 20.0
                    },
                    {
                        "TotalVisits": 1,
                        "Total Time Spent on Website": 50,
                        "Asymmetrique Activity Score": 3.0
                    }
                ],
                "model_type": "coldstart"
            }
        }


class ScoreResponse(BaseModel):
    """
    Output model for a single lead score.
    """
    score: float
    score_pct: str
    priority: str
    model_type: str
    model_version: str
    recommendation: str
    confidence: str
    is_borderline: bool
    explanation: Optional[Dict] = None
    scored_at: str
    error: bool = False


# ============================================================
# ENDPOINTS
# ============================================================

# ── Health Check ──────────────────────────────────────────────
@app.get(
    "/health",
    tags=["System"],
    summary="API Health Check"
)
async def health_check():
    """
    Liveness check endpoint.
    
    Used by load balancers and monitoring systems to verify
    the API is running. Returns immediately with no DB calls.
    
    In production Kubernetes calls this every 30 seconds.
    If it fails — the pod gets restarted automatically.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "environment": ENVIRONMENT
    }


# ── Model Info ────────────────────────────────────────────────
@app.get(
    "/model/info",
    tags=["Model"],
    summary="Model Metadata and Performance"
)
async def model_info():
    """
    Returns metadata for both trained model variants.
    
    Reads from the metadata JSON files saved by train.py.
    Lets stakeholders see exactly which model is deployed,
    when it was trained, and what performance it achieved.
    """
    metadata = {}

    for model_type in ['pipeline', 'coldstart']:
        metadata_path = os.path.join(
            MODEL_DIR,
            f'{model_type}_metadata.json'
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata[model_type] = json.load(f)
        else:
            metadata[model_type] = {
                "error": f"Metadata not found — run train.py first"
            }

    return {
        "models": metadata,
        "priority_labels": PRIORITY_LABELS,
        "thresholds": {
            "hot": HIGH_PRIORITY_THRESHOLD,
            "warm": MEDIUM_PRIORITY_THRESHOLD
        }
    }


# ── Score Single Lead ─────────────────────────────────────────
@app.post(
    "/score",
    tags=["Scoring"],
    summary="Score a Single Lead"
)
async def score_single_lead(request: LeadScoringRequest):
    """
    Scores a single lead and returns conversion probability.
    
    **What it does:**
    1. Validates input data
    2. Auto-detects which model to use (pipeline vs coldstart)
    3. Scores the lead using XGBoost
    4. Generates SHAP explanation (why this score)
    5. Returns priority label and recommendation
    6. Logs score to monitoring database
    
    **Model selection:**
    - If lead has sales history features → pipeline model (AUC 0.93)
    - If brand new lead → coldstart model (AUC 0.91)
    - Override with model_type parameter
    
    **Response time:** ~100ms
    """
    try:
        # Convert Pydantic model to dict
        # Use aliases to get original column names
        lead_data = request.dict(by_alias=True)

        # Remove API-specific fields before scoring
        model_type = lead_data.pop('model_type', None)
        explain = lead_data.pop('explain', True)

        # Remove None values — let model handle missing
        lead_data = {
            k: v for k, v in lead_data.items()
            if v is not None
        }

        # Score the lead
        result = score_lead(
            lead_data=lead_data,
            model_type=model_type,
            explain=explain
        )

        # Handle validation errors from predict.py
        if result.get('error'):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Invalid lead data",
                    "errors": result.get('errors', [])
                }
            )

        # Convert numpy types to Python native for JSON serialization
        return convert_to_serializable(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(e)}"
        )


# ── Score Batch ───────────────────────────────────────────────
@app.post(
    "/score/batch",
    tags=["Scoring"],
    summary="Score Multiple Leads"
)
async def score_leads_batch(request: BatchScoringRequest):
    """
    Scores multiple leads in a single request.
    
    More efficient than calling /score repeatedly because
    the model loads once and scores all leads in one pass.
    
    **Use cases:**
    - Nightly CRM batch updates
    - Scoring all leads after model retrain
    - Bulk lead import scoring
    
    **Limit:** 10,000 leads per request
    **Response time:** ~1ms per lead
    """
    if len(request.leads) == 0:
        raise HTTPException(
            status_code=400,
            detail="leads list cannot be empty"
        )

    if len(request.leads) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10,000 leads per batch request"
        )

    try:
        results_df = score_batch(
            leads=request.leads,
            model_type=request.model_type
        )

        # Convert DataFrame to list of dicts for JSON response
        results = results_df.to_dict(orient='records')

        # Summary statistics
        hot_count = sum(
            1 for r in results
            if r['priority'] == PRIORITY_LABELS['high']
        )
        warm_count = sum(
            1 for r in results
            if r['priority'] == PRIORITY_LABELS['medium']
        )
        cold_count = sum(
            1 for r in results
            if r['priority'] == PRIORITY_LABELS['low']
        )

        return convert_to_serializable({
            "total_leads": len(results),
            "model_type": request.model_type,
            "summary": {
                "hot_leads": hot_count,
                "warm_leads": warm_count,
                "cold_leads": cold_count,
                "hot_rate": f"{hot_count/len(results):.1%}",
                "warm_rate": f"{warm_count/len(results):.1%}",
                "cold_rate": f"{cold_count/len(results):.1%}"
            },
            "results": results,
            "scored_at": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch scoring failed: {str(e)}"
        )


# ── Monitoring ────────────────────────────────────────────────
@app.get(
    "/monitoring",
    tags=["Monitoring"],
    summary="Model Health and Score Distribution"
)
async def get_monitoring():
    """
    Returns current model health metrics.
    
    Reads from the SQLite monitoring database to show:
    - Total leads scored
    - Score distribution (hot/warm/cold rates)
    - Recent drift status
    - Any active alerts
    
    In production this feeds a Grafana dashboard.
    """
    try:
        from src.monitoring import (
            get_monitoring_summary,
            compute_daily_snapshot
        )

        summary = get_monitoring_summary()
        totals = summary['totals']

        # Get latest drift snapshot
        snapshots = summary['snapshots']
        latest_snapshot = (
            snapshots.iloc[0].to_dict()
            if not snapshots.empty
            else {}
        )

        # Active alerts
        alerts = summary['active_alerts']
        active_alerts = (
            alerts.to_dict(orient='records')
            if not alerts.empty
            else []
        )

        return {
            "status": latest_snapshot.get('status', 'Unknown'),
            "total_scored": int(totals.get('total_scored', 0)),
            "overall_mean_score": round(
                float(totals.get('overall_mean', 0)), 4
            ),
            "outcomes_known": int(
                totals.get('outcomes_known', 0)
            ),
            "latest_snapshot": latest_snapshot,
            "active_alerts": active_alerts,
            "thresholds": {
                "hot": HIGH_PRIORITY_THRESHOLD,
                "warm": MEDIUM_PRIORITY_THRESHOLD
            },
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Monitoring error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring failed: {str(e)}"
        )


# ── Root ──────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["System"],
    summary="API Root"
)
async def root():
    """API root — redirects to docs."""
    return {
        "name": "FIN-Score API",
        "version": "1.0.0",
        "description": "B2B Lead Scoring System",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "GET  /health",
            "GET  /model/info",
            "POST /score",
            "POST /score/batch",
            "GET  /monitoring"
        ]
    }

# ============================================================
# Run directly for development
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
