"""
Fraud Detection REST API — Production Grade

Endpoints:
    GET  /health            — liveness check
    GET  /metrics           — system statistics
    POST /predict           — score a single transaction
    POST /predict/batch     — score up to 100 transactions

Security:
    All /predict endpoints require X-API-Key header.
    Set API_KEY environment variable before running.

Usage:
    API_KEY=secret uvicorn fraud_api:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config import CONFIG
from src.ml.predict                    import predict_transaction
from src.risk_engine.fraud_rules       import apply_fraud_rules
from src.risk_engine.risk_scoring      import calculate_risk
from src.risk_engine.explain_fraud     import explain_transaction
from src.monitoring.transaction_logger import log_transaction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ─── Security ─────────────────────────────────────────────────────────────────

API_KEY        = os.getenv("API_KEY", "changeme-before-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(key: str = Security(api_key_header)) -> str:
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return key


# ─── Model state ──────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path  = CONFIG["model"]["path"]
    scaler_path = CONFIG["model"]["scaler_path"]

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise RuntimeError(
            "Model files not found. Run: python -m src.ml.train_model"
        )

    _state["model"]   = joblib.load(model_path)
    _state["scaler"]  = joblib.load(scaler_path)
    _state["started"] = datetime.now(timezone.utc).isoformat()
    _state["requests"] = 0
    _state["errors"]   = 0

    log.info("Models loaded. API ready.")
    yield
    log.info("API shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AI Fraud Detection API",
    description = "Real-time transaction fraud scoring with ML + rules engine.",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    user_id          : str   = Field(..., example="USER_1234")
    merchant_id      : str   = Field(..., example="MERCHANT_7")
    vendor_id        : str   = Field(..., example="VENDOR_42")
    merchant_category: str   = Field(..., example="supermarket")
    country          : str   = Field(..., example="IE")
    Amount           : float = Field(..., gt=0, le=100_000, example=149.99)
    Time             : float = Field(default=0.0)

    # V1–V28 PCA features — optional, default 0 for rule-only scoring
    V1 : float = 0.0; V2 : float = 0.0; V3 : float = 0.0; V4 : float = 0.0
    V5 : float = 0.0; V6 : float = 0.0; V7 : float = 0.0; V8 : float = 0.0
    V9 : float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0

    @field_validator("country")
    @classmethod
    def upper_country(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("merchant_category", "user_id", "merchant_id", "vendor_id")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        return v.strip()


class FraudPredictionResponse(BaseModel):
    fraud_probability : float
    total_risk_score  : float
    risk_level        : str
    action            : str
    explanation       : str
    processed_at      : str


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    results    : list[FraudPredictionResponse]
    total      : int
    high_risk  : int
    duration_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status"  : "ok",
        "started" : _state.get("started"),
        "model"   : "XGBoost",
        "version" : "2.0.0",
    }


@app.get("/metrics", tags=["System"])
def metrics(_: str = Depends(verify_api_key)):
    return {
        "total_requests": _state.get("requests", 0),
        "total_errors"  : _state.get("errors",   0),
        "uptime_since"  : _state.get("started"),
    }


@app.post("/predict", response_model=FraudPredictionResponse, tags=["Scoring"])
def predict(
    req: TransactionRequest,
    _  : str = Depends(verify_api_key),
):
    _state["requests"] = _state.get("requests", 0) + 1
    txn = req.model_dump()

    try:
        ml_prob          = predict_transaction(txn, _state["model"], _state["scaler"])
        total_risk_score = apply_fraud_rules(txn, ml_prob)
        risk_level, action = calculate_risk(
            total_risk_score,
            ml_probability=ml_prob,
        )
        explanation      = explain_transaction(txn, total_risk_score)

        result = {
            "fraud_probability": round(ml_prob, 6),
            "total_risk_score" : round(total_risk_score, 6),
            "risk_level"       : risk_level,
            "action"           : action,
            "explanation"      : explanation,
        }

        log_transaction(
            transaction=txn,
            result=result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return {**result, "processed_at": datetime.now(timezone.utc).isoformat()}

    except Exception as exc:
        _state["errors"] = _state.get("errors", 0) + 1
        log.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal scoring error.")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Scoring"])
def predict_batch(
    req: BatchRequest,
    _  : str = Depends(verify_api_key),
):
    t0      = time.perf_counter()
    results = []

    for transaction in req.transactions:
        txn = transaction.model_dump()
        _state["requests"] = _state.get("requests", 0) + 1

        try:
            ml_prob          = predict_transaction(txn, _state["model"], _state["scaler"])
            total_risk_score = apply_fraud_rules(txn, ml_prob)
            risk_level, action = calculate_risk(
                total_risk_score,
                ml_probability=ml_prob,
            )
            explanation      = explain_transaction(txn, total_risk_score)

            result = {
                "fraud_probability": round(ml_prob, 6),
                "total_risk_score" : round(total_risk_score, 6),
                "risk_level"       : risk_level,
                "action"           : action,
                "explanation"      : explanation,
                "processed_at"     : datetime.now(timezone.utc).isoformat(),
            }

            log_transaction(txn, result, result["processed_at"])
            results.append(result)

        except Exception as exc:
            _state["errors"] = _state.get("errors", 0) + 1
            log.exception("Batch item error: %s", exc)
            results.append({
                "fraud_probability": 0.0,
                "total_risk_score" : 0.0,
                "risk_level"       : "ERROR",
                "action"           : "REVIEW",
                "explanation"      : f"Processing error: {str(exc)}",
                "processed_at"     : datetime.now(timezone.utc).isoformat(),
            })

    duration_ms = (time.perf_counter() - t0) * 1000
    high_risk   = sum(1 for r in results if r["risk_level"] in ("HIGH", "VERY_HIGH"))

    return {
        "results"    : results,
        "total"      : len(results),
        "high_risk"  : high_risk,
        "duration_ms": round(duration_ms, 2),
    }
