# FraudShield AI — Real-Time Fraud Detection System

A production-grade fraud detection platform combining an XGBoost ML model,
a multi-module rules engine, Kafka streaming, a FastAPI scoring API, and a
premium live dashboard.

---

## Model results

Held-out test set (56,962 transactions, 98 frauds, 0.17% fraud rate):

| Metric | Value |
|---|---|
| ROC-AUC | 0.976 |
| PR-AUC | 0.684 |
| Precision @ threshold 0.80 | 0.78 |
| Recall @ threshold 0.80 | 0.79 |
| Confusion | 77 caught, 21 missed, 22 false alarms |

**Modelling decisions**
- **No SMOTE.** An earlier version combined SMOTE with class weighting, which double-corrected
  the imbalance and dropped precision to ~0.29. The model now uses `scale_pos_weight`
  set from the true class ratio (~577:1).
- **Isotonic calibration** on a held-out calibration split, so probabilities reflect the real fraud rate.
- **Precision-weighted threshold:** chosen by maximising F0.5 with a 70% recall floor, saved to
  `models/threshold.json` and loaded at inference time.
- **Leakage control:** the scaler is fit on the training split only; the test split is never used
  for training, early stopping, calibration or threshold selection.

---

## Architecture

```
kafka_producer.py
      │  (transactions topic)
      ▼
   Kafka Broker
      │
      ▼
kafka_consumer.py
      │
      ├─► predict_transaction()     ← XGBoost ML model
      ├─► apply_fraud_rules()       ← Rules engine (6 modules)
      │       ├─ country_rules
      │       ├─ merchant_rules
      │       ├─ vendor_rules
      │       ├─ user_behavior
      │       ├─ velocity_rules
      │       └─ network_detection
      ├─► calculate_risk()          ← Risk classification
      ├─► log_transaction()         ← CSV logger
      └─► fraud_results topic       ← Kafka output

fraud_api.py  ← FastAPI REST endpoint (same pipeline, HTTP entry)
dashboard_app.py  ← Streamlit live dashboard
```

---

## Project Structure

```
fraud_detection/
├── config/
│   ├── __init__.py          # Config loader
│   └── config.yaml          # All thresholds, paths, Kafka settings
├── src/
│   ├── ml/
│   │   ├── train_model.py   # XGBoost training pipeline
│   │   ├── predict.py       # ML inference
│   │   └── feature_engineering.py
│   ├── risk_engine/
│   │   ├── fraud_rules.py   # Orchestrator — apply_fraud_rules()
│   │   ├── risk_scoring.py  # Risk → action mapping
│   │   ├── country_rules.py
│   │   ├── merchant_rules.py
│   │   ├── vendor_rules.py
│   │   ├── velocity_rules.py
│   │   ├── user_behavior.py
│   │   ├── network_detection.py
│   │   └── explain_fraud.py
│   ├── streaming/
│   │   ├── kafka_producer.py
│   │   └── kafka_consumer.py
│   └── monitoring/
│       └── transaction_logger.py
├── models/                  # fraud_model.pkl + scaler.pkl (git-ignored)
├── logs/                    # transactions.csv (git-ignored)
├── data/raw/                # creditcard.csv (git-ignored)
├── fraud_api.py             # FastAPI app
├── dashboard_app.py         # Streamlit dashboard
├── run_streaming_pipeline.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Docker + Docker Compose | Latest |
| Kaggle creditcard.csv | [Download here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## Setup — Step by Step

### 1. Clone and create virtual environment

```bash
git clone <your-repo>
cd fraud_detection

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the dataset

Download `creditcard.csv` from Kaggle and place it at:

```
data/raw/creditcard.csv
```

### 3. Train the model

This step must be completed before running the API or consumer.

```bash
python -m src.ml.train_model
```

Expected output (abridged, from a real run on `creditcard.csv`):
```
Split: train=(199364, 30) | cal=(28481, 30) | test=(56962, 30)
Scaler fit on training only -> models/scaler.pkl
scale_pos_weight = 576.87  (199019 neg / 345 pos)
Training XGBoost (SPW=576.9, early_stopping=30)...
Calibrating with isotonic regression on held-out cal set...
Optimal threshold=0.8000  F0.5=0.8295  (recall floor 70%)
FINAL EVALUATION  (threshold=0.8000)
ROC-AUC    : 0.9764
PR-AUC     : 0.6839
Precision  : 0.7778
Recall     : 0.7857
TP=77 FP=22 FN=21 TN=56842
Model saved -> models/fraud_model.pkl
```

### 4. Start Kafka (Docker)

```bash
docker compose up -d zookeeper kafka
```

Wait ~20 seconds for Kafka to be ready. Verify:

```bash
docker compose ps
# zookeeper and kafka should show "healthy"
```

### 5. Set your API key

```bash
export API_KEY=your-secret-key-here
```

On Windows PowerShell:
```powershell
$env:API_KEY="your-secret-key-here"
```

On Windows Command Prompt:
```cmd
set API_KEY=your-secret-key-here
```

### 6. Start each service

Open **three separate terminals**:

**Terminal 1 — Consumer (processes transactions)**
```bash
python -m src.streaming.kafka_consumer
```

**Terminal 2 — Producer (generates synthetic transactions)**
```bash
python -m src.streaming.kafka_producer
```

**Terminal 3 — Dashboard**
```bash
streamlit run dashboard_app.py
```

Open [http://localhost:8501](http://localhost:8501) — you should see the live command center.

### 7. (Optional) Start the REST API

```bash
uvicorn fraud_api:app --host 0.0.0.0 --port 8000 --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Usage

### Health check (no auth)

```bash
curl http://localhost:8000/health
```

### Score a transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{
    "user_id": "USER_1234",
    "merchant_id": "MERCHANT_7",
    "vendor_id": "VENDOR_42",
    "merchant_category": "supermarket",
    "country": "NG",
    "Amount": 2500.00,
    "Time": 45000
  }'
```

Response:
```json
{
  "fraud_probability": 0.0021,
  "total_risk_score": 0.6021,
  "risk_level": "HIGH",
  "action": "FREEZE_CARD",
  "explanation": "High-risk country (NG)",
  "processed_at": "2024-01-15T10:30:00Z"
}
```

### Batch scoring (up to 100 transactions)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{"transactions": [...]}'
```

---

## Full Docker Deployment

To run everything in containers:

```bash
# Build and start all services
API_KEY=your-secret-key docker compose up --build

# View logs
docker compose logs -f fraud-consumer
docker compose logs -f fraud-producer

# Stop all
docker compose down
```

On Windows PowerShell:
```powershell
$env:API_KEY="your-secret-key"
docker compose up --build
```

Services:
| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Kafka | localhost:9092 |

---

## Risk Scoring Logic

| Score Range | Risk Level | Action |
|-------------|------------|--------|
| < 0.30 | LOW | ALLOW |
| 0.30 – 0.55 | MEDIUM | SOFT_CHECK (step-up auth) |
| 0.55 – 0.75 | HIGH | REVIEW |
| 0.75 – 0.90 | HIGH | FREEZE_CARD |
| ≥ 0.90 | VERY_HIGH | BLOCK_CARD |

Thresholds live in `config/config.yaml`. A confidence gate caps hard actions: if the
ML probability is below `min_confidence_for_hard_action` (0.05), rules alone can only
trigger SOFT_CHECK or REVIEW, never a freeze or block.

**Score components (all additive, capped at 1.0):**

| Module | Max Contribution | Trigger |
|--------|-----------------|---------|
| XGBoost ML | variable | Learned from creditcard.csv |
| Country risk | +0.40 | NG, PK, RU, IR, KP |
| Merchant limit | +0.35 | Amount exceeds category limit |
| Vendor volume | +0.30–0.40 | > 20/30 transactions |
| User behavior | +0.30 | Amount > 5× user average |
| Velocity | +0.40 | > 10 user txns in 10s window |
| Network detection | +0.40 | User cluster > 5 connected users |

---

## Bugs Fixed vs Original Codebase

| # | Bug | Original | Fixed |
|---|-----|----------|-------|
| 1 | `apply_fraud_rules` not defined | `ImportError` on startup | Defined in `fraud_rules.py` |
| 2 | Scaler fit before train/test split | Evaluation metrics inflated | Split first, scale after |
| 3 | Vendor counter double-increment | 2× false positives | Separate counters per function |
| 4 | Country as graph node | All NG/RU users flagged as fraud ring | Country removed from graph |
| 5 | Logger signature mismatch | `TypeError` on every transaction | Unified `(transaction, result, timestamp)` |
| 6 | Action label drift (`NOTIFY` vs `NOTIFY_USER`) | Dashboard NOTIFY count = 0 | Single source in `config.yaml` |
| 7 | `total_risk_score` missing in queue path | `None` written to CSV | Full result dict in both paths |

---

## .gitignore

```
venv/
models/
logs/
data/
__pycache__/
*.pkl
*.pyc
.env
```

---

## Next Steps (Upgrade Roadmap)

1. **Replace in-memory state with Redis** — velocity, user behavior, and vendor counters reset on restart; Redis fixes this
2. **Add model retraining pipeline** — scheduled weekly retraining on accumulated logged data
3. **Add SHAP explainability** — per-feature contribution to the ML score
4. **Prometheus + Grafana metrics** — expose `/metrics` in Prometheus format
5. **Feedback loop** — mark false positives via API; feed confirmed labels back into training
