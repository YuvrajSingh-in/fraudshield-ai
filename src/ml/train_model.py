"""
Production-grade XGBoost training pipeline.

KEY CHANGES FROM v1:
─────────────────────────────────────────────────────────────────────────────
1. SMOTE REMOVED — SMOTE + scale_pos_weight=10 was double-correcting.
   The model saw a 50/50 SMOTE distribution, so threshold=0.5 gave high
   recall but ~0.29 precision. Root cause: probabilities calibrated to
   synthetic distribution, not real 0.17% fraud rate.

   FIX: Use scale_pos_weight=577 (true ratio) without SMOTE.

2. PROBABILITY CALIBRATION — CalibratedClassifierCV(isotonic) on a
   held-out calibration set. Standard at Stripe/Visa-level systems.

3. DYNAMIC THRESHOLD — F-beta(beta=0.5) maximisation on validation set.
   Beta=0.5 weights precision 2x over recall. Threshold saved to
   models/threshold.json and read at inference time.

4. EARLY STOPPING — prevents overfitting without hyperparameter sweeps.

5. THREE-WAY SPLIT — train / calibration / test. The calibration split is
   used for early stopping, isotonic calibration and threshold selection;
   the test split is never touched until final evaluation.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

DATA_PATH      = Path(CONFIG["training"]["data_path"])
MODEL_DIR      = Path("models")
MODEL_PATH     = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH    = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_validate() -> pd.DataFrame:
    log.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("Dataset shape: %s", df.shape)
    for col in ["Amount", "Class"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    nulls = int(df.isnull().sum().sum())
    if nulls:
        log.warning("Filling %d null values with 0", nulls)
        df = df.fillna(0)
    fraud_count = int(df["Class"].sum())
    log.info(
        "Fraud: %d / %d  (%.4f%%)",
        fraud_count, len(df), fraud_count / len(df) * 100,
    )
    return df


def split_data(X, y, cfg):
    rs = cfg["random_state"]
    X_tc, X_test, y_tc, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=rs
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tc, y_tc, test_size=0.125, stratify=y_tc, random_state=rs
    )
    log.info("Split: train=%s | cal=%s | test=%s", X_train.shape, X_cal.shape, X_test.shape)
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def scale(X_train, X_cal, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xc  = scaler.transform(X_cal)
    Xte = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    log.info("Scaler fit on training only → %s", SCALER_PATH)
    return Xtr, Xc, Xte, scaler


def compute_spw(y) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    spw   = n_neg / n_pos
    log.info("scale_pos_weight = %.2f  (%d neg / %d pos)", spw, n_neg, n_pos)
    return spw


def train_base(X_train, y_train, X_val, y_val, spw) -> XGBClassifier:
    cfg   = CONFIG["training"]["xgboost"]
    model = XGBClassifier(
        n_estimators          = cfg["n_estimators"],
        max_depth             = cfg["max_depth"],
        learning_rate         = cfg["learning_rate"],
        subsample             = cfg["subsample"],
        colsample_bytree      = cfg["colsample_bytree"],
        scale_pos_weight      = spw,
        random_state          = CONFIG["training"]["random_state"],
        eval_metric           = "aucpr",
        early_stopping_rounds = 30,
        use_label_encoder     = False,
        n_jobs                = -1,
    )
    log.info("Training XGBoost (SPW=%.1f, early_stopping=30)...", spw)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    log.info("Best iteration: %d", model.best_iteration)
    return model


def calibrate(base_model, X_cal, y_cal):
    log.info("Calibrating with isotonic regression on held-out cal set...")
    cal = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    log.info("Calibration complete.")
    return cal


def find_threshold(model, X_val, y_val, beta=0.5) -> float:
    """
    Find operating threshold maximising F-beta (beta=0.5 → precision-heavy).
    Enforces recall floor of 70% to avoid purely precision-chasing.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    precs, recs, thresholds = precision_recall_curve(y_val, y_prob)

    best_t  = 0.5
    best_fb = 0.0
    for t, p, r in zip(thresholds, precs[:-1], recs[:-1]):
        if r < 0.70:
            continue
        denom = beta**2 * p + r
        fb    = (1 + beta**2) * p * r / denom if denom > 0 else 0.0
        if fb > best_fb:
            best_fb, best_t = fb, float(t)

    log.info("Optimal threshold=%.4f  F%.1f=%.4f  (recall floor 70%%)", best_t, beta, best_fb)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_t, "beta": beta, "fbeta": best_fb}, f, indent=2)
    log.info("Threshold → %s", THRESHOLD_PATH)
    return best_t


def evaluate(model, X_test, y_test, threshold) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    roc    = roc_auc_score(y_test, y_prob)
    prauc  = average_precision_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    log.info("=" * 60)
    log.info("FINAL EVALUATION  (threshold=%.4f)", threshold)
    log.info("ROC-AUC    : %.4f", roc)
    log.info("PR-AUC     : %.4f", prauc)
    log.info("Precision  : %.4f", tp / (tp + fp) if (tp+fp) else 0)
    log.info("Recall     : %.4f", tp / (tp + fn) if (tp+fn) else 0)
    log.info("TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)
    log.info("Est.cost: $%.0f  (FN×$100 + FP×$0.5)", fn*100 + fp*0.5)
    log.info("\n%s", classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))


def run_training_pipeline() -> None:
    cfg = CONFIG["training"]
    df  = load_and_validate()
    X, y = df.drop("Class", axis=1), df["Class"]

    X_train, X_cal, X_test, y_train, y_cal, y_test = split_data(X, y, cfg)
    X_tr_s, X_cal_s, X_te_s, _ = scale(X_train, X_cal, X_test)

    spw        = compute_spw(y_train)
    base       = train_base(X_tr_s, y_train, X_cal_s, y_cal, spw)
    cal_model  = calibrate(base, X_cal_s, y_cal)
    threshold  = find_threshold(cal_model, X_cal_s, y_cal, beta=0.5)

    evaluate(cal_model, X_te_s, y_test, threshold)

    joblib.dump(cal_model, MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    run_training_pipeline()
