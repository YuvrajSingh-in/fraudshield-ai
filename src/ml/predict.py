"""
ML inference — reads dynamic threshold from models/threshold.json.

The threshold is no longer hardcoded to 0.5. It is computed during training
by maximising F-beta(0.5) on the calibration set and saved to a JSON file.
This means you can retune the operating point without retraining.
"""

import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

MODEL_FEATURES: list[str] = CONFIG["model"]["features"]

_THRESHOLD_PATH = Path("models/threshold.json")
_DEFAULT_THRESHOLD = 0.5


def load_threshold() -> float:
    try:
        with open(_THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        t = float(data["threshold"])
        log.info("Loaded inference threshold: %.4f", t)
        return t
    except (FileNotFoundError, KeyError, ValueError):
        log.warning(
            "threshold.json not found or invalid — using default %.2f",
            _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD


# Loaded once at import time — fast at inference
INFERENCE_THRESHOLD: float = load_threshold()


def predict_transaction(transaction: dict, model, scaler) -> float:
    """
    Returns a fraud probability in [0.0, 1.0].
    The returned value is a calibrated probability — not a raw XGBoost score.
    Compare against INFERENCE_THRESHOLD to get a binary decision,
    or pass the raw float to the rules engine for additive scoring.
    """
    df = pd.DataFrame([transaction])

    for feature in MODEL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0

    df = df[MODEL_FEATURES]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X        = df.to_numpy()
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    return float(np.clip(prob, 0.0, 1.0))
