"""
Risk scoring — four-tier decision engine with ML confidence gate.

UPGRADE: Added SOFT_CHECK tier and ML confidence gate.

Decision tiers:
    ALLOW       → < 0.30  → pass through
    SOFT_CHECK  → 0.30-0.55 → trigger OTP / step-up auth
    REVIEW      → 0.55-0.75 → human review queue
    FREEZE_CARD → 0.75-0.90 → card freeze + notify
    BLOCK_CARD  → >= 0.90   → hard block

ML CONFIDENCE GATE:
    If ml_probability < 0.05 (model has near-zero confidence it's fraud),
    the maximum action is capped at SOFT_CHECK regardless of rules.
    This prevents pure-rules false positives where NG/RU transactions
    get BLOCK_CARD with ml_prob=0.00001.
"""

from src.risk_engine.weighted_scorer import MIN_ML_CONFIDENCE_FOR_HARD_ACTION
from config import CONFIG

_T = CONFIG["risk"]["thresholds"]
LOW_THRESHOLD = float(_T.get("low", 0.30))
MEDIUM_THRESHOLD = float(_T.get("medium", 0.55))
HIGH_THRESHOLD = float(_T.get("high", 0.75))
VERY_HIGH_THRESHOLD = float(_T.get("very_high", 0.90))


def calculate_risk(
    total_risk_score: float,
    ml_probability:   float = 1.0,   # default=1.0 → gate disabled if not passed
) -> tuple[str, str]:
    """
    Maps an aggregate risk score to (risk_level, action).

    Parameters
    ----------
    total_risk_score : weighted combined score [0, 1]
    ml_probability   : raw ML output — used for confidence gate
    """
    p = float(total_risk_score)

    # ── ML confidence gate ────────────────────────────────────────────────
    # If the model has near-zero confidence, cap at SOFT_CHECK
    ml_low_confidence = float(ml_probability) < MIN_ML_CONFIDENCE_FOR_HARD_ACTION

    if p < LOW_THRESHOLD:
        return "LOW", "ALLOW"
    elif p < MEDIUM_THRESHOLD:
        return "MEDIUM", "SOFT_CHECK"
    elif p < HIGH_THRESHOLD:
        if ml_low_confidence:
            return "MEDIUM", "SOFT_CHECK"   # gate: downgrade REVIEW to SOFT_CHECK
        return "HIGH", "REVIEW"
    elif p < VERY_HIGH_THRESHOLD:
        if ml_low_confidence:
            return "MEDIUM", "SOFT_CHECK"   # gate: downgrade FREEZE to SOFT_CHECK
        return "HIGH", "FREEZE_CARD"
    else:
        if ml_low_confidence:
            return "HIGH", "REVIEW"          # gate: downgrade BLOCK to REVIEW
        return "VERY_HIGH", "BLOCK_CARD"
