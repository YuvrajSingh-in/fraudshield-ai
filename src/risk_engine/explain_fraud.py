"""
Fully auditable fraud explanation generator.

UPGRADE: Now uses ScoringResult for precise, rule-specific explanations
instead of re-running checks or guessing from amounts.
"""

from src.risk_engine.country_rules   import HIGH_RISK_COUNTRIES
from src.risk_engine.merchant_rules  import MERCHANT_LIMITS


def explain_from_scoring_result(
    transaction:   dict,
    scoring_result,  # ScoringResult — avoid circular import with type hint
) -> str:
    """Generate human-readable explanation from a ScoringResult."""
    reasons = []
    rs      = scoring_result.rule_scores
    ml      = scoring_result.ml_probability

    if ml >= 0.30:
        reasons.append(f"ML model confidence: {ml:.1%}")

    if rs.country > 0:
        country = transaction.get("country", "?")
        reasons.append(f"High-risk country ({country})")

    if rs.merchant > 0:
        cat    = transaction.get("merchant_category", "?")
        amount = float(transaction.get("Amount", 0))
        limit  = MERCHANT_LIMITS.get(str(cat).lower(), "?")
        reasons.append(f"Amount {amount:.2f} exceeds {cat} limit ({limit})")

    if rs.velocity > 0:
        reasons.append("Velocity threshold exceeded")

    if rs.behavior > 0:
        reasons.append("Unusual spending pattern for this user")

    if rs.network > 0:
        reasons.append("User in suspicious transaction network")

    if rs.vendor > 0:
        reasons.append("Vendor high-volume anomaly")

    if rs.attack > 0:
        reasons.append("Coordinated attack pattern detected")

    if scoring_result.rules_only_flag:
        reasons.append("[Low ML confidence — rules only]")

    if not reasons:
        reasons.append("Normal transaction profile")

    return " | ".join(reasons)


def explain_transaction(transaction: dict, total_risk_score: float = 0.0) -> str:
    """Legacy interface for callers that don't have a ScoringResult."""
    reasons = []
    amount  = float(transaction.get("Amount", 0) or 0)
    cat     = str(transaction.get("merchant_category", "")).lower()
    country = str(transaction.get("country", "")).upper()

    if country in HIGH_RISK_COUNTRIES:
        reasons.append(f"High-risk country ({country})")
    if cat in MERCHANT_LIMITS and amount > MERCHANT_LIMITS[cat]:
        reasons.append(f"Amount {amount:.2f} exceeds {cat} limit")
    if amount > 5000:
        reasons.append("Very large transaction")
    if total_risk_score >= 0.85:
        reasons.append("Multiple high-severity risk signals")

    return " | ".join(reasons) if reasons else "ML model detected unusual pattern"
