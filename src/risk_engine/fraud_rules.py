"""
Fraud rules orchestrator v2 — weighted scoring engine.

UPGRADE from v1:
────────────────────────────────────────────────────────────────────────────
v1: apply_fraud_rules() naively summed all scores and capped at 1.0.
    Problem: Rules could dominate over ML. NG transaction with ml_prob=0.00001
    would get total_risk_score=0.95 → BLOCK_CARD (false positive).

v2: Uses weighted_scorer.compute_weighted_score() which:
    1. Normalises rule scores against their max possible contribution
    2. Blends 60% ML + 40% rules (configurable)
    3. Returns a ScoringResult with full audit trail
    4. Passes ml_probability to calculate_risk() for confidence gate

The public interface is unchanged: apply_fraud_rules(transaction, ml_prob)
still returns a float (total_risk_score). The ScoringResult detail is
available via apply_fraud_rules_detailed() for logging and explanation.
"""

from src.risk_engine.vendor_rules      import check_vendor_rules, vendor_velocity_check
from src.risk_engine.country_rules     import check_country_risk
from src.risk_engine.merchant_rules    import merchant_amount_check
from src.risk_engine.user_behavior     import analyze_user_behavior
from src.risk_engine.velocity_rules    import velocity_check
from src.risk_engine.network_detection import detect_fraud_ring
from src.risk_engine.weighted_scorer   import RuleScores, compute_weighted_score, ScoringResult


def _collect_rule_scores(transaction: dict) -> RuleScores:
    """Run all rule modules and collect raw scores."""
    return RuleScores(
        country  = check_country_risk(transaction),
        merchant = merchant_amount_check(transaction),
        vendor   = check_vendor_rules(transaction),
        behavior = analyze_user_behavior(transaction),
        velocity = velocity_check(transaction),
        network  = detect_fraud_ring(transaction),
        attack   = _attack_score(transaction),
    )


def _attack_score(transaction: dict) -> float:
    """Coordinated attack patterns: vendor velocity + country co-occurrence."""
    score = 0.0
    if vendor_velocity_check(transaction):
        score += 0.30
    if check_country_risk(transaction) > 0:
        score += 0.10
    return min(score, 0.35)


def apply_fraud_rules(transaction: dict, ml_probability: float) -> float:
    """
    Public API — returns total_risk_score float.
    Use apply_fraud_rules_detailed() when you need the full audit trail.
    """
    return apply_fraud_rules_detailed(transaction, ml_probability).total_risk_score


def apply_fraud_rules_detailed(transaction: dict, ml_probability: float) -> ScoringResult:
    """
    Full scoring with audit trail.
    Returns ScoringResult containing all component scores and flags.
    """
    rule_scores = _collect_rule_scores(transaction)
    return compute_weighted_score(ml_probability, rule_scores)
