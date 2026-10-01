"""
Weighted risk scoring engine — replaces naive score addition.

WHY NAIVE ADDITION FAILS:
────────────────────────────────────────────────────────────────────────────
The original system added all risk scores directly:
    total = ml_prob + country(0.40) + merchant(0.35) + vendor(0.30) + ...

Problems:
1. Country "NG" always adds +0.40 regardless of ML confidence.
   A transaction with ml_prob=0.001 from Nigeria gets 0.401 → MEDIUM.
   That's a false positive generated entirely by a static rule.

2. Scores are unbounded before the cap — a transaction can accumulate
   1.85 before capping to 1.0, losing all proportionality information.

3. Rules with equal weights (0.30, 0.35, 0.40) have no calibration basis.
   Why is merchant_limit worth 0.35 and country worth 0.40? No data backs it.

WEIGHTED APPROACH:
────────────────────────────────────────────────────────────────────────────
1. ML probability is the anchor — rules only boost, never dominate.
2. Each rule has a configurable weight and a minimum ML confidence gate.
   Below ml_conf_gate, country/merchant rules contribute less.
3. Final score uses a weighted blend: 60% ML + 40% rules.
4. ML confidence gate: if ml_prob < MIN_ML_CONFIDENCE, the result is
   flagged as RULES_ONLY and the action is capped at SOFT_CHECK.

MINIMUM ML CONFIDENCE CUTOFF:
────────────────────────────────────────────────────────────────────────────
Transactions where the ML model has near-zero confidence (< 0.01) should
NOT be escalated to FREEZE/BLOCK purely on rules. This is a post-processing
filter that dramatically reduces false positives for high-rule-score,
low-ML-confidence cases (the exact profile that was dominating the original
transactions.csv with NG/RU transactions at ml_prob=0.00001, total=0.95).
"""

from dataclasses import dataclass, field

from config import CONFIG

_SCORING_CFG = CONFIG.get("scoring", {})
_RISK_CFG = CONFIG.get("risk", {})

# Minimum ML probability before rules can push to FREEZE or BLOCK
MIN_ML_CONFIDENCE_FOR_HARD_ACTION = float(
    _RISK_CFG.get("min_confidence_for_hard_action", 0.05)
)

# Rule weights — tunable without retraining
_DEFAULT_RULE_WEIGHTS = {
    "country": 0.35,
    "merchant": 0.25,
    "vendor": 0.20,
    "behavior": 0.25,
    "velocity": 0.35,
    "network": 0.30,
    "attack": 0.20,
}
RULE_WEIGHTS = {
    name: float(_SCORING_CFG.get("rule_weights", {}).get(name, default))
    for name, default in _DEFAULT_RULE_WEIGHTS.items()
}

# Each rule module returns a bounded raw score. Normalize against that ceiling
# instead of sum(weights), otherwise even stacked rule hits stay artificially low.
RULE_SCORE_CAPS = {
    "country": 0.40,
    "merchant": 0.35,
    "vendor": 0.30,
    "behavior": 0.30,
    "velocity": 0.70,
    "network": 0.40,
    "attack": 0.35,
}

# ML vs rules blend ratio (must sum to 1.0)
ML_WEIGHT = float(_SCORING_CFG.get("ml_weight", 0.60))
RULES_WEIGHT = float(_SCORING_CFG.get("rules_weight", 0.40))
_BLEND_TOTAL = ML_WEIGHT + RULES_WEIGHT
if _BLEND_TOTAL > 0:
    ML_WEIGHT /= _BLEND_TOTAL
    RULES_WEIGHT /= _BLEND_TOTAL
else:
    ML_WEIGHT = 0.60
    RULES_WEIGHT = 0.40


@dataclass
class RuleScores:
    country  : float = 0.0
    merchant : float = 0.0
    vendor   : float = 0.0
    behavior : float = 0.0
    velocity : float = 0.0
    network  : float = 0.0
    attack   : float = 0.0

    def to_dict(self) -> dict:
        return {
            "country"  : self.country,
            "merchant" : self.merchant,
            "vendor"   : self.vendor,
            "behavior" : self.behavior,
            "velocity" : self.velocity,
            "network"  : self.network,
            "attack"   : self.attack,
        }


@dataclass
class ScoringResult:
    ml_probability      : float
    rule_scores         : RuleScores
    weighted_rule_score : float
    total_risk_score    : float
    ml_anchored_score   : float
    rules_only_flag     : bool        # True if ML confidence too low for hard actions
    active_rules        : list[str] = field(default_factory=list)


def compute_weighted_score(
    ml_probability: float,
    rule_scores:    RuleScores,
) -> ScoringResult:
    """
    Combines ML probability and rule scores into a calibrated total.

    Algorithm:
    1. Compute weighted rule contribution (each rule × its weight, normalised)
    2. Blend: total = ML_WEIGHT × ml_prob + RULES_WEIGHT × rule_contribution
    3. Apply ML confidence gate for hard actions
    4. Cap at 1.0

    Returns a ScoringResult with full audit trail.
    """
    scores   = rule_scores.to_dict()
    weights  = RULE_WEIGHTS

    # Weighted rule score normalized against each rule's real ceiling.
    max_possible_rules = sum(RULE_SCORE_CAPS[k] * weights[k] for k in weights)
    raw_rule_total = sum(
        min(max(float(scores[k]), 0.0), RULE_SCORE_CAPS[k]) * weights[k]
        for k in scores
    )
    normalised_rules = (
        raw_rule_total / max_possible_rules if max_possible_rules > 0 else 0.0
    )

    # Blended score
    blended = ML_WEIGHT * ml_probability + RULES_WEIGHT * normalised_rules
    blended = min(blended, 1.0)

    # Active rules (for explanation)
    active = [k for k, v in scores.items() if v > 0]

    # ML confidence gate
    rules_only = ml_probability < MIN_ML_CONFIDENCE_FOR_HARD_ACTION

    return ScoringResult(
        ml_probability      = ml_probability,
        rule_scores         = rule_scores,
        weighted_rule_score = normalised_rules,
        total_risk_score    = blended,
        ml_anchored_score   = blended,
        rules_only_flag     = rules_only,
        active_rules        = active,
    )
