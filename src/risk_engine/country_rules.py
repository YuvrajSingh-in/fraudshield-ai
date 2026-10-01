"""
Country-based risk rules.

Returns a risk score additive to the ML probability.
Scores are calibrated so country risk alone cannot exceed the HIGH threshold.
"""

HIGH_RISK_COUNTRIES: set[str] = {"NG", "PK", "IR", "KP", "RU"}
MEDIUM_RISK_COUNTRIES: set[str] = {"UA", "BY", "VE", "SY"}

COUNTRY_RISK_SCORE: dict[str, float] = {
    "high":   0.40,
    "medium": 0.20,
}


def check_country_risk(transaction: dict) -> float:
    country = str(transaction.get("country", "")).upper().strip()
    if not country:
        return 0.0
    if country in HIGH_RISK_COUNTRIES:
        return COUNTRY_RISK_SCORE["high"]
    if country in MEDIUM_RISK_COUNTRIES:
        return COUNTRY_RISK_SCORE["medium"]
    return 0.0
