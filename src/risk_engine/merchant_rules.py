"""Merchant category spend-limit rules."""

MERCHANT_LIMITS: dict[str, float] = {
    "supermarket":       300.0,
    "restaurant":        200.0,
    "fuel_station":      250.0,
    "gas_station":       250.0,
    "electronics":      3000.0,
    "luxury":          10000.0,
    "online_marketplace": 2000.0,
    "travel":           5000.0,
}


def merchant_amount_check(transaction: dict) -> float:
    category = str(transaction.get("merchant_category", "")).lower().strip()
    amount   = float(transaction.get("Amount", 0) or 0)

    limit = MERCHANT_LIMITS.get(category)
    if limit is None:
        return 0.0

    return 0.35 if amount > limit else 0.0
