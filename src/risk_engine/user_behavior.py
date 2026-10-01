"""
User spending behaviour analysis.

Tracks a rolling history of each user's transaction amounts in memory.
Flags transactions that deviate significantly from the user's baseline.

Note: This state resets on process restart. For production persistence,
replace the in-memory dict with a Redis hash.
"""

from threading import Lock
from config import CONFIG

_CFG = CONFIG["user_behavior"]

HISTORY_LIMIT      : int   = int(_CFG["history_limit"])
ANOMALY_MULTIPLIER : float = float(_CFG["anomaly_multiplier"])

_user_histories: dict[str, list[float]] = {}
_lock = Lock()


def analyze_user_behavior(transaction: dict) -> float:
    user_id = str(transaction.get("user_id", "")).strip()
    amount  = float(transaction.get("Amount", 0) or 0)

    if not user_id:
        return 0.0

    with _lock:
        history = _user_histories.setdefault(user_id, [])

        risk = 0.0
        if history:
            avg = sum(history) / len(history)
            if avg > 0 and amount > avg * ANOMALY_MULTIPLIER:
                risk = 0.30

        history.append(amount)
        if len(history) > HISTORY_LIMIT:
            history.pop(0)

    return risk
