"""
Vendor-based risk rules.

BUG FIXED: The original code shared VENDOR_TRANSACTION_COUNT between
check_vendor_rules() and vendor_velocity_check(), causing double increments
when both were called in the same request. They now use separate counters.

kafka_consumer.py calls check_vendor_rules() directly.
fraud_rules.py / detect_attack_patterns() calls vendor_velocity_check().
These are now independent and do NOT share state.
"""

from threading import Lock

# ─── Independent counters ─────────────────────────────────────────────────────

_VENDOR_COUNT_RULES: dict[str, int] = {}
_VENDOR_COUNT_VELOCITY: dict[str, int] = {}

_RULES_LOCK    = Lock()
_VELOCITY_LOCK = Lock()

VENDOR_THRESHOLD_RULES    = 20
VENDOR_THRESHOLD_VELOCITY = 30


def _normalise_vendor_id(transaction: dict) -> str | None:
    vendor_id = transaction.get("vendor_id")
    if isinstance(vendor_id, list):
        vendor_id = vendor_id[0] if vendor_id else None
    if vendor_id is None:
        return None
    return str(vendor_id).strip()


def check_vendor_rules(transaction: dict) -> float:
    """
    Flags a vendor that has processed more than VENDOR_THRESHOLD_RULES
    transactions since process start. Used by the Kafka consumer directly.
    """
    vendor_id = _normalise_vendor_id(transaction)
    if not vendor_id:
        return 0.0

    with _RULES_LOCK:
        _VENDOR_COUNT_RULES[vendor_id] = _VENDOR_COUNT_RULES.get(vendor_id, 0) + 1
        count = _VENDOR_COUNT_RULES[vendor_id]

    return 0.30 if count > VENDOR_THRESHOLD_RULES else 0.0


def vendor_velocity_check(transaction: dict) -> float:
    """
    Flags a vendor that has processed more than VENDOR_THRESHOLD_VELOCITY
    transactions since process start. Used exclusively by fraud_rules.py.
    """
    vendor_id = _normalise_vendor_id(transaction)
    if not vendor_id:
        return 0.0

    with _VELOCITY_LOCK:
        _VENDOR_COUNT_VELOCITY[vendor_id] = _VENDOR_COUNT_VELOCITY.get(vendor_id, 0) + 1
        count = _VENDOR_COUNT_VELOCITY[vendor_id]

    return 0.40 if count > VENDOR_THRESHOLD_VELOCITY else 0.0
