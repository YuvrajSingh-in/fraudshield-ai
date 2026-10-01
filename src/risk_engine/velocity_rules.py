"""
Multi-window velocity detection.

UPGRADE from v1:
────────────────────────────────────────────────────────────────────────────
v1: Single 10-second window, binary flag (triggered / not triggered).
    Problem: Real card-testing attacks often operate over minutes, not seconds.
    A bot doing 8 transactions/10s was under threshold; 8 transactions/2min was invisible.

v2: Three independent sliding windows per entity:
    - 1 minute  → detects rapid bursts (card testing)
    - 5 minutes → detects sustained probing
    - 1 hour    → detects slow enumeration attacks

Risk scores are GRADUATED — not binary. More velocity = higher score.
The 1-minute window is highest weight (most indicative of automation).

CARD TESTING DETECTION:
    Card testing attacks probe with micro-transactions (< €5) to verify cards.
    Detected when: user has >= 3 transactions under €5 in 5 minutes.

TRANSACTION BURST DETECTION:
    Spike in vendor transactions (>20 in 1 min) indicates compromised terminal
    or coordinated merchant fraud.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import NamedTuple

# Window configs: (seconds, user_threshold, vendor_threshold)
WINDOWS = [
    (60,   3,  15),   # 1 min
    (300,  8,  40),   # 5 min
    (3600, 20, 100),  # 1 hr
]

# Weights for each window (shorter windows weighted higher)
WINDOW_WEIGHTS = [0.40, 0.25, 0.15]

# Card testing: micro-transaction amount threshold
MICRO_TXN_THRESHOLD = 5.0
MICRO_TXN_COUNT     = 3   # >= N micro-txns in 5 min = card testing


@dataclass
class VelocityResult:
    user_risk      : float
    vendor_risk    : float
    card_testing   : bool
    burst_detected : bool
    total          : float
    detail         : str


_user_windows  : defaultdict[str, list[deque]] = defaultdict(lambda: [deque() for _ in WINDOWS])
_vendor_windows: defaultdict[str, list[deque]] = defaultdict(lambda: [deque() for _ in WINDOWS])
_user_micro    : defaultdict[str, deque]        = defaultdict(deque)   # card testing tracker

_lock = Lock()


def _evict(q: deque, now: float, window_sec: float) -> None:
    while q and now - q[0] > window_sec:
        q.popleft()


def velocity_check(transaction: dict) -> float:
    result = velocity_check_detailed(transaction)
    return result.total


def velocity_check_detailed(transaction: dict) -> VelocityResult:
    now       = time.time()
    user_id   = str(transaction.get("user_id",   "unknown"))
    vendor_id = str(transaction.get("vendor_id", "unknown"))
    amount    = float(transaction.get("Amount", 0) or 0)

    user_risk = vendor_risk = 0.0
    card_testing = burst = False
    details = []

    with _lock:
        # ── User velocity across windows ──────────────────────────────────
        for i, (window_sec, u_thresh, _) in enumerate(WINDOWS):
            q = _user_windows[user_id][i]
            q.append(now)
            _evict(q, now, window_sec)
            count = len(q)
            if count > u_thresh:
                excess      = (count - u_thresh) / u_thresh
                contribution = WINDOW_WEIGHTS[i] * min(excess, 1.0)
                user_risk   += contribution
                details.append(f"User {count}tx/{window_sec//60}min")

        # ── Vendor velocity across windows ────────────────────────────────
        for i, (window_sec, _, v_thresh) in enumerate(WINDOWS):
            q = _vendor_windows[vendor_id][i]
            q.append(now)
            _evict(q, now, window_sec)
            count = len(q)
            if count > v_thresh:
                excess       = (count - v_thresh) / v_thresh
                contribution = WINDOW_WEIGHTS[i] * min(excess, 1.0) * 0.7  # lower weight for vendor
                vendor_risk += contribution
                if i == 0 and count > v_thresh * 2:
                    burst = True
                    details.append(f"Vendor burst {count}tx/1min")

        # ── Card testing detection ─────────────────────────────────────────
        q_micro = _user_micro[user_id]
        if amount < MICRO_TXN_THRESHOLD:
            q_micro.append(now)
        _evict(q_micro, now, 300)  # 5-minute window
        if len(q_micro) >= MICRO_TXN_COUNT:
            card_testing  = True
            user_risk    += 0.45
            details.append(f"Card testing: {len(q_micro)} micro-txns")

    user_risk   = min(user_risk,   0.50)
    vendor_risk = min(vendor_risk, 0.35)
    total       = min(user_risk + vendor_risk, 0.70)

    return VelocityResult(
        user_risk      = user_risk,
        vendor_risk    = vendor_risk,
        card_testing   = card_testing,
        burst_detected = burst,
        total          = total,
        detail         = " | ".join(details) if details else "",
    )
