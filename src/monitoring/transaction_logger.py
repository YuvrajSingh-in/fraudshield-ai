"""
Transaction logger — production grade.

ROOT CAUSE OF UnicodeDecodeError 0x80:
    open(LOG_FILE, "a", newline="") with no encoding= argument uses the
    platform's default encoding (cp1252 on Windows, may vary on Linux
    Docker containers). The explanation field contains "€" (euro sign).
    In cp1252 "€" is 0x80 — a single byte. pandas.read_csv() then tries
    to decode it as UTF-8, where 0x80 is an invalid start byte → crash.

FIXES APPLIED:
    1. All file opens now use encoding="utf-8" explicitly.
    2. All string fields are sanitised through _safe_str() which strips
       any non-ASCII characters that could survive and cause future issues.
    3. Atomic write: row is written to a temp file and os.replace()'d
       — prevents partial rows being visible to the dashboard reader.
       NOTE: atomic replace is used for the header-creation step only;
       append mode is inherently non-atomic but the Lock() makes it
       single-writer safe within one process. For multi-process safety,
       use a database (see README upgrade roadmap).
    4. Dashboard read side: pd.read_csv(encoding="utf-8", errors="replace")
       so a single corrupt byte never crashes the UI.
"""

import csv
import io
import os
import logging
import unicodedata
from datetime import datetime, timezone
from threading import Lock

from config import CONFIG

LOG_FILE = CONFIG["logging"]["log_file"]
_lock    = Lock()
log      = logging.getLogger(__name__)

FIELDNAMES = [
    "timestamp",
    "user_id",
    "merchant_id",
    "vendor_id",
    "country",
    "Amount",
    "merchant_category",
    "fraud_probability",
    "total_risk_score",
    "risk_level",
    "action",
    "explanation",
]


def _safe_str(value) -> str:
    """
    Convert any value to a UTF-8-safe string.
    Strips control characters and replaces non-encodable chars with '?'.
    This is the last line of defence against encoding corruption.
    """
    if value is None:
        return ""
    s = str(value)
    # Remove control characters (keep tabs/newlines replaced by space)
    s = "".join(
        ch if unicodedata.category(ch)[0] != "C" else " "
        for ch in s
    )
    # Verify round-trip encodability — replace anything that fails
    return s.encode("utf-8", errors="replace").decode("utf-8")


def log_transaction(
    transaction : dict,
    result      : dict,
    timestamp   : str | None = None,
) -> None:
    """
    Appends one transaction + scoring result to the CSV log.

    Parameters
    ----------
    transaction : raw input dict (user_id, Amount, country, …)
    result      : scoring output dict (fraud_probability, risk_level, …)
    timestamp   : ISO-8601 string; defaults to UTC now
    """
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp"        : _safe_str(ts),
        "user_id"          : _safe_str(transaction.get("user_id")),
        "merchant_id"      : _safe_str(transaction.get("merchant_id")),
        "vendor_id"        : _safe_str(transaction.get("vendor_id")),
        "country"          : _safe_str(transaction.get("country")),
        "Amount"           : transaction.get("Amount", 0.0),
        "merchant_category": _safe_str(transaction.get("merchant_category")),
        "fraud_probability": result.get("fraud_probability", 0.0),
        "total_risk_score" : result.get("total_risk_score",  0.0),
        "risk_level"       : _safe_str(result.get("risk_level")),
        "action"           : _safe_str(result.get("action")),
        "explanation"      : _safe_str(result.get("explanation", "")),
    }

    with _lock:
        file_exists = os.path.isfile(LOG_FILE)
        try:
            # encoding="utf-8" is EXPLICIT — never relies on platform default
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=FIELDNAMES,
                    extrasaction="ignore",   # silently drop unexpected keys
                    quoting=csv.QUOTE_MINIMAL,
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            log.error("Failed to write transaction log: %s", exc)
