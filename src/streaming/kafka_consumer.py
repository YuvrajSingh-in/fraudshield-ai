"""
Kafka fraud detection consumer v2.

UPGRADES:
1. Uses apply_fraud_rules_detailed() for full audit trail in logs
2. Passes ml_probability to calculate_risk() for confidence gate
3. Dead-letter queue: transactions that fail 3 times go to 'fraud_dlq' topic
4. Idempotency key: transaction_id tracked to prevent double-processing
5. Explanation uses ScoringResult for precise per-rule attribution
"""

import sys, os, json, time, joblib, logging, traceback
from datetime import datetime, timezone
from collections import defaultdict
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import CONFIG
from src.ml.predict                       import predict_transaction
from src.risk_engine.fraud_rules          import apply_fraud_rules_detailed
from src.risk_engine.risk_scoring         import calculate_risk
from src.risk_engine.explain_fraud        import explain_from_scoring_result
from src.monitoring.transaction_logger    import log_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_K        = CONFIG["kafka"]
TOPIC_IN  = _K["topic_transactions"]
TOPIC_OUT = _K["topic_results"]
TOPIC_DLQ = "fraud_dlq"
SERVER    = _K["bootstrap_servers"]
GROUP_ID  = _K["group_id"]

# Idempotency: track recently-processed transaction IDs (capped at 10k)
_processed_ids: dict[str, float] = {}
_IDEMPOTENCY_TTL = 300  # seconds


def _load_models():
    mp, sp = CONFIG["model"]["path"], CONFIG["model"]["scaler_path"]
    if not os.path.exists(mp) or not os.path.exists(sp):
        raise FileNotFoundError(f"Models not found. Run: python -m src.ml.train_model")
    m = joblib.load(mp)
    s = joblib.load(sp)
    log.info("Models loaded.")
    return m, s


def _build_consumer():
    for attempt in range(1, 7):
        try:
            c = KafkaConsumer(
                TOPIC_IN,
                bootstrap_servers=SERVER,
                auto_offset_reset=_K["auto_offset_reset"],
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                group_id=GROUP_ID,
                enable_auto_commit=False,  # manual commit for reliability
                max_poll_records=50,
            )
            log.info("Consumer connected → %s", TOPIC_IN)
            return c
        except NoBrokersAvailable:
            log.warning("Kafka not ready (attempt %d/6) — waiting 5s...", attempt)
            time.sleep(5)
    raise RuntimeError("Cannot connect to Kafka.")


def _build_producer():
    return KafkaProducer(
        bootstrap_servers=SERVER,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        acks="all",
        retries=5,
        retry_backoff_ms=300,
    )


def _sanitise(txn: dict) -> dict:
    out = {}
    for k, v in txn.items():
        out[k] = v[0] if isinstance(v, list) and v else (None if isinstance(v, list) else v)
    for f in ("user_id", "merchant_id", "vendor_id", "merchant_category", "country"):
        out[f] = str(out.get(f) or "unknown").strip()
    return out


def _is_duplicate(txn: dict) -> bool:
    txn_id = txn.get("transaction_id")
    if not txn_id:
        return False
    now = time.time()
    # Evict old entries
    expired = [k for k, t in _processed_ids.items() if now - t > _IDEMPOTENCY_TTL]
    for k in expired:
        del _processed_ids[k]
    if txn_id in _processed_ids:
        return True
    _processed_ids[txn_id] = now
    return False


def process_transaction(txn: dict, model, scaler, producer: KafkaProducer) -> None:
    txn = _sanitise(txn)

    if _is_duplicate(txn):
        log.info("Duplicate transaction_id=%s — skipping", txn.get("transaction_id"))
        return

    # 1. ML inference
    ml_prob = predict_transaction(txn, model, scaler)

    # 2. Weighted rules engine — returns full ScoringResult
    scoring = apply_fraud_rules_detailed(txn, ml_prob)

    # 3. Risk classification with ML confidence gate
    risk_level, action = calculate_risk(
        scoring.total_risk_score,
        ml_probability=ml_prob,
    )

    # 4. Explanation with full audit trail
    explanation = explain_from_scoring_result(txn, scoring)

    result = {
        "fraud_probability"  : round(ml_prob, 6),
        "total_risk_score"   : round(scoring.total_risk_score, 6),
        "weighted_rule_score": round(scoring.weighted_rule_score, 6),
        "risk_level"         : risk_level,
        "action"             : action,
        "explanation"        : explanation,
        "rules_only_flag"    : scoring.rules_only_flag,
        "active_rules"       : scoring.active_rules,
        "rule_scores"        : scoring.rule_scores.to_dict(),
    }

    # 5. Log (UTF-8 safe)
    log_transaction(
        transaction=txn,
        result=result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # 6. Publish to results topic
    producer.send(TOPIC_OUT, {**result, "transaction": txn})

    log.info(
        "%-14s | %-4s | ML=%.4f | Total=%.4f | %-10s | %s",
        txn.get("user_id", "?")[:14],
        txn.get("country", "?"),
        ml_prob,
        scoring.total_risk_score,
        risk_level,
        action,
    )


def start_consumer() -> None:
    model, scaler  = _load_models()
    consumer       = _build_consumer()
    producer       = _build_producer()
    processed = errors = 0
    error_counts: defaultdict[str, int] = defaultdict(int)

    log.info("Fraud detection consumer v2 started.")

    try:
        for message in consumer:
            txn = message.value
            txn_key = str(txn.get("transaction_id", id(txn)))
            try:
                process_transaction(txn, model, scaler, producer)
                processed += 1
                consumer.commit()
            except Exception:
                errors += 1
                error_counts[txn_key] += 1
                log.error("Error on txn %s (attempt %d)", txn_key, error_counts[txn_key])
                traceback.print_exc()

                # Dead-letter queue after 3 failures
                if error_counts[txn_key] >= 3:
                    log.warning("Sending txn %s to DLQ after 3 failures", txn_key)
                    try:
                        producer.send(TOPIC_DLQ, txn)
                    except Exception:
                        log.error("DLQ send failed for %s", txn_key)
                    consumer.commit()  # don't reprocess infinitely

    except KeyboardInterrupt:
        log.info("Stopped. Processed=%d Errors=%d", processed, errors)
    finally:
        consumer.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    start_consumer()
