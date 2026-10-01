"""
Kafka transaction producer.

Generates synthetic transactions and publishes them to the Kafka
'transactions' topic. Each transaction includes the 30 model features
(Time, Amount, V1-V28) plus contextual fields used by the rules engine.
"""

import json
import time
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

_CFG      = CONFIG["kafka"]
_P_CFG    = CONFIG["producer"]

TOPIC     = _CFG["topic_transactions"]
SERVER    = _CFG["bootstrap_servers"]
SLEEP     = float(_P_CFG["sleep_interval"])
COUNTRIES = _P_CFG["countries"]
CATEGORIES= _P_CFG["merchant_categories"]
MODEL_FEATURES = CONFIG["model"]["features"]

ATTACK_SAMPLE_RATE = float(_P_CFG.get("attack_sample_rate", 0.18))
FRAUD_SAMPLE_RATE  = float(_P_CFG.get("fraud_sample_rate", 0.08))
ATTACK_USER_POOL   = max(int(_P_CFG.get("attack_user_pool", 5)), 1)
ATTACK_VENDOR_POOL = max(int(_P_CFG.get("attack_vendor_pool", 3)), 1)

ATTACK_USERS = [f"USER_ATTACK_{i}" for i in range(1, ATTACK_USER_POOL + 1)]
ATTACK_MERCHANTS = [f"MERCHANT_ATTACK_{i}" for i in range(1, ATTACK_VENDOR_POOL + 1)]
ATTACK_VENDORS = [f"VENDOR_ATTACK_{i}" for i in range(1, ATTACK_VENDOR_POOL + 1)]
HIGH_RISK_COUNTRIES = [c for c in COUNTRIES if c in {"NG", "PK", "RU", "IR", "KP"}] or COUNTRIES
ATTACK_CATEGORIES = [c for c in CATEGORIES if c in {"gas_station", "restaurant", "online_marketplace", "supermarket"}] or CATEGORIES


def _load_reference_fraud_rows() -> list[dict]:
    data_path = Path(CONFIG["training"]["data_path"])
    if not data_path.exists():
        log.warning("Reference dataset missing at %s; producer will use synthetic-only features.", data_path)
        return []

    try:
        df = pd.read_csv(data_path, usecols=MODEL_FEATURES + ["Class"])
        fraud_rows = df[df["Class"] == 1][MODEL_FEATURES]
        records = fraud_rows.to_dict(orient="records")
        log.info("Loaded %d reference fraud rows for producer scenarios.", len(records))
        return records
    except Exception as exc:
        log.warning("Could not load reference fraud rows: %s", exc)
        return []


REFERENCE_FRAUD_ROWS = _load_reference_fraud_rows()


def _build_producer() -> KafkaProducer:
    for attempt in range(1, 6):
        try:
            p = KafkaProducer(
                bootstrap_servers=SERVER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            log.info("Kafka producer connected to %s", SERVER)
            return p
        except NoBrokersAvailable:
            log.warning("Kafka not available (attempt %d/5) — retrying in 3s...", attempt)
            time.sleep(3)
    raise RuntimeError("Cannot connect to Kafka after 5 attempts.")


def _synthetic_feature_block() -> dict:
    txn = {
        "Time": int(random.randint(1, 172800)),
        "Amount": round(random.uniform(1.0, 3000.0), 2),
    }
    # PCA-transformed features — normally distributed as in creditcard.csv
    for i in range(1, 29):
        txn[f"V{i}"] = round(float(np.random.normal(0, 1)), 6)
    return txn


def _reference_fraud_feature_block() -> dict:
    if not REFERENCE_FRAUD_ROWS:
        return _synthetic_feature_block()

    row = random.choice(REFERENCE_FRAUD_ROWS)
    features = {}
    for feature in MODEL_FEATURES:
        value = float(row.get(feature, 0.0) or 0.0)
        if feature == "Time":
            features[feature] = int(value)
        elif feature == "Amount":
            features[feature] = round(value, 2)
        else:
            features[feature] = round(value, 6)
    return features


def _benign_transaction() -> dict:
    txn = {
        "user_id"          : f"USER_{random.randint(1000, 5000)}",
        "merchant_id"      : f"MERCHANT_{random.randint(1, 50)}",
        "vendor_id"        : f"VENDOR_{random.randint(1, 200)}",
        "merchant_category": random.choice(CATEGORIES),
        "country"          : random.choice(COUNTRIES),
    }
    txn.update(_synthetic_feature_block())
    return txn


def _attack_transaction(use_reference_fraud_features: bool) -> dict:
    txn = {
        "user_id": random.choice(ATTACK_USERS),
        "merchant_id": random.choice(ATTACK_MERCHANTS),
        "vendor_id": random.choice(ATTACK_VENDORS),
        "merchant_category": random.choice(ATTACK_CATEGORIES),
        "country": random.choice(HIGH_RISK_COUNTRIES),
    }
    txn.update(
        _reference_fraud_feature_block()
        if use_reference_fraud_features
        else _synthetic_feature_block()
    )

    if use_reference_fraud_features:
        txn["Amount"] = round(
            max(float(txn["Amount"]), random.choice([350.0, 650.0, 1200.0])),
            2,
        )
    else:
        if random.random() < 0.35:
            txn["Amount"] = round(random.uniform(1.0, 4.5), 2)
        else:
            txn["Amount"] = round(random.uniform(350.0, 3000.0), 2)

    return txn


def generate_transaction() -> dict:
    roll = random.random()
    if roll < FRAUD_SAMPLE_RATE:
        return _attack_transaction(use_reference_fraud_features=True)
    if roll < FRAUD_SAMPLE_RATE + ATTACK_SAMPLE_RATE:
        return _attack_transaction(use_reference_fraud_features=False)
    return _benign_transaction()


def start_stream() -> None:
    producer = _build_producer()
    log.info("Transaction stream started → topic '%s'", TOPIC)
    sent = 0
    try:
        while True:
            txn = generate_transaction()
            producer.send(TOPIC, txn)
            sent += 1
            if sent % 100 == 0:
                log.info("Produced %d transactions", sent)
            time.sleep(SLEEP)
    except KeyboardInterrupt:
        log.info("Producer stopped. Total sent: %d", sent)
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    start_stream()
