"""
Streaming pipeline runner.

Starts the Kafka producer and consumer in separate threads.
Use this for local development without Docker.

For production: run each service in a separate process or container.
"""

import logging
import signal
import sys
from threading import Thread

from src.streaming.kafka_producer import start_stream
from src.streaming.kafka_consumer import start_consumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def _handle_shutdown(sig, frame):
    log.info("Shutdown signal received — stopping pipeline.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    producer_thread = Thread(target=start_stream,    daemon=True, name="producer")
    consumer_thread = Thread(target=start_consumer,  daemon=True, name="consumer")

    producer_thread.start()
    consumer_thread.start()

    log.info("Pipeline running. Press Ctrl+C to stop.")
    producer_thread.join()
    consumer_thread.join()
