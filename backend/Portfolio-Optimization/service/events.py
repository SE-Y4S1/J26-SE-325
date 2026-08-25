"""Kafka producer for decision events.

The TAF makes Kafka the platform's integration bus: "all four subsystems communicate
asynchronously through a centralized event broker (e.g., Kafka/EventBridge), ensuring each
component can be developed, evaluated, and scaled independently".

FIRE-AND-FORGET, DELIBERATELY. If the broker is down, slow, or simply not running on a
teammate's laptop, this component must still answer requests. A withdrawal plan is worthless
if it arrives late because an audit log was unreachable, so publish failures are logged and
swallowed -- never raised into the request path. NullProducer makes "no broker" a
first-class, silent mode rather than an error condition.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)

DECISIONS_TOPIC = "portfolio.decisions"

# Schema version on every envelope. Components 3 and 4 consume these events and will evolve
# at a different pace than this component; a version field is what lets them reject or adapt
# to an envelope shape they do not recognise instead of mis-parsing it.
EVENT_SCHEMA_VERSION = "1.0"


class DecisionProducer(Protocol):
    def publish(self, payload: dict[str, Any], *, key: str | None = None) -> None: ...
    def flush(self, timeout: float = 1.0) -> None: ...
    @property
    def connected(self) -> bool: ...


class NullProducer:
    """No-op producer used when Kafka is unavailable or disabled (unit tests, solo dev)."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    def publish(self, payload: dict[str, Any], *, key: str | None = None) -> None:
        # Retained in memory so tests and local runs can still assert on what WOULD have
        # been sent, without needing a broker.
        self.published.append(payload)
        logger.debug("NullProducer swallowed event %s", payload.get("event_id"))

    def flush(self, timeout: float = 1.0) -> None:
        return None

    @property
    def connected(self) -> bool:
        return False


class KafkaDecisionProducer:
    """confluent-kafka producer for `portfolio.decisions`."""

    def __init__(self, bootstrap_servers: str, *, topic: str = DECISIONS_TOPIC) -> None:
        from confluent_kafka import Producer

        self.topic = topic
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                # Short timeouts: this is a non-critical audit path and must never become a
                # latency source for the withdrawal endpoint.
                "socket.timeout.ms": 3000,
                "message.timeout.ms": 5000,
                "queue.buffering.max.ms": 100,
            }
        )
        self._delivery_failures = 0

    def _on_delivery(self, error, message) -> None:  # noqa: ANN001 - confluent's callback shape
        if error is not None:
            self._delivery_failures += 1
            logger.warning("Kafka delivery failed (%d total): %s", self._delivery_failures, error)

    def publish(self, payload: dict[str, Any], *, key: str | None = None) -> None:
        """Async produce. Delivery failures are logged in the callback, never raised."""
        try:
            self._producer.produce(
                self.topic,
                key=key.encode("utf-8") if key else None,
                value=json.dumps(payload, default=str).encode("utf-8"),
                callback=self._on_delivery,
            )
            # Serve delivery callbacks without blocking. A full local queue raises
            # BufferError, which is caught below rather than propagating to the caller.
            self._producer.poll(0)
        except Exception as exc:  # noqa: BLE001 - the entire point is not to raise
            logger.warning("failed to enqueue decision event: %s", exc)

    def flush(self, timeout: float = 1.0) -> None:
        try:
            self._producer.flush(timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kafka flush failed: %s", exc)

    @property
    def connected(self) -> bool:
        """Best-effort liveness. Reported on /health, never used to gate a request."""
        try:
            metadata = self._producer.list_topics(timeout=2.0)
            return bool(metadata.brokers)
        except Exception:  # noqa: BLE001
            return False


_producer_cache: dict[str, DecisionProducer] = {}


def get_producer() -> DecisionProducer:
    """Resolve a producer from settings; falls back to NullProducer if Kafka is off.

    Cached because constructing a Producer opens sockets and spawns a background thread --
    doing that per request would leak both.
    """
    if "producer" in _producer_cache:
        return _producer_cache["producer"]

    from service.deps import get_settings

    settings = get_settings()
    if not settings.kafka_enabled:
        logger.info("Kafka disabled by configuration; using NullProducer")
        producer: DecisionProducer = NullProducer()
    else:
        try:
            producer = KafkaDecisionProducer(settings.kafka_bootstrap_servers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not construct Kafka producer (%s); using NullProducer", exc)
            producer = NullProducer()

    _producer_cache["producer"] = producer
    return producer


def reset_producer() -> None:
    """Drop the cached producer. Testing and reconfiguration only."""
    _producer_cache.pop("producer", None)


def build_decision_event(
    decision_type: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    model_version: str,
) -> dict[str, Any]:
    """Wrap a decision in the envelope Components 3 and 4 consume.

    Carries model_version at the TOP level, not just nested in the response, so Component 3
    can resolve it against the registry and anchor provenance without parsing the whole
    payload or calling back into this service.
    """
    return {
        "event_id": str(uuid4()),
        "event_type": f"portfolio.{decision_type}",
        "schema_version": EVENT_SCHEMA_VERSION,
        "component": "component1_portfolio_optimization",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "request": request_payload,
        "response": response_payload,
    }


def publish_decision(
    decision_type: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    model_version: str,
) -> dict[str, Any]:
    """Build and fire an event. Returns the envelope so the caller can log or assert on it."""
    event = build_decision_event(decision_type, request_payload, response_payload, model_version)
    get_producer().publish(event, key=event["event_type"])
    return event
