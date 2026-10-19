"""Replay transactions in time order through an in-process stream or Kafka."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class StreamMessage:
    topic: str
    payload: dict[str, Any]


class LocalStream:
    """Test double for Kafka. Messages stay in this process."""

    def __init__(self) -> None:
        self._topics: dict[str, deque[StreamMessage]] = defaultdict(deque)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._topics[topic].append(StreamMessage(topic, payload))

    def consume(self, topic: str, limit: int | None = None) -> list[StreamMessage]:
        bucket = self._topics[topic]
        messages = []
        while bucket and (limit is None or len(messages) < limit):
            messages.append(bucket.popleft())
        return messages


def transaction_payload(row: pd.Series) -> dict[str, Any]:
    """Build a gateway payload from a table row without copying the fraud label."""
    payload = {
        "transaction_id": str(row["TransactionID"]),
        "amount": float(row["TransactionAmt"]),
        "account_key": None if "card1" not in row or _missing(row["card1"]) else str(row["card1"]),
        "features": {},
        "categories": {},
    }
    for column, value in row.items():
        if column in {"isFraud", "TransactionID", "simulated_signal_origin"}:
            continue
        if _missing(value):
            continue
        if isinstance(value, str):
            payload["categories"][str(column)] = value
        else:
            try:
                payload["features"][str(column)] = float(value)
            except (TypeError, ValueError):
                payload["categories"][str(column)] = str(value)
    return payload


def replay_frame(
    frame: pd.DataFrame,
    publish: Callable[[str, dict[str, Any]], None],
    topic: str,
    limit: int | None = None,
) -> int:
    ordered = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
    sent = 0
    for _index, row in ordered.iterrows():
        if limit is not None and sent >= limit:
            break
        publish(topic, transaction_payload(row))
        sent += 1
    return sent


def score_messages(messages: list[StreamMessage], score_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    return [score_fn(message.payload) for message in messages]


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


@dataclass
class KafkaSettings:
    bootstrap_servers: str
    topic_in: str
    topic_out: str
    group_id: str = "fraud-engine"
    client_id: str = field(default="fraud-engine")


class KafkaStream:
    """Thin Kafka adapter. The import is delayed so tests can run without a broker."""

    def __init__(self, settings: KafkaSettings) -> None:
        self.settings = settings
        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError as exc:
            raise RuntimeError(
                "Kafka client library is not installed. The in-process replay still works. "
                "Install kafka-python or kafka-python-ng to connect to a broker."
            ) from exc
        self._producer = KafkaProducer(
            bootstrap_servers=settings.bootstrap_servers,
            client_id=settings.client_id,
            value_serializer=lambda item: __import__("json").dumps(item).encode("utf-8"),
        )
        self._consumer_cls = KafkaConsumer

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._producer.send(topic, payload)
        self._producer.flush()

    def consume(self, topic: str, limit: int = 10, timeout_ms: int = 2000) -> list[StreamMessage]:
        consumer = self._consumer_cls(
            topic,
            bootstrap_servers=self.settings.bootstrap_servers,
            group_id=self.settings.group_id,
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms,
            value_deserializer=lambda item: __import__("json").loads(item.decode("utf-8")),
        )
        messages: list[StreamMessage] = []
        try:
            for record in consumer:
                messages.append(StreamMessage(topic, record.value))
                if len(messages) >= limit:
                    break
        finally:
            consumer.close()
        return messages
