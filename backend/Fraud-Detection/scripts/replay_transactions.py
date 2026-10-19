"""Replay a CSV through the local stream or Kafka and print decisions.

The CSV needs TransactionID, TransactionDT, and TransactionAmt. isFraud is ignored
so a label cannot be fed into the gateway by accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fraud_engine.config import load_settings
from fraud_engine.service import FraudService
from fraud_engine.streaming.replay import KafkaSettings, KafkaStream, LocalStream, replay_frame, score_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay transactions through the fraud gateway.")
    parser.add_argument("csv_path")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--kafka", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv_path)
    settings = load_settings()
    service = FraudService(settings)
    if args.kafka:
        kafka_settings = settings["kafka"]
        stream = KafkaStream(
            KafkaSettings(
                bootstrap_servers=kafka_settings["bootstrap_servers"],
                topic_in=kafka_settings["topic_in"],
                topic_out=kafka_settings["topic_out"],
                group_id=kafka_settings["group_id"],
            )
        )
        topic = kafka_settings["topic_in"]
    else:
        stream = LocalStream()
        topic = settings["kafka"]["topic_in"]
    sent = replay_frame(frame, stream.publish, topic, limit=args.limit)
    messages = stream.consume(topic, limit=sent)
    decisions = score_messages(messages, service.score)
    for decision in decisions:
        print(
            f"{decision['transaction_id']} score={decision['fraud_score']:.4f} "
            f"decision={decision['decision']} model={decision['model_version']}"
        )


if __name__ == "__main__":
    main()
