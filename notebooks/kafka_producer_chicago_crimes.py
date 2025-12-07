# spark-apps/kafka_producer_chicago_crimes.py
import csv
import json
import time
import argparse
from kafka import KafkaProducer


def build_producer(bootstrap_servers):
    """
    Create a KafkaProducer that sends JSON-encoded messages.
    """
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8") if v is not None else None,
        linger_ms=5,
        batch_size=32 * 1024,
        acks="all",
        retries=5,
    )


def stream_csv_to_kafka(
    csv_path,
    topic,
    bootstrap_servers,
    sleep_sec,
    max_rows,
):
    producer = build_producer(bootstrap_servers)

    sent = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            # Optional key: use ID if present, else row index
            key = row.get("ID", str(row_idx))

            producer.send(topic, key=key, value=row)

            sent += 1
            if sent % 5000 == 0:
                producer.flush()
                print(f"[producer] Sent {sent} records so far...")

            if max_rows is not None and sent >= max_rows:
                print(f"[producer] Reached max_rows={max_rows}, stopping.")
                break

            if sleep_sec > 0:
                time.sleep(sleep_sec)

    producer.flush()
    producer.close()
    print(f"[producer] Done. Total records sent: {sent}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream Chicago crime CSV to Kafka as JSON."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/home/jovyan/data/Chicago_Crimes_2001_to_Present.csv",
        help="Path inside Jupyter is /home/jovyan/data/Chicago_Crimes_2001_to_Present.csv",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="chicago-crime-stream",
        help="Kafka topic to publish to.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        nargs="+",
        default=["kafka:9092"], 
        help="Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional delay between messages (e.g. 0.001 to slow down).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="For testing, limit the number of rows to send.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        f"[producer] Streaming from {args.csv_path} "
        f"to topic={args.topic}, bootstrap={args.bootstrap_servers}"
    )
    stream_csv_to_kafka(
        csv_path=args.csv_path,
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        sleep_sec=args.sleep_sec,
        max_rows=args.max_rows,
    )
