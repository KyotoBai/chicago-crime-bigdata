# spark-apps/kafka_producer_chicago_crimes.py
import csv
import json
import time
import argparse
import os
from datetime import datetime
from kafka import KafkaProducer
import requests

class ChicagoCrimeAPIProducer:
    """Producer that fetches from Chicago Crime API and sends to Kafka"""
    
    def __init__(self, bootstrap_servers, topic, api_endpoint, batch_size=50000):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda v: v.encode("utf-8") if v is not None else None,
            linger_ms=10,
            batch_size=64 * 1024,
            acks="all",
            retries=5,
        )
        self.topic = topic
        self.api_endpoint = api_endpoint
        self.batch_size = batch_size
        self.watermark_file = "/home/jovyan/data/chicago_crime_watermark.json"
    
    def get_last_watermark(self):
        """Read the last processed timestamp from watermark file"""
        try:
            with open(self.watermark_file, 'r') as f:
                data = json.load(f)
            v = data.get("last_date")
            return str(v) if v else None
        except FileNotFoundError:
            return None
    
    def save_watermark(self, last_date):
        """Save the latest processed timestamp as an ISO string (JSON-safe)."""
        if last_date is None:
            return

        # Normalize to ISO-8601 string for JSON + Socrata queries
        if isinstance(last_date, datetime):
            last_date = last_date.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            last_date = str(last_date)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.watermark_file), exist_ok=True)

        with open(self.watermark_file, "w") as f:
            json.dump({"last_date": last_date}, f)
    
    def fetch_data(self, where_clause=None, limit=None, offset=0, order='ASC'):
        """
        Fetch data from Chicago Crime API
        API docs: https://dev.socrata.com/foundry/data.cityofchicago.org/ijzp-q8t2
        
        Args:
            order: 'ASC' for old to new (default), 'DESC' for new to old
        """
        params = {
            '$limit': limit or self.batch_size,
            '$offset': offset,
            '$order': f'date {order}'  # Control sort order
        }
        
        if where_clause:
            params['$where'] = where_clause
        
        try:
            response = requests.get(self.api_endpoint, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            return []
    
    def initial_full_load(self, max_records=None):
        """
        Load all historical data from the API (OLD to NEW order)
        """
        print("=" * 70)
        print("[PRODUCER] Starting FULL LOAD of all historical data...")
        print("[PRODUCER] Reading from OLDEST to NEWEST")
        print("=" * 70)
        
        offset = 0
        total_sent = 0
        last_date_seen = None
        
        while True:
            print(f"[PRODUCER] Fetching batch at offset {offset}...")
            # Fetch in ASC order (old to new)
            records = self.fetch_data(limit=self.batch_size, offset=offset, order='ASC')
            
            if not records:
                print("[PRODUCER] No more records to fetch")
                break
            
            # Send to Kafka
            for i, record in enumerate(records):
                key = record.get("id", str(offset + i))
                self.producer.send(self.topic, key=key, value=record)
                total_sent += 1
                
                # Track the most recent date we've seen
                record_date = record.get('date')
                if record_date:
                    last_date_seen = record_date
            
            # Flush periodically
            if total_sent % 10000 == 0:
                self.producer.flush()
                print(f"[PRODUCER] Sent {total_sent} records so far... (latest: {last_date_seen})")
            
            # Check if we should stop
            if max_records and total_sent >= max_records:
                print(f"[PRODUCER] Reached max_records limit: {max_records}")
                break
            
            offset += len(records)
            
            # If we got fewer records than batch_size, we're done
            if len(records) < self.batch_size:
                break
        
        self.producer.flush()
        
        # Save watermark (the last date we actually processed)
        if last_date_seen:
            self.save_watermark(last_date_seen)
            print(f"[PRODUCER] Watermark saved: {last_date_seen}")
            print(f"[PRODUCER] Next incremental update will fetch records > {last_date_seen}")
        
        print("=" * 70)
        print(f"[PRODUCER] FULL LOAD complete. Total records sent: {total_sent}")
        print("=" * 70)
        return total_sent
    
    def incremental_update(self):
        """
        Fetch only new records since last watermark (OLD to NEW order)
        """
        last_date = self.get_last_watermark()
        
        if not last_date:
            print("[WARNING] No watermark found. Running full load instead...")
            return self.initial_full_load()
        
        print("=" * 70)
        print(f"[PRODUCER] Starting INCREMENTAL UPDATE since {last_date}...")
        print("[PRODUCER] Reading from OLDEST to NEWEST")
        print("=" * 70)
        
        # Socrata API uses ISO 8601 format
        # Query for records where date > last_date
        where_clause = f"date > '{last_date}'"
        
        offset = 0
        total_sent = 0
        last_date_seen = last_date  # Start with the old watermark
        
        while True:
            # Fetch in ASC order (old to new)
            records = self.fetch_data(
                where_clause=where_clause,
                limit=self.batch_size,
                offset=offset,
                order='ASC'
            )
            
            if not records:
                break
            
            # Send to Kafka
            for i, record in enumerate(records):
                key = record.get("id", str(offset + i))
                self.producer.send(self.topic, key=key, value=record)
                total_sent += 1
                
                # Track the most recent date we've seen
                record_date = record.get('date')
                if record_date:
                    last_date_seen = record_date
            
            if total_sent % 5000 == 0:
                self.producer.flush()
                print(f"[PRODUCER] Sent {total_sent} new records... (latest: {last_date_seen})")
            
            offset += len(records)
            
            if len(records) < self.batch_size:
                break
        
        self.producer.flush()
        
        # Update watermark to the last date we actually processed
        if total_sent > 0:
            self.save_watermark(last_date_seen)
            print(f"[PRODUCER] Watermark updated: {last_date_seen}")
            print(f"[PRODUCER] Next update will fetch records > {last_date_seen}")
        else:
            print(f"[PRODUCER] No new records found since {last_date}")
        
        print("=" * 70)
        print(f"[PRODUCER] INCREMENTAL UPDATE complete. New records sent: {total_sent}")
        print("=" * 70)
        return total_sent
    
    def run_continuous(self, update_interval_hours=12):
        """
        Run continuously: initial load, then incremental updates every N hours
        """
        # First run: full load
        print("=" * 70)
        print("[PRODUCER] PHASE 1: Initial Full Load")
        print("=" * 70)
        self.initial_full_load()
        
        # Then: incremental updates
        print("=" * 70)
        print(f"[PRODUCER] PHASE 2: Incremental Updates (every {update_interval_hours} hours)")
        print("=" * 70)
        
        while True:
            sleep_seconds = update_interval_hours * 3600
            print(f"[PRODUCER] Sleeping for {update_interval_hours} hours...")
            time.sleep(sleep_seconds)
            
            print(f"[PRODUCER] Waking up for incremental update...")
            self.incremental_update()

    def run_csv(self, csv_path, max_rows=None):
        total_sent = 0
        last_date_seen = None
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                # Optional key: use ID if present, else row index
                key = row.get("ID", str(row_idx))

                self.producer.send(self.topic, key=key, value=row)

                total_sent += 1

                date_str = row.get("Date") or row.get("date")
                dt = parse_date_flexible(date_str) if date_str else None
                if dt and (last_date_seen  is None or dt > last_date_seen):
                    last_date_seen  = dt
                
                if total_sent % 10000 == 0:
                    self.producer.flush()
                    print(f"[producer] Sent {total_sent} records so far...")

                if max_rows is not None and total_sent >= max_rows:
                    print(f"[producer] Reached max_rows={max_rows}, stopping.")
                    break

        self.producer.flush()

        if total_sent > 0:
            self.save_watermark(last_date_seen)
            print(f"[PRODUCER] Watermark updated: {last_date_seen}")
            print(f"[PRODUCER] Next update will fetch records > {last_date_seen}")

        self.incremental_update()
        
        print(f"[producer] Done. Total records sent: {total_sent}")
    
    def close(self):
        """Close Kafka producer"""
        self.producer.close()


def parse_date_flexible(s: str):
    """Parse common Chicago crime datetime formats into a naive datetime (no tz)."""
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%m/%d/%Y %I:%M:%S %p",   # 12/31/2020 11:59:00 PM
        "%m/%d/%Y %H:%M:%S",      # 12/31/2020 23:59:00
        "%Y-%m-%dT%H:%M:%S.%f",   # 2020-12-31T23:59:00.000
        "%Y-%m-%dT%H:%M:%S",      # 2020-12-31T23:59:00
        "%Y-%m-%d %H:%M:%S",      # 2020-12-31 23:59:00
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream Chicago crime data from API to Kafka"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "incremental", "continuous","csv"],
        default="continuous",
        help="full: initial load only, incremental: update only, continuous: both, csv: CSV then API"
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        default="https://data.cityofchicago.org/resource/ijzp-q8t2.json",
        help="Chicago Crime API endpoint"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/home/jovyan/data/Chicago_Crimes_2001_to_Present.csv",
        help="Path to CSV file (used in csv mode)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="chicago-crime-stream",
        help="Kafka topic to publish to"
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        nargs="+",
        default=["kafka:9092"],
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=12,
        help="Hours between incremental updates (for continuous mode)"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="For testing: limit total records in full load"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="API batch size per request"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    producer = ChicagoCrimeAPIProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        api_endpoint=args.api_endpoint,
        batch_size=args.batch_size
    )
    
    try:
        if args.mode == "full":
            producer.initial_full_load(max_records=args.max_records)
        elif args.mode == "incremental":
            producer.incremental_update()
        elif args.mode == "continuous":
            producer.run_continuous(update_interval_hours=args.update_interval)
        elif args.mode == "csv":
            producer.run_csv(csv_path=args.csv_path, max_rows=args.max_records)
    except KeyboardInterrupt:
        print("\n[PRODUCER] Received interrupt signal, shutting down...")
    finally:
        producer.close()
        print("[PRODUCER] Producer closed")