# spark-apps/kafka_producer_chicago_crimes.py
import csv
import json
import time
import argparse
import os
import random
from datetime import datetime, timedelta
from kafka import KafkaProducer
import requests


# ---------------- Defaults ----------------
DEFAULT_API_ENDPOINT = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
DEFAULT_TOPIC = "chicago-crime-stream"
DEFAULT_BOOTSTRAP_SERVERS = ["kafka:9092"]

DEFAULT_BATCH_SIZE = 50000
DEFAULT_FLUSH_EVERY = 10000

# overlap window on updated_on (duplicates OK because ETL can dedup)
DEFAULT_LOOKBACK_HOURS = 0

# for continuous mode
DEFAULT_UPDATE_INTERVAL_HOURS = 12

DEFAULT_WATERMARK_FILE = "/home/jovyan/data/chicago_crime_watermark.json"
DEFAULT_CSV_PATH = "/home/jovyan/data/Chicago_Crimes_2001_to_Present.csv"


def parse_dt_flexible(s: str):
    """Parse ISO / CSV-ish datetimes into naive datetime (no tz). Returns None if parse fails."""
    # Supports multiple formats:
    # - Chicago CSV "MM/DD/YYYY hh:mm:ss AM/PM"
    if not s:
        return None
    s = str(s).strip()
    fmts = [
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def soql_id_literal(v):
    """SoQL literal: numeric ids unquoted, otherwise quoted."""
    if v is None or v == "":
        return "0"
    s = str(v).strip()
    if s.isdigit():
        return s
    s = s.replace("'", "''")
    return f"'{s}'"


class ChicagoCrimeAPIProducer:
    """
    Cursor paging with tie-breaker on (updated_on, id):
      ORDER BY updated_on ASC, id ASC
      WHERE (updated_on > cursor_u) OR (updated_on = cursor_u AND id > cursor_id)

    Incremental uses rollback window on updated_on
    """

    def __init__(self):
        self.topic = DEFAULT_TOPIC
        self.api_endpoint = DEFAULT_API_ENDPOINT
        self.batch_size = DEFAULT_BATCH_SIZE
        self.flush_every = DEFAULT_FLUSH_EVERY
        self.lookback_hours = DEFAULT_LOOKBACK_HOURS
        self.update_interval_hours = DEFAULT_UPDATE_INTERVAL_HOURS
        self.watermark_file = DEFAULT_WATERMARK_FILE
        self.csv_path = DEFAULT_CSV_PATH

        self.producer = KafkaProducer(
            bootstrap_servers=DEFAULT_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda v: v.encode("utf-8") if v is not None else None,
            linger_ms=10,
            batch_size=64 * 1024,
            acks="all",
            retries=5,
        )

        self.session = requests.Session()

    # ---------------- Watermark (updated_on + id) ----------------

    def get_last_watermark(self):
        """Load the last saved cursor from local JSON file; return None/empty if not found."""
        try:
            with open(self.watermark_file, "r") as f:
                data = json.load(f)
            u = data.get("last_updated_on")
            i = data.get("last_id", "")
            return {"last_updated_on": str(u) if u else None, "last_id": str(i) if i else ""}
        except FileNotFoundError:
            return {"last_updated_on": None, "last_id": ""}

    def save_watermark(self, last_updated_on, last_id):
        """Persist watermark locally so incremental mode can resume after restarts."""
        if last_updated_on is None:
            return
        if isinstance(last_updated_on, datetime):
            # Normalize to a simple ISO-like string without timezone
            last_updated_on = last_updated_on.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            last_updated_on = str(last_updated_on)

        last_id = "" if last_id is None else str(last_id)

        os.makedirs(os.path.dirname(self.watermark_file), exist_ok=True)
        with open(self.watermark_file, "w") as f:
            json.dump({"last_updated_on": last_updated_on, "last_id": last_id}, f)

    # ---------------- API fetch ----------------
    def fetch_data(self, where_clause=None, limit=None, max_retries=8):
        """
        Fetch one page of records from Socrata.
        - Uses deterministic ordering to support cursor paging.
        - Retries on 429 rate-limit and timeouts with exponential backoff + jitter.
        """
        params = {
            "$limit": limit or self.batch_size,
            "$order": "updated_on ASC, id ASC",
        }
        if where_clause:
            params["$where"] = where_clause

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                # timeout=(connect, read)
                r = self.session.get(self.api_endpoint, params=params, timeout=(10, 240))

                # Socrata rate limiting
                if r.status_code == 429:
                    wait = min(120, 2 ** attempt) + random.random()
                    print(f"[WARN] 429 rate limited. Sleep {wait:.1f}s then retry...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json()

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                last_err = e
                wait = min(120, 2 ** attempt) + random.random()
                print(f"[WARN] Timeout. Retry {attempt}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)
            except requests.exceptions.RequestException as e:
                last_err = e
                wait = min(60, 2 ** attempt) + random.random()
                print(f"[WARN] Request failed ({e}). Retry {attempt}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)

        print(f"[ERROR] API failed after retries: {last_err}")
        return []

    def _where_after_cursor(self, cursor_u, cursor_id):
        """
        Build a stable SoQL cursor filter:
        - (updated_on > cursor_u) OR (updated_on == cursor_u AND id > cursor_id)
        """
        cu = str(cursor_u).replace("'", "''")
        id_lit = soql_id_literal(cursor_id)
        return f"(updated_on > '{cu}') OR (updated_on = '{cu}' AND id > {id_lit})"

    # ---------------- Full load ----------------
    def initial_full_load(self):
        print("=" * 70)
        print("[PRODUCER] FULL LOAD (cursor paging, order by updated_on,id ASC)")
        print("=" * 70)

        cursor_u = "1900-01-01T00:00:00"
        cursor_id = 0

        total_sent = 0
        wm_u = None
        wm_id = ""

        while True:
            # Fetch next page strictly after the current cursor
            where_clause = self._where_after_cursor(cursor_u, cursor_id)
            records = self.fetch_data(where_clause=where_clause, limit=self.batch_size)

            if not records:
                break

            for rec in records:
                rid = rec.get("id")
                ru = rec.get("updated_on")

                self.producer.send(self.topic, key=str(rid) if rid is not None else None, value=rec)
                total_sent += 1

                # Advance cursor in deterministic order
                if ru is not None:
                    cursor_u = str(ru)
                if rid is not None:
                    cursor_id = rid

                # Track watermark to persist progress
                wm_u = cursor_u
                wm_id = str(cursor_id) if cursor_id is not None else ""

                # Flush and save watermark for resilience
                if total_sent % self.flush_every == 0:
                    self.producer.flush()
                    self.save_watermark(wm_u, wm_id)
                    print(f"[PRODUCER] Sent {total_sent}... (wm saved: {wm_u}, {wm_id})")

            if len(records) < self.batch_size:
                break

        # Final flush + final watermark
        self.producer.flush()
        if wm_u is not None:
            self.save_watermark(wm_u, wm_id)

        print("=" * 70)
        print(f"[PRODUCER] FULL LOAD complete. Total sent: {total_sent}")
        if wm_u is not None:
            print(f"[PRODUCER] Final watermark: (last_updated_on={wm_u}, last_id={wm_id})")
        print("=" * 70)
        return total_sent

    # ---------------- Incremental ----------------

    def incremental_update(self):
        """
        Incremental sync from persisted watermark.
        """
        wm = self.get_last_watermark()
        last_u = wm["last_updated_on"]
        last_id = wm["last_id"]

        if not last_u:
            print("[WARNING] No updated_on watermark found. Running full load instead...")
            return self.initial_full_load()

        print("=" * 70)
        print(f"[PRODUCER] INCREMENTAL UPDATE from watermark: (updated_on={last_u}, id={last_id})")
        print(f"[PRODUCER] Order: updated_on ASC, id ASC | lookback_hours={self.lookback_hours}")
        print("=" * 70)

        # rollback start cursor
        start_u = last_u
        start_id = last_id or 0

        last_dt = parse_dt_flexible(last_u)
        if last_dt and self.lookback_hours > 0:
            rollback_dt = last_dt - timedelta(hours=self.lookback_hours)
            start_u = rollback_dt.strftime("%Y-%m-%dT%H:%M:%S")
            start_id = 0
            print(f"[PRODUCER] Rollback start cursor: (updated_on={start_u}, id={start_id})")

        cursor_u = start_u
        cursor_id = start_id

        total_sent = 0
        wm_u = last_u
        wm_id = last_id

        while True:
            where_clause = self._where_after_cursor(cursor_u, cursor_id)
            records = self.fetch_data(where_clause=where_clause, limit=self.batch_size)

            if not records:
                break

            for rec in records:
                rid = rec.get("id")
                ru = rec.get("updated_on")

                self.producer.send(self.topic, key=str(rid) if rid is not None else None, value=rec)
                total_sent += 1

                if ru is not None:
                    cursor_u = str(ru)
                if rid is not None:
                    cursor_id = rid

                wm_u = cursor_u
                wm_id = str(cursor_id) if cursor_id is not None else ""

                if total_sent % self.flush_every == 0:
                    self.producer.flush()
                    self.save_watermark(wm_u, wm_id)
                    print(f"[PRODUCER] Sent {total_sent}... (wm saved: {wm_u}, {wm_id})")

            if len(records) < self.batch_size:
                break

        self.producer.flush()

        if total_sent > 0:
            self.save_watermark(wm_u, wm_id)
            print(f"[PRODUCER] Watermark updated: (last_updated_on={wm_u}, last_id={wm_id})")
        else:
            print("[PRODUCER] No records returned.")

        print("=" * 70)
        print(f"[PRODUCER] INCREMENTAL UPDATE complete. Sent: {total_sent}")
        print("=" * 70)
        return total_sent

    # ---------------- Continuous ----------------

    def run_continuous(self):
        wm = self.get_last_watermark()
        if wm["last_updated_on"]:
            print("[PRODUCER] Watermark found. Doing incremental catch-up first...")
            self.incremental_update()
        else:
            print("[PRODUCER] No watermark found. Running full load first...")
            self.initial_full_load()

        while True:
            print(f"[PRODUCER] Sleeping for {self.update_interval_hours} hours...")
            time.sleep(self.update_interval_hours * 3600)
            print("[PRODUCER] Waking up for incremental update...")
            self.incremental_update()

    # ---------------- CSV ----------------
    def run_csv(self, max_rows=None):
        """
        CSV mode:
        - Publish each CSV row as a record to Kafka.
        - Derive watermark from max(Updated On, ID) seen in CSV.
        - Then run incremental_update() to catch new changes from API after CSV watermark.
        """
        total_sent = 0
        best_u = None
        best_id = ""

        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                key = row.get("ID", str(row_idx))
                self.producer.send(self.topic, key=str(key), value=row)
                total_sent += 1

                # Track max watermark candidate in the CSV
                u_str = row.get("Updated On") or row.get("updated_on")
                dt = parse_dt_flexible(u_str) if u_str else None
                rid = str(row.get("ID") or "")

                if dt:
                    # Compare by updated_on, then tie-break by ID
                    if (best_u is None) or (dt > best_u) or (dt == best_u and rid > best_id):
                        best_u = dt
                        best_id = rid

                if total_sent % self.flush_every == 0:
                    self.producer.flush()
                    print(f"[PRODUCER] Sent {total_sent} CSV rows...")

                if max_rows is not None and total_sent >= max_rows:
                    break

        self.producer.flush()
        
        # Save derived watermark from CSV so API incremental starts after it
        if best_u is not None:
            wm_u = best_u.strftime("%Y-%m-%dT%H:%M:%S")
            self.save_watermark(wm_u, best_id)
            print(f"[PRODUCER] Watermark set from CSV: (last_updated_on={wm_u}, last_id={best_id})")

         # After CSV bootstrap, pull the API for anything newer
        self.incremental_update()
        print(f"[PRODUCER] CSV mode complete. Total CSV sent: {total_sent}")
        return total_sent

    def close(self):
        self.producer.flush()
        self.producer.close()


def parse_args():
    p = argparse.ArgumentParser(description="Stream Chicago crime data from API to Kafka (cursor updated_on+id).")
    p.add_argument(
        "--mode",
        type=str,
        choices=["full", "incremental", "continuous", "csv"],
        default="continuous",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    producer = ChicagoCrimeAPIProducer()

    try:
        if args.mode == "full":
            producer.initial_full_load()
        elif args.mode == "incremental":
            producer.incremental_update()
        elif args.mode == "continuous":
            producer.run_continuous()
        elif args.mode == "csv":
            producer.run_csv()
    except KeyboardInterrupt:
        print("\n[PRODUCER] Received interrupt signal, shutting down...")
    finally:
        producer.close()
        print("[PRODUCER] Producer closed")
