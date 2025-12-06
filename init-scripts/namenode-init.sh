#!/bin/bash
set -euo pipefail

# Use the same URI as in hadoop.env; fall back to namenode:9000 if not set
HDFS_URI="${CORE_CONF_fs_defaultFS:-hdfs://namenode:9000}"

echo "[hdfs-init] Using HDFS URI: ${HDFS_URI}"

echo "[hdfs-init] Waiting for HDFS to be ready..."
# IMPORTANT: talk to the real HDFS, not file:/// 
until hdfs dfs -fs "${HDFS_URI}" -ls / >/dev/null 2>&1; do
  echo "[hdfs-init] HDFS not ready yet, sleeping 5s..."
  sleep 5
done
echo "[hdfs-init] HDFS ready"

create_dir() {
  local dir="$1"
  if hdfs dfs -fs "${HDFS_URI}" -test -d "${dir}"; then
    echo "[hdfs-init] Directory ${dir} already exists."
  else
    echo "[hdfs-init] Creating directory ${dir} ..."
    hdfs dfs -fs "${HDFS_URI}" -mkdir -p "${dir}"
    hdfs dfs -fs "${HDFS_URI}" -ls -d "${dir}"
  fi
}

echo "[hdfs-init] Creating HDFS directory structure..."
create_dir "/data"
create_dir "/data/raw"
create_dir "/data/processed"
create_dir "/data/results"
create_dir "/user"
create_dir "/user/spark"

echo "[hdfs-init] Final listing of / on HDFS:"
hdfs dfs -fs "${HDFS_URI}" -ls / || true

echo "[hdfs-init] HDFS directories created (or already existed). Done."

