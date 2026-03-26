#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/runjob_benchmark.sh"
WORKER_COUNTS=(1 2 4 6 8 10 12 16 20 24 28 32)
# WORKER_COUNTS=(16 20 24 28 32)

TIME_LIMIT="${TIME_LIMIT:-04-00:00:00}"
PARTITION="${PARTITION:-earth-3}"
MEMORY="${MEMORY:-128G}"

mkdir -p /cfs/earth/scratch/ulzg/SABCpy/txtout
mkdir -p /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output

for workers in "${WORKER_COUNTS[@]}"; do
  job_name="C14m8W${workers}"
  echo "Submitting $job_name"
  sbatch \
    --job-name="$job_name" \
    --cpus-per-task="$workers" \
    --time="$TIME_LIMIT" \
    --partition="$PARTITION" \
    --mem="$MEMORY" \
    "$JOB_SCRIPT"
done
