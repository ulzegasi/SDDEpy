#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/runjob_benchmark.sh"
WORKER_COUNTS=(1 2 4 6 8 10 12 16 20 24 28 32)

DATASET="${DATASET:-obsSN}"
ALGORITHM="${ALGORITHM:-single_eps}"
TIME_LIMIT="${TIME_LIMIT:-04-00:00:00}"
PARTITION="${PARTITION:-earth-3}"
MEMORY="${MEMORY:-128G}"

mkdir -p /cfs/earth/scratch/ulzg/SABCpy/txtout
mkdir -p /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output

algorithm_label_short="s"
algorithm_label_long="single"
if [[ "$ALGORITHM" == "multi_eps" ]]; then
  algorithm_label_short="m"
  algorithm_label_long="multi"
fi

dataset_label="SN"
if [[ "$DATASET" == "C14" ]]; then
  dataset_label="C14"
fi

benchmark_file="/cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output/benchmark_${DATASET}_${algorithm_label_long}.csv"

for workers in "${WORKER_COUNTS[@]}"; do
  job_name="b${dataset_label}${algorithm_label_short}w${workers}"
  run_id="${dataset_label}_${algorithm_label_long}_w${workers}"
  echo "Submitting $job_name"
  sbatch \
    --job-name="$job_name" \
    --cpus-per-task="$workers" \
    --time="$TIME_LIMIT" \
    --partition="$PARTITION" \
    --mem="$MEMORY" \
    --export=DATASET="$DATASET",ALGORITHM="$ALGORITHM",N_WORKERS="$workers",RUN_ID="$run_id",BENCHMARK_FILE="$benchmark_file" \
    "$JOB_SCRIPT"
done