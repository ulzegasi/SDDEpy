#!/bin/bash
#
#SBATCH --job-name=sdde_bench
#SBATCH --output=/cfs/earth/scratch/ulzg/SABCpy/txtout/bench.%x.%j.%N.log
#SBATCH --error=/cfs/earth/scratch/ulzg/SABCpy/txtout/bench.%x.%j.%N.log
#SBATCH --chdir=/cfs/earth/scratch/ulzg/SABCpy/SDDEpy
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04-00:00:00
#SBATCH --partition=earth-3
#SBATCH --no-requeue
#SBATCH --constraint=rhel8
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=ulzg@zhaw.ch
#SBATCH --mem=128G

# ==============================
# Editable variables
# ==============================
DATASET="obsSN"          # "obsSN" or "C14"
ALGORITHM="single_eps"   # "single_eps" or "multi_eps"
N_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

algorithm_label="single"
if [[ "$ALGORITHM" == "multi_eps" ]]; then
  algorithm_label="multi"
fi

dataset_label="SN"
if [[ "$DATASET" == "C14" ]]; then
  dataset_label="C14"
fi

RUN_ID="${dataset_label}_${algorithm_label}_w${N_WORKERS}"
BENCHMARK_FILE="/cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output/benchmark_${DATASET}_${algorithm_label}.csv"

# ==============================
# Environment setup
# ==============================
. /cfs/earth/scratch/ulzg/SABCpy/load_sddepy_env.sh

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output

export RUN_ID

# ==============================
# Diagnostics
# ==============================
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
PYTHON_BIN="$(command -v python3)"
echo "Python used: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Julia used: $(command -v julia)"
julia -v
echo "JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
echo "DATASET=$DATASET"
echo "ALGORITHM=$ALGORITHM"
echo "N_WORKERS=$N_WORKERS"
echo "RUN_ID=$RUN_ID"
echo "BENCHMARK_FILE=$BENCHMARK_FILE"

# ==============================
# Run
# ==============================
srun "$PYTHON_BIN" SABC_SolarDynamo_BENCHMARK.py \
  --dataset "$DATASET" \
  --algorithm "$ALGORITHM" \
  --n-workers "$N_WORKERS" \
  --benchmark-file "$BENCHMARK_FILE"
