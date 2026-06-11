#!/bin/bash
#
#SBATCH --job-name=SYNs3py
#SBATCH --output=/cfs/earth/scratch/ulzg/SABCpy/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/SABCpy/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/SABCpy/SDDEpy
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
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
DATASET="synthetic"          # "obsSN", "C14", or "synthetic"
SYNTHETIC_DATA_FILE="sn_t6_T7_N12_s002_B8_tobs271_seed1822.csv"
ALGORITHM="single_eps"   # "single_eps" or "multi_eps"
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
SUMMARY_STATS="fft"      # "fft", "enca", or "mlp"
FOURIER_RANGE="1:6:60"   # default in code is 1:6:120; set empty string to use default
TRAIN_RUN_DIR=""         # required when SUMMARY_STATS is "enca" or "mlp"
ENCA_CHECKPOINT_BASENAME="model_best_ckpt"

algorithm_label="single"
if [[ "$ALGORITHM" == "multi_eps" ]]; then
  algorithm_label="multi"
fi

RUN_NAME="${DATASET}_${algorithm_label}_3"

# ==============================
# Environment setup
# ==============================
# IMPORTANT: submit a job using this script from a shell where sddepy_env is NOT already activated.
# Let this script handle conda activation.

. /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/load_sddepy_env.sh

export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
mkdir -p "$JULIA_DEPOT_PATH"

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output

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
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
echo "DATASET=$DATASET"
if [[ "$DATASET" == "synthetic" ]]; then
  echo "SYNTHETIC_DATA_FILE=$SYNTHETIC_DATA_FILE"
fi
echo "ALGORITHM=$ALGORITHM"
echo "N_WORKERS=$N_WORKERS"
echo "RUN_NAME=$RUN_NAME"
echo "SUMMARY_STATS=$SUMMARY_STATS"
if [[ "$SUMMARY_STATS" == "fft" ]]; then
  echo "FOURIER_RANGE=${FOURIER_RANGE:-default 1:6:120}"
elif [[ "$SUMMARY_STATS" == "enca" || "$SUMMARY_STATS" == "mlp" ]]; then
  if [[ -z "$TRAIN_RUN_DIR" ]]; then
    echo "ERROR: TRAIN_RUN_DIR must be set when SUMMARY_STATS=$SUMMARY_STATS" >&2
    exit 1
  fi
  echo "TRAIN_RUN_DIR=$TRAIN_RUN_DIR"
  echo "ENCA_CHECKPOINT_BASENAME=$ENCA_CHECKPOINT_BASENAME"
else
  echo "ERROR: SUMMARY_STATS must be fft, enca, or mlp, got '$SUMMARY_STATS'" >&2
  exit 1
fi

# ==============================
# Run
# ==============================
cmd=(
  srun --cpu-bind=cores "$PYTHON_BIN" SABC_SolarDynamo.py
  --dataset "$DATASET"
  --synthetic-data-file "$SYNTHETIC_DATA_FILE"
  --algorithm "$ALGORITHM"
  --n-workers "$N_WORKERS"
  --run-name "$RUN_NAME"
  --summary-stats "$SUMMARY_STATS"
)

if [[ "$SUMMARY_STATS" == "fft" && -n "$FOURIER_RANGE" ]]; then
  cmd+=(--fourier-range "$FOURIER_RANGE")
fi

if [[ "$SUMMARY_STATS" == "enca" || "$SUMMARY_STATS" == "mlp" ]]; then
  cmd+=(--train-run-dir "$TRAIN_RUN_DIR")
  cmd+=(--enca-checkpoint-basename "$ENCA_CHECKPOINT_BASENAME")
fi

"${cmd[@]}"
