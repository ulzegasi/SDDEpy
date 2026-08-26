#!/bin/bash
#
#SBATCH --job-name=SNsMLP6v1
#SBATCH --output=/cfs/earth/scratch/ulzg/SABCpy/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/SABCpy/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/SABCpy/SDDEpy
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02-00:00:00
#SBATCH --partition=earth-1
#SBATCH --no-requeue
#SBATCH --constraint=rhel8
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=ulzg@zhaw.ch
#SBATCH --mem=128G

# ==============================
# Editable variables
# ==============================
DATASET="obsSN"          # "obsSN", "C14", or "synthetic"
SYNTHETIC_DATA_FILE="sn_t3_T3_N7_s003_B8_tobs271_seed1822.csv"
ALGORITHM="single_eps"   # "single_eps" or "multi_eps"
SUMMARY_STATS="mlp"      # "fft", "enca", "mlp", or "enca_fft_cnn"
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
SIMULATOR_SEED="123"     # forward-model simulations; set "" for fresh randomness
ALGORITHM_SEED="18"      # algorithm randomness; set "" for fresh randomness
PROPOSAL_SEED="22"       # Differential Evolution proposals; set "" for fresh randomness
FOURIER_RANGE=""   # default in code is 1:6:120; set empty string to use default
TRAIN_RUN_DIR="/cfs/earth/scratch/ulzg/enca-inca/sdde_MLP_runs/20260611_mlp_z6_1"         # required when SUMMARY_STATS is "enca", "mlp", or "enca_fft_cnn"
ENCA_CHECKPOINT_BASENAME="model_best_ckpt"

algorithm_label="single"
if [[ "$ALGORITHM" == "multi_eps" ]]; then
  algorithm_label="multi"
fi

RUN_NAME="${DATASET}_${algorithm_label}_mlp6v1"

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
echo "SIMULATOR_SEED=${SIMULATOR_SEED:-random}"
echo "ALGORITHM_SEED=${ALGORITHM_SEED:-random}"
echo "PROPOSAL_SEED=${PROPOSAL_SEED:-random}"
echo "RUN_NAME=$RUN_NAME"
echo "SUMMARY_STATS=$SUMMARY_STATS"
if [[ "$SUMMARY_STATS" == "fft" ]]; then
  echo "FOURIER_RANGE=${FOURIER_RANGE:-default 1:6:120}"
elif [[ "$SUMMARY_STATS" == "enca" || "$SUMMARY_STATS" == "mlp" || "$SUMMARY_STATS" == "enca_fft_cnn" ]]; then
  if [[ -z "$TRAIN_RUN_DIR" ]]; then
    echo "ERROR: TRAIN_RUN_DIR must be set when SUMMARY_STATS=$SUMMARY_STATS" >&2
    exit 1
  fi
  echo "TRAIN_RUN_DIR=$TRAIN_RUN_DIR"
  echo "ENCA_CHECKPOINT_BASENAME=$ENCA_CHECKPOINT_BASENAME"
else
  echo "ERROR: SUMMARY_STATS must be fft, enca, mlp, or enca_fft_cnn, got '$SUMMARY_STATS'" >&2
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

if [[ -n "$SIMULATOR_SEED" ]]; then
  cmd+=(--simulator-seed "$SIMULATOR_SEED")
fi

if [[ -n "$ALGORITHM_SEED" ]]; then
  cmd+=(--algorithm-seed "$ALGORITHM_SEED")
fi

if [[ -n "$PROPOSAL_SEED" ]]; then
  cmd+=(--proposal-seed "$PROPOSAL_SEED")
fi

if [[ "$SUMMARY_STATS" == "fft" && -n "$FOURIER_RANGE" ]]; then
  cmd+=(--fourier-range "$FOURIER_RANGE")
fi

if [[ "$SUMMARY_STATS" == "enca" || "$SUMMARY_STATS" == "mlp" || "$SUMMARY_STATS" == "enca_fft_cnn" ]]; then
  cmd+=(--train-run-dir "$TRAIN_RUN_DIR")
  cmd+=(--enca-checkpoint-basename "$ENCA_CHECKPOINT_BASENAME")
fi

"${cmd[@]}"
