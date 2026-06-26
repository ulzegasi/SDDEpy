#!/bin/bash
#
#SBATCH --job-name=SNsFNOgpu6
#SBATCH --output=/cfs/earth/scratch/ulzg/SABCpy/txtout/info.%x.%j.%N.info
#SBATCH --error=/cfs/earth/scratch/ulzg/SABCpy/txtout/info.%x.%j.%N.info
#SBATCH --chdir=/cfs/earth/scratch/ulzg/SABCpy/SDDEpy
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#SBATCH --partition=earth-5
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
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
SUMMARY_STATS="fno"
FOURIER_RANGE=""
TRAIN_RUN_DIR="/cfs/earth/scratch/ulzg/enca-inca/sdde_FNO_runs/20260622_fno_z6_m32_fourier"
ENCA_CHECKPOINT_BASENAME="model_best_ckpt"

algorithm_label="single"
if [[ "$ALGORITHM" == "multi_eps" ]]; then
  algorithm_label="multi"
fi

RUN_NAME="${DATASET}_${algorithm_label}_fnofft_z6m32"

# ==============================
# Environment setup
# ==============================
# IMPORTANT: submit this from a shell where no conda environment is already activated.
# SUMMARY_STATS must be set before sourcing load_sddepy_env.sh.

. /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/load_sddepy_env.sh

module load cuda/11.6.2

export JULIA_DEPOT_PATH=/cfs/earth/scratch/ulzg/.julia
mkdir -p "$JULIA_DEPOT_PATH"

mkdir -p "$TMPDIR"
mkdir -p /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/output

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export MPLCONFIGDIR=/cfs/earth/scratch/ulzg/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true
if ! "$PYTHON_BIN" - <<'PY'
import sys
import tensorflow as tf

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)
if not gpus:
    print("ERROR: TensorFlow does not see a GPU in sddepy_fno_env.", file=sys.stderr)
    sys.exit(1)
PY
then
  echo "ERROR: Aborting FNO inference because TensorFlow GPU check failed." >&2
  exit 1
fi
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
echo "TRAIN_RUN_DIR=$TRAIN_RUN_DIR"
echo "ENCA_CHECKPOINT_BASENAME=$ENCA_CHECKPOINT_BASENAME"

if [[ -z "$TRAIN_RUN_DIR" ]]; then
  echo "ERROR: TRAIN_RUN_DIR must be set" >&2
  exit 1
fi

# ==============================
# Run
# ==============================
cmd=(
  srun --export=ALL --cpu-bind=cores "$PYTHON_BIN" SABC_SolarDynamo.py
  --dataset "$DATASET"
  --synthetic-data-file "$SYNTHETIC_DATA_FILE"
  --algorithm "$ALGORITHM"
  --n-workers "$N_WORKERS"
  --run-name "$RUN_NAME"
  --summary-stats "$SUMMARY_STATS"
  --train-run-dir "$TRAIN_RUN_DIR"
  --enca-checkpoint-basename "$ENCA_CHECKPOINT_BASENAME"
)

"${cmd[@]}"
