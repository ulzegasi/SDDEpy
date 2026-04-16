#!/bin/bash
#
#SBATCH --job-name=SNs78py
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
DATASET="obsSN"          # "obsSN" or "C14"
ALGORITHM="single_eps"   # "single_eps" or "multi_eps"
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

algorithm_label="single"
if [[ "$ALGORITHM" == "multi_eps" ]]; then
  algorithm_label="multi"
fi

RUN_NAME="${DATASET}_${algorithm_label}_78py"

# ==============================
# Environment setup
# ==============================
# IMPORTANT: submit a job using this script from a shell where sddepy_env is NOT already activated.
# Let this script handle conda activation.

. /cfs/earth/scratch/ulzg/SABCpy/load_sddepy_env.sh

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
echo "ALGORITHM=$ALGORITHM"
echo "N_WORKERS=$N_WORKERS"
echo "RUN_NAME=$RUN_NAME"

# ==============================
# Run
# ==============================
srun --cpu-bind=cores "$PYTHON_BIN" SABC_SolarDynamo.py \
  --dataset "$DATASET" \
  --algorithm "$ALGORITHM" \
  --n-workers "$N_WORKERS" \
  --run-name "$RUN_NAME"
