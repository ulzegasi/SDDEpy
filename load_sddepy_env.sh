#!/bin/bash
#
# run this with:
# . /cfs/earth/scratch/ulzg/SABCpy/SDDEpy/load_sddepy_env.sh
#
# IMPORTANT: use this script in a shell where sddepy_env is NOT already activated.
# Let this script handle conda activation.


module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load lsfm-init-miniconda/1.0.0
module load openmpi/4.1.4

if [[ "${SUMMARY_STATS:-fft}" == "fno" ]]; then
  env_name="sddepy_fno_env"
elif [[ "${SUMMARY_STATS:-fft}" == "enca" || "${SUMMARY_STATS:-fft}" == "mlp" ]]; then
  env_name="sddepy_enca_env"
else
  env_name="sddepy_env"
fi

echo "Activating Conda environment: $env_name"
conda activate "$env_name"

export TMPDIR=/cfs/earth/scratch/ulzg/.tmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export JULIA_NUM_THREADS=1
