#!/bin/bash
#SBATCH --job-name=jdong_gen_full
#SBATCH --array=0-999
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=06:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.out
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.err

set -euo pipefail

# ==============================================================================
#  Configuration Area (Modify here to switch Class 0 / Class 1)
# ==============================================================================

# Mode Selection: "unbiased" (Class 0) or "biased" (Class 1)
MODE="unbiased"
# MODE="biased"  <-- Uncomment this line to run BIASED mode

if [[ "$MODE" == "unbiased" ]]; then
    SIM_NAME="fastpm_hodz"
    echo " Running in UNBIASED mode (Class 0)"
elif [[ "$MODE" == "biased" ]]; then
    SIM_NAME="fastpm_hodzbias"
    echo "Running in BIASED mode (Class 1)"
else
    echo "[FATAL] Unknown MODE: $MODE"; exit 1
fi

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
module load conda
conda activate cmass
module load gsl

export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export OPENBLAS_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
export NUMEXPR_NUM_THREADS=${SLURM_NTASKS:-1}
unset XLA_FLAGS

# Auto-locate code directory
SRCDIR_HOME="$HOME/src/ltu-cmass"
SRCDIR_SCRATCH="/anvil/scratch/x-jdong8/src/ltu-cmass"
if [[ -d "$SRCDIR_HOME" ]]; then
  SRCDIR="$SRCDIR_HOME"
  mkdir -p /anvil/scratch/x-jdong8/src
  ln -sfn "$SRCDIR_HOME" "$SRCDIR_SCRATCH"
elif [[ -d "$SRCDIR_SCRATCH" ]]; then
  SRCDIR="$SRCDIR_SCRATCH"
else
  SRCDIR="/anvil/scratch/x-jdong8/ensemble-ili/maho3/ltu-cmass"
fi

# Base Directory
WDIR="/anvil/scratch/x-jdong8/cmass-ili"

# Input Halo Path (Shared source)
HALO_SRC_ROOT="/anvil/scratch/x-mho1/cmass-ili/abacuslike/fastpm/L2000-N256"

# Output Directory (Changes based on SIM_NAME to avoid conflicts)
OUTDIR="${WDIR}/abacuslike/${SIM_NAME}/L2000-N256"

# Hydra Lock File (Prevents log race conditions)
mkdir -p "${WDIR}/logs"
LOCK="${WDIR}/logs/hydra_logs.lock"

cd "$SRCDIR" || { echo "[FATAL] cannot cd to $SRCDIR"; exit 2; }

# ==============================================================================
# 2. Main Loop
# ==============================================================================
# Offset Control: Set to 0 for 0-999; Set to 1000 for 1000-1999
OFFSET=0
# OFFSET=1000 

lhid=$((SLURM_ARRAY_TASK_ID + OFFSET))
echo "================ LHID=${lhid} (Sim: ${SIM_NAME}) ================"

# Base Hydra parameters
POSTFIX_BASE="nbody=abacuslike sim=${SIM_NAME} nbody.N=256 nbody.lhid=${lhid}"

# Run 20 HOD seeds per LHID
for j in {0..19}; do
  hod_seed=$(( lhid*10 + j ))
  printf -v hod_str "%05d" "${hod_seed}"
  
  lc_dir="${OUTDIR}/${lhid}/mtng_lightcone"
  lc_file="${lc_dir}/hod${hod_str}_aug00000.h5"
  
  # --- Step A: Generate Lightcone (if not exists) ---
  if [[ -f "$lc_file" ]]; then
    echo "  [skip lc]   seed=${hod_seed} exists"
  else
    # Symlink Halo File (Critical requirement for hodlightcone)
    halo_src="${HALO_SRC_ROOT}/${lhid}/halos.h5"
    halo_link="${OUTDIR}/${lhid}/halos.h5"
    mkdir -p "${OUTDIR}/${lhid}"
    
    if [[ ! -f "$halo_src" ]]; then
      echo "  [ERR] Missing halo source: $halo_src"
      continue
    fi
    ln -sfn "$halo_src" "$halo_link"

    echo "  [run lc]    seed=${hod_seed} mode=${MODE}"
    # Replicates parameters from your original run_hodz script
    python -m cmass.survey.hodlightcone \
      $POSTFIX_BASE \
      bias.hod.seed=$hod_seed \
      bias.hod.assem_bias=false \
      bias.hod.vel_assem_bias=false \
      survey.geometry=mtng survey.nomask=true \
      meta.wdir=$WDIR
  fi

  # --- Step B: Run Summary (summ) ---
  if [[ ! -f "$lc_file" ]]; then
    echo "  [err] Lightcone gen failed for seed ${hod_seed}"
    continue
  fi

  # Replicates parameters from your original summ_hodz script
  # Note: survey.randoms=true is required for Pk measurements in summ.py
  SUMM_CMD="python -m cmass.diagnostics.summ \
            $POSTFIX_BASE \
            bias.hod.seed=${hod_seed} \
            meta.wdir=${WDIR} \
            diag.survey_backend=pylians \
            survey.randoms=true"

  echo "  [run summ]  seed=${hod_seed}"
  
  # Run with file lock to prevent Hydra logging conflicts
  if ! flock -x "${LOCK}" -c "$SUMM_CMD"; then
      echo "  [warn] flock contention, retrying..."
      sleep 2
      flock -x "${LOCK}" -c "$SUMM_CMD"
  fi
  
done

echo "[DONE] Task ${SLURM_ARRAY_TASK_ID} finished."