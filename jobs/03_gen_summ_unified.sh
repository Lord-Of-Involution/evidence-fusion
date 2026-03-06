#!/bin/bash
#SBATCH --job-name=jdong_gen_full
#SBATCH --array=912-999
#SBATCH --nodes=1
#SBATCH --ntasks=64             # <--- Kept your update (64 cores)
#SBATCH --time=06:00:00         # <--- Kept your update (6 hours)
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.out
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.err

set -euo pipefail

# ==============================================================================
#  Configuration Area (Modify here to switch Class 0 / Class 1)
# ==============================================================================

# Mode Selection: "unbiased" (Class 0) or "biased" (Class 1)
#MODE="unbiased"
MODE="biased"  # Uncomment this line to run BIASED mode

if [[ "$MODE" == "unbiased" ]]; then
    SIM_NAME="fastpm_hodz"
    echo "🔵 Running in UNBIASED mode (Class 0)"
elif [[ "$MODE" == "biased" ]]; then
    SIM_NAME="fastpm_hodzbias"
    echo "🔴 Running in BIASED mode (Class 1)"
else
    echo "[FATAL] Unknown MODE: $MODE"; exit 1
fi

# ==============================================================================
# 1. Environment & Paths
# ==============================================================================
module load conda
conda activate cmass
module load gsl

# Set thread count based on ntasks (64 in your case)
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
# Input Halo Path
HALO_SRC_ROOT="/anvil/scratch/x-mho1/cmass-ili/abacuslike/fastpm/L2000-N256"
# Output Directory
OUTDIR="${WDIR}/abacuslike/${SIM_NAME}/L2000-N256"

# Hydra Lock File
mkdir -p "${WDIR}/logs"
LOCK="${WDIR}/logs/hydra_logs.lock"
# Failure Log File
FAIL_LOG="${WDIR}/logs/failed_seeds.log"

cd "$SRCDIR" || { echo "[FATAL] cannot cd to $SRCDIR"; exit 2; }

# ==============================================================================
# 2. Main Loop (With Fault Tolerance)
# ==============================================================================
# Offset Control: Set to 0 for 0-999; Set to 1000 for 1000-1999
OFFSET=0
# OFFSET=1000 

lhid=$((SLURM_ARRAY_TASK_ID + OFFSET))
echo "================ LHID=${lhid} (Sim: ${SIM_NAME}) ================"

POSTFIX_BASE="nbody=abacuslike sim=${SIM_NAME} nbody.N=256 nbody.lhid=${lhid}"

# --- CRITICAL: Temporarily disable "Exit on Error" ---
# This prevents a single failed seed (OOM/Crash) from killing the whole job.
set +e

for j in {0..19}; do
  hod_seed=$(( lhid*10 + j ))
  printf -v hod_str "%05d" "${hod_seed}"
  
  lc_dir="${OUTDIR}/${lhid}/mtng_lightcone"
  lc_file="${lc_dir}/hod${hod_str}_aug00000.h5"
  
  # --- Step A: Generate Lightcone ---
  if [[ -f "$lc_file" ]]; then
    echo "  [skip lc]   seed=${hod_seed} exists"
  else
    # Symlink Halo File
    halo_src="${HALO_SRC_ROOT}/${lhid}/halos.h5"
    halo_link="${OUTDIR}/${lhid}/halos.h5"
    mkdir -p "${OUTDIR}/${lhid}"
    
    if [[ ! -f "$halo_src" ]]; then
      echo "  [ERR] Missing halo source: $halo_src"
      continue
    fi
    ln -sfn "$halo_src" "$halo_link"

    echo "  [run lc]    seed=${hod_seed} mode=${MODE}"
    
    # Run Python and capture the exit code (don't crash script)
    python -m cmass.survey.hodlightcone \
      $POSTFIX_BASE \
      bias.hod.seed=$hod_seed \
      bias.hod.assem_bias=false \
      bias.hod.vel_assem_bias=false \
      survey.geometry=mtng survey.nomask=true \
      meta.wdir=$WDIR
      
    RET_CODE=$?
    
    # Check if failed (e.g. Memory Error, OOM Kill, or other crash)
    if [ $RET_CODE -ne 0 ]; then
        echo "  [FAIL] Lightcone gen failed for seed=${hod_seed} (Exit Code: $RET_CODE)"
        echo "         -> Skipping this seed."
        # Log the failure for later analysis
        echo "${lhid},${hod_seed},LC_FAIL_${RET_CODE}" >> "$FAIL_LOG"
        continue
    fi
  fi

  # --- Step B: Run Summary ---
  if [[ ! -f "$lc_file" ]]; then
    echo "  [err] Lightcone gen failed (file missing), skipping summ..."
    continue
  fi

  # Summary parameters (kept per your request)
  SUMM_CMD="python -m cmass.diagnostics.summ \
            $POSTFIX_BASE \
            bias.hod.seed=${hod_seed} \
            meta.wdir=${WDIR} \
            diag.survey_backend=pylians \
            survey.randoms=true \
            diag.from_scratch=false \
            diag.mtng=true \
            diag.galaxy=false \
            diag.halo=false \
            diag.summaries=\"['Pk','Bk']\""  

  echo "  [run summ]  seed=${hod_seed}"
  
  # Use flock to prevent Hydra logging race conditions
  if ! flock -x "${LOCK}" -c "$SUMM_CMD"; then
      echo "  [FAIL] Summary failed for seed=${hod_seed}"
      echo "${lhid},${hod_seed},SUMM_FAIL" >> "$FAIL_LOG"
      continue
  fi
  
done

# --- Re-enable "Exit on Error" (Safety) ---
set -e

echo "[DONE] Task ${SLURM_ARRAY_TASK_ID} finished."