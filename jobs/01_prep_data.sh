#!/bin/bash
#SBATCH --job-name=evidence_prep
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%j.out
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%j.err

module load conda
conda activate ensemble-ili

RAW_DIR_0="/anvil/scratch/x-jdong8/cmass-ili/quijote/nbody/L1000-N128"
RAW_DIR_1="/anvil/scratch/x-jdong8/cmass-ili/quijote/nbody_bias/L1000-N128"

# --- OUTPUT PATHS ---
PROJECT_ROOT="/home/x-jdong8/src/evidence-fusion"
cd $PROJECT_ROOT

echo "Starting Data Prep on $(hostname)"
echo "Project Root: $PROJECT_ROOT"

# 1. Prep Vectors (Fast)
echo "------------------------------------------------"
echo "[Step 1] Prepping Vectors..."
python scripts/1_prep_vectors.py \
    --dir_0 "$RAW_DIR_0" \
    --dir_1 "$RAW_DIR_1" \
    --out_dir "/anvil/scratch/x-jdong8/fusion_data/vectors"

# 2. Prep Grids (IO Heavy)
echo "------------------------------------------------"
echo "[Step 2] Prepping Grids..."
python scripts/2_prep_grids.py \
    --dir_0 "$RAW_DIR_0" \
    --dir_1 "$RAW_DIR_1" \
    --vector_dir "/anvil/scratch/x-jdong8/fusion_data/vectors" \
    --out_dir "/anvil/scratch/x-jdong8/fusion_data/grids"

echo "Data Prep Complete!"