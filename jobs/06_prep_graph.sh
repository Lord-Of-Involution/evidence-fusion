#!/bin/bash
#SBATCH --job-name=prep_graph
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=04:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%j.out
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%j.err

# ==============================================================================
# 0. Environment Setup
# ==============================================================================
set -e  # Fail fast on errors
set -u  # Uninitialized variables cause errors

module load conda
conda activate ensemble-ili

# ==============================================================================
# 1. Define Paths
# ==============================================================================
PROJECT_ROOT="/home/x-jdong8/src/evidence-fusion"

# 你的 Quijote 数据存放路径
RAW_DIR_0="/anvil/scratch/x-jdong8/cmass-ili/quijote/nbody/L1000-N128"
RAW_DIR_1="/anvil/scratch/x-jdong8/cmass-ili/quijote/nbody_bias/L1000-N128"

# Vector (MLP输入) 所在的路径，用来做样本对齐
VECTOR_DIR="/anvil/scratch/x-jdong8/fusion_data/vectors"

# [NEW] 存放提取好的点云 Catalogs 的路径
OUT_DIR="/anvil/scratch/x-jdong8/fusion_data/catalogs"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

# ==============================================================================
# 2. Execution
# ==============================================================================
echo "================================================================"
echo "Starting Galaxy Catalog Extraction & RSD Application on $(hostname)"
echo "Date: $(date)"
echo "Project Root: $PROJECT_ROOT"
echo "Output Dir: $OUT_DIR"
echo "================================================================"

python scripts/8_prep_graph.py \
    --dir_0 "$RAW_DIR_0" \
    --dir_1 "$RAW_DIR_1" \
    --vector_dir "$VECTOR_DIR" \
    --out_dir "$OUT_DIR"

echo "================================================================"
echo "Data Prep Complete! HDF5 point clouds are ready."
echo "Finished at: $(date)"
echo "================================================================"