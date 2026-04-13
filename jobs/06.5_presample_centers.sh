#!/bin/bash
#SBATCH --job-name=presample_centers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00               
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=/work/hdd/bdne/jdong8/jobout/%x_%j.out
#SBATCH --error=/work/hdd/bdne/jdong8/jobout/%x_%j.err

# ==============================================================================
# 0. Environment Setup
# ==============================================================================
set -e  # Fail fast on errors
set -u  # Uninitialized variables cause errors (Strict Mode)

source /projects/bdne/jdong8/miniforge3_arm64/bin/activate gnn-gh200

PROJECT_ROOT="/projects/bdne/jdong8/src/evidence-fusion"
cd $PROJECT_ROOT

export PYTHONUNBUFFERED=1  
# 【核心修复】：优雅处理未初始化的 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "================================================================"
echo "Starting Density Grid Presampling on $(hostname)"
echo "Date: $(date)"
echo "Project Root: $PROJECT_ROOT"
echo "================================================================"

time python scripts/2.5_presample_centers.py

echo "================================================================"
echo ">>> Presampling Complete! 600GB payload successfully dropped."
echo "Finished at: $(date)"
echo "================================================================"