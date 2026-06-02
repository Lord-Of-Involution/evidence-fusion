#!/bin/bash
#SBATCH --job-name=ultimate_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00               
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=/work/hdd/bdne/jdong8/jobout/%x_%j.out
#SBATCH --error=/work/hdd/bdne/jdong8/jobout/%x_%j.err

set -euo pipefail

# ==============================================================================
# SYSTEM-LEVEL PARANOIA (DID YOU ALREADY FORGET THE FUTEX CRASH?)
# ==============================================================================
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1  

source /projects/bdne/jdong8/miniforge3_arm64/bin/activate gnn-gh200
cd /projects/bdne/jdong8/src/evidence-fusion
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "================================================================"
echo ">>> [Phase 1] Synced Evidence Extraction (100% Alignment)..."
python scripts/10_extract_sync.py \
    --subbox_size 60.0 \
    --num_subboxes 8

echo "================================================================"
echo ">>> [Phase 2] Light-speed Additive Bayesian Optimization..."
python scripts/11_fast_fusion.py \
    --out_dir "/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion" \
    --epochs 10000 \
    --lr 0.01 \
    --patience 1000

echo "================================================================"
echo ">>> [Phase 3] Plotting Comparison & Fusion Geometry..."
python scripts/12_plot_comparison.py \
    --fusion_dir "/work/hdd/bdne/jdong8/fusion_models/ultimate_fusion"

echo "================================================================"
echo ">>> ALL DONE! Let's see if you actually broke the physics limit."