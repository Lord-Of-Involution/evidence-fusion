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

source /projects/bdne/jdong8/miniforge3_arm64/bin/activate gnn-gh200
cd /projects/bdne/jdong8/src/evidence-fusion

export PYTHONUNBUFFERED=1  
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "================================================================"
echo ">>> [Phase 1] Synced Evidence Extraction (100% Alignment)..."
python scripts/10_extract_sync.py

echo "================================================================"
echo ">>> [Phase 2] Light-speed Additive Bayesian Optimization..."
python scripts/11_fast_fusion.py

echo "================================================================"
echo ">>> ALL DONE! You just broke the physics limit."