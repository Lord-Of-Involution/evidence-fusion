#!/bin/bash
#SBATCH --job-name=train_mlp_abacus
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

set -e
set -u
set -o pipefail

source /projects/bdne/jdong8/miniforge3_arm64/bin/activate gnn-gh200
cd /projects/bdne/jdong8/src/evidence-fusion

export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1  
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

nvidia-smi

# Target the newly synced Abacus Lightcone directory
DATA_ROOT="/work/hdd/bdne/jdong8/fusion_data_abacus/vectors"
OUT_ROOT="/work/hdd/bdne/jdong8/fusion_models_abacus/mlp"

python scripts/3_train_mlp.py \
    --data_dir "$DATA_ROOT" \
    --out_dir "$OUT_ROOT" \
    --hidden_sizes 512 512 256 128 \
    --epochs 240 \
    --bce_epochs 30 \
    --batch_size 512 \
    --lr 2e-4 \
    --dropout 0.2