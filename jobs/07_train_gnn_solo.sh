#!/bin/bash
#SBATCH --job-name=gnn_solo_gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00               
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1   
export PYTHONUNBUFFERED=1  
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

nvidia-smi

echo "================================================================"
echo "Starting Anisotropic GNN Solo Training on $(hostname)"
echo "Date: $(date)"
echo "================================================================"

DATA_BASE="/work/hdd/bdne/jdong8/fusion_data"
if [ ! -f "$DATA_BASE/catalogs/train_centers.pt" ]; then
    echo "[FATAL ERROR] 找不到脱壳预计算文件: train_centers.pt"
    exit 1
fi

# 【加回来了！】明确指定 --num_subboxes 4
python scripts/9_train_gnn.py \
    --vector_dir "$DATA_BASE/vectors" \
    --catalog_dir "$DATA_BASE/catalogs" \
    --out_dir "/work/hdd/bdne/jdong8/fusion_models/gnn_solo" \
    --epochs 60 \
    --batch_size 128 \
    --num_subboxes 8 \
    --r_link 20.0 \
    --lr 5e-4 

echo "================================================================"
echo ">>> Training flawlessly completed."
echo "Finished at: $(date)"
echo "================================================================"