#!/bin/bash
#SBATCH --job-name=train_mlp_400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00               
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=/work/hdd/bdne/jdong8/jobout/%x_%j.out
#SBATCH --error=/work/hdd/bdne/jdong8/jobout/%x_%j.err

set -e
set -u
set -o pipefail

# 激活 DeltaAI 的 ARM 环境
source /projects/bdne/jdong8/miniforge3_arm64/bin/activate gnn-gh200
cd /projects/bdne/jdong8/src/evidence-fusion

export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1  
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

nvidia-smi

echo "================================================================"
echo ">>> Starting MLP Teacher Training on $(hostname) (400-dim cut)"
echo ">>> Date: $(date)"
echo "================================================================"

# 【重要】直接指向刚刚生成的 vectors_400 目录！
# 并且输出模型到一个全新的 mlp_400 文件夹，防止覆盖你的旧心血
python scripts/3_train_mlp.py \
    --data_dir "/work/hdd/bdne/jdong8/fusion_data/vectors" \
    --out_dir "/work/hdd/bdne/jdong8/fusion_models/mlp_400" \
    --hidden_sizes 512 512 128 \
    --epochs 60 \
    --bce_epochs 20 \
    --batch_size 512 \
    --lr 2e-4 \
    --dropout 0.2

echo "================================================================"
echo ">>> MLP Training flawlessly completed."
echo ">>> Finished at: $(date)"
echo "================================================================"