#!/bin/bash
#SBATCH --job-name=early_fusion_abacus
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

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1   
export PYTHONUNBUFFERED=1  
export PYTHONPATH="$PWD:${PYTHONPATH:-}"


echo "================================================================"
echo "Starting Anisotropic GNN Solo Training on $(hostname) [ABACUS LIGHTCONE]"
echo "Date: $(date)"
echo "================================================================"

# [CRITICAL] 指向全新生成的光锥数据集
DATA_BASE="/work/hdd/bdne/jdong8/fusion_data_purified"

if [ ! -f "$DATA_BASE/catalogs/train_centers.pt" ]; then
    echo "[FATAL ERROR] 找不到脱壳预计算文件: $DATA_BASE/catalogs/train_centers.pt"
    exit 1
fi

# [CRITICAL] 注入光锥拓扑开关与新的输出路径
python scripts/train_gnn.py \
    --vector_dir "$DATA_BASE/vectors" \
    --catalog_dir "$DATA_BASE/catalogs" \
    --out_dir "/work/hdd/bdne/jdong8/fusion_models/abacus_early_fusion" \
    --bce_epochs 20 \
    --epochs 120 \
    --batch_size 512 \
    --num_subboxes 8 \
    --r_link 20.0 \
    --lr 1e-3 \
    --geometry "lightcone" \
    --early_fusion 
#    --resume "/work/hdd/bdne/jdong8/fusion_models/abacus_early_fusion/gnn_best.pt"

echo "================================================================"
echo ">>> Training flawlessly completed."
echo "Finished at: $(date)"
echo "================================================================"