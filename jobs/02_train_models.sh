#!/bin/bash
#SBATCH --job-name=evidence_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%j.out
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%j.err

module load conda
conda activate ensemble-ili

PROJECT_ROOT="/home/x-jdong8/src/evidence-fusion"
cd $PROJECT_ROOT

echo "Starting Training on $(hostname)"
nvidia-smi

# 1. Train MLP (The Teacher)
echo "------------------------------------------------"
echo "[Step 3] Training MLP Teacher..."
python scripts/3_train_mlp.py \
    --data_dir "./data/vectors" \
    --out_dir "./models/mlp_paper" \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4

# 2. Train Fusion (The Student)
# Note: MLP is frozen, so we can use a smaller LR for the CNN part
echo "------------------------------------------------"
echo "[Step 4] Training Fusion Student..."
python scripts/4_train_fusion.py \
    --vector_dir "./data/vectors" \
    --grid_dir "./data/grids" \
    --mlp_dir "./models/mlp_paper" \
    --out_dir "./models/fusion_paper" \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-5

echo "Training Complete!"