#!/bin/bash
#SBATCH --job-name=cnn_solo_gpu         # Job name 换成了 cnn_solo
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=64                     # Number of tasks
#SBATCH --gpus-per-node=2               # Number of GPUs
#SBATCH --time=04:30:00                 # Time limit
#SBATCH --partition=gpu                 # Partition name
#SBATCH --account=phy240043-gpu         # Account name
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.out   # Error file for each array task

module load conda
conda activate ensemble-ili

PROJECT_ROOT="/home/x-jdong8/src/evidence-fusion"
cd $PROJECT_ROOT

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Starting CNN Solo Training on $(hostname)"
nvidia-smi

echo "------------------------------------------------"
echo "[Diagnostic Step] Training CNN Solo (No MLP)..."
python scripts/7_train_cnn_solo.py \
    --vector_dir "/anvil/scratch/x-jdong8/fusion_data/vectors" \
    --grid_dir "/anvil/scratch/x-jdong8/fusion_data/grids" \
    --out_dir "/anvil/scratch/x-jdong8/fusion_models/cnn_solo" \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --loss_type bce