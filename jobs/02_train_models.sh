#!/bin/bash
#SBATCH --job-name=evidence_train_gpu   # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=64            # Number of tasks
#SBATCH --gpus-per-node=2    # Number of GPUs
#SBATCH --time=04:30:00         # Time limit
#SBATCH --partition=gpu      # Partition name
#SBATCH --account=phy240043-gpu   # Account name
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/%x_%A_%a.out   # Error file for each array task

module load conda
conda activate ensemble-ili

PROJECT_ROOT="/home/x-jdong8/src/evidence-fusion"
cd $PROJECT_ROOT

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Starting Training on $(hostname)"
nvidia-smi

echo "------------------------------------------------"
echo "[Step 4] Training Fusion Student (Fresh Retrain)..."
python scripts/4_train_fusion.py \
    --vector_dir "/anvil/scratch/x-jdong8/fusion_data/vectors" \
    --grid_dir "/anvil/scratch/x-jdong8/fusion_data/grids" \
    --mlp_dir "/anvil/scratch/x-jdong8/fusion_models/mlp" \
    --out_dir "/anvil/scratch/x-jdong8/fusion_models/fusion" \
    --epochs 30 \
    --batch_size 16 \
    --accum_steps 2 \
    --lr 2e-5
    # --resume "/anvil/scratch/x-jdong8/fusion_models/fusion/fusion_best.pt"