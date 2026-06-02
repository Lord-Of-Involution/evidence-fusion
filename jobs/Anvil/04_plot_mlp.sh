#!/bin/bash
#SBATCH --job-name=plot_mlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-jdong8/jobout/plot_%j.out
#SBATCH --error=/anvil/scratch/x-jdong8/jobout/plot_%j.err

module load conda
conda activate ensemble-ili

PROJECT_ROOT="/home/x-jdong8/src/evidence-fusion"
cd $PROJECT_ROOT


DATA_ROOT="/anvil/scratch/x-jdong8/fusion_data/vectors"

MODEL_DIR="/anvil/scratch/x-jdong8/fusion_models/mlp"

# MODEL_DIR="./models/mlp_paper"

echo "Generating plots..."
python scripts/6_eval_mlp.py \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_ROOT"

echo "Done! Check the 'plots' folder inside $MODEL_DIR"