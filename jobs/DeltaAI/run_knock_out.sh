#!/bin/bash
#SBATCH --job-name=cosmic_knockout
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
# SYSTEM-LEVEL PARANOIA (ABSOLUTE SHIELDS)
# ==============================================================================
# Prevent Lustre filesystem deadlock during parallel DataLoader reads
export HDF5_USE_FILE_LOCKING=FALSE

# Suppress OpenMP/MKL thread explosion on Grace Hopper CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Debugging flags
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1  

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
source /projects/bdne/jdong8/miniforge3_arm64/bin/activate gnn-gh200
cd /projects/bdne/jdong8/src/evidence-fusion
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "================================================================"
echo ">>> INITIATING THE KNOCKOUT EXPERIMENT"
echo "================================================================"

python scripts/analysis/knock_out_exp.py \
    --vector_dir "/work/hdd/bdne/jdong8/fusion_data_purified/vectors" \
    --catalog_dir "/work/hdd/bdne/jdong8/fusion_data_purified/catalogs" \
    --mlp_dir "/work/hdd/bdne/jdong8/fusion_models/abacus_mlp_solo" \
    --fus_dir "/work/hdd/bdne/jdong8/fusion_models/abacus_early_fusion" \
    --out_dir "/projects/bdne/jdong8/src/evidence-fusion/plots" \
    --subbox_size 60.0 \
    --num_subboxes 8 \
    --r_link 20.0 \
    --shuffle_mode "knockout_graph"

echo "================================================================"
echo ">>> EXPERIMENT CONCLUDED. CHECK YOUR PLOTS."
echo "================================================================"