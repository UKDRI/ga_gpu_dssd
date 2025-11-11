#!/bin/bash
#SBATCH --job-name=ga_gather
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH -A AIRR-P13-DAWN-GPU
#SBATCH --mem=16G # Give it enough RAM to load data
# --- ADDED: This line is required by the pvc9 partition ---
#SBATCH --gres=gpu:1

# --- Get Output Directory from command line ---
if [ -z "$1" ]; then
  echo "Error: No output directory specified."
  echo "Usage: sbatch run_gather.slurm /path/to/results"
  exit 1
fi
OUTPUT_DIR=$1
BASE_DIR=${SLURM_SUBMIT_DIR}

# --- Load Environment ---
echo "Purging modules and loading DAWN environment..."
module purge
module load default-dawn
# --- Load the full conda env ---
module load intelpython-conda
conda activate pytorch-gpu

echo "Python path: $(which python3)"

# --- Run the script ---
echo "Starting Python gather_results.py script..."
# Run from BASE_DIR, pass the OUTPUT_DIR as the argument
python3 ${BASE_DIR}/gather_results.py ${OUTPUT_DIR}

echo "Gather script complete."

