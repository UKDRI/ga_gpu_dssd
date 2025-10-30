#!/bin/bash
#SBATCH --job-name=gpu_ga_10k
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-999             # <<< Run 1000 jobs
#SBATCH -A AIRR-P13-DAWN-GPU
#SBATCH --time=10:15:00           # <<< Set to 10 hours 15 mins

# --- Get Output Directory from command line argument $1 ---
if [ -z "$1" ]; then
  echo "Error: No output directory specified."
  exit 1
fi
OUTPUT_DIR=$1
BASE_DIR=${SLURM_SUBMIT_DIR}

# --- Use the provided OUTPUT_DIR for SLURM logs ---
#SBATCH --output=${OUTPUT_DIR}/slurm_output_%a.log
#SBATCH --error=${OUTPUT_DIR}/slurm_output_%a.log

# --- Load Modules and Environment ---
echo "Task $SLURM_ARRAY_TASK_ID: Purging modules and loading DAWN environment..."
module purge
module load default-dawn
module load intelpython-conda
conda activate pytorch-gpu

# --- Verify Environment ---
echo "Task $SLURM_ARRAY_TASK_ID: Running on node: $(hostname)"
echo "Task $SLURM_ARRAY_TASK_ID: Python path: $(which python)"
echo "Task $SLURM_ARRAY_TASK_ID: Current directory: $(pwd)" # Will be BASE_DIR

# --- Run the script ---
echo "Task $SLURM_ARRAY_TASK_ID: Starting Python GA script..."
python3 ${BASE_DIR}/simple_ga_array.py $SLURM_ARRAY_TASK_ID ${BASE_DIR} ${OUTPUT_DIR}

echo "Task $SLURM_ARRAY_TASK_ID complete."

