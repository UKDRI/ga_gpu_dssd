#!/bin/bash
#SBATCH --job-name=gpu_ga_TEST
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A AIRR-P13-DAWN-GPU

# --- 10 MINUTE TEST CONFIG ---
# (Run 20 jobs for 10 minutes each)
#SBATCH --array=0-19
#SBATCH --time=00:10:00
export JOB_TIME_MINS=10


# --- Get Output Directory from submit_ga.sh ---
if [ -z "$1" ]; then
  echo "Error: This script is intended to be run by submit_ga.sh"
  echo "It requires an output directory as an argument."
  exit 1
fi
OUTPUT_DIR=$1
BASE_DIR=${SLURM_SUBMIT_DIR}

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
# Pass the new JOB_TIME_MINS argument
python3 ${BASE_DIR}/simple_ga_array.py $SLURM_ARRAY_TASK_ID ${BASE_DIR} ${OUTPUT_DIR} ${JOB_TIME_MINS}

echo "Task $SLURM_ARRAY_TASK_ID complete."
