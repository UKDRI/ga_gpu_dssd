#!/bin/bash
#SBATCH --job-name=gpu_ga_array
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00             # Keep time short for test
#SBATCH --array=0-99
#SBATCH -A AIRR-P13-DAWN-GPU

# --- Get Output Directory from command line argument $1 ---
if [ -z "$1" ]; then
  echo "Error: No output directory specified."
  echo "Usage: sbatch run_ga_array.slurm <output_directory_path>"
  exit 1
fi
OUTPUT_DIR=$1
BASE_DIR=${SLURM_SUBMIT_DIR} # Directory where sbatch was run

# --- Use the provided OUTPUT_DIR for SLURM logs ---
#SBATCH --output=${OUTPUT_DIR}/slurm_output_%a.log
#SBATCH --error=${OUTPUT_DIR}/slurm_output_%a.log

# --- Check if Output Directory exists (optional but good practice) ---
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Error: Output directory '${OUTPUT_DIR}' does not exist. Please create it first."
  # Optionally create it here? No, let's stick to user creating it.
  # mkdir -p ${OUTPUT_DIR}
  exit 1
fi

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
# Pass TaskID, SUBMIT_DIR (for data), and the user-provided OUTPUT_DIR
python3 ${BASE_DIR}/simple_ga_array.py $SLURM_ARRAY_TASK_ID ${BASE_DIR} ${OUTPUT_DIR}

echo "Task $SLURM_ARRAY_TASK_ID complete."
