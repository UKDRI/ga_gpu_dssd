#!/bin/bash
#SBATCH --job-name=gpu_ga_10k
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-999         # <<< Run 1000 jobs
#SBATCH -A AIRR-P13-DAWN-GPU

# --- 10 MINUTE TEST CONFIG ---
# (Uncomment these two lines for a quick test)
SBATCH --time=00:10:00       # <<< 10 Minute time limit
export JOB_TIME_MINS=10       # <<< Pass "10" to Python

# --- 10 HOUR+ PRODUCTION CONFIG ---
# (Uncomment these two lines for production)
#SBATCH --time=10:15:00       # <<< 10 Hour 15 Minute time limit
#export JOB_TIME_MINS=615      # <<< Pass "615" (10h 15m) to Python


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
# Pass the new JOB_TIME_MINS argument
python3 ${BASE_DIR}/simple_ga_array.py $SLURM_ARRAY_TASK_ID ${BASE_DIR} ${OUTPUT_DIR} ${JOB_TIME_MINS}

echo "Task $SLURM_ARRAY_TASK_ID complete."
