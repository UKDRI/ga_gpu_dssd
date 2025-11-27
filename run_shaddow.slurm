#!/bin/bash
#SBATCH --job-name=shadow_run_100x
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -A AIRR-P13-DAWN-GPU

# --- SHADOW RUN CONFIG ---
# Full statistical power (100 jobs)
#SBATCH --array=0-99
# Shorter duration (10 hours) for rapid validation
#SBATCH --time=10:15:00
export JOB_TIME_MINS=600  # 10 Hours

# --- Directory Setup ---
BASE_DIR=${SLURM_SUBMIT_DIR}
OUTPUT_DIR="${BASE_DIR}/shadow_results_100job"
mkdir -p ${OUTPUT_DIR}

# --- Environment ---
module purge
module load default-dawn
module load intelpython-conda
conda activate pytorch-gpu

echo "Shadow Run Task $SLURM_ARRAY_TASK_ID"
echo "Reading blinded data from: sc_data_blinded"

# --- Run the SHADOW script ---
# Uses 'shadow_ga.py' which points to the blinded data
python3 ${BASE_DIR}/shadow_ga.py $SLURM_ARRAY_TASK_ID ${BASE_DIR} ${OUTPUT_DIR} ${JOB_TIME_MINS}