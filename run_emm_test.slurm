#!/bin/bash
#SBATCH --job-name=dssd_EMM_TEST
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00   # 15 min is enough to see if it crashes
#SBATCH --mem=32G
#SBATCH --gres=gpu:1      # Required by pvc9
#SBATCH -A AIRR-P13-DAWN-GPU

BASE_DIR=${SLURM_SUBMIT_DIR}
CONTAINER_SIF="${BASE_DIR}/wine-test-dssd.sif"

# The config file we are using for this test
TEST_CONF_FILE="${BASE_DIR}/emc_emotions_test.conf"

# --- Load Environment ---
echo "Loading default-dawn..."
module purge
module load default-dawn

# --- Run the EMM "emotions" Test ---
echo "--- Starting EMM 'emotions' control experiment ---"
echo "--- This will test if the container's EMM code is stable ---"

apptainer run \
    --pwd /opt/dssd/bin \
    --bind ${TEST_CONF_FILE}:/opt/dssd/bin/emc.conf \
    ${CONTAINER_SIF}

echo "--- EMM Test Complete ---"
