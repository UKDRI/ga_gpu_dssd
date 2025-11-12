#!/bin/bash
#SBATCH --job-name=dssd_emm_run
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00   # 4 hours (DSSD can be slow)
#SBATCH --mem=32G         # DSSD is memory hungry
#SBATCH --gres=gpu:1      # CRITICAL: Required by the pvc9 partition
#SBATCH -A AIRR-P13-DAWN-GPU

# --- Define Our Paths ---
# This is the directory where you are running the job
BASE_DIR=${SLURM_SUBMIT_DIR}

# These are the directories created by prepare_emm_data.py
PD_DATA_DIR="${BASE_DIR}/emm_pd_data"
CTRL_DATA_DIR="${BASE_DIR}/emm_ctrl_data"
EMM_RESULTS_DIR="${BASE_DIR}/emm_results"

# These are our config files
PD_CONF_FILE="${BASE_DIR}/dssd_pd.conf"
CTRL_CONF_FILE="${BASE_DIR}/dssd_ctrl.conf"

# This is our container
CONTAINER_SIF="${BASE_DIR}/wine-test-dssd.sif"

# --- DSSD Internal Paths (from our Nextflow/README investigation) ---
# DSSD will look for data in /opt/dssd/data/
# --- UPDATED: Added the 'datasets' subdirectory ---
INTERNAL_DATA_BASE="/opt/dssd/data/datasets"
# DSSD will write results to /opt/dssd/xps/dssd/
INTERNAL_OUTPUT_DIR="/opt/dssd/xps/dssd"
# DSSD program is at /opt/dssd/bin/
INTERNAL_BIN_DIR="/opt/dssd/bin"

# --- Load Environment (for Apptainer) ---
echo "Loading default-dawn..."
module purge
module load default-dawn

# --- Run 1: PD Patients ---
echo "--- Starting EMM for PD group ---"
apptainer exec \
    --bind ${PD_DATA_DIR}:${INTERNAL_DATA_BASE}/pd_run \
    --bind ${PD_CONF_FILE}:${INTERNAL_BIN_DIR}/dssd_pd.conf \
    --bind ${EMM_RESULTS_DIR}:${INTERNAL_OUTPUT_DIR} \
    ${CONTAINER_SIF} \
    wine ${INTERNAL_BIN_DIR}/dssd64.exe dssd_pd

echo "--- PD Run Complete. Results are in ${EMM_RESULTS_DIR} ---"


# --- Run 2: Control Patients ---
echo "--- Starting EMM for Control group ---"
apptainer exec \
    --bind ${CTRL_DATA_DIR}:${INTERNAL_DATA_BASE}/ctrl_run \
    --bind ${CTRL_CONF_FILE}:${INTERNAL_BIN_DIR}/dssd_ctrl.conf \
    --bind ${EMM_RESULTS_DIR}:${INTERNAL_OUTPUT_DIR} \
    ${CONTAINER_SIF} \
    wine ${INTERNAL_BIN_DIR}/dssd64.exe dssd_ctrl

echo "--- Control Run Complete. Results are in ${EMM_RESULTS_DIR} ---"
echo "--- EMM Analysis Finished ---"

