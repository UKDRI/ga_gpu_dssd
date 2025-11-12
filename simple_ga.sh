#!/usr/bin/env bash
set -euo pipefail

# --- Usage ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_dir> [extra sbatch args...]"
  echo "Example: $0 /scratch/dn-neav1/gpu_ga_results --time=01:00:00"
  exit 1
fi

# --- Main Output Directory ---
OUTDIR=$(realpath "$1")
shift  # shift off first argument so extra sbatch args can follow

# --- Create subdirectories for logs ---
mkdir -p "${OUTDIR}/logs/output"
mkdir -p "${OUTDIR}/logs/error"

echo "Submitting array job..."
echo "  Output directory : ${OUTDIR}"
echo "  Stdout logs      : ${OUTDIR}/logs/output/"
echo "  Stderr logs      : ${OUTDIR}/logs/error/"
echo

# --- Submit the array job ---
# 1. Submit the main GA array job and capture its Job ID
#    We use --parsable to get *only* the job ID
#    We also give it a unique log file name "slurm_GA..."
GA_JOB_ID=$(sbatch \
  --parsable \
  --job-name=gpu_ga_array \
  --output="${OUTDIR}/logs/output/slurm_GA_%A_%a.out" \
  --error="${OUTDIR}/logs/error/slurm_GA_%A_%a.err" \
  "$@" \
  run_ga_array.slurm "${OUTDIR}")

echo "GA array job submitted with ID: ${GA_JOB_ID}"

# --- Submit the gather job, dependent on the array ---
# 2. Submit the gather job to run *after* the array finishes
#    --dependency=afterok:${GA_JOB_ID}
#    This job won't start until the *entire* array (jobid_*) is done.
echo "Submitting gather job (will run after ${GA_JOB_ID} completes)..."
GATHER_JOB_ID=$(sbatch \
  --parsable \
  --job-name=ga_gather \
  --output="${OUTDIR}/logs/output/slurm_GATHER_%A.out" \
  --error="${OUTDIR}/logs/error/slurm_GATHER_%A.err" \
  --dependency=afterok:${GA_JOB_ID} \
  run_gather.slurm "${OUTDIR}")

echo "Gather job submitted with ID: ${GATHER_JOB_ID}"
echo "Workflow successfully launched."

