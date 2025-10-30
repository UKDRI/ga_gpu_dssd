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
sbatch \
  --output="${OUTDIR}/logs/output/slurm_%A_%a.out" \
  --error="${OUTDIR}/logs/error/slurm_%A_%a.err" \
  "$@" \
  run_ga_array.slurm "${OUTDIR}"
