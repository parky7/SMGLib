#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=cpu
#SBATCH --job-name=fine_tune
#SBATCH --mem=30GB
#SBATCH --output=./slurm-%j.out
#SBATCH --error=./slurm-%j.err
#SBATCH --cpus-per-task=4

set -euo pipefail

# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$PROJECT_ROOT"

NUM_ROBOTS="${NUM_ROBOTS:-2}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
MAX_COMBINATIONS="${MAX_COMBINATIONS:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/CBF-RM/fine_tune}"

# Safety cap requested for cluster runs.
if [[ "$MAX_COMBINATIONS" =~ ^[0-9]+$ ]] && (( MAX_COMBINATIONS > 10000 )); then
	echo "MAX_COMBINATIONS=${MAX_COMBINATIONS} is above 10000, clamping to 10000."
	MAX_COMBINATIONS=10000
fi

QUICK_FLAG=()
if [[ "${QUICK:-0}" == "1" ]]; then
	QUICK_FLAG=(--quick)
fi

echo "Running src/fine_tune.py"
echo "num_robots=${NUM_ROBOTS} progress_every=${PROGRESS_EVERY} max_combinations=${MAX_COMBINATIONS}"
echo "output_dir=${OUTPUT_DIR}"

source ./.venv/bin/activate

python src/fine_tune.py \
	"${QUICK_FLAG[@]}" \
	--num-robots "$NUM_ROBOTS" \
	--progress-every "$PROGRESS_EVERY" \
	--max-combinations "$MAX_COMBINATIONS" \
	--output-dir "$OUTPUT_DIR"

