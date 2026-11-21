#!/bin/bash
# Submit one copy of a job per GPU type; whichever starts first runs, the rest are cancelled.
# Usage: slurm/submit_race.sh slurm/detect.sbatch [hydra overrides...]
set -euo pipefail
script=$1; shift
GPUS=${GPUS:-"h200 a100"}
mkdir -p slurm/logs slurm/.race
race=$(mktemp -d "$PWD/slurm/.race/race.XXXXXX")
ids=()
for gpu in $GPUS; do
  est=$(sbatch --test-only --gres=gpu:$gpu:1 "$script" 2>&1 | grep -o 'start at [^ ]*' || true)
  id=$(sbatch --parsable --gres=gpu:$gpu:1 --export=ALL,RACE_DIR="$race" "$script" "$@")
  ids+=("$id"); echo "submitted $id  gpu=$gpu  ($est)"
done
echo "${ids[*]}" > "$race/jobs"
echo "race dir: $race"
