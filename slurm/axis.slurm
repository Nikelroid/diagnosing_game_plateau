#!/bin/bash
# Oracle-observation probe along the hidden-information axis.
#
# One array task per (game, seed). Each task is single-core and minutes long, but there are
# 16 of them and they are real computation, so they belong in the scheduler rather than on a
# login node.
#
#   sbatch --array=0-15 slurm/axis.slurm
#
#SBATCH --job-name=probe-axis
#SBATCH --account=${SLURM_ACCOUNT:-your_account_here}
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output="${REPO_ROOT:-$HOME/diagnosing_game_plateau}"/slurm/logs/axis_%A_%a.out
#SBATCH --error="${REPO_ROOT:-$HOME/diagnosing_game_plateau}"/slurm/logs/axis_%A_%a.err
set -euo pipefail

module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-$HOME/envs/coev}"
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR"

cd "${REPO_ROOT:-$HOME/diagnosing_game_plateau}"

# tasks 0-7 -> kuhn seeds 0-7 ; tasks 8-15 -> leduc seeds 0-7
T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 8 ]; then
  GAME=kuhn_poker;  SEED=$T;         EPISODES=300000
else
  GAME=leduc_poker; SEED=$((T - 8)); EPISODES=800000
fi

echo "task $T -> $GAME seed $SEED, $EPISODES episodes"
python scripts/probe_axis.py --game "$GAME" --seed "$SEED" \
       --episodes "$EPISODES" --eval-games 40000 --cfr-iters 3000
