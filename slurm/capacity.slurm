#!/bin/bash
# Capacity probe: how few distinct situations can a learner tell apart before it pays?
#
# Buckets are the capacity knob. At 0 the learner is exact; below the game's information-state
# count it is forced to share a policy across situations it cannot distinguish. Sweeping this
# against a CFR optimum shows what a genuinely capacity-bound plateau looks like, which is the
# reference the Gin Rummy capacity null has never had.
#
#   sbatch --array=0-95 slurm/capacity.slurm
#
#SBATCH --job-name=capacity
#SBATCH --account=${SLURM_ACCOUNT:-your_account_here}
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output="${REPO_ROOT:-$HOME/diagnosing_game_plateau}"/slurm/logs/cap_%A_%a.out
#SBATCH --error="${REPO_ROOT:-$HOME/diagnosing_game_plateau}"/slurm/logs/cap_%A_%a.err
set -euo pipefail
module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-$HOME/envs/coev}"
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd "${REPO_ROOT:-$HOME/diagnosing_game_plateau}"

GAMES=( "kuhn_poker" "leduc_poker" "liars_dice(numdice=1,dice_sides=4)" "dark_hex(board_size=2)" )
BUCKETS=( 2 4 8 16 32 128 512 0 )        # 0 = exact, no folding
T=${SLURM_ARRAY_TASK_ID}
G=$(( T / 24 )); R=$(( T % 24 )); B=$(( R / 3 )); S=$(( R % 3 ))
python scripts/probe_capacity.py --game "${GAMES[$G]}" --buckets "${BUCKETS[$B]}" \
       --seed "$S" --episodes 400000 --eval-games 40000 --cfr-iters 3000
