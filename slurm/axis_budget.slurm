#!/bin/bash
# Does the oracle advantage depend on the training budget?
#
# The Gin Rummy probe ran at one budget. If the oracle observation only helps early, or only
# late, a single-budget null says less than it appears to. This sweeps four budgets on both
# small games, where the optimum is computable, so the answer is checkable.
#
#   sbatch --array=0-31 slurm/axis_budget.slurm
#
#SBATCH --job-name=axis-budget
#SBATCH --account=REDACTED_SLURM_ACCOUNT
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=$HOME/diagnosing_game_plateau/slurm/logs/budget_%A_%a.out
#SBATCH --error=$HOME/diagnosing_game_plateau/slurm/logs/budget_%A_%a.err
set -euo pipefail

module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/coev
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR"
cd $HOME/diagnosing_game_plateau

GAMES=(kuhn_poker leduc_poker)
BUDGETS=(25000 100000 400000 1600000)
T=${SLURM_ARRAY_TASK_ID}
G=$(( T / 16 )); R=$(( T % 16 )); B=$(( R / 4 )); S=$(( R % 4 ))
GAME=${GAMES[$G]}; EPISODES=${BUDGETS[$B]}

echo "task $T -> $GAME seed $S budget $EPISODES"
python scripts/probe_axis.py --game "$GAME" --seed "$S" \
       --episodes "$EPISODES" --eval-games 40000 --cfr-iters 3000 \
       --out $HOME/diagnosing_game_plateau/data/axis_budget/b${EPISODES}
