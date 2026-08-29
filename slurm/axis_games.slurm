#!/bin/bash
# The oracle-observation probe across every OpenSpiel variant small enough to solve exactly.
#
# Eleven variants spanning four families (poker, dice, fog-of-war hex) at sizes from 12 to
# 24576 information states. Each carries its own exact hidden-bit count, so the probe result
# and its position on the axis come from the same run.
#
#   sbatch --array=0-103 slurm/axis_games.slurm   # 13 variants x 8 seeds
#
#SBATCH --job-name=axis-games
#SBATCH --account=REDACTED_SLURM_ACCOUNT
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=$HOME/diagnosing_game_plateau/slurm/logs/games_%A_%a.out
#SBATCH --error=$HOME/diagnosing_game_plateau/slurm/logs/games_%A_%a.err
set -euo pipefail
module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/coev
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR"
cd $HOME/diagnosing_game_plateau

SPECS=(
  "kuhn_poker"
  "liars_dice(numdice=1,dice_sides=3)"
  "liars_dice_ir(numdice=1,dice_sides=4)"
  "leduc_poker"
  "liars_dice(numdice=1,dice_sides=4)"
  "leduc_poker(players=2,suit_isomorphism=true)"
  "liars_dice_ir(numdice=1,dice_sides=5)"
  "liars_dice(numdice=1,dice_sides=5)"
  "liars_dice_ir(numdice=1,dice_sides=6)"
  "liars_dice(numdice=1,dice_sides=6)"
  "leduc_poker(action_mapping=true)"
  "liars_dice(numdice=2,dice_sides=3)"
  "liars_dice_ir(numdice=1,dice_sides=8)"
)
# episode budget scaled to the size of the game, so every variant gets a comparable
# number of updates per information state rather than a flat and unfair budget
BUDGETS=( 300000 400000 500000 800000 500000 800000 700000 900000 1200000 1500000 800000 2000000 2000000 )

T=${SLURM_ARRAY_TASK_ID}
G=$(( T / 8 )); SEED=$(( T % 8 ))
SPEC=${SPECS[$G]}; EP=${BUDGETS[$G]}

echo "task $T -> $SPEC seed $SEED budget $EP"
python scripts/probe_axis.py --game "$SPEC" --seed "$SEED" \
       --episodes "$EP" --eval-games 40000 --cfr-iters 3000 \
       --out $HOME/diagnosing_game_plateau/data/axis_games
