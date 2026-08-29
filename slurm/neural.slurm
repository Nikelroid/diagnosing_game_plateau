#!/bin/bash
# The apparatus-matched probe: same fixed-width tensor and network as the Gin Rummy learner.
#
# The tabular probe differs from the Gin Rummy experiment on three axes at once: a table versus
# a network, a key multiplier versus an extra channel, and a fixed opponent versus a self-play
# pool. This closes the first two. Budget is 200k episodes because the learner is still at
# random after 20k and only reaches -0.26 (against an optimum of -0.14) by 150k.
#
#   sbatch --array=0-35 slurm/neural.slurm
#
#SBATCH --job-name=neural
#SBATCH --account=REDACTED_SLURM_ACCOUNT
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=$HOME/diagnosing_game_plateau/slurm/logs/nn_%A_%a.out
#SBATCH --error=$HOME/diagnosing_game_plateau/slurm/logs/nn_%A_%a.err
set -euo pipefail
module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/coev
export PYTHONNOUSERSITE= OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd $HOME/diagnosing_game_plateau

GAMES=( "leduc_poker" "kuhn_poker" "liars_dice(numdice=1,dice_sides=4)" )
ARMS=( baseline oracle placebo )
T=${SLURM_ARRAY_TASK_ID}
G=$(( T / 12 )); R=$(( T % 12 )); A=$(( R / 4 )); S=$(( R % 4 ))
python scripts/probe_neural.py --game "${GAMES[$G]}" --arm "${ARMS[$A]}" --seed "$S" \
       --episodes 200000 --eval-games 10000 --cfr-iters 2000
