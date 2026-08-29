#!/bin/bash
# The run the paper actually needs.
#
# The oracle-observation null on Gin Rummy rests on 4 seeds, giving a 95% interval of
# [-4.5, +2.7] percentage points. That cannot rule out the oracle helping by 2.7pp, which is an
# 11% relative gain, so the null is currently under-powered rather than established.
#
# Two things change here. Twelve seeds per arm instead of four, which is what 2pp at 80% power
# needs. And a placebo arm: the same fifth plane carrying no information about the opponent's
# hand. Without the placebo, "the oracle plane changed nothing" cannot be told apart from "the
# information's benefit exactly cancelled the cost of widening the observation".
#
#   sbatch --array=0-23 slurm/gin_power.slurm
#
#SBATCH --job-name=gin-power
#SBATCH --account=REDACTED_SLURM_ACCOUNT
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/diagnosing_game_plateau/slurm/logs/ginpow_%A_%a.out
#SBATCH --error=$HOME/diagnosing_game_plateau/slurm/logs/ginpow_%A_%a.err
set -euo pipefail
module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/coev
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd $HOME/diagnosing_game_plateau

T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 12 ]; then ARM=oracle;  SEED=$T;          else ARM=placebo; SEED=$((T - 12)); fi
echo "task $T -> arm=$ARM seed=$SEED"
python scripts/train_oracle.py --arm "$ARM" --seed "$SEED" --steps 3000000
