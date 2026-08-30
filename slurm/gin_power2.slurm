#!/bin/bash
# Second power wave for the Gin Rummy oracle-observation probe: seeds 12-23, all three arms.
#
# The paper currently rests on 12 seeds per arm, giving oracle-minus-baseline
# -0.25 pp with a 95% interval of [-1.70, +1.20]. Doubling to 24 seeds per arm shrinks the
# interval by about sqrt(2) and tightens the ruled-out gain from ~1.2 points to ~0.75.
#
# Task map (36 tasks): 0-11 oracle s12-23, 12-23 placebo s12-23, 24-35 baseline s12-23.
#
#   sbatch --array=0-35 slurm/gin_power2.slurm
#
#SBATCH --job-name=gin-pow2
#SBATCH --account=REDACTED_SLURM_ACCOUNT
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=$HOME/diagnosing_game_plateau/slurm/logs/ginpow2_%A_%a.out
#SBATCH --error=$HOME/diagnosing_game_plateau/slurm/logs/ginpow2_%A_%a.err
set -euo pipefail
module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/coev
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd $HOME/diagnosing_game_plateau

T=${SLURM_ARRAY_TASK_ID}
if   [ "$T" -lt 12 ]; then ARM=oracle;   SEED=$((T + 12))
elif [ "$T" -lt 24 ]; then ARM=placebo;  SEED=$((T - 12 + 12))
else                       ARM=baseline; SEED=$((T - 24 + 12))
fi
echo "task $T -> arm=$ARM seed=$SEED"
python scripts/train_oracle.py --arm "$ARM" --seed "$SEED" --steps 3000000
