#!/bin/bash
# Baseline arm at the same power: four planes, seeds 4-11, same recipe and budget.
# The oracle and placebo arms need something to be compared against at twelve seeds too,
# otherwise the comparison is twelve against four.
#
#   sbatch --array=4-11 slurm/gin_baseline.slurm
#
#SBATCH --job-name=gin-base
#SBATCH --account=REDACTED_SLURM_ACCOUNT
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/diagnosing_game_plateau/slurm/logs/ginbase_%A_%a.out
#SBATCH --error=$HOME/diagnosing_game_plateau/slurm/logs/ginbase_%A_%a.err
set -euo pipefail
module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/coev
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd $HOME/Adversarial-CoEvolution
python sweep/curriculum_train.py --config sweep/ppoarch_cfgs/ppoarch_mlp_s${SLURM_ARRAY_TASK_ID}.json 2>/dev/null \
  || python -m sweep.curriculum_train --config sweep/ppoarch_cfgs/ppoarch_mlp_s${SLURM_ARRAY_TASK_ID}.json
