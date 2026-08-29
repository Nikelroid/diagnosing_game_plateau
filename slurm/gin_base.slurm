#!/bin/bash
# Baseline arm at matched power: the original four planes, seeds 4-11, same recipe and budget.
#   sbatch --array=4-11 slurm/gin_base.slurm
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
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
cd $HOME/diagnosing_game_plateau
python scripts/train_oracle.py --arm baseline --seed ${SLURM_ARRAY_TASK_ID} --steps 3000000
