#!/bin/bash
# Oracle-observation probe: seeds as an array job.
#
# Runs the baseline recipe with the opponent's hand appended as a fifth observation plane.
# Submitted as a batch job rather than run interactively because an interactive session's
# allocation ends when the session does, which is exactly how seeds 1-3 were lost once already.
#
#   sbatch --array=1-3 slurm/oracle.slurm          # the three remaining seeds
#   sbatch --array=0-3 slurm/oracle.slurm          # all four from scratch
#
#SBATCH --job-name=oracle-obs
#SBATCH --account=${SLURM_ACCOUNT:-your_account_here}
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=$HOME/oracle-probes/slurm/logs/oracle_%A_%a.out
#SBATCH --error=$HOME/oracle-probes/slurm/logs/oracle_%A_%a.err
set -euo pipefail

module purge
source ~/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-$HOME/envs/coev}"

# PYTHONNOUSERSITE: the conda env has NumPy 2.4.6 while ~/.local has 1.26.4 and shadows it; the
# curated seed models are NumPy 2 pickles and fail to load under NumPy 1.
# OMP_NUM_THREADS=1: anything higher deadlocks the forked vectorised envs after the first eval.
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_GROUP=oracle-probes NUM_ENV=16

cd $HOME/oracle-probes
python scripts/train_oracle.py --seed "${SLURM_ARRAY_TASK_ID}" --steps 3000000
