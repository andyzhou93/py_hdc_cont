#!/bin/bash
#
#SBATCH --job-name=four_trial
#SBATCH --partition=savio2_htc
#SBATCH --account=fc_flexemg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andyz@berkeley.edu
#
module load python/3.7
source activate hdcpar
python four_trial.py 1 1
python four_trial.py 1 2
python four_trial.py 1 4