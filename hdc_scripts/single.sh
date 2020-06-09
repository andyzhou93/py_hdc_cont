#!/bin/bash
#
#SBATCH --job-name=single_trial
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
python single_trial.py 1 1
python single_trial.py 1 2
python single_trial.py 1 4
python single_trial.py 1 8
python single_trial.py 1 16
python single_trial.py 1 32
python single_trial.py 1 64
