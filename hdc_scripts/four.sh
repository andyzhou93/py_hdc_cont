#!/bin/bash
#
#SBATCH --job-name=four_trial
#SBATCH --partition=savio2_htc
#SBATCH --account=fc_flexemg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andyz@berkeley.edu
#
module load python/3.6
source activate hdc
python four_trial.py > four_trial.pyout
