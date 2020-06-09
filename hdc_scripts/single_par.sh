#!/bin/bash
#
#SBATCH --job-name=single_trial
#SBATCH --partition=savio3_bigmem
#SBATCH --account=fc_flexemg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=60:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andyz@berkeley.edu
#
module load python/3.7
source activate hdcpar
ipcluster start -n $SLURM_NTASKS &
sleep 45
python single_trial_par.py 1
python single_trial_par.py 2
python single_trial_par.py 4
python single_trial_par.py 8
python single_trial_par.py 16
python single_trial_par.py 32
python single_trial_par.py 64
ipcluster stop
