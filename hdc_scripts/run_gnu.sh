#!/bin/bash
#
#SBATCH --job-name=single_trial_gnu_parallel
#SBATCH --partition=savio3_bigmem
#SBATCH --account=fc_flexemg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --time=71:30:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andyz@berkeley.edu
#
module load python/3.7
source activate hdcpar

module load gnu-parallel/2019.03.22

parallel --progress --jobs 30 < gnu_jobs.txt
