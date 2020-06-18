#!/bin/bash
#
#SBATCH --job-name=arm_positions_gnu_parallel
#SBATCH --partition=savio2_htc
#SBATCH --account=fc_flexemg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=71:30:00
#SBATCH --mail-type=all
#SBATCH --mail-user=andyz@berkeley.edu
#
module load python/3.7
source activate hdcpar

module load gnu-parallel/2019.03.22

export HDF5_USE_FILE_LOCKING='FALSE'

parallel --progress --jobs 10 < gnu_test.txt
