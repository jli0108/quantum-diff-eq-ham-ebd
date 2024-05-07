#!/bin/bash
#SBATCH --output=resource_output_2d_burgers.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 48:00:00
#SBATCH --partition=serial

module load python

python resource_analysis_2d_burgers.py