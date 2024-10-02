#!/bin/bash
#SBATCH --output=resource_analysis_nonlinear_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --partition=serial
#SBATCH --mem-per-cpu=15000

python resource_analysis_nonlinear.py