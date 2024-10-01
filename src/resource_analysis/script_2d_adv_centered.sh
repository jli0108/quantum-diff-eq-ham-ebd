#!/bin/bash
#SBATCH --output=resource_analysis_2d_adv_centered_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --partition=serial

python resource_analysis_2d_adv_centered.py