#!/bin/bash
#SBATCH --output=resource_output_1d_tfim_lchs.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 24:00:00
#SBATCH --partition=serial

source ~/.bashrc  # <- Required to give you access to conda command if auto-activate has been turned off
conda activate base

python resource_analysis_1d_tfim.py lchs