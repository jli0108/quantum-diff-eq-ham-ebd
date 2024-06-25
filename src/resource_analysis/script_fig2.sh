#!/bin/bash
#SBATCH --output=fig2_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 72:00:00
#SBATCH --partition=serial

python fig2.py