#!/bin/bash
#SBATCH --output=fig2_nonseparable_output.out
#SBATCH --ntasks=1
#SBATCH -t 72:00:00
#SBATCH --partition=serial

python fig2_nonseparable.py