#!/bin/bash
#SBATCH --output=test32.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0:10:00
#SBATCH --partition=debug

python test.py