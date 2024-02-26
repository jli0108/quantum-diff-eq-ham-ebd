#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --mem=1000000
#SBATCH --partition=serial

module load python

python simulate_pde_2d.py