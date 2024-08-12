#!/bin/bash
#SBATCH --output=fig2_separable_pauli_5_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH --partition=serial

python fig2_separable_pauli_5.py