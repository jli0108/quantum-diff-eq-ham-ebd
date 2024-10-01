#!/bin/bash
#SBATCH --output=resource_analysis_2d_adv_upwind_pauli_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH --partition=serial

python resource_analysis_2d_adv_upwind_pauli.py