#!/bin/bash
#SBATCH --output=resource_analysis_nonlinear_first_order_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --partition=serial
#SBATCH --mem-per-cpu=8000

python resource_analysis_nonlinear_first_order.py