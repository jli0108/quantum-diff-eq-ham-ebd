#!/bin/bash
#SBATCH --output=fig2_nonseparable_first_order_output.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH --partition=serial

python fig2_nonseparable_first_order.py