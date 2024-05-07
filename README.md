# Quantum simulation of differential equations

Project for solving differential equations using quantum computers.

This is a joint work by [Joseph Li](https://jli0108.github.io/), [Gengzhi Yang](https://github.com/Genz17), [Jiaqi Leng](https://jiaqileng.github.io/), and [Xiaodi Wu](https://www.cs.umd.edu/~xwu/).

# Code organization

The source code is organized as follows.

- `figures/` contains the experiment-related figures presented in the paper.
- `src/` contains all scripts used to run the experiments and resource analysis, as well as generating the figures. This directory is subdivided into the following five subdirectories:

    - `src/experiments/` contains Jupyter notebooks used to run the experiments (LCHS and Schrodingerization for non-Hermitian TFIM, and the 2D advection equation).
    - `src/experiment_data` contains the experiment data.
    - `src/resource_analysis` contains scripts for running the empirical resource comparison between different embedding schemes.
    - `src/plot` contains scripts for generating figures, which are saved in `figures/`

# Usage
The code has been tested with Python 3.10 but should also work with some earlier versions such as 3.8 or 3.9.

The experiments are run using [SimuQ](https://pickspeng.github.io/SimuQ/), which can be installed from the [GitHub repository](https://github.com/PicksPeng/SimuQ) or through pip.


