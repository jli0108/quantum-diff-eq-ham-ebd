# Quantum simulation of differential equations

Project for solving differential equations using quantum computers.

This is a joint work by [Joseph Li](https://jli0108.github.io/), [Gengzhi Yang](https://github.com/Genz17), [Jiaqi Leng](https://jiaqileng.github.io/), and [Xiaodi Wu](https://www.cs.umd.edu/~xwu/).

# Code organization

The source code is organized as follows.

- `figures/` contains all figures presented in the paper.
- `src/` contains all scripts used to run the experiments and resource analysis, as well as generating the figures. This directory is subdivided into the following five subdirectories:

    - `src/experiments/` contains files used to run the 2D advection equation experiment.
    - `src/experiment_data` contains the experimental data.
    - `src/resource_analysis` contains scripts for running the empirical resource comparison between different embedding schemes.
    Specifically, the scripts `resource_analysis_2d_adv_upwind_{scheme}.py` are used to obtain data shown in Figure 3.
    The script `resource_analysis_nonlinear_fourier.py` is used to obtain data shown in Figure 4.
    - `src/plot` contains scripts for generating Figures 3 and 4, which are saved in `figures/`.

# Usage
The code has been tested with Python 3.10 but should also work with some earlier versions such as 3.8 or 3.9.

The experiments are run using [SimuQ](https://pickspeng.github.io/SimuQ/), which can be installed from the [GitHub repository](https://github.com/PicksPeng/SimuQ) or through pip.

There are a few other dependencies used in this project. Below are the relevant packages, along with the tested versions.
- numpy 1.23.5
- scipy 1.11.1
- qiskit 1.0.2
- pytket 1.26.0
- pytket-qiskit 0.51.0

For plotting the simulation of a nonlinear hyperbolic PDE, we use Mathematica and [ReadNumPy](https://github.com/lr94/NumPyArray) for reading `.npy` files.
