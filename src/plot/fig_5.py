import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from os.path import join
import sys
sys.path.append(join(".", ".."))
from utils import *


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


trotter_method = "first_order"
data = np.load(f"../resource_analysis_data/nonlinear_data.npz")
N_vals = data["N_vals"]

pauli_basis_trotter_steps = data["pauli_basis_trotter_steps"]
pauli_basis_single_qubit_gates = data["pauli_basis_single_qubit_gates"]
pauli_basis_two_qubit_gates = data["pauli_basis_two_qubit_gates"]
pauli_basis_gate_count = pauli_basis_trotter_steps * (pauli_basis_single_qubit_gates + pauli_basis_two_qubit_gates)
pauli_basis_circ_depth_per_trotter_step = data["pauli_basis_circ_depth"]
pauli_basis_circ_depth = pauli_basis_trotter_steps * pauli_basis_circ_depth_per_trotter_step

one_hot_trotter_steps = data["one_hot_trotter_steps"]
one_hot_single_qubit_gates = data["one_hot_single_qubit_gates"]
one_hot_two_qubit_gates = data["one_hot_two_qubit_gates"]
one_hot_gate_count = one_hot_trotter_steps * (one_hot_single_qubit_gates + one_hot_two_qubit_gates)
one_hot_circ_depth_per_trotter_step = data["one_hot_circ_depth"]
one_hot_circ_depth = one_hot_trotter_steps * one_hot_circ_depth_per_trotter_step

unary_trotter_steps = data["unary_trotter_steps"]
unary_single_qubit_gates = data["unary_single_qubit_gates"]
unary_two_qubit_gates = data["unary_two_qubit_gates"]
unary_gate_count = unary_trotter_steps * (unary_single_qubit_gates + unary_two_qubit_gates)
unary_circ_depth_per_trotter_step = data["one_hot_circ_depth"]
unary_circ_depth = unary_trotter_steps * unary_circ_depth_per_trotter_step

fig = plt.figure(figsize=(10,3.4))
gs = fig.add_gridspec(1, 2, wspace=0.25)
axs = gs.subplots()

fig.suptitle(r"Resource analysis for 1D nonlinear PDE:", y=1.055, size=14)
plt.figtext(x=0.25, y=0.93, s=r"$\frac{\partial u}{\partial t} - u^3 (1-u^4)\frac{\partial u}{\partial x} - \left[x^3(1-x^2)+1\right]\left[u^3(1-u^2)+1\right]=0$", size=14)

axs[0].plot(N_vals, pauli_basis_gate_count, '-s', linewidth=1, color="red", label="Std binary (Pauli basis)")
axs[0].plot(N_vals, one_hot_gate_count, '-s', linewidth=1, color="green", label="One-hot", alpha=0.9)
axs[0].plot(N_vals, unary_gate_count, '-s', linewidth=1, color="blue", label="Circulant unary", alpha=0.4)
axs[0].set_ylabel("Total gate count")
axs[0].set_xlabel(rf"$N$ (number of grid points per dimension)")
axs[0].set_yscale('log')
axs[0].set_xscale('log', base=2)
axs[0].legend()


axs[1].plot(N_vals, pauli_basis_circ_depth, '-s', linewidth=1, color="red", label="Std binary (Pauli basis)")
axs[1].plot(N_vals, one_hot_circ_depth, '-s', linewidth=1, color="green", label="One-hot", alpha=0.9)
axs[1].plot(N_vals, unary_circ_depth, '-s', linewidth=1, color="blue", label="Circulant unary", alpha=0.4)
axs[1].set_ylabel("Circuit depth")
axs[1].set_xlabel(rf"$N$ (number of grid points per dimension)")
axs[1].set_yscale('log')
axs[1].set_xscale('log', base=2)
axs[1].legend()

plt.savefig(join("..", "..", "figures", "Fig_5.pdf"), bbox_inches='tight')
plt.show()