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
data = np.load(f"../resource_analysis_data/nonlinear/nonlinear_data.npz")
N_vals = data["N_vals"]

pauli_basis_trotter_steps = data["pauli_basis_trotter_steps"]
pauli_basis_circ_depth_per_trotter_step = data["pauli_basis_circ_depth"]
pauli_basis_two_qubit_gates_per_trotter_step = data["pauli_basis_two_qubit_gates"]
pauli_basis_circ_depth = pauli_basis_trotter_steps * pauli_basis_circ_depth_per_trotter_step + 2 * np.log2(N_vals).astype(int)
pauli_basis_two_qubit_gates = pauli_basis_trotter_steps * pauli_basis_two_qubit_gates_per_trotter_step + 2 * np.log2(N_vals).astype(int)

one_hot_trotter_steps = data["one_hot_trotter_steps"]
one_hot_circ_depth_per_trotter_step = data["one_hot_circ_depth"]
one_hot_two_qubit_gates_per_trotter_step = data["one_hot_two_qubit_gates"]
one_hot_circ_depth = one_hot_trotter_steps * one_hot_circ_depth_per_trotter_step + 2 * (N_vals)
one_hot_two_qubit_gates = one_hot_trotter_steps * one_hot_two_qubit_gates_per_trotter_step + 2 * (N_vals)

unary_trotter_steps = data["unary_trotter_steps"]
unary_circ_depth_per_trotter_step = data["unary_circ_depth"]
unary_two_qubit_gates_per_trotter_step = data["unary_two_qubit_gates"]
unary_circ_depth = unary_trotter_steps * unary_circ_depth_per_trotter_step + 2 * (N_vals // 2)
unary_two_qubit_gates = unary_trotter_steps * unary_two_qubit_gates_per_trotter_step + 2 * (N_vals // 2)

fig = plt.figure(figsize=(5,3.4))

plt.plot(N_vals, pauli_basis_circ_depth, '-s', linewidth=1, color="red", label="Std binary (Pauli basis)")
plt.plot(N_vals, one_hot_circ_depth, '-s', linewidth=1, color="green", label="One-hot")
plt.plot(N_vals, unary_circ_depth, '-s', linewidth=1, color="blue", label="Circulant unary")
plt.ylabel("Circuit depth")
plt.xlabel(rf"$N$ (number of grid points per dimension)")
plt.yscale('log')
plt.xscale('log', base=2)
plt.xticks([2 ** j for j in range(2, 7)], [str(2 ** j) for j in range(2, 7)])
plt.legend()
plt.savefig(join("..", "..", "figures", "Fig_5A.pdf"), bbox_inches='tight')
# plt.show()


fig = plt.figure(figsize=(5,3.4))

plt.plot(N_vals, pauli_basis_two_qubit_gates, '-s', linewidth=1, color="red", label="Std binary (Pauli basis)")
plt.plot(N_vals, one_hot_two_qubit_gates, '-s', linewidth=1, color="green", label="One-hot")
plt.plot(N_vals, unary_two_qubit_gates, '-s', linewidth=1, color="blue", label="Circulant unary")
plt.ylabel("Two-qubit gates")
plt.xlabel(rf"$N$ (number of grid points per dimension)")
plt.yscale('log')
plt.xscale('log', base=2)
plt.xticks([2 ** j for j in range(2, 7)], [str(2 ** j) for j in range(2, 7)])
plt.legend()
plt.savefig(join("..", "..", "figures", "Fig_5B.pdf"), bbox_inches='tight')