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
data = np.load(f"../resource_analysis_data/nonlinear/nonlinear_fourier_data.npz")
N_q_vals = data["N_q_vals"]

pauli_basis_circ_depth = data["pauli_basis_circ_depth"]
pauli_basis_two_qubit_gates = data["pauli_basis_two_qubit_gates"]

one_hot_circ_depth = data["one_hot_circ_depth"]
one_hot_two_qubit_gates = data["one_hot_two_qubit_gates"]

unary_circ_depth = data["unary_circ_depth"]
unary_two_qubit_gates = data["unary_two_qubit_gates"]

fig = plt.figure(figsize=(5.2,3.5))

plt.plot(N_q_vals, pauli_basis_circ_depth, '-s', linewidth=1, color="red", label="Std binary (Pauli basis)")
plt.plot(N_q_vals, one_hot_circ_depth, '-s', linewidth=1, color="green", label="One-hot")
plt.plot(N_q_vals, unary_circ_depth, '-s', linewidth=1, color="blue", label="Unary")
plt.ylabel("Circuit depth")
plt.xlabel(rf"$N_q$ (number of grid points per dimension)")
plt.yscale('log')
plt.xscale('log', base=2)
plt.xticks(N_q_vals, [str(j) for j in N_q_vals])
plt.legend()
plt.savefig(join("..", "..", "figures", "Fig_4B.pdf"), bbox_inches='tight')
# plt.show()


fig = plt.figure(figsize=(5.2,3.5))

plt.plot(N_q_vals, pauli_basis_two_qubit_gates, '-s', linewidth=1, color="red", label="Std binary (Pauli basis)")
plt.plot(N_q_vals, one_hot_two_qubit_gates, '-s', linewidth=1, color="green", label="One-hot")
plt.plot(N_q_vals, unary_two_qubit_gates, '-s', linewidth=1, color="blue", label="Unary")
plt.ylabel("Two-qubit gates")
plt.xlabel(rf"$N_q$ (number of grid points per dimension)")
plt.yscale('log')
plt.xscale('log', base=2)
plt.xticks(N_q_vals, [str(j) for j in N_q_vals])
plt.legend(loc="upper left")
plt.legend()
plt.savefig(join("..", "..", "figures", "Fig_4C.pdf"), bbox_inches='tight')