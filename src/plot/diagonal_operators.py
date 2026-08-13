import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

if __name__ == "__main__":
    # Script for generating Figure 5 in the Appendix, showing one-hot vs standard binary for simulating diagonal operators.

    n_vals = np.arange(2, 14)
    k_vals = np.arange(1, 7)
    std_binary_gate_counts = np.zeros((len(n_vals), len(k_vals)))
    T = 1

    for j1, n in enumerate(n_vals):
        N = 2**n
        print(f"N={N}")

        pauli_op_list = []
        for i in range(n):
            op = n * ["I"]
            op[n - 1 - i] = "Z"
            pauli_op_list.append(("".join(op), -(2 ** (i))))

        op = n * ["I"]
        op[0] = "Z"
        pauli_op_list.append(("".join(n * ["I"]), (N - 1)))
        P = SparsePauliOp.from_list(pauli_op_list)

        for j2, k in enumerate(k_vals):
            print(f"k = {k} / {k_vals[-1]}", end="\r")
            pauli_op = ((P / (2 * N)) ** k).simplify()

            circuit = LieTrotter(reps=1).synthesize(
                PauliEvolutionGate((T * pauli_op).group_commuting())
            )
            compiled_circuit = transpile(
                circuit, basis_gates=["rxx", "rx", "ry", "rz"], optimization_level=3
            )
            ops = compiled_circuit.count_ops()
            num_single_qubit_gates = 0
            num_two_qubit_gates = 0
            for op in ops:
                if op == "rx" or op == "ry" or op == "rz":
                    num_single_qubit_gates += ops[op]
                elif op == "rxx":
                    num_two_qubit_gates += ops[op]

            std_binary_gate_counts[j1, j2] = (
                num_single_qubit_gates + num_two_qubit_gates
            )
        print(f"k = {k} / {k_vals[-1]}")

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
        }
    )

    one_hot_gate_counts = np.outer(2**n_vals, np.ones(len(k_vals)))

    plt.matshow((std_binary_gate_counts < one_hot_gate_counts).T, cmap="Set3")
    plt.gca().invert_yaxis()
    plt.xlabel(r"System size $N$", fontsize=14)
    plt.ylabel(r"Polynomial degree $K$", fontsize=14)
    plt.yticks(np.arange(len(k_vals)), k_vals)
    plt.xticks(np.arange(len(n_vals)), 2**n_vals)
    plt.gca().tick_params(axis="x", labelbottom=True, labeltop=False, top=False)
    plt.gca().tick_params(axis="x", labelbottom=True, labeltop=False)
    plt.text(x=6, y=-0.15, s="Standard binary is better", fontsize=18)
    plt.text(x=3, y=3.4, s="One-hot is better", fontsize=18)
    plt.savefig("../../figures/diagonal_operators.pdf", bbox_inches="tight")
