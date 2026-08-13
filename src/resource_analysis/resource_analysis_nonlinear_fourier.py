import sys
from os.path import join
from time import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from scipy.sparse import lil_matrix

from resource_estimate_utils import *

sys.path.append(join(".", ".."))
import resource
import warnings

from utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)


def parallelize_ctrl_circuit(circuit):
    # Given an n-qubit circuit, perform a controlled version of the circuit using n+1 ancillas:.
    # For each layer of the circuit, perform the controlled version in parallel.
    # Output: the depth and two qubit gate count
    overall_depth = 0
    two_qubit_gate_count = 0
    dag = circuit_to_dag(circuit)

    for layer in dag.layers():
        layer_as_circuit = dag_to_circuit(layer["graph"])
        # In between the fan-out gates, perform the controlled version of each gate
        # The overall depth only depends on the maximum depth of any controlled gate in the layer
        max_depth = 0
        for gate in layer_as_circuit.data:
            qc = QuantumCircuit.from_instructions([gate])
            ctrl_qc = transpile(
                qc, basis_gates=["rx", "ry", "rz", "rxx"], optimization_level=0
            ).control(1, ctrl_state="1")
            transpiled_qc = transpile(
                ctrl_qc, basis_gates=["rxx", "rx", "ry", "rz"], optimization_level=3
            )
            max_depth = max(max_depth, transpiled_qc.depth(lambda instr: len(instr.qubits) > 1))
            two_qubit_gate_count += transpiled_qc.num_nonlocal_gates()
        overall_depth += max_depth
    return overall_depth, two_qubit_gate_count


def get_H_std_binary(N_q):

    D_q = lil_matrix((N_q, N_q), dtype=np.complex128)
    q_vals = np.linspace(-1, 1, N_q, endpoint=False)
    for j in range(N_q):
        q = q_vals[j]
        D_q[j, j] = q**3 * (1 - q**4)

    D_q_pauli_op = SparsePauliOp.from_operator(D_q.toarray())
    return D_q_pauli_op


def get_H_one_hot(N_q):

    D_q = lil_matrix((N_q, N_q), dtype=np.complex128)
    q_vals = np.linspace(-1, 1, N_q, endpoint=False)
    for j in range(N_q):
        q = q_vals[j]
        D_q[j, j] = q**3 * (1 - q**4)

    D_q_pauli_list = []
    for j in range(N_q):
        D_q_pauli_list.append((N_q * "I", 0.5 * D_q[j, j].real))
        op = N_q * ["I"]
        op[j] = "Z"
        D_q_pauli_list.append(("".join(op), -0.5 * D_q[j, j].real))
    D_q_pauli_op = SparsePauliOp.from_list(D_q_pauli_list).simplify()

    return D_q_pauli_op


def get_H_unary(N_q):
    n_q = N_q - 1

    D_q = lil_matrix((N_q, N_q), dtype=np.complex128)
    q_vals = np.linspace(-1, 1, N_q, endpoint=False)
    for j in range(N_q):
        q = q_vals[j]
        D_q[j, j] = q**3 * (1 - q**4)

    D_q_pauli_list = []
    op = n_q * ["I"]
    D_q_pauli_list.append(("".join(op), D_q[0, 0].real))
    for j in range(1, N_q):
        D_q_pauli_list.append((n_q * "I", 0.5 * (D_q[j, j] - D_q[j - 1, j - 1])))
        op = n_q * ["I"]
        op[j - 1] = "Z"
        D_q_pauli_list.append(("".join(op), -0.5 * (D_q[j, j] - D_q[j - 1, j - 1])))

    D_q_pauli_op = SparsePauliOp.from_list(D_q_pauli_list).simplify()

    return D_q_pauli_op


def get_resource_estimate(n_x, n_q, H):
    depth = 0
    two_qubit_gate_count = 0
    # N_x = 2 ** n_x
    # D_x = np.diag(2 * np.pi * np.fft.fftfreq(n=N_x) * N_x)
    # D_x_pauli_op = SparsePauliOp.from_operator(D_x)

    circuit = QuantumCircuit(n_q)
    circuit.append(
        LieTrotter(reps=1).synthesize(PauliEvolutionGate(H)),
        qargs=np.arange(n_q).tolist(),
    )
    compiled_circuit = transpile(
        circuit, basis_gates=["rxx", "rx", "ry", "rz"], optimization_level=3
    )
    # Non-controlled version
    depth += compiled_circuit.depth(lambda instr: len(instr.qubits) > 1)
    two_qubit_gate_count += compiled_circuit.num_nonlocal_gates()

    # Do a controlled version
    depth_tmp, two_qubit_gate_count_tmp = parallelize_ctrl_circuit(compiled_circuit)
    # Add the cost of fan-out
    fanout_circ = QuantumCircuit(n_q)
    for i in range(n_q - 1):
        fanout_circ.cx(i, i + 1)
    fanout_compiled_circ = transpile(
        fanout_circ, basis_gates=["rxx", "rx", "ry", "rz"], optimization_level=3
    )
    depth_tmp += 2 * fanout_compiled_circ.depth(lambda instr: len(instr.qubits) > 1)
    two_qubit_gate_count_tmp += 2 * fanout_compiled_circ.num_nonlocal_gates()

    # The Pauli decomposition of Fourier frequencies involves n_x + 1 terms, one of which is the identity.
    # The n_x non-identity terms are Pauli-Z, which are basically controls (controlling on 0 and 1).
    # Thus, we multiply by a factor of (2 * n_x)
    # The other factor of 2 is since we consider two spatial variables.
    depth += 2 * (2 * n_x) * depth_tmp
    two_qubit_gate_count += 2 * (2 * n_x) * two_qubit_gate_count_tmp

    return depth, two_qubit_gate_count


def qft_cost(N, encoding):
    if encoding == "std_binary":
        transpiled_qc = transpile(
            QFT(num_qubits=int(np.log(N))),
            basis_gates=["rx", "ry", "rz", "rxx"],
            optimization_level=0,
        )
        depth = transpiled_qc.depth(lambda instr: len(instr.qubits) > 1)
        two_qubit_gate_count = transpiled_qc.num_nonlocal_gates()
        # Multiply gate count by two since we consider a 2D problem
        return depth, 2 * two_qubit_gate_count

    elif encoding == "one_hot":
        circ = one_hot_qft(N, theta=2 * np.pi / N)

        transpiled_qc = transpile(
            circ, basis_gates=["rx", "ry", "rz", "rxx"], optimization_level=0
        )
        depth = transpiled_qc.depth(lambda instr: len(instr.qubits) > 1)
        two_qubit_gate_count = transpiled_qc.num_nonlocal_gates()
        # Multiply gate count by two since we consider a 2D problem
        return depth, 2 * two_qubit_gate_count

    elif encoding == "unary":
        circ = QuantumCircuit(N)
        for j in range(N - 1):
            circ.cx(j + 1, j)

        circ.append(one_hot_qft(N, theta=2 * np.pi / N), qargs=np.arange(N).tolist())

        for j in range(N - 1)[::-1]:
            circ.cx(j + 1, j)

        transpiled_qc = transpile(
            circ, basis_gates=["rx", "ry", "rz", "rxx"], optimization_level=0
        )
        depth = transpiled_qc.depth(lambda instr: len(instr.qubits) > 1)
        two_qubit_gate_count = transpiled_qc.num_nonlocal_gates()
        # Multiply gate count by two since we consider a 2D problem
        return depth, 2 * two_qubit_gate_count


def one_hot_qft(N, theta):

    circ = QuantumCircuit(N)

    if N == 2:
        circ.rz(-theta, 0)
    else:
        circ.append(one_hot_qft(N // 2, theta), qargs=np.arange(0, N // 2).tolist())
        circ.append(one_hot_qft(N // 2, theta), qargs=np.arange(N // 2, N).tolist())

    for j in range(N // 2):
        circ.rz(-theta * j, j)
    for j in range(N // 2):
        circ.rxx(-np.pi / 4, j, j + N // 2)
        circ.ryy(np.pi / 4, j, j + N // 2)

    return circ


if __name__ == "__main__":
    print("Running resource analysis for nonlinear PDE", flush=True)
    start_time = time()
    n_x = 7
    N_x = 2**n_x
    N_q_vals = 2 ** np.arange(3, 9)

    pauli_basis_two_qubit_gates = np.zeros_like(N_q_vals)
    pauli_basis_circ_depth = np.zeros_like(N_q_vals)

    one_hot_two_qubit_gates = np.zeros_like(N_q_vals)
    one_hot_circ_depth = np.zeros_like(N_q_vals)

    unary_two_qubit_gates = np.zeros_like(N_q_vals)
    unary_circ_depth = np.zeros_like(N_q_vals)

    for i, N_q in enumerate(N_q_vals):
        print(f"Running resource analysis for N_q={N_q}.")
        """Standard binary"""
        H_std_binary = get_H_std_binary(N_q)
        print("Computing gates for Pauli basis.", flush=True)
        pauli_basis_circ_depth[i], pauli_basis_two_qubit_gates[i] = (
            get_resource_estimate(n_x, int(np.log2(N_q)), H_std_binary)
        )
        qft_depth, qft_gates = qft_cost(N_x, encoding="std_binary")
        pauli_basis_circ_depth[i] += 2 * qft_depth
        pauli_basis_two_qubit_gates[i] += 2 * qft_gates

        """One-hot encoding (ours)"""
        H_one_hot = get_H_one_hot(N_q)
        print("Computing gates for one-hot.", flush=True)
        one_hot_circ_depth[i], one_hot_two_qubit_gates[i] = get_resource_estimate(
            n_x, N_q, H_one_hot
        )
        qft_depth, qft_gates = qft_cost(N_x, encoding="one_hot")
        one_hot_circ_depth[i] += 2 * qft_depth
        one_hot_two_qubit_gates[i] += 2 * qft_gates

        """Unary encoding (ours)"""
        H_unary = get_H_unary(N_q)
        print("Computing gates for unary.", flush=True)
        unary_circ_depth[i], unary_two_qubit_gates[i] = get_resource_estimate(
            n_x, N_q - 1, H_unary
        )
        qft_depth, qft_gates = qft_cost(N_x, encoding="unary")
        unary_circ_depth[i] += 2 * qft_depth
        unary_two_qubit_gates[i] += 2 * qft_gates

        np.savez(
            join(
                "../resource_analysis_data", "nonlinear", "nonlinear_fourier_data.npz"
            ),
            N_q_vals=N_q_vals[: i + 1],
            pauli_basis_two_qubit_gates=pauli_basis_two_qubit_gates[: i + 1],
            pauli_basis_circ_depth=pauli_basis_circ_depth[: i + 1],
            one_hot_two_qubit_gates=one_hot_two_qubit_gates[: i + 1],
            one_hot_circ_depth=one_hot_circ_depth[: i + 1],
            unary_two_qubit_gates=unary_two_qubit_gates[: i + 1],
            unary_circ_depth=unary_circ_depth[: i + 1],
        )
        print(f"Finished N_q={N_q}.")
        print()

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)
    print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
