import numpy as np
from scipy.sparse import eye, lil_matrix
from resource_estimate_utils import *
from os.path import join
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate, RZGate
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk

from os.path import join, dirname
from utils import *

def get_binary_resource_estimate(N, T, dimension, pauli_op_P_list, error_tol, trotter_method, num_samples, num_jobs):
    
    A_1d = np.zeros((N,N), dtype=np.complex128)
    for i in range(N):
        A_1d[i,(i+1)%N] = -1j
        A_1d[(i+1)%N,i] = 1j

    A_1d_padded = np.pad(A_1d, (0, 2 ** int(np.ceil(np.log2(N))) - N))

    pauli_op_A_list = SparsePauliOp.from_operator(A_1d_padded).to_list()

    pauli_op_list = []
    for i in range(len(pauli_op_A_list)):
        for j in range(len(pauli_op_P_list)):
            pauli_op_list.append((pauli_op_A_list[i][0] + pauli_op_P_list[j][0], pauli_op_A_list[i][1] * pauli_op_P_list[j][1]))
    pauli_op = SparsePauliOp.from_list(pauli_op_list)

    # Compute number of gates per Trotter step
    if trotter_method == "first_order" or trotter_method == "randomized_first_order":
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    elif trotter_method == "second_order":
        circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    else:
        raise ValueError(f"{trotter_method} not supported")
    
    compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
    tket_circuit = qiskit_to_tk(compiled_circuit)
    gateset = {OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}
    rebase = auto_rebase_pass(gateset) 
    comp = SequencePass([FullPeepholeOptimise(), CommuteThroughMultis(), RemoveRedundancies(), rebase])
    comp.apply(tket_circuit)

    # Gates per Trotter step
    num_single_qubit_gates, num_two_qubit_gates = tket_circuit.n_1qb_gates(), tket_circuit.n_2qb_gates()
    print(f"1q gates: {num_single_qubit_gates}, 2q gates: {num_two_qubit_gates}")

    # Estimate number of Trotter steps required
    r_min, r_max = 1, 10
    while r_max * std_bin_trotter_error_sampling(pauli_op.to_matrix(sparse=True), pauli_op, T / r_max, 1, trotter_method, num_samples, num_jobs) > error_tol / dimension:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if r * std_bin_trotter_error_sampling(pauli_op.to_matrix(sparse=True), pauli_op, T / r, 1, trotter_method, num_samples, num_jobs) > error_tol / dimension:
            r_min = r
        else:
            r_max = r
    
    print(f"Finished N={N}, num two qubit gates={num_two_qubit_gates}, trotter steps={r_max}", flush=True)
    return dimension * num_single_qubit_gates, dimension * num_two_qubit_gates, r_max

def estimate_trotter_error_1d_adv(N, T, r):
    A_1d_even = np.zeros((N,N), dtype=np.complex128)
    A_1d_odd = np.zeros((N,N), dtype=np.complex128)
    for i in range(N - 1):
        if i % 2 == 0:
            A_1d_even[i,(i+1)%N] = -1j
            A_1d_even[(i+1)%N,i] = 1j
        else:
            A_1d_odd[i,(i+1)%N] = -1j
            A_1d_odd[(i+1)%N,i] = 1j
    if N % 2 == 0:
        A_1d_odd[0,-1] = 1j
        A_1d_odd[-1,0] = -1j
    else:
        A_1d_extra_edge = np.zeros((N,N), dtype=np.complex128)
        A_1d_extra_edge[0,-1] = 1j
        A_1d_extra_edge[-1,0] = -1j


    if N % 2 == 0:
        U = expm(-1j * T * (A_1d_even + A_1d_odd))

        U_trotter = np.eye(N)
        for i in range(r):
            U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_even, U_trotter)
            U_trotter = expm_multiply(-1j * (T / r) * A_1d_odd, U_trotter)
            U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_even, U_trotter)
    else:
        U = expm(-1j * T * (A_1d_even + A_1d_odd + A_1d_extra_edge))

        U_trotter = np.eye(N)
        for i in range(r):
            U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_extra_edge, U_trotter)
            U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_even, U_trotter)
            U_trotter = expm_multiply(-1j * (T / r) * A_1d_odd, U_trotter)
            U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_even, U_trotter)
            U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_extra_edge, U_trotter)

    error = np.linalg.norm(U - U_trotter, ord=2)
    return error

def get_trotter_number(N, T, error_tol):
    r_min, r_max = 1, 10
    while r_max * estimate_trotter_error_1d_adv(N, T / r_max, 1) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if r * estimate_trotter_error_1d_adv(N, T / r, 1) > error_tol:
            r_min = r
        else:
            r_max = r
    return r_max

if __name__ == "__main__":

    DATA_DIR = join(dirname(__file__), "..", "data")
    TASK_DIR = "2d_burgers"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for 2d Burgers' equation.")

    num_jobs = 64
    print("Number of jobs:", num_jobs)
    num_samples = 1000

    n_p = 8
    N_p = 2 ** n_p
    error_tol = 5e-2
    trotter_method = "second_order"
    dimension = 2

    print(f"Error tolerance: {error_tol : 0.4f}.")
    print(f"Method: {trotter_method}")

    N_vals_binary = np.arange(3, 129)
    binary_trotter_steps = np.zeros(len(N_vals_binary), dtype=int)
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary), dtype=int)
    binary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary), dtype=int)

    N_vals_one_hot = np.arange(3, 129)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)

    N_vals_unary = np.arange(4, 129, 2)
    unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    unary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary), dtype=int)


    pauli_op_P_list = []
    for j in range(n_p-1):
        op = n_p * ['I']
        op[n_p-1-j] = 'Z'
        pauli_op_P_list.append((''.join(op), -2 ** (j-1) / N_p))
    pauli_op_P_list.append((''.join(n_p * ['I']), (2 ** (n_p-1) - 0.5) / N_p)) 

    print("\nRunning resource estimation for standard binary encoding")
    for i, N in enumerate(N_vals_binary):
        T = 0.2 * N
        start_time = time()
        print(f"N = {N}")
        binary_one_qubit_gate_count_per_trotter_step[i], binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(N, T, dimension, pauli_op_P_list, error_tol, trotter_method, num_samples, num_jobs)

        np.savez(join(CURR_DIR, f"std_binary_{trotter_method}.npz"),
                N_vals_binary=N_vals_binary[:i+1],
                binary_trotter_steps=binary_trotter_steps[:i+1],
                binary_one_qubit_gate_count_per_trotter_step=binary_one_qubit_gate_count_per_trotter_step[:i+1],
                binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step[:i+1])
        
        print(f"Time = {time() - start_time} seconds.", flush=True)

    # One hot encoding
    print("\nRunning resource estimation for one-hot encoding", flush=True)
    encoding = "one-hot"

    for i, N in enumerate(N_vals_one_hot):
        T = 0.2 * N
        start_time = time()

        pauli_op_A_list = []
        for j in range(N):
            op = N * ['I']
            op[j] = 'X'
            op[(j+1)%N] = 'Y'
            pauli_op_A_list.append((''.join(op), 1/2))
            op = N * ['I']
            op[j] = 'Y'
            op[(j+1)%N] = 'X'
            pauli_op_A_list.append((''.join(op), -1/2))

        pauli_op_list = []
        for i in range(len(pauli_op_A_list)):
            for j in range(len(pauli_op_P_list)):
                pauli_op_list.append((pauli_op_A_list[i][0] + pauli_op_P_list[j][0], pauli_op_A_list[i][1] * pauli_op_P_list[j][1]))
        pauli_op = SparsePauliOp.from_list(pauli_op_list)

        # Compute number of gates per Trotter step
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
        elif trotter_method == "second_order":
            circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
        else:
            raise ValueError(f"{trotter_method} not supported")

        compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
        tket_circuit = qiskit_to_tk(compiled_circuit)
        gateset = {OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}
        rebase = auto_rebase_pass(gateset) 
        comp = SequencePass([FullPeepholeOptimise(), CommuteThroughMultis(), RemoveRedundancies(), rebase])
        comp.apply(tket_circuit)

        # Gates per Trotter step
        num_single_qubit_gates, num_two_qubit_gates = tket_circuit.n_1qb_gates(), tket_circuit.n_2qb_gates()
        print(f"1q gates: {num_single_qubit_gates}, 2q gates: {num_two_qubit_gates}")

        # Computes the Trotter error per dimension, so total error will be error_tol
        one_hot_trotter_steps[i] = get_trotter_number(N, T, error_tol / dimension)
        one_hot_one_qubit_gate_count_per_trotter_step[i], one_hot_two_qubit_gate_count_per_trotter_step[i] = dimension * num_single_qubit_gates, dimension * num_two_qubit_gates

        # Save data
        np.savez(join(CURR_DIR, f"one_hot_{trotter_method}.npz"),
                 N_vals_one_hot=N_vals_one_hot[:i+1],
                 one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                 one_hot_one_qubit_gate_count_per_trotter_step=one_hot_one_qubit_gate_count_per_trotter_step[:i+1],
                 one_hot_two_qubit_gate_count_per_trotter_step=one_hot_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

    # Unary encoding
    print("\nRunning resource estimation for unary encoding", flush=True)
    encoding = "unary"

    for i, N in enumerate(N_vals_unary):
        assert N % 2 == 0
        T = 0.2 * N
        start_time = time()

        pauli_op_A_list = []
        n = N // 2
        for j in range(N):

            if 1 <= j <= N // 2:
                a = 1
            else:
                a = 0

            if n - 1 <= j < N - 1:
                b = 1
            else:
                b = 0
                    
            if j >= n:
                c = 1
            else:
                c = 0

            op = n * ['I']
            op[j%n] = 'Y'
            pauli_op_A_list.append((''.join(op), (-1) ** c /4))

            op = n * ['I']
            op[j%n] = 'Y'
            op[(j-1)%n] = 'Z'
            pauli_op_A_list.append((''.join(op), - (-1) ** (a+c) /4))

            op = n * ['I']
            op[j%n] = 'Y'
            op[(j+1)%n] = 'Z'
            pauli_op_A_list.append((''.join(op), - (-1) ** (b+c) /4))

            op = n * ['I']
            op[(j-1)%n] = 'Z'
            op[j%n] = 'Y'
            op[(j+1)%n] = 'Z'
            pauli_op_A_list.append((''.join(op), (-1) ** (a+b+c) /4))

        pauli_op_list = []
        for i in range(len(pauli_op_A_list)):
            for j in range(len(pauli_op_P_list)):
                pauli_op_list.append((pauli_op_A_list[i][0] + pauli_op_P_list[j][0], pauli_op_A_list[i][1] * pauli_op_P_list[j][1]))
        pauli_op = SparsePauliOp.from_list(pauli_op_list)

        # Compute number of gates per Trotter step
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
        elif trotter_method == "second_order":
            circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
        else:
            raise ValueError(f"{trotter_method} not supported")

        compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
        tket_circuit = qiskit_to_tk(compiled_circuit)
        gateset = {OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}
        rebase = auto_rebase_pass(gateset) 
        comp = SequencePass([FullPeepholeOptimise(), CommuteThroughMultis(), RemoveRedundancies(), rebase])
        comp.apply(tket_circuit)

        # Gates per Trotter step
        num_single_qubit_gates, num_two_qubit_gates = tket_circuit.n_1qb_gates(), tket_circuit.n_2qb_gates()
        print(f"1q gates: {num_single_qubit_gates}, 2q gates: {num_two_qubit_gates}")

        # Computes the Trotter error per dimension, so total error will be error_tol
        unary_trotter_steps[i] = get_trotter_number(N, T, error_tol / dimension)
        unary_one_qubit_gate_count_per_trotter_step[i], unary_two_qubit_gate_count_per_trotter_step[i] = dimension * num_single_qubit_gates, dimension * num_two_qubit_gates

        # Save data
        np.savez(join(CURR_DIR, f"unary_{trotter_method}.npz"),
                 N_vals_unary=N_vals_unary[:i+1],
                 unary_trotter_steps=unary_trotter_steps[:i+1],
                 unary_one_qubit_gate_count_per_trotter_step=unary_one_qubit_gate_count_per_trotter_step[:i+1],
                 unary_two_qubit_gate_count_per_trotter_step=unary_two_qubit_gate_count_per_trotter_step[:i+1])

        print(f"Finished N = {N}, time = {time() - start_time} seconds.", flush=True)

