import numpy as np
from scipy.sparse import eye, lil_matrix
from resource_estimate_utils import *
from os.path import join
from time import time
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk
from braket.devices import LocalSimulator

from os.path import join, dirname
from utils import *

def get_binary_resource_estimate(N, T, dimension, error_tol, trotter_method, num_samples, num_jobs):
    
    A_1d = np.zeros((N,N), dtype=np.complex128)
    for i in range(N):
        A_1d[i,(i+1)%N] = -1j
        A_1d[(i+1)%N,i] = 1j

    A_1d_padded = np.pad(A_1d, (0, 2 ** int(np.ceil(np.log2(N))) - N))

    # print(A_1d_padded)
    # print(SparsePauliOp.from_operator(A_1d_padded))

    pauli_op_1d_list = SparsePauliOp.from_operator(A_1d_padded).to_list()
    pauli_op_1d = SparsePauliOp.from_list(pauli_op_1d_list)
    # pauli_op_2d_list = []
    # for i in range(len(pauli_op_1d_list)):
    #     n = int(np.ceil(np.log2(N)))
    #     op = pauli_op_1d_list[i]
    #     pauli_op_2d_list.append((op[0] + n * 'I', op[1]))
    #     pauli_op_2d_list.append((n * 'I' + op[0], op[1]))
    # # print(pauli_op_2d_list)
    # pauli_op_2d = SparsePauliOp.from_list(pauli_op_2d_list)
    # print(pauli_op_2d)

    # Compute number of gates per Trotter step
    if trotter_method == "first_order" or trotter_method == "randomized_first_order":
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op_1d.group_commuting()))
    elif trotter_method == "second_order":
        circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op_1d.group_commuting()))
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
    while std_bin_trotter_error_sampling(pauli_op_1d.to_matrix(), pauli_op_1d, T, r_max, trotter_method, num_samples, num_jobs) > error_tol / dimension:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if std_bin_trotter_error_sampling(pauli_op_1d.to_matrix(), pauli_op_1d, T, r, trotter_method, num_samples, num_jobs) > error_tol / dimension:
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

    U = expm(-1j * T * (A_1d_even + A_1d_odd))

    U_trotter = np.eye(N)
    for i in range(r):
        U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_even, U_trotter)
        U_trotter = expm_multiply(-1j * (T / r) * A_1d_odd, U_trotter)
        U_trotter = expm_multiply(-1j * (T / (2 * r)) * A_1d_even, U_trotter)

    error = np.linalg.norm(U - U_trotter, ord=2)
    return error

def get_trotter_number(N, T, error_tol):
    r_min, r_max = 1, 10
    while estimate_trotter_error_1d_adv(N, T, r_max) > error_tol:
        r_max *= 2

    # binary search for r
    while r_max - r_min > 1:
        r = (r_min + r_max) // 2
        if estimate_trotter_error_1d_adv(N, T, r) > error_tol:
            r_min = r
        else:
            r_max = r
    return r_max

# Std binary implementation using Bell basis
def get_w_circ(j, n, lamb, T):
    
    circuit = QuantumCircuit(n)
    if j == 0:
        circuit.p(-lamb, j)
        circuit.rx(2 * T, j)
        circuit.p(lamb, j)
        return circuit
    for i in range(j):
        circuit.cx(control_qubit=j, target_qubit=i)
    circuit.p(-lamb, j)
    circuit.h(j)
    circuit.x(j)
    circuit.mcp(-T, control_qubits=np.arange(0, j).tolist(), target_qubit=j)
    circuit.x(j)
    circuit.mcp(T, control_qubits=np.arange(0, j).tolist(), target_qubit=j)
    circuit.h(j)
    circuit.p(lamb, j)
    for i in range(j):
        circuit.cx(control_qubit=j, target_qubit=j-1-i)
    return circuit

def get_v_circ(n, lamb, T, periodic=True):
    circuit = QuantumCircuit(n)
    
    # Second order Trotter
    for j in range(n):
        circuit.append(get_w_circ(j, n, lamb, T / 2), qargs=np.arange(n).tolist())

    if periodic:
        for j in range(n-1):
            circuit.cx(control_qubit=n-1, target_qubit=j)
        circuit.p(lamb, n-1)
        circuit.h(n-1)
        for j in range(n-1):
            circuit.x(j)
        circuit.x(n-1)
        circuit.mcp(-T, control_qubits=np.arange(0, n-1).tolist(), target_qubit=n-1)
        circuit.x(n-1)
        circuit.mcp(T, control_qubits=np.arange(0, n-1).tolist(), target_qubit=n-1)
        for j in range(n-1):
            circuit.x(j)
        circuit.h(n-1)
        circuit.p(-lamb, n-1)
        for j in range(n-1):
            circuit.cx(control_qubit=n-1, target_qubit=n-2-j)
    for j in range(n):
        circuit.append(get_w_circ(n-1-j, n, lamb, T / 2), qargs=np.arange(n).tolist())
    return circuit

def get_trotterized_circ_bell_basis(n, lamb, T, r, periodic=True):
    circuit = QuantumCircuit(n)
    for _ in range(r):
        circuit.append(get_v_circ(n, lamb, T/r, periodic), qargs=np.arange(n).tolist())
    return circuit

def get_bell_basis_trotter_error(n, lamb, T, r, U_A, periodic=True):
    circuit = get_trotterized_circ_bell_basis(n, lamb, T, r, periodic)
    get_trotterized_circ_bell_basis(n, lamb, T, r, periodic)
    U_B = np.array(Operator(circuit))
    error = np.linalg.norm(U_A - U_B, ord=2)
    error_bound = (T ** 2) * (n-1) / 2
    assert error <= error_bound + 1e-8, "Error larger than error bound?"
    return error

if __name__ == "__main__":

    DATA_DIR = join(dirname(__file__), "..", "data")
    TASK_DIR = "2d_advection"

    CURR_DIR = DATA_DIR
    check_and_make_dir(CURR_DIR)
    CURR_DIR = join(CURR_DIR, TASK_DIR)
    check_and_make_dir(CURR_DIR)
    
    print("Resource estimation for 2d advection equation.")

    num_jobs = 64
    print("Number of jobs:", num_jobs)
    num_samples = 1000

    error_tol = 1e-3
    trotter_method = "second_order"
    dimension = 2

    print(f"Error tolerance: {error_tol : 0.4f}.")
    print(f"Method: {trotter_method}")

    N_vals_binary = np.arange(3, 128)
    binary_trotter_steps = np.zeros(len(N_vals_binary), dtype=int)
    binary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary), dtype=int)
    binary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_binary), dtype=int)

    n_vals_bell_basis = np.arange(2, 8)
    N_vals_bell_basis = 2 ** n_vals_bell_basis
    bell_basis_trotter_steps = np.zeros(len(N_vals_bell_basis), dtype=int)
    bell_basis_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_bell_basis), dtype=int)
    bell_basis_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_bell_basis), dtype=int)

    N_vals_one_hot = np.arange(3, 128)
    one_hot_trotter_steps = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)
    one_hot_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_one_hot), dtype=int)

    N_vals_unary = np.arange(3, 128)
    unary_trotter_steps = np.zeros(len(N_vals_unary), dtype=int)
    unary_one_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary), dtype=int)
    unary_two_qubit_gate_count_per_trotter_step = np.zeros(len(N_vals_unary), dtype=int)


    # print("\nRunning resource estimation for standard binary encoding")
    # for i, N in enumerate(N_vals_binary):
    #     T = N
    #     start_time = time()
    #     print(f"N = {N}")
    #     binary_one_qubit_gate_count_per_trotter_step[i], binary_two_qubit_gate_count_per_trotter_step[i], binary_trotter_steps[i] = get_binary_resource_estimate(N, T, dimension, error_tol, trotter_method, num_samples, num_jobs)

    #     np.savez(join(CURR_DIR, f"std_binary_{trotter_method}.npz"),
    #             N_vals_binary=N_vals_binary[:i+1],
    #             binary_trotter_steps=binary_trotter_steps[:i+1],
    #             binary_one_qubit_gate_count_per_trotter_step=binary_one_qubit_gate_count_per_trotter_step[:i+1],
    #             binary_two_qubit_gate_count_per_trotter_step=binary_two_qubit_gate_count_per_trotter_step[:i+1])
        
    #     print(f"Time = {time() - start_time} seconds.", flush=True)


    print("\nRunning resource estimation for standard binary encoding w/ Bell basis")
    for i, n in enumerate(n_vals_bell_basis):
        N = N_vals_bell_basis[i]
        T = N
        start_time = time()
        print(f"N = {N}")

        lamb = np.pi / 2
        periodic = True
        A = np.zeros((N,N), dtype=np.complex128)
        for j in range(N-1):
            A[j,(j+1)%N] = np.exp(-1j * lamb)
            A[(j+1)%N,j] = np.exp(1j * lamb)
        if periodic:
            A[N-1,0] = np.exp(-1j * lamb)
            A[0,N-1] = np.exp(1j * lamb)

        # Ground truth
        U_A = expm(-1j * A * T)

        # Binary search to find Trotter number

        r_min, r_max = 1, 10
        while get_bell_basis_trotter_error(n, lamb, T, r_max, U_A, periodic=True) > error_tol / dimension:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if get_bell_basis_trotter_error(n, lamb, T, r, U_A, periodic=True) > error_tol / dimension:
                r_min = r
            else:
                r_max = r

        print("Trotter steps:", r_max)
        # Get the circuit (for 1D)
        circuit = get_trotterized_circ_bell_basis(n, lamb, T, 1, periodic)
        compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
        tket_circuit = qiskit_to_tk(compiled_circuit)
        gateset = {OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}
        rebase = auto_rebase_pass(gateset) 
        comp = SequencePass([FullPeepholeOptimise(), CommuteThroughMultis(), RemoveRedundancies(), rebase])
        comp.apply(tket_circuit)

        # Gates per Trotter step
        num_single_qubit_gates, num_two_qubit_gates = tket_circuit.n_1qb_gates(), tket_circuit.n_2qb_gates()
        print(f"1q gates: {num_single_qubit_gates}, 2q gates: {num_two_qubit_gates}")

        bell_basis_trotter_steps[i] = r_max
        bell_basis_one_qubit_gate_count_per_trotter_step[i] = dimension * num_single_qubit_gates
        bell_basis_two_qubit_gate_count_per_trotter_step[i] = dimension * num_two_qubit_gates

        np.savez(join(CURR_DIR, f"bell_basis_{trotter_method}.npz"),
                n_vals_bell_basis=n_vals_bell_basis[:i+1],
                N_vals_bell_basis=N_vals_bell_basis[:i+1],
                bell_basis_trotter_steps=bell_basis_trotter_steps[:i+1],
                bell_basis_one_qubit_gate_count_per_trotter_step=bell_basis_one_qubit_gate_count_per_trotter_step[:i+1],
                bell_basis_two_qubit_gate_count_per_trotter_step=bell_basis_two_qubit_gate_count_per_trotter_step[:i+1])
        print(f"Time = {time() - start_time} seconds.", flush=True)


    # One hot encoding
    print("\nRunning resource estimation for one-hot encoding", flush=True)
    encoding = "one-hot"
    device = LocalSimulator()

    for i, N in enumerate(N_vals_one_hot):
        T = N
        start_time = time()

        pauli_op_1d_list = []
        for j in range(N):
            op = N * ['I']
            op[j] = 'X'
            op[(j+1)%N] = 'Y'
            pauli_op_1d_list.append((''.join(op), 1/2))
            op = N * ['I']
            op[j] = 'Y'
            op[(j+1)%N] = 'X'
            pauli_op_1d_list.append((''.join(op), -1/2))

        pauli_op_1d = SparsePauliOp.from_list(pauli_op_1d_list)
        # pauli_op_2d_list = []
        # for j in range(len(pauli_op_1d_list)):
        #     op = pauli_op_1d_list[j]
        #     pauli_op_2d_list.append((op[0] + N * 'I', op[1]))
        #     pauli_op_2d_list.append((N * 'I' + op[0], op[1]))
        # # print(pauli_op_2d_list)
        # pauli_op_2d = SparsePauliOp.from_list(pauli_op_2d_list)
        # # print(pauli_op_2d)

        # Compute number of gates per Trotter step
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op_1d.group_commuting()))
        elif trotter_method == "second_order":
            circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op_1d.group_commuting()))
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
    device = LocalSimulator()

    for i, N in enumerate(N_vals_unary):
        T = N
        start_time = time()

        pauli_op_1d_list = []
        n = N // 2
        for j in range(N):

            if 1 <= j <= N // 2:
                # print(f"n_{j-1}^(1)")
                a = 1
            else:
                # print(f"n_{j-1}^(0)")
                a = 0

            # print(f"X_{j}")
            if n - 1 <= j < N - 1:
                # print(f"n_{j+1}^(1)")
                b = 1
            else:
                # print(f"n_{j+1}^(0)")
                b = 0
            
            op = n * ['I']
            op[j%n] = 'X'
            pauli_op_1d_list.append((''.join(op), 1/4))

            op = n * ['I']
            op[j%n] = 'X'
            op[(j-1)%n] = 'Z'
            pauli_op_1d_list.append((''.join(op), - (-1) ** a /4))

            op = n * ['I']
            op[j%n] = 'X'
            op[(j+1)%n] = 'Z'
            pauli_op_1d_list.append((''.join(op), - (-1) ** b /4))

            op = n * ['I']
            op[(j-1)%n] = 'Z'
            op[j%n] = 'X'
            op[(j+1)%n] = 'Z'
            pauli_op_1d_list.append((''.join(op), (-1) ** (a+b) /4))

        pauli_op_1d = SparsePauliOp.from_list(pauli_op_1d_list)
        # pauli_op_2d_list = []
        # for j in range(len(pauli_op_1d_list)):
        #     op = pauli_op_1d_list[j]
        #     pauli_op_2d_list.append((op[0] + N * 'I', op[1]))
        #     pauli_op_2d_list.append((N * 'I' + op[0], op[1]))
        # # print(pauli_op_2d_list)
        # pauli_op_2d = SparsePauliOp.from_list(pauli_op_2d_list)
        # # print(pauli_op_2d)

        # Compute number of gates per Trotter step
        if trotter_method == "first_order" or trotter_method == "randomized_first_order":
            circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op_1d.group_commuting()))
        elif trotter_method == "second_order":
            circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(pauli_op_1d.group_commuting()))
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

