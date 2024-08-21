import numpy as np

from scipy.sparse import eye, lil_matrix, diags
from scipy.linalg.interpolative import estimate_spectral_norm
from joblib import Parallel, delayed
from resource_estimate_utils import *
from os.path import join
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate

from os.path import join
import sys
sys.path.append(join(".", ".."))
from utils import *
import resource

def get_xi_pauli_op(n_p, R):
    xi_pauli_list = []
    op = n_p * ['I']
    xi_pauli_list.append((''.join(op), 1))

    for i in range(n_p - 1):
        op = n_p * ['I']
        op[n_p-1-i] = 'Z'
        xi_pauli_list.append((''.join(op), 2 ** i))
    
    op = n_p * ['I']
    op[0] = 'Z'
    xi_pauli_list.append((''.join(op), - 2 ** (n_p - 1)))

    xi_pauli_op = 0.5 * (2 * np.pi / (2 * R)) * SparsePauliOp.from_list(xi_pauli_list)
    return xi_pauli_op

def get_H_std_binary(N, n_p, R):
    h = 1 / N

    A = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        x = j / N
        A[j,(j+1)%N] = (x * (1-x) + 1)
        A[j,j] = - (x * (1-x) + 1)
    A /= h

    D = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        x = j / N
        D[j,j] = 1 + x * (1 - x)

    H_1 = (A + np.conj(A.T)) / 2
    H_2 = (A - np.conj(A.T)) / 2j

    H_1_pauli_op_grouped = SparsePauliOp.from_operator(H_1.toarray()).group_commuting()
    H_2_pauli_op_grouped = SparsePauliOp.from_operator(H_2.toarray()).group_commuting()
    
    D_pauli_op = SparsePauliOp.from_operator(D.toarray())
    H_F_pauli_op = (-get_xi_pauli_op(n_p, R))
    Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])

    H = []
    H_sp_mats = []

    # Hermitian part
    for j in range(len(H_1_pauli_op_grouped)):
        H.append(H_1_pauli_op_grouped[j].tensor(D_pauli_op).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([H_1_pauli_op_grouped[j].to_matrix(sparse=True), D, H_F_pauli_op.to_matrix(sparse=True)]))
        H.append(D_pauli_op.tensor(H_1_pauli_op_grouped[j]).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([D, H_1_pauli_op_grouped[j].to_matrix(sparse=True), H_F_pauli_op.to_matrix(sparse=True)]))
    # Anti-Hermitian part
    for j in range(len(H_2_pauli_op_grouped)):
        H.append(H_2_pauli_op_grouped[j].tensor(D_pauli_op).tensor(Id_n_p))
        H_sp_mats.append(tensor([H_2_pauli_op_grouped[j].to_matrix(sparse=True), D, identity(N_p)]))
        H.append(D_pauli_op.tensor(H_2_pauli_op_grouped[j]).tensor(Id_n_p))
        H_sp_mats.append(tensor([D, H_2_pauli_op_grouped[j].to_matrix(sparse=True), identity(N_p)]))

    return H, H_sp_mats

def one_hot_projection(H_term):
    N = H_term.num_qubits
    pauli_str = H_term.to_list()[0][0]
    pauli_indices = []
    for j in range(N):
        if pauli_str[j] != 'I':
            pauli_indices.append(j)
    if len(pauli_indices) == 0:
        mat = np.identity(N, dtype=np.complex128)
    elif len(pauli_indices) == 1:
        mat = np.identity(N, dtype=np.complex128)
        assert pauli_str[pauli_indices[0]] == 'Z'
        mat[pauli_indices[0], pauli_indices[0]] = -1
    elif len(pauli_indices) == 2:
        mat = np.zeros((N,N), dtype=np.complex128)
        if pauli_str[pauli_indices[0]] + pauli_str[pauli_indices[1]] == 'XX' or pauli_str[pauli_indices[0]] + pauli_str[pauli_indices[1]] == 'YY':
            mat[pauli_indices[0], pauli_indices[1]] = 1
            mat[pauli_indices[1], pauli_indices[0]] = 1
        elif pauli_str[pauli_indices[0]] + pauli_str[pauli_indices[1]] == 'XY':
            mat[pauli_indices[0], pauli_indices[1]] *= -1j
            mat[pauli_indices[1], pauli_indices[0]] *= 1j
        elif pauli_str[pauli_indices[0]] + pauli_str[pauli_indices[1]] == 'YX':
            mat[pauli_indices[0], pauli_indices[1]] *= -1j
            mat[pauli_indices[1], pauli_indices[0]] *= 1j
        else:
            raise Exception()
    else:
        raise Exception()
    return H_term.coeffs[0] * csc_matrix(mat)

def get_H_one_hot(N, n_p, R):
    h = 1 / N
    N_p = 2 ** n_p

    A = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        x = j / N
        A[j,(j+1)%N] = (x * (1-x) + 1)
        A[j,j] = - (x * (1-x) + 1)
    A /= h

    D = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        x = j / N
        D[j,j] = 1 + x * (1 - x)

    H_1 = (A + np.conj(A.T)) / 2
    H_2 = (A - np.conj(A.T)) / 2j


    H_1_pauli_list = []
    H_2_pauli_list = []
    D_pauli_list = []

    for j in range(N):
        op = N * ['I']
        op[j] = 'X'
        op[(j+1)%N] = 'X'
        H_1_pauli_list.append((''.join(op), 0.5*H_1[(j+1)%N,j].real))

        op = N * ['I']
        op[j] = 'Y'
        op[(j+1)%N] = 'Y'
        H_1_pauli_list.append((''.join(op), 0.5*H_1[(j+1)%N,j].real))

        H_1_pauli_list.append((N * 'I', 0.5 * H_1[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        H_1_pauli_list.append((''.join(op), -0.5 * H_1[j,j].real))

    for j in range(N):
        op = N * ['I']
        op[j] = 'X'
        op[(j+1)%N] = 'Y'
        H_2_pauli_list.append((''.join(op), 0.5*H_2[(j+1)%N,j].imag))

        op = N * ['I']
        op[j] = 'Y'
        op[(j+1)%N] = 'X'
        H_2_pauli_list.append((''.join(op), -0.5*H_2[(j+1)%N,j].imag))

        H_2_pauli_list.append((N * 'I', 0.5 * H_2[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        H_2_pauli_list.append((''.join(op), -0.5 * H_2[j,j].real))

    for j in range(N):
        D_pauli_list.append((N * 'I', 0.5 * D[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_pauli_list.append((''.join(op), -0.5 * D[j,j].real))

    H_1_pauli_op_grouped = SparsePauliOp.from_list(H_1_pauli_list).group_commuting()
    H_2_pauli_op_grouped = SparsePauliOp.from_list(H_2_pauli_list).group_commuting()
    D_pauli_op = SparsePauliOp.from_list(D_pauli_list)
    H_F_pauli_op = (-get_xi_pauli_op(n_p, R))
    Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])
    
    H_1_sp_mats = []
    for j in range(len(H_1_pauli_op_grouped)):
        sp_mat = csc_matrix((N,N), dtype=np.complex128)
        for k in range(len(H_1_pauli_op_grouped[j])):
            sp_mat += one_hot_projection(H_1_pauli_op_grouped[j][k])
        H_1_sp_mats.append(sp_mat)
    H_2_sp_mats = []
    for j in range(len(H_2_pauli_op_grouped)):
        sp_mat = csc_matrix((N,N), dtype=np.complex128)
        for k in range(len(H_2_pauli_op_grouped[j])):
            sp_mat += one_hot_projection(H_2_pauli_op_grouped[j][k])
        H_2_sp_mats.append(sp_mat)

    H = []
    H_sp_mats = []
    # Hermitian part
    for j in range(len(H_1_pauli_op_grouped)):
        H.append(H_1_pauli_op_grouped[j].tensor(D_pauli_op).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([H_1_sp_mats[j], D, H_F_pauli_op.to_matrix(sparse=True)]))
        H.append(D_pauli_op.tensor(H_1_pauli_op_grouped[j]).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([D, H_1_sp_mats[j], H_F_pauli_op.to_matrix(sparse=True)]))
    # Anti-Hermitian part
    for j in range(len(H_2_pauli_op_grouped)):
        H.append(H_2_pauli_op_grouped[j].tensor(D_pauli_op).tensor(Id_n_p))
        H_sp_mats.append(tensor([H_2_sp_mats[j], D, identity(N_p)]))
        H.append(D_pauli_op.tensor(H_2_pauli_op_grouped[j]).tensor(Id_n_p))
        H_sp_mats.append(tensor([D, H_2_sp_mats[j], identity(N_p)]))

    return H, H_sp_mats

def get_gate_count_per_trotter_step(H, trotter_method="second_order"):
    num_single_qubit_gates, num_two_qubit_gates = 0, 0
    L = len(H)
    for j in range(L):
        print(f"Getting gate counts per Trotter step: [{j}/{L}]", end="\r")
        for pauli_op in H[j]:
            circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op))
            compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
            ops = compiled_circuit.count_ops()
            for key in ops.keys():
                if key == 'rx' or key == 'ry' or key == 'rz':
                    num_single_qubit_gates += ops[key]
                elif key == 'rxx':
                    num_two_qubit_gates += ops[key]
    
    print(f"Getting gate counts per Trotter step: [{L}/{L}]")
    if trotter_method == "first_order":
        return num_single_qubit_gates, num_two_qubit_gates
    elif trotter_method == "second_order":
        return 2 * num_single_qubit_gates, 2 * num_two_qubit_gates
    
def sparse_comm(H1, H2):
    return H1 @ H2 - H2 @ H1

def get_first_order_trotter_steps(t, H, epsilon):
    L = len(H)
    coeff = 0
    for j in range(L-1):
        print(f"Computing Trotter steps: [{j} / {L-1}]", end="\r", flush=True)
        comm_term = 0
        for k in np.arange(j+1, L):
            comm_term += sparse_comm(H[j], H[k])
        coeff += estimate_spectral_norm(comm_term)
    print(f"Computing Trotter steps: [{L-1} / {L-1}]", flush=True)
    return np.ceil((t ** 2 * coeff / (2 * epsilon)))

def get_second_order_trotter_steps(t, H, epsilon):
    L = len(H)
    coeff = 0
    for j in range(L-1):
        print(f"Computing Trotter steps: [{j} / {L-1}]", end="\r", flush=True)
        comm_term = 0
        for k in np.arange(j+1, L):
            for l in np.arange(j+1, L):
                comm_term += sparse_comm(H[l], sparse_comm(H[k], H[j]))
        coeff += estimate_spectral_norm(comm_term)

        comm_term = 0
        for k in np.arange(j+1, L):
            comm_term += sparse_comm(H[j], sparse_comm(H[j], H[k]))
        coeff += 0.5 * estimate_spectral_norm(comm_term)
    print(f"Computing Trotter steps: [{L-1} / {L-1}]", flush=True)

    return np.ceil(np.sqrt(t ** 3 * coeff / (12 * epsilon)))

if __name__ == "__main__":

    print("Running Fig 2 script (nonseparable)", flush=True)
    start_time = time()
    error_tols = np.exp(-np.linspace(np.log(10), np.log(1000), 10))
    N = 128                                     # grid points along each dimension
    n_x = int(np.log2(N))
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    trotter_method="first_order"

    print("Error tolerances:", error_tols, flush=True)

    print("Computing Trotter steps.", flush=True)
    pauli_basis_trotter_steps = np.zeros((len(error_tols)))
    one_hot_trotter_steps = np.zeros((len(error_tols)))

    # Gate counts per Trotter step
    print("Getting Hamiltonian w/ Pauli basis.", flush=True)
    H_std_binary, H_std_binary_sp_mats = get_H_std_binary(N, n_p, R)
    print("Computing Trotter steps for Pauli basis.", flush=True)
    pauli_basis_trotter_steps = get_first_order_trotter_steps(T, H_std_binary_sp_mats, error_tols)
    print("Computing gates per Trotter step for Pauli basis.", flush=True)
    pauli_basis_single_qubit_gates, pauli_basis_two_qubit_gates = get_gate_count_per_trotter_step(H_std_binary, trotter_method)

    '''One-hot encoding (ours)'''
    print("Getting Hamiltonian w/ one-hot encoding.", flush=True)
    H_one_hot, H_one_hot_sp_mats = get_H_one_hot(N, n_p, R)
    print("Computing Trotter steps for one-hot encoding.", flush=True)
    one_hot_trotter_steps = get_first_order_trotter_steps(T, H_one_hot_sp_mats, error_tols)
    print("Computing gates per Trotter step for one-hot.", flush=True)
    one_hot_single_qubit_gates, one_hot_two_qubit_gates = get_gate_count_per_trotter_step(H_one_hot, trotter_method)


    np.savez(join("../resource_analysis_data", f"fig2_nonseparable_data_{trotter_method}.npz"),
            error_tols=error_tols,
            pauli_basis_trotter_steps=pauli_basis_trotter_steps,
            pauli_basis_single_qubit_gates=pauli_basis_single_qubit_gates,
            pauli_basis_two_qubit_gates=pauli_basis_two_qubit_gates,
            one_hot_trotter_steps=one_hot_trotter_steps,
            one_hot_single_qubit_gates=one_hot_single_qubit_gates,
            one_hot_two_qubit_gates=one_hot_two_qubit_gates)

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)
    print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

