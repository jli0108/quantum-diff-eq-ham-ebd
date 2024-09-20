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
    N_p = 2 ** n_p

    F_x = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        F_x[j,(j+1)%N] = 1
        F_x[j,j] = -1
    F_x /= h

    F_p = lil_matrix((N, N), dtype=np.complex128)
    p_vals = np.linspace(-1, 1, N, endpoint=False)
    for j in range(N):
        p = p_vals[j]
        F_p[j,(j+1)%N] = (p**3*(1-p**2)+1)
        F_p[j,j] = - (p**3*(1-p**2)+1)
    F_p /= h

    D_x = lil_matrix((N, N), dtype=np.complex128)
    x_vals = np.linspace(-1, 1, N, endpoint=False)
    for j in range(N):
        x = x_vals[j]
        D_x[j,j] = x**3 * (1-x**2) + 1

    D_p = lil_matrix((N, N), dtype=np.complex128)
    p_vals = np.linspace(-1, 1, N, endpoint=False)
    for j in range(N):
        p = p_vals[j]
        D_p[j,j] = p ** 3 * (1 - p ** 4)


    F_x_1 = (F_x + np.conj(F_x.T)) / 2
    F_x_2 = (F_x - np.conj(F_x.T)) / 2j
    F_p_1 = (F_p + np.conj(F_p.T)) / 2
    F_p_2 = (F_p - np.conj(F_p.T)) / 2j

    F_x_1_pauli_op_grouped = SparsePauliOp.from_operator(F_x_1.toarray()).simplify().group_commuting()
    F_x_2_pauli_op_grouped = SparsePauliOp.from_operator(F_x_2.toarray()).simplify().group_commuting()
    F_p_1_pauli_op_grouped = SparsePauliOp.from_operator(F_p_1.toarray()).simplify().group_commuting()
    F_p_2_pauli_op_grouped = SparsePauliOp.from_operator(F_p_2.toarray()).simplify().group_commuting()
    D_x_pauli_op = SparsePauliOp.from_operator(D_x.toarray())
    D_p_pauli_op = SparsePauliOp.from_operator(D_p.toarray())
    H_F_pauli_op = (-get_xi_pauli_op(n_p, R))
    Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])

    H = []
    H_sp_mats = []

    # Hermitian part
    for j in range(len(F_x_1_pauli_op_grouped)):
        H.append(F_x_1_pauli_op_grouped[j].tensor(D_p_pauli_op).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([F_x_1_pauli_op_grouped[j].to_matrix(sparse=True), D_p, H_F_pauli_op.to_matrix(sparse=True)]))
                
    for j in range(len(F_p_1_pauli_op_grouped)):
        H.append(D_x_pauli_op.tensor(F_p_1_pauli_op_grouped[j]).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([D_x, F_p_1_pauli_op_grouped[j].to_matrix(sparse=True), H_F_pauli_op.to_matrix(sparse=True)]))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_op_grouped)):
        H.append(F_x_2_pauli_op_grouped[j].tensor(D_p_pauli_op).tensor(Id_n_p))
        H_sp_mats.append(-tensor([F_x_2_pauli_op_grouped[j].to_matrix(sparse=True), D_p, identity(N_p)]))
    for j in range(len(F_p_2_pauli_op_grouped)):
        H.append(D_x_pauli_op.tensor(F_p_2_pauli_op_grouped[j]).tensor(Id_n_p))
        H_sp_mats.append(-tensor([D_x, F_p_2_pauli_op_grouped[j].to_matrix(sparse=True), identity(N_p)]))

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

    F_x = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        F_x[j,(j+1)%N] = 1
        F_x[j,j] = -1
    F_x /= h

    F_p = lil_matrix((N, N), dtype=np.complex128)
    p_vals = np.linspace(-1, 1, N, endpoint=False)
    for j in range(N):
        p = p_vals[j]
        F_p[j,(j+1)%N] = (p**3*(1-p**2)+1)
        F_p[j,j] = - (p**3*(1-p**2)+1)
    F_p /= h

    D_x = lil_matrix((N, N), dtype=np.complex128)
    x_vals = np.linspace(-1, 1, N, endpoint=False)
    for j in range(N):
        x = x_vals[j]
        D_x[j,j] = x**3 * (1-x**2) + 1

    D_p = lil_matrix((N, N), dtype=np.complex128)
    p_vals = np.linspace(-1, 1, N, endpoint=False)
    for j in range(N):
        p = p_vals[j]
        D_p[j,j] = p ** 3 * (1 - p ** 4)


    F_x_1 = (F_x + np.conj(F_x.T)) / 2
    F_x_2 = (F_x - np.conj(F_x.T)) / 2j
    F_p_1 = (F_p + np.conj(F_p.T)) / 2
    F_p_2 = (F_p - np.conj(F_p.T)) / 2j



    F_x_1_pauli_list = []
    F_x_2_pauli_list = []
    F_p_1_pauli_list = []
    F_p_2_pauli_list = []
    D_x_pauli_list = []
    D_p_pauli_list = []

    for j in range(N):
        op = N * ['I']
        op[j] = 'X'
        op[(j+1)%N] = 'X'
        F_x_1_pauli_list.append((''.join(op), 0.5*F_x_1[(j+1)%N,j].real))
        F_p_1_pauli_list.append((''.join(op), 0.5*F_p_1[(j+1)%N,j].real))

        op = N * ['I']
        op[j] = 'Y'
        op[(j+1)%N] = 'Y'
        F_x_1_pauli_list.append((''.join(op), 0.5*F_x_1[(j+1)%N,j].real))
        F_p_1_pauli_list.append((''.join(op), 0.5*F_p_1[(j+1)%N,j].real))

        F_x_1_pauli_list.append((N * 'I', 0.5 * F_x_1[j,j].real))
        F_p_1_pauli_list.append((N * 'I', 0.5 * F_p_1[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        F_x_1_pauli_list.append((''.join(op), -0.5 * F_x_1[j,j].real))
        F_p_1_pauli_list.append((''.join(op), -0.5 * F_p_1[j,j].real))

    for j in range(N):
        op = N * ['I']
        op[j] = 'X'
        op[(j+1)%N] = 'Y'
        F_x_2_pauli_list.append((''.join(op), 0.5*F_x_2[(j+1)%N,j].imag))
        F_p_2_pauli_list.append((''.join(op), 0.5*F_p_2[(j+1)%N,j].imag))

        op = N * ['I']
        op[j] = 'Y'
        op[(j+1)%N] = 'X'
        F_x_2_pauli_list.append((''.join(op), -0.5*F_x_2[(j+1)%N,j].imag))
        F_p_2_pauli_list.append((''.join(op), -0.5*F_p_2[(j+1)%N,j].imag))

        F_x_2_pauli_list.append((N * 'I', 0.5 * F_x_1[j,j].real))
        F_p_2_pauli_list.append((N * 'I', 0.5 * F_p_1[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        F_x_2_pauli_list.append((''.join(op), -0.5 * F_x_1[j,j].real))
        F_p_2_pauli_list.append((''.join(op), -0.5 * F_p_1[j,j].real))

    for j in range(N):
        D_x_pauli_list.append((N * 'I', 0.5 * D_x[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_x_pauli_list.append((''.join(op), -0.5 * D_x[j,j].real))

        D_p_pauli_list.append((N * 'I', 0.5 * D_p[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_p_pauli_list.append((''.join(op), -0.5 * D_p[j,j].real))

    F_x_1_pauli_op_grouped = SparsePauliOp.from_list(F_x_1_pauli_list).simplify().group_commuting()
    F_x_2_pauli_op_grouped = SparsePauliOp.from_list(F_x_2_pauli_list).simplify().group_commuting()
    F_p_1_pauli_op_grouped = SparsePauliOp.from_list(F_p_1_pauli_list).simplify().group_commuting()
    F_p_2_pauli_op_grouped = SparsePauliOp.from_list(F_p_2_pauli_list).simplify().group_commuting()
    D_x_pauli_op = SparsePauliOp.from_list(D_x_pauli_list).simplify()
    D_p_pauli_op = SparsePauliOp.from_list(D_p_pauli_list).simplify()
    H_F_pauli_op = (-get_xi_pauli_op(n_p, R))
    Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])

    F_x_1_sp_mats = []
    for j in range(len(F_x_1_pauli_op_grouped)):
        sp_mat = csc_matrix((N,N), dtype=np.complex128)
        for k in range(len(F_x_1_pauli_op_grouped[j])):
            sp_mat += one_hot_projection(F_x_1_pauli_op_grouped[j][k])
        F_x_1_sp_mats.append(sp_mat)
    F_p_1_sp_mats = []
    for j in range(len(F_p_1_pauli_op_grouped)):
        sp_mat = csc_matrix((N,N), dtype=np.complex128)
        for k in range(len(F_p_1_pauli_op_grouped[j])):
            sp_mat += one_hot_projection(F_p_1_pauli_op_grouped[j][k])
        F_p_1_sp_mats.append(sp_mat)
    F_x_2_sp_mats = []
    for j in range(len(F_x_2_pauli_op_grouped)):
        sp_mat = csc_matrix((N,N), dtype=np.complex128)
        for k in range(len(F_x_2_pauli_op_grouped[j])):
            sp_mat += one_hot_projection(F_x_2_pauli_op_grouped[j][k])
        F_x_2_sp_mats.append(sp_mat)
    F_p_2_sp_mats = []
    for j in range(len(F_p_2_pauli_op_grouped)):
        sp_mat = csc_matrix((N,N), dtype=np.complex128)
        for k in range(len(F_p_2_pauli_op_grouped[j])):
            sp_mat += one_hot_projection(F_p_2_pauli_op_grouped[j][k])
        F_p_2_sp_mats.append(sp_mat)

    H = []
    H_sp_mats = []

    # Hermitian part
    for j in range(len(F_x_1_pauli_op_grouped)):
        H.append(F_x_1_pauli_op_grouped[j].tensor(D_p_pauli_op).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([F_x_1_sp_mats[j], D_p, H_F_pauli_op.to_matrix(sparse=True)]))
                
    for j in range(len(F_p_1_pauli_op_grouped)):
        H.append(D_x_pauli_op.tensor(F_p_1_pauli_op_grouped[j]).tensor(H_F_pauli_op))
        H_sp_mats.append(tensor([D_x, F_p_1_sp_mats[j], H_F_pauli_op.to_matrix(sparse=True)]))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_op_grouped)):
        H.append(F_x_2_pauli_op_grouped[j].tensor(D_p_pauli_op).tensor(Id_n_p))
        H_sp_mats.append(-tensor([F_x_2_sp_mats[j], D_p, identity(N_p)]))
    for j in range(len(F_p_2_pauli_op_grouped)):
        H.append(D_x_pauli_op.tensor(F_p_2_pauli_op_grouped[j]).tensor(Id_n_p))
        H_sp_mats.append(-tensor([D_x, F_p_2_sp_mats[j], identity(N_p)]))

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
    
def get_circuit_depth(n, H):
    L = len(H)
    depth = 0
    for j in range(L):
        circuit = QuantumCircuit(n)
        for pauli_op in H[j]:
            circuit.append(LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op)), qargs=np.arange(n).tolist())
        compiled_circ = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
        depth += compiled_circ.depth()

    return depth

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

    print("Running resource analysis for nonlinear PDE", flush=True)
    start_time = time()
    error_tols = np.exp(-np.linspace(np.log(10), np.log(1000), 10))
    N = 64                                     # grid points along each dimension
    n_x = int(np.log2(N))
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    trotter_method="first_order"

    print("Error tolerances:", error_tols, flush=True)

    pauli_basis_trotter_steps = np.zeros((len(error_tols)))
    one_hot_trotter_steps = np.zeros((len(error_tols)))

    # Gate counts per Trotter step
    H_std_binary, H_std_binary_sp_mats = get_H_std_binary(N, n_p, R)
    print("Computing Trotter steps for Pauli basis.", flush=True)
    pauli_basis_trotter_steps = get_first_order_trotter_steps(T, H_std_binary_sp_mats, error_tols)
    print("Computing gates per Trotter step for Pauli basis.", flush=True)
    pauli_basis_single_qubit_gates, pauli_basis_two_qubit_gates = get_gate_count_per_trotter_step(H_std_binary, trotter_method)
    pauli_basis_circ_depth = get_circuit_depth(2 * n_x + n_p, H_std_binary)
    
    '''One-hot encoding (ours)'''
    H_one_hot, H_one_hot_sp_mats = get_H_one_hot(N, n_p, R)
    print("Computing Trotter steps for one-hot encoding.", flush=True)
    one_hot_trotter_steps = get_first_order_trotter_steps(T, H_one_hot_sp_mats, error_tols)
    print("Computing gates per Trotter step for one-hot.", flush=True)
    one_hot_single_qubit_gates, one_hot_two_qubit_gates = get_gate_count_per_trotter_step(H_one_hot, trotter_method)
    one_hot_circ_depth = get_circuit_depth(2 * N + n_p, H_one_hot)

    np.savez(join("../resource_analysis_data", f"nonlinear_data_{trotter_method}.npz"),
            error_tols=error_tols,
            pauli_basis_trotter_steps=pauli_basis_trotter_steps,
            pauli_basis_single_qubit_gates=pauli_basis_single_qubit_gates,
            pauli_basis_two_qubit_gates=pauli_basis_two_qubit_gates,
            pauli_basis_circ_depth=pauli_basis_circ_depth,
            one_hot_trotter_steps=one_hot_trotter_steps,
            one_hot_single_qubit_gates=one_hot_single_qubit_gates,
            one_hot_two_qubit_gates=one_hot_two_qubit_gates,
            one_hot_circ_depth=one_hot_circ_depth)

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)
    print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)