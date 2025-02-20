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
from qiskit.converters import circuit_to_dag, dag_to_circuit


from os.path import join
import sys
sys.path.append(join(".", ".."))
from utils import *
import resource

def parallelize_ctrl_circuit(circuit):
    # Given an n-qubit circuit, perform a controlled version of the circuit using n+1 ancillas:.
    # For each layer of the circuit, perform the controlled version in parallel.
    # Output: the depth and measurement rounds
    overall_depth = 0
    dag = circuit_to_dag(circuit)

    for layer in dag.layers():

        layer_as_circuit = dag_to_circuit(layer['graph'])
        # In between the fan-out gates, perform the controlled version of each gate
        # The overall depth only depends on the maximum depth of any controlled gate in the layer
        max_depth = 0
        for gate in layer_as_circuit.data:
            qc = QuantumCircuit.from_instructions([gate])
            ctrl_qc = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'rxx'], optimization_level=0).control(1, ctrl_state='1')
            transpiled_qc = transpile(ctrl_qc, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
            if transpiled_qc.depth(lambda instr: len(instr.qubits) > 1) > max_depth:
                max_depth = transpiled_qc.depth(lambda instr: len(instr.qubits) > 1)

        overall_depth += max_depth
    return overall_depth

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
    # Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])

    # Store the Pauli decomposition separately for the Hermitian and anti-Hermitian parts,
    #  (without the ancilla register for Schrodingerization) to estimate the circuit depth per Trotter step
    # Note that for the anti-Hermitian part, we can fan-out the Schrodingerization ancilla register
    H_herm, H_antiherm = [], []
    # Store as sparse matrices (with the ancilla register for Schrodingerization), to estimate the Trotter steps
    H_herm_sp_mats, H_antiherm_sp_mats = [], []

    # Hermitian part
    for j in range(len(F_x_1_pauli_op_grouped)):
        H_herm.append(F_x_1_pauli_op_grouped[j].tensor(D_p_pauli_op))
        H_herm_sp_mats.append(tensor([F_x_1_pauli_op_grouped[j].to_matrix(sparse=True), D_p, H_F_pauli_op.to_matrix(sparse=True)]))
                
    for j in range(len(F_p_1_pauli_op_grouped)):
        H_herm.append(D_x_pauli_op.tensor(F_p_1_pauli_op_grouped[j]))
        H_herm_sp_mats.append(tensor([D_x, F_p_1_pauli_op_grouped[j].to_matrix(sparse=True), H_F_pauli_op.to_matrix(sparse=True)]))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_op_grouped)):
        H_antiherm.append(F_x_2_pauli_op_grouped[j].tensor(D_p_pauli_op))
        H_antiherm_sp_mats.append(-tensor([F_x_2_pauli_op_grouped[j].to_matrix(sparse=True), D_p, identity(N_p)]))
    for j in range(len(F_p_2_pauli_op_grouped)):
        H_antiherm.append(D_x_pauli_op.tensor(F_p_2_pauli_op_grouped[j]))
        H_antiherm_sp_mats.append(-tensor([D_x, F_p_2_pauli_op_grouped[j].to_matrix(sparse=True), identity(N_p)]))

    return H_herm, H_antiherm, H_herm_sp_mats, H_antiherm_sp_mats

def decompose_tridiag_mat(A):
    # Splits an N-by-N (where N is even) tridiagonal matrix into an even part, odd part, and the diagonal part
    N = A.shape[0]
    sp_mat_even = np.zeros((N,N), dtype=np.complex128)
    sp_mat_odd = np.zeros((N,N), dtype=np.complex128)
    sp_mat_diag = np.zeros((N,N), dtype=np.complex128)

    for j in range(N):
        if j % 2 == 0:
            sp_mat_even[j,(j+1)%N] = A[j,(j+1)%N]
            sp_mat_even[(j+1)%N,j] = A[(j+1)%N,j]
        else:
            sp_mat_odd[j,(j+1)%N] = A[j,(j+1)%N]
            sp_mat_odd[(j+1)%N,j] = A[(j+1)%N,j]
        sp_mat_diag[j,j] = A[j,j]

    return csc_matrix(sp_mat_even), csc_matrix(sp_mat_odd), csc_matrix(sp_mat_diag)

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



    F_x_1_pauli_list_even = []
    F_x_1_pauli_list_odd = []
    F_x_1_pauli_list_diag = []
    F_x_2_pauli_list_even = []
    F_x_2_pauli_list_odd = []
    F_p_1_pauli_list_even = []
    F_p_1_pauli_list_odd = []
    F_p_1_pauli_list_diag = []
    F_p_2_pauli_list_even = []
    F_p_2_pauli_list_odd = []
    D_x_pauli_list = []
    D_p_pauli_list = []

    for j in range(N):
        op = N * ['I']
        op[j] = 'X'
        op[(j+1)%N] = 'X'
        if j % 2 == 0:
            F_x_1_pauli_list_even.append((''.join(op), 0.5*F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_even.append((''.join(op), 0.5*F_p_1[(j+1)%N,j].real))
        else:
            F_x_1_pauli_list_odd.append((''.join(op), 0.5*F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_odd.append((''.join(op), 0.5*F_p_1[(j+1)%N,j].real))
            

        op = N * ['I']
        op[j] = 'Y'
        op[(j+1)%N] = 'Y'
        if j % 2 == 0:
            F_x_1_pauli_list_even.append((''.join(op), 0.5*F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_even.append((''.join(op), 0.5*F_p_1[(j+1)%N,j].real))
        else:
            F_x_1_pauli_list_odd.append((''.join(op), 0.5*F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_odd.append((''.join(op), 0.5*F_p_1[(j+1)%N,j].real))

        F_x_1_pauli_list_diag.append((N * 'I', 0.5 * F_x_1[j,j].real))
        F_p_1_pauli_list_diag.append((N * 'I', 0.5 * F_p_1[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        F_x_1_pauli_list_diag.append((''.join(op), -0.5 * F_x_1[j,j].real))
        F_p_1_pauli_list_diag.append((''.join(op), -0.5 * F_p_1[j,j].real))

    for j in range(N):
        op = N * ['I']
        op[j] = 'X'
        op[(j+1)%N] = 'Y'
        if j % 2 == 0:
            F_x_2_pauli_list_even.append((''.join(op), 0.5*F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_even.append((''.join(op), 0.5*F_p_2[(j+1)%N,j].imag))
        else:
            F_x_2_pauli_list_odd.append((''.join(op), 0.5*F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_odd.append((''.join(op), 0.5*F_p_2[(j+1)%N,j].imag))

        op = N * ['I']
        op[j] = 'Y'
        op[(j+1)%N] = 'X'
        if j % 2 == 0:
            F_x_2_pauli_list_even.append((''.join(op), -0.5*F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_even.append((''.join(op), -0.5*F_p_2[(j+1)%N,j].imag))
        else:
            F_x_2_pauli_list_odd.append((''.join(op), -0.5*F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_odd.append((''.join(op), -0.5*F_p_2[(j+1)%N,j].imag))

    for j in range(N):
        D_x_pauli_list.append((N * 'I', 0.5 * D_x[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_x_pauli_list.append((''.join(op), -0.5 * D_x[j,j].real))

        D_p_pauli_list.append((N * 'I', 0.5 * D_p[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_p_pauli_list.append((''.join(op), -0.5 * D_p[j,j].real))

    F_x_1_pauli_op_grouped = [SparsePauliOp.from_list(F_x_1_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_x_1_pauli_list_odd).simplify(),
                                SparsePauliOp.from_list(F_x_1_pauli_list_diag).simplify()]
    F_x_2_pauli_op_grouped = [SparsePauliOp.from_list(F_x_2_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_x_2_pauli_list_odd).simplify()]
    F_p_1_pauli_op_grouped = [SparsePauliOp.from_list(F_p_1_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_p_1_pauli_list_odd).simplify(),
                                SparsePauliOp.from_list(F_p_1_pauli_list_diag).simplify()]
    F_p_2_pauli_op_grouped = [SparsePauliOp.from_list(F_p_2_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_p_2_pauli_list_odd).simplify()]
    
    D_x_pauli_op = SparsePauliOp.from_list(D_x_pauli_list).simplify()
    D_p_pauli_op = SparsePauliOp.from_list(D_p_pauli_list).simplify()
    H_F_pauli_op = (-get_xi_pauli_op(n_p, R))
    # Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])

    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_x_1)
    F_x_1_sp_mats = [sp_mat_even, sp_mat_odd, sp_mat_diag]
    
    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_p_1)
    F_p_1_sp_mats = [sp_mat_even, sp_mat_odd, sp_mat_diag]
    
    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_x_2)
    F_x_2_sp_mats = [sp_mat_even, sp_mat_odd]
    
    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_p_2)
    F_p_2_sp_mats = [sp_mat_even, sp_mat_odd]

    # Store the Pauli decomposition separately for the Hermitian and anti-Hermitian parts,
    #  (without the ancilla register for Schrodingerization) to estimate the circuit depth per Trotter step
    # Note that for the anti-Hermitian part, we can fan-out the Schrodingerization ancilla register
    H_herm, H_antiherm = [], []
    # Store as sparse matrices (with the ancilla register for Schrodingerization), to estimate the Trotter steps
    H_herm_sp_mats, H_antiherm_sp_mats = [], []

    # Hermitian part
    for j in range(len(F_x_1_pauli_op_grouped)):
        H_herm.append(F_x_1_pauli_op_grouped[j].tensor(D_p_pauli_op))
        H_herm_sp_mats.append(tensor([F_x_1_sp_mats[j], D_p, H_F_pauli_op.to_matrix(sparse=True)]))
                
    for j in range(len(F_p_1_pauli_op_grouped)):
        H_herm.append(D_x_pauli_op.tensor(F_p_1_pauli_op_grouped[j]))
        H_herm_sp_mats.append(tensor([D_x, F_p_1_sp_mats[j], H_F_pauli_op.to_matrix(sparse=True)]))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_op_grouped)):
        H_antiherm.append(F_x_2_pauli_op_grouped[j].tensor(D_p_pauli_op))
        H_antiherm_sp_mats.append(-tensor([F_x_2_sp_mats[j], D_p, identity(N_p)]))
    for j in range(len(F_p_2_pauli_op_grouped)):
        H_antiherm.append(D_x_pauli_op.tensor(F_p_2_pauli_op_grouped[j]))
        H_antiherm_sp_mats.append(-tensor([D_x, F_p_2_sp_mats[j], identity(N_p)]))

    return H_herm, H_antiherm, H_herm_sp_mats, H_antiherm_sp_mats

def get_H_unary(N, n_p, R):
    n = N // 2
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



    F_x_1_pauli_list_even = []
    F_x_1_pauli_list_odd = []
    F_x_1_pauli_list_diag = []
    F_x_2_pauli_list_even = []
    F_x_2_pauli_list_odd = []
    F_p_1_pauli_list_even = []
    F_p_1_pauli_list_odd = []
    F_p_1_pauli_list_diag = []
    F_p_2_pauli_list_even = []
    F_p_2_pauli_list_odd = []
    D_x_pauli_list = []
    D_p_pauli_list = []

    for j in range(N):
        if n - 1 <= j < N - 1:
            a = 1
        else:
            a = 0

        if 1 <= j <= N // 2:
            b = 1
        else:
            b = 0
                
        if j >= n:
            c = 1
        else:
            c = 0

        # Real part
        op = n * ['I']
        op[j%n] = 'X'
        if j % 2 == 0:
            F_x_1_pauli_list_even.append((''.join(op), ((-1) ** 0 / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_even.append((''.join(op), ((-1) ** 0 / 4) * F_p_1[(j+1)%N,j].real))
        else:
            F_x_1_pauli_list_odd.append((''.join(op), ((-1) ** 0 / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_odd.append((''.join(op), ((-1) ** 0 / 4) * F_p_1[(j+1)%N,j].real))


        op = n * ['I']
        op[j%n] = 'X'
        op[(j-1)%n] = 'Z'
        if j % 2 == 0:
            F_x_1_pauli_list_even.append((''.join(op), ((-1) ** (a+0) / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_even.append((''.join(op), ((-1) ** (a+0) / 4) * F_p_1[(j+1)%N,j].real))
        else:
            F_x_1_pauli_list_odd.append((''.join(op), ((-1) ** (a+0) / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_odd.append((''.join(op), ((-1) ** (a+0) / 4) * F_p_1[(j+1)%N,j].real))

        op = n * ['I']
        op[j%n] = 'X'
        op[(j+1)%n] = 'Z'
        if j % 2 == 0:
            F_x_1_pauli_list_even.append((''.join(op), ((-1) ** (b+0) / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_even.append((''.join(op), ((-1) ** (b+0) / 4) * F_p_1[(j+1)%N,j].real))
        else:
            F_x_1_pauli_list_odd.append((''.join(op), ((-1) ** (b+0) / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_odd.append((''.join(op), ((-1) ** (b+0) / 4) * F_p_1[(j+1)%N,j].real))

        op = n * ['I']
        op[(j-1)%n] = 'Z'
        op[j%n] = 'X'
        op[(j+1)%n] = 'Z'
        if j % 2 == 0:
            F_x_1_pauli_list_even.append((''.join(op), ((-1) ** (a+b+0) / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_even.append((''.join(op), ((-1) ** (a+b+0) / 4) * F_p_1[(j+1)%N,j].real))
        else:
            F_x_1_pauli_list_odd.append((''.join(op), ((-1) ** (a+b+0) / 4) * F_x_1[(j+1)%N,j].real))
            F_p_1_pauli_list_odd.append((''.join(op), ((-1) ** (a+b+0) / 4) * F_p_1[(j+1)%N,j].real))


        # Diagonal part
        op = n * ['I']
        F_x_1_pauli_list_diag.append((''.join(op), F_x_1[j,j].real))
        F_p_1_pauli_list_diag.append((''.join(op), F_p_1[j,j].real))
        op = n * ['I']
        op[(j-1)%n] = 'Z'
        F_x_1_pauli_list_diag.append((''.join(op), (-1) ** (b) * F_x_1[j,j].real))
        F_p_1_pauli_list_diag.append((''.join(op), (-1) ** (b) * F_p_1[j,j].real))
        op = n * ['I']
        op[j%n] = 'Z'
        F_x_1_pauli_list_diag.append((''.join(op), (-1) ** (c) * F_x_1[j,j].real))
        F_p_1_pauli_list_diag.append((''.join(op), (-1) ** (c) * F_p_1[j,j].real))
        op = n * ['I']
        op[j%n] = 'Z'
        op[(j-1)%n] = 'Z'
        F_x_1_pauli_list_diag.append((''.join(op), (-1) ** (c + b) * F_x_1[j,j].real))
        F_p_1_pauli_list_diag.append((''.join(op), (-1) ** (c + b) * F_p_1[j,j].real))

    for j in range(N):
        if n - 1 <= j < N - 1:
            a = 1
        else:
            a = 0

        if 1 <= j <= N // 2:
            b = 1
        else:
            b = 0
                
        if j >= n:
            c = 1
        else:
            c = 0

        # Imag part
        op = n * ['I']
        op[j%n] = 'Y'
        if j % 2 == 0:
            F_x_2_pauli_list_even.append((''.join(op), ((-1) ** c / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_even.append((''.join(op), ((-1) ** c / 4) * F_p_2[(j+1)%N,j].imag))
        else:
            F_x_2_pauli_list_odd.append((''.join(op), ((-1) ** c / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_odd.append((''.join(op), ((-1) ** c / 4) * F_p_2[(j+1)%N,j].imag))

        op = n * ['I']
        op[j%n] = 'Y'
        op[(j-1)%n] = 'Z'
        if j % 2 == 0:
            F_x_2_pauli_list_even.append((''.join(op), (- (-1) ** (a+c) / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_even.append((''.join(op), (- (-1) ** (a+c) / 4) * F_p_2[(j+1)%N,j].imag))
        else:
            F_x_2_pauli_list_odd.append((''.join(op), (- (-1) ** (a+c) / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_odd.append((''.join(op), (- (-1) ** (a+c) / 4) * F_p_2[(j+1)%N,j].imag))

        op = n * ['I']
        op[j%n] = 'Y'
        op[(j+1)%n] = 'Z'
        if j % 2 == 0:
            F_x_2_pauli_list_even.append((''.join(op), (- (-1) ** (b+c) / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_even.append((''.join(op), (- (-1) ** (b+c) / 4) * F_p_2[(j+1)%N,j].imag))
        else:
            F_x_2_pauli_list_odd.append((''.join(op), (- (-1) ** (b+c) / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_odd.append((''.join(op), (- (-1) ** (b+c) / 4) * F_p_2[(j+1)%N,j].imag))

        op = n * ['I']
        op[(j-1)%n] = 'Z'
        op[j%n] = 'Y'
        op[(j+1)%n] = 'Z'
        if j % 2 == 0:
            F_x_2_pauli_list_even.append((''.join(op), ((-1) ** (a+b+c) / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_even.append((''.join(op), ((-1) ** (a+b+c) / 4) * F_p_2[(j+1)%N,j].imag))
        else:
            F_x_2_pauli_list_odd.append((''.join(op), ((-1) ** (a+b+c) / 4) * F_x_2[(j+1)%N,j].imag))
            F_p_2_pauli_list_odd.append((''.join(op), ((-1) ** (a+b+c) / 4) * F_p_2[(j+1)%N,j].imag))



    for j in range(N):
        # Diagonal part
        op = n * ['I']
        D_x_pauli_list.append((''.join(op), D_x[j,j].real))
        D_p_pauli_list.append((''.join(op), D_p[j,j].real))
        op = n * ['I']
        op[(j-1)%n] = 'Z'
        D_x_pauli_list.append((''.join(op), (-1) ** (b) * D_x[j,j].real))
        D_p_pauli_list.append((''.join(op), (-1) ** (b) * D_p[j,j].real))
        op = n * ['I']
        op[j%n] = 'Z'
        D_x_pauli_list.append((''.join(op), (-1) ** (c) * D_x[j,j].real))
        D_p_pauli_list.append((''.join(op), (-1) ** (c) * D_p[j,j].real))
        op = n * ['I']
        op[j%n] = 'Z'
        op[(j-1)%n] = 'Z'
        D_x_pauli_list.append((''.join(op), (-1) ** (c + b) * D_x[j,j].real))
        D_p_pauli_list.append((''.join(op), (-1) ** (c + b) * D_p[j,j].real))


    F_x_1_pauli_op_grouped = [SparsePauliOp.from_list(F_x_1_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_x_1_pauli_list_odd).simplify(),
                                SparsePauliOp.from_list(F_x_1_pauli_list_diag).simplify()]
    F_x_2_pauli_op_grouped = [SparsePauliOp.from_list(F_x_2_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_x_2_pauli_list_odd).simplify()]
    F_p_1_pauli_op_grouped = [SparsePauliOp.from_list(F_p_1_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_p_1_pauli_list_odd).simplify(),
                                SparsePauliOp.from_list(F_p_1_pauli_list_diag).simplify()]
    F_p_2_pauli_op_grouped = [SparsePauliOp.from_list(F_p_2_pauli_list_even).simplify(),
                                SparsePauliOp.from_list(F_p_2_pauli_list_odd).simplify()]
    
    D_x_pauli_op = SparsePauliOp.from_list(D_x_pauli_list).simplify()
    D_p_pauli_op = SparsePauliOp.from_list(D_p_pauli_list).simplify()
    H_F_pauli_op = (-get_xi_pauli_op(n_p, R))
    # Id_n_p = SparsePauliOp.from_list([(n_p * 'I', 1)])

    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_x_1)
    F_x_1_sp_mats = [sp_mat_even, sp_mat_odd, sp_mat_diag]
    
    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_p_1)
    F_p_1_sp_mats = [sp_mat_even, sp_mat_odd, sp_mat_diag]
    
    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_x_2)
    F_x_2_sp_mats = [sp_mat_even, sp_mat_odd]
    
    sp_mat_even, sp_mat_odd, sp_mat_diag = decompose_tridiag_mat(F_p_2)
    F_p_2_sp_mats = [sp_mat_even, sp_mat_odd]

    # Store the Pauli decomposition separately for the Hermitian and anti-Hermitian parts,
    #  (without the ancilla register for Schrodingerization) to estimate the circuit depth per Trotter step
    # Note that for the anti-Hermitian part, we can fan-out the Schrodingerization ancilla register
    H_herm, H_antiherm = [], []
    # Store as sparse matrices (with the ancilla register for Schrodingerization), to estimate the Trotter steps
    H_herm_sp_mats, H_antiherm_sp_mats = [], []

    # Hermitian part
    for j in range(len(F_x_1_pauli_op_grouped)):
        H_herm.append(F_x_1_pauli_op_grouped[j].tensor(D_p_pauli_op))
        H_herm_sp_mats.append(tensor([F_x_1_sp_mats[j], D_p, H_F_pauli_op.to_matrix(sparse=True)]))
                
    for j in range(len(F_p_1_pauli_op_grouped)):
        H_herm.append(D_x_pauli_op.tensor(F_p_1_pauli_op_grouped[j]))
        H_herm_sp_mats.append(tensor([D_x, F_p_1_sp_mats[j], H_F_pauli_op.to_matrix(sparse=True)]))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_op_grouped)):
        H_antiherm.append(F_x_2_pauli_op_grouped[j].tensor(D_p_pauli_op))
        H_antiherm_sp_mats.append(-tensor([F_x_2_sp_mats[j], D_p, identity(N_p)]))
    for j in range(len(F_p_2_pauli_op_grouped)):
        H_antiherm.append(D_x_pauli_op.tensor(F_p_2_pauli_op_grouped[j]))
        H_antiherm_sp_mats.append(-tensor([D_x, F_p_2_sp_mats[j], identity(N_p)]))

    return H_herm, H_antiherm, H_herm_sp_mats, H_antiherm_sp_mats
    
def get_circuit_depth(n, n_p, H_herm, H_antiherm):
    depth = 0

    # First do Hermitian part (need fanout)
    for j in range(len(H_herm)):
        circuit = QuantumCircuit(2 * n + n_p)
        for pauli_op in H_herm[j]:
            circuit.append(LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op)), qargs=np.arange(2 * n).tolist())
        # Do a controlled version

        compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
        depth += parallelize_ctrl_circuit(compiled_circuit)
    
    # Next do anti-Hermitian part (just simulate normally)
    for j in range(len(H_antiherm)):
        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(H_antiherm))
        compiled_circ = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
        # multiply by n_p since we need to do this for every qubit
        depth += n_p * compiled_circ.depth(lambda instr: len(instr.qubits) > 1)

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
    error_tol = 5e-2
    n_vals = np.arange(2, 7)
    N_vals = 2 ** n_vals                        # grid points along each dimension
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    trotter_method="first_order"

    print("Error tolerance:", error_tol, flush=True)

    pauli_basis_trotter_steps = np.zeros_like(N_vals)
    pauli_basis_single_qubit_gates = np.zeros_like(N_vals)
    pauli_basis_two_qubit_gates = np.zeros_like(N_vals)
    pauli_basis_circ_depth = np.zeros_like(N_vals)

    one_hot_trotter_steps = np.zeros_like(N_vals)
    one_hot_single_qubit_gates = np.zeros_like(N_vals)
    one_hot_two_qubit_gates = np.zeros_like(N_vals)
    one_hot_circ_depth = np.zeros_like(N_vals)

    unary_trotter_steps = np.zeros_like(N_vals)
    unary_single_qubit_gates = np.zeros_like(N_vals)
    unary_two_qubit_gates = np.zeros_like(N_vals)
    unary_circ_depth = np.zeros_like(N_vals)

    for i, N in enumerate(N_vals):
        n_x = n_vals[i]
        # Gate counts per Trotter step
        H_std_binary_herm, H_std_binary_antiherm, H_std_binary_sp_mats_herm, H_std_binary_sp_mats_antiherm = get_H_std_binary(N, n_p, R)
        print("Computing Trotter steps for Pauli basis.", flush=True)
        pauli_basis_trotter_steps[i] = get_first_order_trotter_steps(T, H_std_binary_sp_mats_herm + H_std_binary_sp_mats_antiherm, error_tol)
        print("Computing gates per Trotter step for Pauli basis.", flush=True)
        # pauli_basis_single_qubit_gates[i], pauli_basis_two_qubit_gates[i] = get_gate_count_per_trotter_step(H_std_binary, trotter_method)
        pauli_basis_circ_depth[i] = get_circuit_depth(n_x, n_p, H_std_binary_herm, H_std_binary_antiherm)
        
        '''One-hot encoding (ours)'''
        H_one_hot_herm, H_one_hot_antiherm, H_one_hot_sp_mats_herm, H_one_hot_sp_mats_antiherm = get_H_one_hot(N, n_p, R)
        print("Computing Trotter steps for one-hot encoding.", flush=True)
        one_hot_trotter_steps[i] = get_first_order_trotter_steps(T, H_one_hot_sp_mats_herm + H_one_hot_sp_mats_antiherm, error_tol)
        print("Computing gates per Trotter step for one-hot.", flush=True)
        # one_hot_single_qubit_gates[i], one_hot_two_qubit_gates[i] = get_gate_count_per_trotter_step(H_one_hot, trotter_method)
        one_hot_circ_depth[i] = get_circuit_depth(N, n_p, H_one_hot_herm, H_one_hot_antiherm)

        '''Unary encoding (ours)'''
        H_unary_herm, H_unary_antiherm, H_unary_sp_mats_herm, H_unary_sp_mats_antiherm = get_H_unary(N, n_p, R)
        print("Computing Trotter steps for unary encoding.", flush=True)
        unary_trotter_steps[i] = get_first_order_trotter_steps(T, H_unary_sp_mats_herm + H_unary_sp_mats_antiherm, error_tol)
        print("Computing gates per Trotter step for unary.", flush=True)
        # unary_single_qubit_gates[i], unary_two_qubit_gates[i] = get_gate_count_per_trotter_step(H_unary, trotter_method)
        unary_circ_depth[i] = get_circuit_depth((N // 2), n_p, H_unary_herm, H_unary_antiherm)

        np.savez(join("../resource_analysis_data", "nonlinear", f"nonlinear_data.npz"),
                N_vals=N_vals[:i+1],
                pauli_basis_trotter_steps=pauli_basis_trotter_steps[:i+1],
                # pauli_basis_single_qubit_gates=pauli_basis_single_qubit_gates[:i+1],
                # pauli_basis_two_qubit_gates=pauli_basis_two_qubit_gates[:i+1],
                pauli_basis_circ_depth=pauli_basis_circ_depth[:i+1],
                one_hot_trotter_steps=one_hot_trotter_steps[:i+1],
                # one_hot_single_qubit_gates=one_hot_single_qubit_gates[:i+1],
                # one_hot_two_qubit_gates=one_hot_two_qubit_gates[:i+1],
                one_hot_circ_depth=one_hot_circ_depth[:i+1],
                unary_trotter_steps=unary_trotter_steps[:i+1],
                # unary_single_qubit_gates=unary_single_qubit_gates[:i+1],
                # unary_two_qubit_gates=unary_two_qubit_gates[:i+1],
                unary_circ_depth=unary_circ_depth[:i+1])

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)
    print("Memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)