import numpy as np

from scipy.sparse.linalg import expm_multiply, expm, norm
# from braket.circuits import Circuit

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator

from joblib import Parallel, delayed
import sys
from os.path import join, dirname
sys.path.append(join(dirname(__file__), ".."))
from utils import *

def get_gate_counts(ops):
    num_single_qubit_gates = 0
    num_two_qubit_gates = 0
    single_qubit_gates = ['rx', 'ry', 'rz', 'h']
    two_qubit_gates = ['rxx', 'ryy', 'rzz']

    for gate in single_qubit_gates:
        if gate in ops:
            num_single_qubit_gates += ops[gate]
    
    for gate in two_qubit_gates:
        if gate in ops:
            num_two_qubit_gates += ops[gate]
    
    return num_single_qubit_gates, num_two_qubit_gates

def commutator(A, B):
    return A @ B - B @ A

def get_randomized_trotter_error(H, t, r):
    L = len(H)
    lamb = np.max(np.abs([np.linalg.norm(H[j].simplify().coeffs, ord=1) for j in range(L)]))
    return ((lamb * t * L) ** 4 / (r ** 3)) * np.exp(2 * lamb * t * L / r) + 2 * ((lamb * t * L) ** 3 / (3 * r ** 2)) * np.exp(lamb * t * L / r)

def get_trotter_number(pauli_op, t, epsilon, trotter_method):
    '''Uses analytical bound to compute the Trotter number to reach error threshold epsilon'''
    H = pauli_op.group_commuting()
    L = len(H)
    error = 0
    if trotter_method == "first_order":
        for j in range(L):
            for k in np.arange(j + 1, L):
                error += np.linalg.norm(commutator(H[k], H[j]).simplify().coeffs, ord=1)
        
        return max(1, int(np.ceil((t ** 2 / (2 * epsilon)) * error)))
    elif trotter_method == "second_order":    
        for j in range(L):
            for k in np.arange(j+1, L):
                for l in np.arange(j+1, L):
                    error += np.linalg.norm(commutator(H[l], commutator(H[k], H[j])).simplify().coeffs, ord=1)

                error += 0.5 * np.linalg.norm(commutator(H[j], commutator(H[j], H[k])).simplify().coeffs, ord=1)

        return max(1, int(np.ceil(np.sqrt((t ** 3 / (12 * epsilon)) * error))))
    elif trotter_method == "randomized_first_order":
        r_min, r_max = 1, 10

        while get_randomized_trotter_error(H, t, r_max) > epsilon:
            r_max *= 2

        # binary search for r
        while r_max - r_min > 1:
            r = (r_min + r_max) // 2
            if get_randomized_trotter_error(H, t, r) > epsilon:
                r_min = r
            else:
                r_max = r
        return r_max
    else:
        raise ValueError(f"{trotter_method} not supported")
    
def std_bin_trotter_fidelity(H, r):
    pauli_op = SparsePauliOp.from_operator(H / r)  

    circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(pauli_op))

    U_one_trotter_layer = Operator(circuit).data
    U_trotter = np.linalg.matrix_power(U_one_trotter_layer, r)
    U_exact = expm(-1j * H)
    return np.abs((U_exact @ np.conj(U_trotter.T)).trace()) / H.shape[0]
    
def std_bin_trotter_error_one_sample(H, pauli_op, t, r, trotter_method):
    pauli_op_grouped = pauli_op.group_commuting()
    psi = np.random.randn(H.shape[0]) + 1j * np.random.randn(H.shape[0])
    psi /= np.linalg.norm(psi)

    psi_no_trotter = expm_multiply(-1j * H * t, psi)
    psi_trotter = psi

    dt = t / r

    if trotter_method == "first_order":
        for _ in range(r):
            for j in range(len(pauli_op_grouped)):
                H_j = pauli_op_grouped[j]
                psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) * dt, psi_trotter)

    elif trotter_method == "second_order":
        for _ in range(r):
            for j in range(len(pauli_op_grouped)):
                H_j = pauli_op_grouped[j]
                psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) * dt / 2, psi_trotter)
            for j in range(len(pauli_op_grouped))[::-1]:
                H_j = pauli_op_grouped[j]
                psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) * dt / 2, psi_trotter)
    elif trotter_method == "randomized_first_order":
        np.random.seed(int(t * r))
        for _ in range(r):
            if np.random.rand() < 0.5:
                for j in range(len(pauli_op_grouped)):
                    H_j = pauli_op_grouped[j]
                    psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) * dt, psi_trotter)
            else:
                for j in range(len(pauli_op_grouped))[::-1]:
                    H_j = pauli_op_grouped[j]
                    psi_trotter = expm_multiply(-1j * H_j.to_matrix(sparse=True) * dt, psi_trotter)
    else:
        raise ValueError(f"{trotter_method} not supported")
        
    error = np.linalg.norm(psi_no_trotter - psi_trotter, ord=2)
    return error

def std_bin_trotter_error_sampling(H, pauli_op, t, r, trotter_method, num_samples, num_jobs):
    '''Uses sampling to compute the Trotter error'''

    res = Parallel(n_jobs=num_jobs)(delayed(std_bin_trotter_error_one_sample)(H, pauli_op, t, r, trotter_method) for _ in range(num_samples))
  
    return max(res)

def subspace_fidelity(n, t, H_ebd, H, codewords):
    U_exact = expm(-1j * H * t)
    U_subspace = expm(-1j * H_ebd * t)[codewords][:,codewords]
    return np.abs((U_exact @ np.conj(U_subspace.T)).trace()) / U_exact.shape[0]

def subspace_trotter_error(A, B, T, r, n, encoding):

    U_exact = expm(-1j * csc_matrix(A+B) * T)

    U_one_trotter_layer = expm(-1j * csc_matrix(A) * (T / r)) @ expm(-1j * csc_matrix(B) * (T / r))

    U_trotter = U_one_trotter_layer ** r
    return subspace_error(U_exact, U_trotter, n, encoding)

def subspace_error(U1, U2, n, encoding):
    
    diff = U1 - U2
    codewords = get_codewords_1d(n, encoding=encoding, periodic=False)
    diff_subspace = diff[codewords][:,codewords]

    if encoding == "one-hot":
        assert diff_subspace.shape == (n, n)
    
    return norm(diff_subspace, ord=2)

def get_gate_counts(braket_circuit):

    one_qubit_gates = 0
    two_qubit_gates = 0

    for instruction in braket_circuit.instructions:
        if len(instruction.target) == 1:
            one_qubit_gates += 1
        elif len(instruction.target) == 2:
            two_qubit_gates += 1
        else:
            raise ValueError("Error counting gates")

    assert one_qubit_gates + two_qubit_gates == len(braket_circuit.instructions)

    return one_qubit_gates, two_qubit_gates