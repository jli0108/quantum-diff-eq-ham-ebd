
import numpy as np

from scipy.sparse import eye, lil_matrix, diags
from joblib import Parallel, delayed
from resource_estimate_utils import *
from os.path import join
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
# from pytket import OpType
# from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
# from pytket.extensions.qiskit import qiskit_to_tk

from os.path import join
import sys
sys.path.append(join(".", ".."))
from utils import *

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

    F_x = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        F_x[j,(j+1)%N] = 1
        F_x[j,j] = -1
    F_x /= h

    F_p = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        p = j / N
        F_p[j,(j+1)%N] = (p * (1-p) + 1)
        F_p[j,j] = - (p * (1-p) + 1)
    F_p /= h

    D_x = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        x = j / N
        D_x[j,j] = x * (1-x) + 1

    D_p = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        p = j / N
        D_p[j,j] = p


    F_x_1 = (F_x + np.conj(F_x.T)) / 2
    F_x_2 = (F_x - np.conj(F_x.T)) / 2j
    F_p_1 = (F_p + np.conj(F_p.T)) / 2
    F_p_2 = (F_p - np.conj(F_p.T)) / 2j

    F_x_1_pauli_list = SparsePauliOp.from_operator(F_x_1.toarray()).to_list()
    F_x_2_pauli_list = SparsePauliOp.from_operator(F_x_2.toarray()).to_list()
    F_p_1_pauli_list = SparsePauliOp.from_operator(F_p_1.toarray()).to_list()
    F_p_2_pauli_list = SparsePauliOp.from_operator(F_p_2.toarray()).to_list()
    D_x_pauli_list = SparsePauliOp.from_operator(D_x.toarray()).to_list()
    D_p_pauli_list = SparsePauliOp.from_operator(D_p.toarray()).to_list()
    H_F_pauli_list = (-get_xi_pauli_op(n_p, R)).to_list()

    H_pauli_list = []

    # Hermitian part
    for j in range(len(F_x_1_pauli_list)):
        for k in range(len(D_p_pauli_list)):
            for l in range(len(H_F_pauli_list)):
                H_pauli_list.append((F_x_1_pauli_list[j][0] + D_p_pauli_list[k][0] + H_F_pauli_list[l][0], (F_x_1_pauli_list[j][1] * D_p_pauli_list[k][1] * H_F_pauli_list[l][1]).real))
                
    for j in range(len(F_p_1_pauli_list)):
        for k in range(len(D_x_pauli_list)):
            for l in range(len(H_F_pauli_list)):
                H_pauli_list.append((F_p_1_pauli_list[j][0] + D_x_pauli_list[k][0] + H_F_pauli_list[l][0], (F_p_1_pauli_list[j][1] * D_x_pauli_list[k][1] * H_F_pauli_list[l][1]).real))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_list)):
        for k in range(len(D_p_pauli_list)):
            H_pauli_list.append((F_x_2_pauli_list[j][0] + D_p_pauli_list[k][0] + n_p * 'I', -(F_x_2_pauli_list[j][1] * D_p_pauli_list[k][1]).real))
    for j in range(len(F_p_2_pauli_list)):
        for k in range(len(D_x_pauli_list)):
            H_pauli_list.append((F_p_2_pauli_list[j][0] + D_x_pauli_list[k][0] + n_p * 'I', -(F_p_2_pauli_list[j][1] * D_x_pauli_list[k][1]).real))
    
    H = SparsePauliOp.from_list(H_pauli_list)
    return H

def get_H_one_hot(N, n_p, R):

    h = 1 / N

    F_x = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        F_x[j,(j+1)%N] = 1
        F_x[j,j] = -1
    F_x /= h

    F_p = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        p = j / N
        F_p[j,(j+1)%N] = (p * (1-p) + 1)
        F_p[j,j] = - (p * (1-p) + 1)
    F_p /= h

    D_x = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        x = j / N
        D_x[j,j] = x * (1-x) + 1

    D_p = lil_matrix((N, N), dtype=np.complex128)
    for j in range(N):
        p = j / N
        D_p[j,j] = p


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
    H_F_pauli_list = (-get_xi_pauli_op(n_p, R)).to_list()

    F_x_1_pauli_list.append((N * 'I', F_x_1[0,0].real))
    F_p_1_pauli_list.append((N * 'I', F_p_1[0,0].real))
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

    for j in range(N):
        D_x_pauli_list.append((N * 'I', 0.5 * D_x[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_x_pauli_list.append((''.join(op), -0.5 * D_x[j,j].real))

        D_p_pauli_list.append((N * 'I', 0.5 * D_p[j,j].real))
        op = N * ['I']
        op[j] = 'Z'
        D_p_pauli_list.append((''.join(op), -0.5 * D_p[j,j].real))


    H_pauli_list = []

    # Hermitian part
    for j in range(len(F_x_1_pauli_list)):
        for k in range(len(D_p_pauli_list)):
            for l in range(len(H_F_pauli_list)):
                H_pauli_list.append((F_x_1_pauli_list[j][0] + D_p_pauli_list[k][0] + H_F_pauli_list[l][0], (F_x_1_pauli_list[j][1] * D_p_pauli_list[k][1] * H_F_pauli_list[l][1]).real))
                
    for j in range(len(F_p_1_pauli_list)):
        for k in range(len(D_x_pauli_list)):
            for l in range(len(H_F_pauli_list)):
                H_pauli_list.append((F_p_1_pauli_list[j][0] + D_x_pauli_list[k][0] + H_F_pauli_list[l][0], (F_p_1_pauli_list[j][1] * D_x_pauli_list[k][1] * H_F_pauli_list[l][1]).real))
    
    # Anti-Hermitian part
    for j in range(len(F_x_2_pauli_list)):
        for k in range(len(D_p_pauli_list)):
            H_pauli_list.append((F_x_2_pauli_list[j][0] + D_p_pauli_list[k][0] + n_p * 'I', -(F_x_2_pauli_list[j][1] * D_p_pauli_list[k][1]).real))
    for j in range(len(F_p_2_pauli_list)):
        for k in range(len(D_x_pauli_list)):
            H_pauli_list.append((F_p_2_pauli_list[j][0] + D_x_pauli_list[k][0] + n_p * 'I', -(F_p_2_pauli_list[j][1] * D_x_pauli_list[k][1]).real))
    
    H = SparsePauliOp.from_list(H_pauli_list)
    return H

# def get_gate_count_per_trotter_step_std_binary(H, trotter_method="second_order"):
#     '''Constructs the full circuit for a single Trotter step'''
#     # Compute number of gates per Trotter step
#     if trotter_method == "first_order" or trotter_method == "randomized_first_order":
#         circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(H.simplify()))
#     elif trotter_method == "second_order":
#         circuit = SuzukiTrotter(order=2, reps=1).synthesize(PauliEvolutionGate(H.simplify()))
#     else:
#         raise ValueError(f"{trotter_method} not supported")

#     compiled_circuit = transpile(circuit, basis_gates=['rxx', 'rx', 'ry', 'rz'], optimization_level=3)
#     print(compiled_circuit.count_ops())

#     tket_circuit = qiskit_to_tk(compiled_circuit)
#     gateset = {OpType.Rx, OpType.Ry, OpType.Rz, OpType.XXPhase}
#     rebase = auto_rebase_pass(gateset) 
#     comp = SequencePass([FullPeepholeOptimise(), CommuteThroughMultis(), RemoveRedundancies(), rebase])
#     comp.apply(tket_circuit)

#     # Gates per Trotter step
#     num_single_qubit_gates, num_two_qubit_gates = tket_circuit.n_1qb_gates(), tket_circuit.n_2qb_gates()

#     return num_single_qubit_gates, num_two_qubit_gates

def get_gate_count_per_trotter_step(H, trotter_method="second_order"):
    num_single_qubit_gates, num_two_qubit_gates = 0, 0
    L = len(H)
    for j in range(L):
        if j % 1000 == 0:
            print(f"Getting gate counts per Trotter step: [{j}/{L}]", end="\r")

        circuit = LieTrotter(reps=1).synthesize(PauliEvolutionGate(H[j]))
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
        return 2 * num_single_qubit_gates, num_two_qubit_gates

def accumulator_sum(generator):
    result = 0
    for value in generator:
        result += value
    return result

def get_first_order_trotter_steps(t, H, epsilon, n_jobs=16):
    L = len(H)
    res = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(get_comm_term_first_order)(H, L, j) for j in range(L-1))
    coeff = accumulator_sum(res)

    return np.ceil((t ** 2 * coeff / (2 * epsilon)))

def get_comm_term_first_order(H, L, j):
    comm_term = 0
    for k in np.arange(j+1, L):
        comm_term += commutator(H[k], H[j])
    comm_term.simplify()
    return np.linalg.norm(comm_term.coeffs, ord=1)


def get_second_order_trotter_steps(t, H, epsilon, n_jobs=16):
    L = len(H)
    res = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(get_comm_term_second_order)(H, L, j) for j in range(L-1))
    coeff = accumulator_sum(res)

    return np.ceil(np.sqrt(t ** 3 * coeff / (12 * epsilon)))

def get_comm_term_second_order(H, L, j):
    comm_term = 0
    for k in np.arange(j+1, L):
        for l in np.arange(j+1, L):
            comm_term += commutator(H[l], commutator(H[k], H[j]))
    comm_term.simplify()
    coeff1 = np.linalg.norm(comm_term.coeffs, ord=1)

    comm_term = 0
    for k in np.arange(j+1, L):
        comm_term += commutator(H[j], commutator(H[j], H[k]))
    comm_term.simplify()
    coeff2 = 0.5 * np.linalg.norm(comm_term.coeffs, ord=1)

    return coeff1 + coeff2


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

    print("Computing Trotter steps.")
    pauli_basis_trotter_steps = np.zeros((len(error_tols)))
    one_hot_trotter_steps = np.zeros((len(error_tols)))

    # Gate counts per Trotter step
    H_std_binary = get_H_std_binary(N, n_p, R)
    H_one_hot = get_H_one_hot(N, n_p, R)
    pauli_basis_single_qubit_gates, pauli_basis_two_qubit_gates = get_gate_count_per_trotter_step(H_std_binary, trotter_method)
    one_hot_single_qubit_gates, one_hot_two_qubit_gates = get_gate_count_per_trotter_step(H_one_hot, trotter_method)

    '''Schrodingerization w/ Pauli basis'''
    pauli_basis_trotter_steps = get_first_order_trotter_steps(T, H_std_binary, error_tols)

    '''One-hot encoding (ours)'''
    one_hot_trotter_steps = get_first_order_trotter_steps(T, H_one_hot, error_tols)

    np.savez(join("../resource_analysis_data", f"nonlinear_data_{trotter_method}.npz"),
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


