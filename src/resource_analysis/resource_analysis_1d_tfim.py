import numpy as np
from scipy.sparse.linalg import expm_multiply
from joblib import Parallel, delayed

from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk

from os.path import join, dirname
import sys
sys.path.append(join(".", ".."))
from utils import *
import argparse

def get_lchs_hamiltonian(n, J, h, gamma, k1, k2):

    pauli_op_list = []
    # Hermitian part
    for i in range(n):
        op = (n+1) * ['I']
        op[i] = 'X'
        pauli_op_list.append((''.join(op), h))
    for i in range(n-1):
        op = (n+1) * ['I']
        op[i] = 'Z'
        op[i+1] = 'Z'
        pauli_op_list.append((''.join(op), J))

    # Anti-Hermitian part
    for i in range(n):
        '''Controlled on \ket{0}'''
        op = (n+1) * ['I']
        op[i] = 'Z'
        op[n] = 'I'
        pauli_op_list.append((''.join(op), -k1 * gamma / 2))

        op = (n+1) * ['I']
        op[i] = 'I'
        op[n] = 'Z'
        pauli_op_list.append((''.join(op), k1 * gamma / 2))

        op = (n+1) * ['I']
        op[i] = 'Z'
        op[n] = 'Z'
        pauli_op_list.append((''.join(op), -k1 * gamma / 2))

        '''Controlled on \ket{1}'''
        op = (n+1) * ['I']
        op[i] = 'Z'
        op[n] = 'I'
        pauli_op_list.append((''.join(op), -k2 * gamma / 2))

        op = (n+1) * ['I']
        op[i] = 'I'
        op[n] = 'Z'
        pauli_op_list.append((''.join(op), -k2 * gamma / 2))

        op = (n+1) * ['I']
        op[i] = 'Z'
        op[n] = 'Z'
        pauli_op_list.append((''.join(op), k2 * gamma / 2))

    return pauli_op_list

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

def get_schrodingerization_hamiltonian(n, J, h, gamma, n_p, R):

    H_1_pauli_list = []
    H_2_pauli_list = []

    # Hermitian part
    for i in range(n):
        op = n * ['I']
        op[i] = 'X'
        H_2_pauli_list.append((''.join(op), -h))
    for i in range(n-1):
        op = n * ['I']
        op[i] = 'Z'
        op[i+1] = 'Z'
        H_2_pauli_list.append((''.join(op), -J))
    
    # Anti-Hermitian part
    for i in range(n):
        op = n * ['I']
        H_1_pauli_list.append((''.join(op), -gamma))
        op = n * ['I']
        op[i] = 'Z'
        H_1_pauli_list.append((''.join(op), gamma))

    xi_pauli_list = get_xi_pauli_op(n_p, R).to_list()

    pauli_op_list = []
    for i in range(len(H_1_pauli_list)):
        for j in range(len(xi_pauli_list)):
            pauli_op_list.append((H_1_pauli_list[i][0] + xi_pauli_list[j][0], -H_1_pauli_list[i][1] * xi_pauli_list[j][1]))
    for i in range(len(H_2_pauli_list)):
        pauli_op_list.append((H_2_pauli_list[i][0] + ''.join(n_p * ['I']), -H_2_pauli_list[i][1]))

    return pauli_op_list

def estimate_trotter_error(n, T, r, pauli_op, num_samples, num_jobs=16):
    # print(f"Estimating Trotter error w/ {num_jobs} jobs", flush=True)
    return max(Parallel(n_jobs=num_jobs)(delayed(estimate_trotter_error_one_sample)(n, T, r, pauli_op) for _ in range(num_samples)))

def estimate_trotter_error_one_sample(n, T, r, pauli_op):

    pauli_op_grouped = pauli_op.group_commuting()
    H = pauli_op.to_matrix(sparse=True)
    psi_0 = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
    psi_0 /= np.linalg.norm(psi_0)
    psi_true = expm_multiply(-1j * H * T, psi_0)

    # Compute Trotter error
    psi_trot = np.copy(psi_0)
    for _ in range(r):
        for j in range(len(pauli_op_grouped)):
            psi_trot = expm_multiply(-1j * pauli_op_grouped[j].to_matrix(sparse=True) * (T / (2 * r)), psi_trot)
        for j in range(len(pauli_op_grouped))[::-1]:
            psi_trot = expm_multiply(-1j * pauli_op_grouped[j].to_matrix(sparse=True) * (T / (2 * r)), psi_trot)

    error = np.linalg.norm(psi_true - psi_trot, ord=2)
        
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resource analysis for LCHS/Schrodingerization")
    parser.add_argument('algorithm', help="Algorithm (lchs or schrodingerization)")
    args = parser.parse_args()
    print("Algorithm:", args.algorithm)
    assert args.algorithm == "lchs" or args.algorithm == "schrodingerization"

    CURR_DIR = join(dirname(__file__), "..", "resource_analysis_data", "1d_tfim")

    T = 2
    J = 1
    h = 1
    gamma = 0.05
    R = 8

    error_tol = 5e-2
    num_samples = 1000
    num_jobs = 32
    trotter_method = "second_order"

    n_vals = np.arange(2, 17)

    if args.algorithm == "lchs":

        lchs_trotter_steps = np.zeros(len(n_vals), dtype=int)
        lchs_one_qubit_gate_count_per_trotter_step = np.zeros(len(n_vals), dtype=int)
        lchs_two_qubit_gate_count_per_trotter_step = np.zeros(len(n_vals), dtype=int)

        '''Resource analysis for LCHS'''
        print("Running resource analysis for LCHS")
        for i, n_x in enumerate(n_vals):

            print(f"n={n_x}")
            k1 = R
            k2 = -R
            pauli_op = SparsePauliOp.from_list(get_lchs_hamiltonian(n_x, J, h, gamma, k1, k2))

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
            while r_max * estimate_trotter_error(n_x + 1, T / r_max, 1, pauli_op, num_samples, num_jobs) > error_tol:
                r_max *= 2

            # binary search for r
            while r_max - r_min > 1:
                r = (r_min + r_max) // 2
                if r * estimate_trotter_error(n_x + 1, T / r, 1, pauli_op, num_samples, num_jobs) > error_tol:
                    r_min = r
                else:
                    r_max = r
            print(f"Trotter steps: {r_max}")

            lchs_one_qubit_gate_count_per_trotter_step[i] = num_single_qubit_gates
            lchs_two_qubit_gate_count_per_trotter_step[i] = num_two_qubit_gates
            lchs_trotter_steps[i] = r_max

            np.savez(join(CURR_DIR, "lchs.npz"),
                    n_vals=n_vals[:i+1],
                    lchs_trotter_steps=lchs_trotter_steps[:i+1],
                    lchs_one_qubit_gate_count_per_trotter_step=lchs_one_qubit_gate_count_per_trotter_step[:i+1],
                    lchs_two_qubit_gate_count_per_trotter_step=lchs_two_qubit_gate_count_per_trotter_step[:i+1])

    elif args.algorithm == "schrodingerization":
            
        n_p = 5
        N_p = 2 ** n_p
        print(f"N_p={N_p}")

        schrodingerization_trotter_steps = np.zeros(len(n_vals), dtype=int)
        schrodingerization_one_qubit_gate_count_per_trotter_step = np.zeros(len(n_vals), dtype=int)
        schrodingerization_two_qubit_gate_count_per_trotter_step = np.zeros(len(n_vals), dtype=int)
        
        '''Resource analysis for Schrodingerization'''
        print("Running resource analysis for Schrodingerization")
        for i, n_x in enumerate(n_vals):

            print(f"n_x={n_x}", flush=True)
            pauli_op = SparsePauliOp.from_list(get_schrodingerization_hamiltonian(n_x, J, h, gamma, n_p, R))

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
            print(f"1q gates: {num_single_qubit_gates}, 2q gates: {num_two_qubit_gates}", flush=True)

            # Estimate number of Trotter steps required
            r_min, r_max = 1, 10
            while r_max * estimate_trotter_error(n_x + n_p, T / r_max, 1, pauli_op, num_samples, num_jobs) > error_tol:
                print(f"r_max = {r_max}", flush=True)
                r_max *= 2

            # binary search for r
            while r_max - r_min > 1:
                r = (r_min + r_max) // 2
                if r * estimate_trotter_error(n_x + n_p, T / r, 1, pauli_op, num_samples, num_jobs) > error_tol:
                    r_min = r
                else:
                    r_max = r
            print(f"Trotter steps: {r_max}", flush=True)

            schrodingerization_one_qubit_gate_count_per_trotter_step[i] = num_single_qubit_gates
            schrodingerization_two_qubit_gate_count_per_trotter_step[i] = num_two_qubit_gates
            schrodingerization_trotter_steps[i] = r_max

            np.savez(join(CURR_DIR, "schrodingerization.npz"),
                    n_p=n_p,
                    n_vals=n_vals[:i+1],
                    schrodingerization_trotter_steps=schrodingerization_trotter_steps[:i+1],
                    schrodingerization_one_qubit_gate_count_per_trotter_step=schrodingerization_one_qubit_gate_count_per_trotter_step[:i+1],
                    schrodingerization_two_qubit_gate_count_per_trotter_step=schrodingerization_two_qubit_gate_count_per_trotter_step[:i+1])