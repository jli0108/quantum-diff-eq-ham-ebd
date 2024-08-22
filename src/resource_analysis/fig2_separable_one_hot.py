
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
from pytket import OpType
from pytket.passes import RemoveRedundancies, CommuteThroughMultis, SequencePass, FullPeepholeOptimise, auto_rebase_pass
from pytket.extensions.qiskit import qiskit_to_tk

from qclib.gates.ldmcsu import Ldmcsu

from os.path import join
import sys
sys.path.append(join(".", ".."))
from utils import *
from fig2_separable_utils import *

if __name__ == "__main__":

    print("Running Fig 2 script", flush=True)
    start_time = time()
    dimensions = np.arange(1, 10)
    error_tols = np.exp(-np.linspace(np.log(10), np.log(1000), 10))
    N = 128                                     # grid points along each dimension
    n_x = int(np.log2(N))
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    num_samples = 1000
    num_jobs = 16
    trotter_method="second_order"

    print("Dimensions:", dimensions, flush=True)
    print("Error tolerances:", error_tols, flush=True)

    print("Computing Trotter steps.")
    # pauli_basis_trotter_steps = np.zeros((len(dimensions), len(error_tols)))
    # bell_basis_trotter_steps = np.zeros((len(dimensions), len(error_tols)))
    one_hot_trotter_steps = np.zeros((len(dimensions), len(error_tols)))

    # Gate counts per Trotter step
    # pauli_basis_single_qubit_gates, pauli_basis_two_qubit_gates = pauli_basis_gate_count_per_trotter_step(n_x, n_p, R, T)
    # bell_basis_single_qubit_gates, bell_basis_two_qubit_gates = bell_basis_gate_count_per_trotter_step(n_x, n_p, R, T)
    one_hot_single_qubit_gates, one_hot_two_qubit_gates = one_hot_gate_count_per_trotter_step(N, n_p, R, trotter_method)

    for dim_idx, dimension in enumerate(dimensions):
        for error_tol_idx, error_tol in enumerate(error_tols):
            print(f"Estimating gate counts for dimension {dimension}, error_tol={error_tol:0.2e}", flush=True)

            # '''Schrodingerization w/ Pauli basis'''
            # pauli_basis_trotter_steps[dim_idx, error_tol_idx] = get_trotter_number_pauli_basis(n_x, n_p, R, T, error_tol / dimension, num_samples, num_jobs)
            # print("Pauli basis Trotter steps:", pauli_basis_trotter_steps[dim_idx, error_tol_idx], flush=True)

            # '''Schrodingerization w/ Bell basis'''
            # bell_basis_trotter_steps[dim_idx, error_tol_idx] = get_trotter_number_bell_basis(n_x, n_p, R, T, error_tol / dimension, num_samples, num_jobs)
            # print("Bell basis Trotter steps:", bell_basis_trotter_steps[dim_idx, error_tol_idx], flush=True)
            
            '''One-hot encoding (ours)'''
            one_hot_trotter_steps[dim_idx, error_tol_idx] = get_trotter_number_one_hot(N, N_p, R, T, error_tol / dimension, num_samples, num_jobs)
            print("One-hot Trotter steps:", one_hot_trotter_steps[dim_idx, error_tol_idx], flush=True)

    np.savez(join("../resource_analysis_data", "fig2_separable_one_hot_data.npz"),
            dimensions=dimensions,
            error_tols=error_tols,
            one_hot_trotter_steps=one_hot_trotter_steps,
            one_hot_single_qubit_gates=one_hot_single_qubit_gates,
            one_hot_two_qubit_gates=one_hot_two_qubit_gates)

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)


