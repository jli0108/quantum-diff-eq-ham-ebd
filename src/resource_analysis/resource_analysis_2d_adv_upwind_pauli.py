
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
from resource_analysis_2d_adv_upwind_utils import *

if __name__ == "__main__":

    print("Running resource analysis for 2d advection (upwind)", flush=True)
    start_time = time()
    dimension = 2
    error_tol = 5e-2
    n_vals_binary = np.arange(2, 9)
    N_vals_binary = 2 ** n_vals_binary
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    num_samples = 1000
    num_jobs = 16
    trotter_method="second_order"


    print("Computing Trotter steps.")
    pauli_basis_circ_depth = np.zeros_like(N_vals_binary)
    pauli_basis_trotter_steps = np.zeros_like(N_vals_binary)

    for i, N in enumerate(N_vals_binary):
        print(f"Estimating gate counts for dimension {dimension}, error_tol={error_tol:0.2e}", flush=True)

        '''Schrodingerization w/ Pauli basis'''
        pauli_basis_circ_depth[i] = pauli_basis_depth_per_trotter_step(n_vals_binary[i], n_p, R, T, trotter_method)
        pauli_basis_trotter_steps[i] = get_trotter_number_pauli_basis(n_vals_binary[i], n_p, R, T, error_tol / dimension, num_samples, num_jobs)
        print("Pauli basis Trotter steps:", pauli_basis_trotter_steps[i], flush=True)

        np.savez(join("../resource_analysis_data", "2d_advection_upwind", "separable_pauli_data.npz"),
                n_vals_binary=n_vals_binary[:i+1],
                N_vals_binary=N_vals_binary[:i+1],
                pauli_basis_trotter_steps=pauli_basis_trotter_steps[:i+1],
                pauli_basis_circ_depth=pauli_basis_circ_depth[:i+1])

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)


