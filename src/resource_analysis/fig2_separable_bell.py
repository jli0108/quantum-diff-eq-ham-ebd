
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
    dimension = 2
    error_tol = 5e-2
    n_vals_bell_basis = np.arange(2, 9)
    N_vals_bell_basis = 2 ** n_vals_bell_basis
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    num_samples = 1000
    num_jobs = 16
    trotter_method="second_order"

    print("Computing Trotter steps.")
    bell_basis_single_qubit_gates = np.zeros_like(N_vals_bell_basis)
    bell_basis_two_qubit_gates = np.zeros_like(N_vals_bell_basis)
    bell_basis_circ_depth = np.zeros_like(N_vals_bell_basis)
    bell_basis_trotter_steps = np.zeros_like(N_vals_bell_basis)

    # Gate counts per Trotter step
    for i, n_x in enumerate(n_vals_bell_basis):
        bell_basis_single_qubit_gates[i], bell_basis_two_qubit_gates[i], bell_basis_circ_depth[i] = bell_basis_gate_count_per_trotter_step(n_x, n_p, R, T)
        '''Schrodingerization w/ Bell basis'''
        bell_basis_trotter_steps[i] = get_trotter_number_bell_basis(n_x, n_p, R, T, error_tol / dimension, num_samples, num_jobs)
        print("Bell basis Trotter steps:", bell_basis_trotter_steps[i], flush=True)
                

    np.savez(join("../resource_analysis_data", "2d_advection_upwind", "fig2_separable_bell_data.npz"),
            n_vals_bell_basis=n_vals_bell_basis[:i+1],
            N_vals_bell_basis=N_vals_bell_basis[:i+1],
            bell_basis_trotter_steps=bell_basis_trotter_steps[:i+1],
            bell_basis_single_qubit_gates=bell_basis_single_qubit_gates[:i+1],
            bell_basis_two_qubit_gates=bell_basis_two_qubit_gates[:i+1],
            bell_basis_circ_depth=bell_basis_circ_depth[:i+1])

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)


