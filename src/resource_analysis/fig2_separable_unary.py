
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
    N_vals_unary = np.arange(4, 257, 2)
    n_p = 5                                     # num qubits for p
    N_p = 2 ** n_p
    T = 1
    R = 8
    num_samples = 1000
    num_jobs = 16
    trotter_method="second_order"


    print("Computing Trotter steps.")
    unary_single_qubit_gates = np.zeros_like(N_vals_unary)
    unary_two_qubit_gates = np.zeros_like(N_vals_unary)
    unary_circ_depth = np.zeros_like(N_vals_unary)
    unary_trotter_steps = np.zeros_like(N_vals_unary)

    for i, N in enumerate(N_vals_unary):
        print(f"Estimating gate counts for dimension {dimension}, error_tol={error_tol:0.2e}", flush=True)

        unary_single_qubit_gates[i], unary_two_qubit_gates[i], unary_circ_depth[i] = unary_gate_count_per_trotter_step(N, n_p, R, trotter_method)
        unary_trotter_steps[i] = get_trotter_number_one_hot_or_unary(N, N_p, R, T, error_tol / dimension, num_samples, num_jobs)
        print("Unary Trotter steps:", unary_trotter_steps[i], flush=True)

        np.savez(join("../resource_analysis_data", "fig2_separable_unary_data.npz"),
                dimension=dimension,
                error_tol=error_tol,
                unary_trotter_steps=unary_trotter_steps[:i+1],
                unary_single_qubit_gates=unary_single_qubit_gates[:i+1],
                unary_two_qubit_gates=unary_two_qubit_gates[:i+1],
                unary_circ_depth=unary_circ_depth[:i+1])

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)


