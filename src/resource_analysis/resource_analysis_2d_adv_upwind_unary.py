import sys
from os.path import join
from time import time

import numpy as np

from resource_estimate_utils import *

sys.path.append(join(".", ".."))
from utils import *

from resource_analysis_2d_adv_upwind_utils import *

if __name__ == "__main__":
    print("Running resource analysis for 2d advection (upwind)", flush=True)
    start_time = time()
    dimension = 2
    error_tol = 5e-2
    N_vals_unary = np.arange(4, 257, 4)
    n_p = 5  # num qubits for p
    N_p = 2**n_p
    T = 1
    R = 8
    num_samples = 1000
    num_jobs = 16
    trotter_method = "second_order"

    print("Computing Trotter steps.")
    unary_circ_depth = np.zeros_like(N_vals_unary)
    unary_two_qubit_gates = np.zeros_like(N_vals_unary)
    unary_trotter_steps = np.zeros_like(N_vals_unary)

    for i, N in enumerate(N_vals_unary):
        print(
            f"Estimating gate counts for dimension {dimension}, error_tol={error_tol:0.2e}",
            flush=True,
        )

        unary_circ_depth[i], unary_two_qubit_gates[i] = (
            unary_resources_per_trotter_step(N, n_p, R, trotter_method)
        )
        unary_trotter_steps[i] = get_trotter_number_one_hot_or_unary(
            N, N_p, R, T, error_tol / dimension, num_samples, num_jobs
        )
        print("Unary Trotter steps:", unary_trotter_steps[i], flush=True)

        np.savez(
            join(
                "../resource_analysis_data",
                "2d_advection_upwind",
                "separable_unary_data.npz",
            ),
            N_vals_unary=N_vals_unary[: i + 1],
            unary_trotter_steps=unary_trotter_steps[: i + 1],
            unary_circ_depth=unary_circ_depth[: i + 1],
            unary_two_qubit_gates=unary_two_qubit_gates[: i + 1],
        )

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)
