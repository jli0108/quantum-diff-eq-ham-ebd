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
    N_vals_one_hot = np.arange(4, 257, 4)
    n_p = 5  # num qubits for p
    N_p = 2**n_p
    T = 1
    R = 8
    num_samples = 1000
    num_jobs = 16
    trotter_method = "second_order"

    print("Computing Trotter steps.")
    one_hot_circ_depth = np.zeros_like(N_vals_one_hot)
    one_hot_two_qubit_gates = np.zeros_like(N_vals_one_hot)
    one_hot_trotter_steps = np.zeros_like(N_vals_one_hot)

    for i, N in enumerate(N_vals_one_hot):
        print(
            f"Estimating gate counts for dimension {dimension}, error_tol={error_tol:0.2e}",
            flush=True,
        )

        """One-hot encoding (ours)"""
        one_hot_circ_depth[i], one_hot_two_qubit_gates[i] = (
            one_hot_resources_per_trotter_step(N, n_p, R, trotter_method)
        )
        one_hot_trotter_steps[i] = get_trotter_number_one_hot_or_unary(
            N, N_p, R, T, error_tol / dimension, num_samples, num_jobs
        )
        print("One-hot Trott~er steps:", one_hot_trotter_steps[i], flush=True)

        np.savez(
            join(
                "../resource_analysis_data",
                "2d_advection_upwind",
                "separable_one_hot_data.npz",
            ),
            N_vals_one_hot=N_vals_one_hot[: i + 1],
            one_hot_trotter_steps=one_hot_trotter_steps[: i + 1],
            one_hot_circ_depth=one_hot_circ_depth[: i + 1],
            one_hot_two_qubit_gates=one_hot_two_qubit_gates[: i + 1],
        )

    end_time = time()
    print(f"Runtime: {end_time - start_time}", flush=True)

    print("Finished!", flush=True)
