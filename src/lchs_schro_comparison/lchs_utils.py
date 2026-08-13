import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, operators
from qiskit import transpile
from qiskit_aer import Aer


def get_cdf(c_normalized):
    """Returns the cumulative distribution function for distribution c"""
    assert np.abs(np.sum(c_normalized) - 1) < 1e-5
    cdf = np.zeros_like(c_normalized)
    cdf[0] = c_normalized[0]
    for i in np.arange(1, len(c_normalized)):
        cdf[i] = cdf[i - 1] + c_normalized[i]
    return cdf


def sample_from_dist(cdf):
    """Returns a random index with probability proportional to the distribution according to cdf"""
    u = np.random.rand()
    return np.argmax(u < cdf)


def hamming_weight(bitstring):
    weight = 0
    for bit in bitstring:
        if bit == "1":
            weight += 1
    return weight


def diagonal_circuit(n, k, t, J, h, gamma_x=0, gamma_z=0, r=1):
    # Second order Trotter with r Trotter steps
    circuit = QuantumCircuit(n)
    dt = t / r

    # Start with Hadamard layer
    # for i in range(n):
    #     circuit.x(i)

    for _ in range(r):
        # Anti-Hermitian part
        for i in range(n):
            circuit.rx(-dt * gamma_x * k, i)
            circuit.rz(-dt * gamma_z * k, i)

        # TFIM part
        for i in range(n):
            circuit.rx(2 * dt * h, i)
        for i in range(n - 1):
            circuit.rzz(2 * dt * J, i, i + 1)

        # Anti-Hermitian part
        for i in range(n):
            circuit.rz(-dt * gamma_z * k, i)
            circuit.rx(-dt * gamma_x * k, i)

    return circuit


def off_diagonal_circuit(
    n, k1, k2, t, J, h, gamma_x=0, gamma_z=0, r=1, compute_real_part=True
):
    assert k1 != k2

    circuit = QuantumCircuit(n + 1)
    circuit.h(0)
    if not compute_real_part:
        circuit.rz(-np.pi / 2, 0)

    # Start with Hadamard layer
    # for i in range(n):
    #     circuit.x(1 + i)

    dt = t / r
    # Second-order Trotter
    for _ in range(r):
        # TFIM part
        for i in range(n):
            circuit.rx(dt * h, 1 + i)
        for i in range(n - 1):
            circuit.rzz(dt * J, 1 + i, 1 + i + 1)

        # Anti-Hermitian part
        a_z = -2 * dt * gamma_z * (k1 + k2) / 2
        b_z = -2 * dt * gamma_z * (k1 - k2) / 2
        for i in range(n):
            circuit.rz(-np.sign(gamma_z) * b_z, 0)
            circuit.rz(a_z, 1 + i)
            circuit.h(0)
            circuit.h(1 + i)
            circuit.rxx(b_z, 0, 1 + i)
            circuit.h(0)
            circuit.h(1 + i)

        a_x = -2 * dt * gamma_x * (k1 + k2) / 2
        b_x = -2 * dt * gamma_x * (k1 - k2) / 2
        for i in range(n):
            circuit.rz(-np.sign(gamma_x) * b_x, 0)
            circuit.rx(a_x, 1 + i)
            circuit.h(0)
            circuit.rxx(b_x, 0, 1 + i)
            circuit.h(0)

        # TFIM part
        for i in range(n):
            circuit.rx(dt * h, 1 + i)
        for i in range(n - 1):
            circuit.rzz(dt * J, 1 + i, 1 + i + 1)

    circuit.h(0)
    return circuit


def run_lchs(n_x, t, J, h, gamma_x, gamma_z, r, K, M, num_samples_lchs, num_shots_lchs):

    # LCHS functions
    k = np.linspace(-K, K, M, endpoint=False)
    c = 2 * K / (M * np.pi * (1 + k**2))
    cdf = get_cdf(c / np.sum(c))

    # LCHS
    sampled_indices = []
    obs_lchs = 0
    obs_sq_lchs = 0
    circ_depths = []
    for _ in range(num_samples_lchs):
        j1, j2 = sample_from_dist(cdf), sample_from_dist(cdf)
        sampled_indices.append((int(j1), int(j2)))

        if j1 == j2:
            """Diagonal terms"""
            circuit = diagonal_circuit(n_x, k[j1], t, J, h, gamma_x, gamma_z, r)
        else:
            """Off-diagonal terms"""
            circuit = off_diagonal_circuit(
                n_x, k[j1], k[j2], t, J, h, gamma_x, gamma_z, r
            )
        circuit.measure_all()
        # Run on Qiskit Aer simulator
        compiled_circuit = transpile(
            circuit, basis_gates=["rx", "ry", "rz", "rxx"], optimization_level=3
        )

        circ_depths.append(compiled_circuit.depth())
        # Run the quantum circuit on a statevector simulator backend
        backend = Aer.get_backend("statevector_simulator")
        job = backend.run(compiled_circuit, shots=num_shots_lchs)
        # job = backend.run(compiled_circuit)
        result = job.result()

        # Note: if sampling, this is the counts (not freq)
        freq = result.get_counts(compiled_circuit)

        if j1 == j2:

            for string in freq.keys():
                assert len(string) == n_x
                bitstring = np.binary_repr(int(string, 2), n_x)
                # Observable is Hamming weight

                obs_lchs += hamming_weight(bitstring) * freq[string]
                obs_sq_lchs += (hamming_weight(bitstring) ** 2) * freq[string]
                # normalization_observable[i] += freq[string] / num_shots_lchs

        else:
            # normalization_zero_obs, normalization_one_obs = 0, 0
            for string in freq.keys():
                assert len(string) == n_x + 1
                bitstring = np.binary_repr(int(string, 2), n_x + 1)
                if bitstring[-1] == "0":
                    obs_lchs += hamming_weight(bitstring[:n_x]) * freq[string]
                    obs_sq_lchs += (hamming_weight(bitstring[:n_x]) ** 2) * freq[string]
                    # normalization_zero_obs += freq[string] / num_shots_lchs
                else:
                    obs_lchs -= hamming_weight(bitstring[:n_x]) * freq[string]
                    obs_sq_lchs += (hamming_weight(bitstring[:n_x]) ** 2) * freq[string]
                    # normalization_one_obs += freq[string] / num_shots_lchs
            # normalization_observable[i] += (normalization_zero_obs - normalization_one_obs)

    print("max LCHS depth:", np.max(circ_depths))
    obs_lchs *= (np.sum(c) ** 2) / (num_samples_lchs * num_shots_lchs)
    obs_sq_lchs *= (np.sum(c) ** 2) ** 2 / (num_samples_lchs * num_shots_lchs)
    # normalization_observable *= (np.sum(c) ** 2) / experiment_info["num_samples"]
    # obs_lchs /= num_samples_lchs
    # normalization_observable /= num_samples_lchs
    var_lchs = obs_sq_lchs - obs_lchs**2
    return obs_lchs, var_lchs
