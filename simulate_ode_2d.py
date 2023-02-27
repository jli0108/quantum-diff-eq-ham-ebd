import numpy as np
from scipy.sparse.linalg import expm_multiply
from utils import *
from os.path import join
from time import time

start = time()
DATA_DIR = "data_2d"

C6 = 862690 * 2 * np.pi # Rydberg interaction constant (MHz⋅μm^6)
R = 4.0 # Lattice scale / μm
Ω_0 = 0.25 # (Global) Rabi frequency (MHz)
ts = 4.0 # Simulation time (μs)

CURR_DIR = DATA_DIR
check_and_make_dir(CURR_DIR)

print("Running simulation of ODE/PDE solver on Rydberg atoms.")

# number of qubits
n = 11
# number of chains
d = 2
# number of grid points in along one axis
N = (n+1)
# size of interval
L = 1
# mesh spacing
h = L/(N-1)
# parameters for discretized delta
m = 1
omega = (m * h) ** (1/2)
q = np.linspace(0, L, N)
num_time_points = 64
times = np.linspace(0, ts, num_time_points)
print(f"Simulating n={n} atoms on each chain.")
print(f"Mesh resolution: h ={h: 0.3f}")

x0_tmp, x1_tmp = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N), indexing='ij')
X = np.stack([x0_tmp, x1_tmp], axis=-1)

# Initial condition in the codeword subspace
X_0 = np.array([0.3, 0.7])
psi_0_subspace = delta_multivar(X - X_0, omega)
psi_0_subspace /= np.linalg.norm(psi_0_subspace)

x0 = np.linspace(0, L, N)
x1 = np.linspace(0, L, N)

x0_midpoints = np.zeros(N-1)
x1_midpoints = np.zeros(N-1)
for i in range(N-1):
    x0_midpoints[i] = (x0[i] + x0[i+1]) / 2
    x1_midpoints[i] = (x1[i] + x1[i+1]) / 2

# Functions in the ODE problem
F0 = 0.5 * x0_midpoints + 0.2 * x1_midpoints
F1 = - 0.1 * x0_midpoints - 0.5 * x1_midpoints

# Rydberg atoms
chain_locations = np.zeros(n)
for k in range(n):
    chain_locations[k] = R * (n - 1 - k)

qubit_locations = np.concatenate([chain_locations, chain_locations])

detuning = np.zeros(n)
for i in range(n):
    if i % 2 == 0:
        for j in range(int((n-i)/2)):
            detuning[i] += C6 / np.linalg.norm(qubit_locations[i] - qubit_locations[i + (2 * j + 1)]) ** 6
        for j in range(int(i/2)):
            detuning[i] += C6 / np.linalg.norm(qubit_locations[i] - qubit_locations[i - (2 * (j + 1))]) ** 6
    else:
        for j in range(int((n-1-i)/2)):
            detuning[i] += C6 / np.linalg.norm(qubit_locations[i] - qubit_locations[i + (2 * (j + 1))]) ** 6
        for j in range(int((i+1)/2)):
            detuning[i] += C6 / np.linalg.norm(qubit_locations[i] - qubit_locations[i - (2 * j + 1)]) ** 6

# Ignore interactions between two chains
V = np.zeros((d * n, d * n))
for i in range(n):
    for j in range(i):
        V[i,j] = C6 / np.linalg.norm(qubit_locations[i] - qubit_locations[j]) ** 6
        V[i+n,j+n] = C6 / np.linalg.norm(qubit_locations[i] - qubit_locations[j]) ** 6

delta = np.concatenate([detuning, detuning])

alternating_sign = np.array([(-1)**i for i in range(n)])
rabi_freq = Ω_0 * np.concatenate([F0 * alternating_sign, F1 * alternating_sign]) / h
H = driving_term(d * n, rabi_freq) - sum_delta_n(d * n, delta) + sum_V_nn(d * n, V)

# Initialize the full psi_0
psi_0 = np.zeros(2 ** (d * n))
bitstring_1 = [k % 2 for k in range(n)]
for i in range(n+1):

    bitstring_2 = [k % 2 for k in range(n)]
    for j in range(n+1):

        full_bitstring = np.concatenate([bitstring_1, bitstring_2])
        psi_0[bitstring_to_int(full_bitstring)] = psi_0_subspace[i,j]

        if j < n:
            bitstring_2[j] = 1 - bitstring_2[j]

    if i < n:
        bitstring_1[i] = 1 - bitstring_1[i]

psi_0 /= np.linalg.norm(psi_0)

# faster but should require more memory to store state vector for all time points
psi = expm_multiply(-1j * H, psi_0, start=0, stop=ts, num=num_time_points)

psi_subspace = np.zeros((num_time_points,n+1,n+1), dtype=np.complex128)
bitstring_1 = [k % 2 for k in range(n)]
for i in range(n+1):

    bitstring_2 = [k % 2 for k in range(n)]
    for j in range(n+1):

        full_bitstring = np.concatenate([bitstring_1, bitstring_2])
        psi_subspace[:,i,j] = psi[:,bitstring_to_int(full_bitstring)]

        if j < n:
            bitstring_2[j] = 1 - bitstring_2[j]

    if i < n:
        bitstring_1[i] = 1 - bitstring_1[i]

filename = join(CURR_DIR, f"{n}_qubits.npz")
np.savez(filename,
    psi_subspace=psi_subspace,
    times=times,
    Ω_0=Ω_0
    )

runtime = time() - start

print(f"Runtime: {runtime / 3600} hours")