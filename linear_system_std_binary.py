import numpy as np
from scipy.linalg import ishermitian, eigvals, eigvalsh
from scipy.sparse import kron, eye, diags, csc_matrix, hstack, vstack, random
from scipy.sparse.linalg import expm, expm_multiply, spsolve, norm
from utils import *

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.library import QFT


def schrodingerization(A, u_0, N, R, T, N_t, lambda_min, r=None):
    dimension = u_0.shape[0]
    h = (2 * R) / (N - 1)
    # Convert ODE to PDE
    H1 = (A + np.conj(A).T) / 2
    print("Eigvals of H1:", eigvalsh(H1.toarray()))
    # assert np.alltrue(np.linalg.eigvalsh(H1) <= 0)
    H2 = (A - np.conj(A).T) / 2j
    # print("H1:", H1.toarray())
    # print("H2:", H2.toarray())

    p_vals = np.linspace(-R, R, N)
    v_0 = kron(u_0, np.exp(-np.abs(p_vals))).toarray().flatten() # second axis is for p
    # Write out the Hamiltonian
    # centered differences
    H = 1j * (- kron(H1, (1 / (2 * h)) * diags([np.ones(N-1), -np.ones(N-1)], offsets=[1,-1])) + 1j * kron(H2, eye(N)))
    # Solve the PDE
    v = expm_multiply(-1j * H, v_0, start=0, stop=T, num=N_t).real

    # One issue: how to recover the amplitudes? (need to take absolute value?)
    # u_recover = np.sum(np.abs(np.reshape(v, newshape=(N_t, dimension, N))[:,:,N//2:]), axis=2) * h

    u_recover = np.zeros((N_t, dimension))
    for i, t in enumerate(np.linspace(0, T, N_t)):
        a = -lambda_min * t
        a_idx = N // 2 + int(a / h)
        # u_recover[i] = np.exp(a) * np.sum(np.reshape(v, newshape=(N_t, dimension, N))[i,:,a_idx:], axis=1) * h
        u_recover[i] = np.exp(a) * np.sum(np.reshape(np.abs(v), newshape=(N_t, dimension, N))[i,:,a_idx:], axis=1) * h
    
    return u_recover

def schrodingerization_ft(A, u_0, N, R, T, N_t, lambda_min, r=None):
    dimension = u_0.shape[0]
    h = (2 * R) / (N - 1)
    # Convert ODE to PDE
    H1 = (A + np.conj(A).T) / 2
    print("Eigvals of H1:", eigvalsh(H1.toarray()))
    # assert np.alltrue(np.linalg.eigvalsh(H1) <= 0)
    H2 = (A - np.conj(A).T) / 2j
    # print("H1:", H1.toarray())
    # print("H2:", H2.toarray())
    p = np.linspace(-R, R, N)
    v_0 = kron(u_0, np.fft.fft(np.exp(-np.abs(p)), norm="ortho")).toarray().flatten() # second axis is for p
    # v_0 = kron(u_0, np.sqrt(2 / np.pi) / (xi ** 2 + 1)).toarray().flatten() # second axis is for p
    # Write out the Hamiltonian
    H = - (kron(H1, -diags(np.fft.fftfreq(N, d = 1/(N * 2 * np.pi / (2 * R))))) + kron(H2, eye(N)))
    print(H.shape)
    # Solve the PDE
    v = expm_multiply(-1j * H, v_0, start=0, stop=T, num=N_t)
    
    IFFT = np.fft.ifft(np.eye(N), N, norm="ortho")
    v = ((kron(np.eye(dimension), IFFT) @ v.T).real).T
    
    # Issue: how to recover the amplitudes? (need to take absolute value?)
    u_recover = np.zeros((N_t, dimension))
    for i, t in enumerate(np.linspace(0, T, N_t)):
        a = -lambda_min * t
        a_idx = N // 2 + int(a / h)
        # u_recover[i] = np.exp(a) * np.sum(np.reshape(v, newshape=(N_t, dimension, N))[i,:,a_idx:], axis=1) * h
        u_recover[i] = np.exp(a) * np.sum(np.reshape(np.abs(v), newshape=(N_t, dimension, N))[i,:,a_idx:], axis=1) * h
    

    return u_recover

def schrodingerization_ft_trot(A, u_0, N, R, T, N_t, lambda_min, r):
    assert type(r) == int and r >= 1
    dimension = u_0.shape[0]
    n_p = int(np.log2(N))
    h = (2 * R) / (N - 1)
    # Convert ODE to PDE
    H1 = (A + np.conj(A).T) / 2
    print("Eigvals of H1:", eigvalsh(H1.toarray()))
    # assert np.alltrue(np.linalg.eigvalsh(H1) <= 0)
    H2 = (A - np.conj(A).T) / 2j
    # print("H1:", H1.toarray())
    # print("H2:", H2.toarray())
    p = np.linspace(-R, R, N, dtype=np.complex128)
    v_0 = kron(u_0, np.fft.fft(np.exp(-np.abs(p)), norm="ortho")).toarray().flatten() # second axis is for p
    # v_0 = kron(u_0, np.sqrt(2 / np.pi) / (xi ** 2 + 1)).toarray().flatten() # second axis is for p
    # Write out the Hamiltonian
    H = - (kron(H1, -diags(np.fft.fftfreq(N, d = 1/(N * 2 * np.pi / (2 * R))))) + kron(H2, eye(N)))
    H1_pauli_list = SparsePauliOp.from_operator(H1.toarray()).to_list()
    xi_pauli_list = SparsePauliOp.from_operator(-diags(np.fft.fftfreq(N, d = 1/(N * 2 * np.pi / (2 * R)))).toarray()).to_list()
    H2_pauli_list = SparsePauliOp.from_operator(H2.toarray()).to_list()

    pauli_list = []
    for i in range(len(H1_pauli_list)):
        for j in range(len(xi_pauli_list)):
            pauli_list.append((H1_pauli_list[i][0] + xi_pauli_list[j][0], -H1_pauli_list[i][1] * xi_pauli_list[j][1]))
    for i in range(len(H2_pauli_list)):
        pauli_list.append((H2_pauli_list[i][0] + ''.join(n_p * ['I']), -H2_pauli_list[i][1]))
    pauli_op = SparsePauliOp.from_list(pauli_list)
    pauli_op_commuting = pauli_op.simplify().group_commuting()
    print(f"Norm of diff: {np.linalg.norm(pauli_op.to_matrix() - H.toarray()) : 0.2f}")

    # Hamiltonian simulation
    t_vals = np.linspace(0, T, N_t)
    v = []
    for t in t_vals:
        v_t = v_0
        dt = t / r
        for _ in range(r):
            for i in range(len(pauli_op_commuting)):
                v_t = expm_multiply(-1j * pauli_op_commuting[i].to_matrix() * dt / 2, v_t)
            for i in range(len(pauli_op_commuting))[::-1]:
                v_t = expm_multiply(-1j * pauli_op_commuting[i].to_matrix() * dt / 2, v_t)
        
        v.append(v_t)
    v = np.array(v)
    # Solve the PDE
    v_no_trot = expm_multiply(-1j * H * T, v_0)
    print(f"Trotter error: {np.linalg.norm(v[-1] - v_no_trot) : 0.2f}")
    
    IFFT = np.fft.ifft(np.eye(N), N, norm="ortho")
    v = ((kron(np.eye(dimension), IFFT) @ v.T)).T
    # Issue: how to recover the amplitudes? (need to take absolute value?)
    u_recover = np.zeros((N_t, dimension))
    for i, t in enumerate(np.linspace(0, T, N_t)):
        a = -lambda_min * t
        a_idx = N // 2 + int(a / h)
        # u_recover[i] = np.exp(a) * np.sum(np.reshape(v, newshape=(N_t, dimension, N))[i,:,a_idx:], axis=1) * h
        # print(np.reshape(np.abs(v), newshape=(N_t, dimension, N))[i,:,a_idx:])
        u_recover[i] = np.exp(a) * np.sum(np.reshape(np.abs(v), newshape=(N_t, dimension, N))[i,:,a_idx:], axis=1) * h
    
    return u_recover

def solve_gradient_flow(A, b, x_0, N, R, T, N_t, schrodingerization_method, r=None):
    # The gradient of the optimization problem is Ax-b, so the gradient flow ODE is x'(t) = - Ax + b.
    A_tilde = vstack([hstack([A, -b]), csc_matrix((1, A.shape[0]+1))])
    # print(A_tilde.toarray())
    print("Eigvals of A_tilde: ", eigvals(A_tilde.toarray()))

    lambda_min = np.min(eigvalsh((A_tilde + A_tilde.T).toarray() / 2))
    print("Min eval:", lambda_min)
    u_recover = schrodingerization_method(-A_tilde, vstack([x_0, 1]), N, R, T, N_t, lambda_min=-norm(b)/2, r=r)
    return u_recover[:,:-1]

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

def get_pauli_weight(pauli_str):
    weight = 0
    for pauli in pauli_str:
        if pauli == 'X' or pauli == 'Y' or pauli == 'Z':
            weight += 1
        else:
            assert pauli == 'I'
    return weight

def get_pauli_index(pauli_str):
    indices, paulis = get_pauli_indices(pauli_str)
    assert len(indices) == 1, "Pauli string should have weight 1"
    return indices[0], paulis[0]

def get_pauli_indices(pauli_str):
    n = len(pauli_str)
    indices = []
    paulis = []
    for i in range(n):
        pauli = pauli_str[n-1-i]
        if pauli != 'I':
            paulis.append(pauli)
            indices.append(i)
    return indices, paulis

def pauli_to_int(pauli):
    if pauli == 'X':
        return 0
    elif pauli == 'Y':
        return 1
    elif pauli == 'Z':
        return 2
    else:
        raise ValueError("Invalid Pauli, should be one of X, Y, or Z")
    
def get_pauli_rotation(pauli_str, coeff):
    n = len(pauli_str)
    circuit = QuantumCircuit(n)
    match get_pauli_weight(pauli_str):
        case 0:
            return circuit
        case 1:
            index, pauli = get_pauli_index(pauli_str)
            if pauli == 'X':
                circuit.rx(2 * coeff, index)
            elif pauli == 'Y':
                circuit.ry(2 * coeff, index)
            elif pauli == 'Z':
                circuit.rz(2 * coeff, index)
            return circuit
        case 2:
            indices, paulis = get_pauli_indices(pauli_str)
            for i in range(2):
                if paulis[i] == 'Y':
                    circuit.rz(-np.pi/2, indices[i])
                elif paulis[i] == 'Z':
                    circuit.h(indices[i])
                else:
                    assert paulis[i] == 'X'
            circuit.rxx(2 * coeff, indices[0], indices[1])
            for i in range(2):
                if paulis[i] == 'Y':
                    circuit.rz(np.pi/2, indices[i])
                elif paulis[i] == 'Z':
                    circuit.h(indices[i])
                else:
                    assert paulis[i] == 'X'

            return circuit
        case _:
            raise ValueError("Only implemented for locality up to 2")
    
def get_qft(n_p):
    # Returns QFT circuit
    # Decompose QFT so two-qubit gates are XX rotations
    qft_circuit = QuantumCircuit(n_p)
    for i in np.arange(n_p)[::-1]:
        # myqft.ry(-np.pi/4, n_p - 1 - i)
        # myqft.rz(np.pi, n_p - 1 - i)
        qft_circuit.h(i)
        for j in range(i):
            theta = np.pi / (2 ** (j + 1))
            # Controlled phase gate
            qft_circuit.rz(theta / 2, i)
            qft_circuit.rz(theta / 2, i - 1 - j)
            qft_circuit.h(i)
            qft_circuit.h(i - 1 - j)
            qft_circuit.rxx(-theta / 2, i, i - 1 - j)
            qft_circuit.h(i)
            qft_circuit.h(i - 1 - j)
    for i in range(n_p // 2):
        qft_circuit.swap(i, n_p - 1 - i)
    return qft_circuit

def get_full_circuit_naive_trotter(n_x, n_p, t, H_1, H_2, r, R):
    N_p = 2 ** n_p
    p = np.linspace(-R, R, N_p)

    amplitude_vector_left = np.exp(-np.abs(p))[:2**(n_p - 1)]
    amplitude_vector_left /= np.linalg.norm(amplitude_vector_left)
    q = QuantumRegister(n_p)
    state_prep_circuit = QuantumCircuit(q)
    state_prep_circuit.initialize(amplitude_vector_left, [q[j] for j in range(n_p-1)])
    # Make symmetric
    state_prep_circuit.h(n_p - 1)
    for i in range(n_p-1):
        state_prep_circuit.cnot(n_p - 1, i)

    state_prep_circuit.append(get_qft(n_p).inverse(), qargs=list(range(n_p)))

    xi_pauli_list = get_xi_pauli_op(n_p, R).to_list()
    H_1_pauli_list = SparsePauliOp.from_operator(H_1.toarray()).to_list()
    H_2_pauli_list = SparsePauliOp.from_operator(H_2.toarray()).to_list()
    # print(H_1_pauli_list)
    # print(H_2_pauli_list)

    pauli_list = []
    for i in range(len(H_1_pauli_list)):
        for j in range(len(xi_pauli_list)):
            pauli_list.append((H_1_pauli_list[i][0] + xi_pauli_list[j][0], -H_1_pauli_list[i][1] * xi_pauli_list[j][1]))
    for i in range(len(H_2_pauli_list)):
        pauli_list.append((H_2_pauli_list[i][0] + ''.join(n_p * ['I']), -H_2_pauli_list[i][1]))

    # print(pauli_list)

    pauli_op = SparsePauliOp.from_list(pauli_list)
    # trot_circuit = LieTrotter(reps=r).synthesize(PauliEvolutionGate(pauli_op.group_commuting()))
    trot_circuit = SuzukiTrotter(order=2, reps=r).synthesize(PauliEvolutionGate((t * pauli_op).group_commuting()))

    full_circuit = QuantumCircuit(n_p + n_x)
    full_circuit.append(state_prep_circuit, qargs=range(n_p))
    # Initial state for x register is [0,0,...,0,1]
    for i in np.arange(n_p, n_p + n_x):
        full_circuit.x(i)

    full_circuit.append(trot_circuit, qargs=range(n_p+n_x))
    full_circuit.append(get_qft(n_p), qargs=range(n_p))

    return full_circuit

def get_full_circuit(n_x, n_p, t, H_1, H_2, r, R):
    N_p = 2 ** n_p
    h = (2 * R) / (N_p - 1)

    state_prep_circuit = QuantumCircuit(n_p)
    for i in range(n_p-1):
        theta = 2 * np.arccos(1 / np.sqrt(1 + np.exp(-2 * (2 ** i) * h)))
        state_prep_circuit.ry(theta, i)
    # Make symmetric
    state_prep_circuit.h(n_p - 1)
    for i in range(n_p-1):
        state_prep_circuit.cnot(n_p - 1, i)
    state_prep_circuit.x(n_p - 1)

    state_prep_circuit.append(get_qft(n_p).inverse(), qargs=list(range(n_p)))

    xi_pauli_list = get_xi_pauli_op(n_p, R).to_list()
    H_1_pauli_list = SparsePauliOp.from_operator(H_1.toarray()).to_list()
    H_2_pauli_list = SparsePauliOp.from_operator(H_2.toarray()).to_list()
    # print(H_1_pauli_list)
    # print(H_2_pauli_list)

    trot_circuit = QuantumCircuit(n_p + n_x)
    dt = t / r

    pauli_list_1 = []
    # Deal with special case separately, where there is a 1-site Pauli on the p register and a 1-site Pauli on the x register
    # Store the coefficients in an array
    two_qubit_pauli_coeffs = np.zeros((n_p, n_x, 3))
    for i in range(len(H_1_pauli_list)):
        for j in range(len(xi_pauli_list)):
            if get_pauli_weight(H_1_pauli_list[i][0]) == 1 and get_pauli_weight(xi_pauli_list[j][0]) == 1:
                # Figure out index and which Pauli operator for H_1
                H_1_index, H_1_pauli = get_pauli_index(H_1_pauli_list[i][0])
                xi_index, xi_pauli = get_pauli_index(xi_pauli_list[j][0])
                assert xi_pauli == 'Z'
                two_qubit_pauli_coeffs[xi_index, H_1_index, pauli_to_int(H_1_pauli)] = (-H_1_pauli_list[i][1] * xi_pauli_list[j][1]).real
            else:
                pauli_list_1.append((H_1_pauli_list[i][0] + xi_pauli_list[j][0], -H_1_pauli_list[i][1] * xi_pauli_list[j][1]))

    pauli_list_2 = []
    for i in range(len(H_2_pauli_list)):
        pauli_list_2.append((H_2_pauli_list[i][0] + ''.join(n_p * ['I']), -H_2_pauli_list[i][1]))

    
    # First-order Trotter
    for _ in range(r):
        for pauli_str, coeff in pauli_list_1:
            trot_circuit.append(get_pauli_rotation(pauli_str, dt * coeff.real), qargs=list(range(n_p+n_x)))

        for pauli_str, coeff in pauli_list_2:
            trot_circuit.append(get_pauli_rotation(pauli_str, dt * coeff.real), qargs=list(range(n_p+n_x)))

        # Hadamard on p register
        for i in range(n_p):
            trot_circuit.h(i)
        for i in range(n_p):
            for j in range(n_x):
                assert np.count_nonzero(two_qubit_pauli_coeffs[i,j]) == 2
                x, y, z = two_qubit_pauli_coeffs[i,j]
                assert np.abs(y) < 1e-6
                norm = np.linalg.norm(two_qubit_pauli_coeffs[i,j])
                phi = np.arctan2(z, x)
                trot_circuit.ry(phi, n_p + j)
                trot_circuit.rxx(2 * norm * dt, i, n_p + j)
                trot_circuit.ry(-phi, n_p + j)
        # Hadamard on p register
        for i in range(n_p):
            trot_circuit.h(i)

    full_circuit = QuantumCircuit(n_p + n_x)
    full_circuit.append(state_prep_circuit, qargs=range(n_p))
    # Initial state for x register is [0,0,...,0,1]
    for i in np.arange(n_p, n_p + n_x):
        full_circuit.x(i)

    full_circuit.append(trot_circuit, qargs=range(n_p+n_x))
    full_circuit.append(get_qft(n_p), qargs=range(n_p))

    return full_circuit