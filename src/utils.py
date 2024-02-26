import numpy as np
from scipy.sparse import csc_matrix, kron, identity
from scipy.optimize import minimize
from functools import reduce
from os import mkdir
from os.path import exists

PAULI_X = csc_matrix(np.array([[0, 1], [1, 0]]))
PAULI_Y = csc_matrix(np.array([[0,-1j],[1j,0]]))
PAULI_Z = csc_matrix(np.array([[1, 0], [0, -1]]))
IDENTITY = csc_matrix(np.eye(2))
NUMBER = csc_matrix(np.array([[0, 0], [0, 1]]))

def tensor(op_list : list):
    return reduce(kron, op_list, 1).tocsc()

def uniform_superposition(n : int):
    return np.ones(2 ** n) / np.sqrt(2 ** n)

def sum_x(n : int) -> np.ndarray:
    '''
    Returns `\sum_i \sigma_x^{(i)}` where `\sigma_x^{(i)}` 
    is the Pauli-x operator on the ith qubit.
    '''
    assert n > 0
    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), PAULI_X, identity(dims[i], format='csr')])
        
    return res

def sum_y(n : int) -> np.ndarray:
    '''
    Returns `\sum_i \sigma_x^{(i)}` where `\sigma_x^{(i)}` 
    is the Pauli-x operator on the ith qubit.
    '''
    assert n > 0
    dims = [2 ** i for i in range(n)]
#     return np.sum([tensor([identity(dims[i], format='csr'), PAULI_Y, identity(dims[n-i-1], format='csr')]) for i in range(n)])
    
    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), PAULI_Y, identity(dims[i], format='csr')])
        
    return res

def sum_h_x(n : int, h : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_i h_i \sigma_x^{(i)}` where `\sigma_x^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0
    assert len(h) == n

    dims = [2 ** i for i in range(n)]
    

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), h[i] * PAULI_X, identity(dims[i], format='csr')])
        
    return res

def sum_h_y(n : int, h : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_i h_i \sigma_y^{(i)}` where `\sigma_y^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0
    assert len(h) == n

    dims = [2 ** i for i in range(n)]
    

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), h[i] * PAULI_Y, identity(dims[i], format='csr')])
        
    return res


def driving_term(n : int, omega : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_i h_i \sigma_y^{(i)}` where `\sigma_y^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0

    dims = [2 ** i for i in range(n)]
    

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += omega[i] * tensor([identity(dims[n-i-1], format='csr'), PAULI_Y, identity(dims[i], format='csr')])
        
    return 0.5 * res

def sum_delta_n(n : int, delta : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_i \Delta_i \hat{n}^{(i)}` where `\hat{n}^{(i)}` 
    is the number operator on the ith qubit.
    '''
    assert n > 0
    assert len(delta) == n

    dims = [2 ** i for i in range(n)]
    
#     return np.sum([tensor([identity(dims[i], format='csr'), delta[i] * NUMBER, identity(dims[n-i-1], format='csr')]) for i in range(n)])
    
    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), delta[i] * NUMBER, identity(dims[i], format='csr')])
        
    return res

def sum_h_z(n : int, h : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_i h_i \sigma_z^{(i)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert len(h) == n

    dims = [2 ** i for i in range(n)]
    
    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        res += tensor([identity(dims[n-i-1], format='csr'), h[i] * PAULI_Z, identity(dims[i], format='csr')])
        
    return res
def sum_V_nn(n : int, V : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_{i>j} V_{i,j} \sigma_z^{(i)} \sigma_z^{(j)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert V.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += V[i,j] * tensor([identity(dims[n-i-1], format='csr'), NUMBER, identity(dims[i-j-1], format='csr'), NUMBER, identity(dims[j], format='csr')])
            
    return res

def sum_J_xx(n : int, J : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_{i>j} J_{i,j} \sigma_x^{(i)} \sigma_x^{(j)}` where `\sigma_x^{(i)}` 
    is the Pauli-x operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (J[i,j] + J[j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_X, identity(dims[i-j-1], format='csr'), PAULI_X, identity(dims[j], format='csr')])
            
    return res

def sum_J_yy(n : int, J : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_{i>j} J_{i,j} \sigma_y^{(i)} \sigma_y^{(j)}` where `\sigma_y^{(i)}` 
    is the Pauli-y operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (J[i,j] + J[j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_Y, identity(dims[i-j-1], format='csr'), PAULI_Y, identity(dims[j], format='csr')])
            
    return res

def sum_J_zz(n : int, J : np.ndarray) -> np.ndarray:
    '''
    Returns `\sum_{i>j} J_{i,j} \sigma_z^{(i)} \sigma_z^{(j)}` where `\sigma_z^{(i)}` 
    is the Pauli-z operator on the ith qubit.
    '''
    assert n > 0
    assert J.shape == (n,n)

    dims = [2 ** i for i in range(n)]

    res = csc_matrix((2 ** n, 2 ** n))
    for i in range(n):
        for j in range(i):
            res += (J[i,j] + J[j,i]) * tensor([identity(dims[n-i-1], format='csr'), PAULI_Z, identity(dims[i-j-1], format='csr'), PAULI_Z, identity(dims[j], format='csr')])
            
    return res

def check_and_make_dir(dir_name):
    if not exists(dir_name):
        mkdir(dir_name)

def bitstring_to_int(bitstring):
    sum = 0
    for i in range(len(bitstring)):
        sum += bitstring[i] * 2 ** i
    
    return sum

# a single point is arranged along the last axis [i_1,i_2,...,i_d,:]
def delta_multivar(x, omega):
    # computes multivariate delta function on each point
    assert omega > 0
    return np.prod(delta_1d(x, omega), axis=-1)

def delta_1d(x, omega):
    assert omega > 0
    return (np.abs(x) <= omega) *  beta_2(x / omega) / omega

def beta(x):
    return 1 - np.abs(x)

def beta_2(x):
    return (1 + np.cos(np.pi * x)) / 2

def beta_3(x):
    return 2

def get_codewords_1d(n : int, encoding, periodic):
    
    codewords = []

    if encoding == "unary" or encoding == "antiferromagnetic":
        if encoding == "unary":
            bitstring = 0
        elif encoding == "antiferromagnetic":
            bitstring = 0
            for k in range(n):
                if k % 2 == 1:
                    bitstring += 1 << k

        if periodic:
            for i in range(2 * n):
                codewords.append(bitstring)
                bitstring ^= 1 << (i % n)
        else:
            for i in range(n+1):
                codewords.append(bitstring)

                if i < n:
                    bitstring ^= 1 << i
    
    elif encoding == "one-hot":

        bitstring = 1

        for i in range(n):
            codewords.append(bitstring)

            if i < n - 1:
                bitstring ^= (1 << i)
                bitstring ^= (1 << (i+1))
    return codewords

def get_codewords(N : int, dimension: int, encoding, periodic=False):
    '''Returns codewords for a given encoding.'''

    n = num_qubits_per_dim(N, encoding)
    codewords_1d = get_codewords_1d(n, encoding, periodic)
    codewords = []

    indices = np.zeros(dimension, dtype=int)
    for i in range(N ** dimension):
        assert np.all(indices <= N - 1)

        codeword = 0
        for j in range(dimension):
            codeword += (2 ** (j * n)) * codewords_1d[indices[j]]
        codewords.append(codeword)
        
        if i < N ** dimension - 1:
            # Increment indices
            indices[-1] += 1
            for j in np.arange(dimension):
                if (indices[dimension - 1 - j] >= N):
                    indices[dimension - 1 - j - 1] += 1
                    indices[dimension - 1 - j] %= N

    return codewords

def num_qubits_per_dim(N, encoding):
    if encoding == "one-hot":
        return N
    elif encoding == "unary" or encoding == "antiferromagnetic":
        return N - 1
    else:
        raise ValueError("Encoding not supported. Valid encodings: unary, antiferromagnetic, one-hot")
    
def get_bitstrings_1d(N, encoding):
    bitstrings = []

    if encoding == "unary":
        bitstring = (N-1) * ["0"]
        for i in range(N):
            bitstrings.append("".join(bitstring))
            if i < N - 1:
                bitstring[i] = "1"

        return bitstrings
    
    elif encoding == "antiferromagnetic":
        bitstring = []
        for i in range(N-1):
            if i % 2 == 0:
                bitstring.append("0")
            else:
                bitstring.append("1")

        for i in range(N):
            bitstrings.append("".join(bitstring))
            if i < N - 1:
                if i % 2 == 0:
                    bitstring[i] = "1"
                else:
                    bitstring[i] = "0"

        return bitstrings
        
    elif encoding == "one-hot":
        bitstring = N * ["0"]
        for i in range(N):
            bitstring[i] = "1"
            if i > 0:
                bitstring[i-1] = "0"

            bitstrings.append("".join(bitstring))

        return bitstrings
    else:
        return ValueError("Encoding not supported.")

def get_bitstrings(N, dimension, encoding):
    bitstrings_1d = get_bitstrings_1d(N, encoding)
    N = len(bitstrings_1d)

    bitstrings = []
    indices = np.zeros(dimension, dtype=int)
    for i in range(N ** dimension):
        
        assert np.all(indices <= N - 1)
        bitstring = []
        for j in range(dimension):
            bitstring.append(bitstrings_1d[indices[j]])
        bitstrings.append("".join(bitstring))
        
        if i < N ** dimension - 1:
            # Increment indices
            indices[-1] += 1
            for j in np.arange(dimension):
                if (indices[dimension - 1 - j] >= N):
                    indices[dimension - 1 - j - 1] += 1
                    indices[dimension - 1 - j] %= N
                
    return bitstrings