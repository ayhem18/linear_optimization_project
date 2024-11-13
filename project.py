"""
Jupyter Kernel might crush with intensive computations. Scripts are more robust on this aspect.
"""

# Import useful libraries
import numpy as np
import scipy.io as io
import scipy.optimize as opt
import os

from tqdm import tqdm
from itertools import combinations


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
X_STANDARD_SOL_PATH = os.path.join(SCRIPT_DIR, 'x_standard.npy')


############################ UTILITY FUNCTIONS INITIALLY PROVIDED ############################
def encoding_bin(mess):
    # Convert each character to its ASCII value and then to binary
    xi = [format(ord(char), '08b') for char in mess]

    # Get the number of characters
    m = len(xi)

    # Initialize an empty list for the binary vector
    x = []

    # Convert each binary string to a binary vector
    for i in range(m):
        x.append([int(bit) for bit in xi[i]])

    # Convert the list to a numpy array for easier manipulation
    x = np.array(x)


    # Return the binary vector and its dimensions
    d = x.shape[1]  # Number of bits per character
    x = x.flatten() # convert into a 1-d vector
    return x, d

def decoding_bin(x, d):
    # Ensure x is a binary vector (0s and 1s)
    x = np.clip(x, 0, 1)  # Clip values to be between 0 and 1
    x = np.round(x)        # Round values to the nearest integer

    # Initialize the output array
    y = np.zeros((len(x) // d, d), dtype=int)

    k = 0
    for i in range(len(x) // d):
        for j in range(d):
            y[i, j] = int(x[k])  # Fill the binary matrix
            k += 1

    # Convert binary to decimal and then to characters
    mess = ''.join(chr(int(''.join(map(str, row)), 2)) for row in y)

    return mess, y


# Disrupts percenterror% of y entries randomly
def noisychannel(y, percenterror):
    m = len(y)                               # Length of the message
    K = int(np.floor(m * percenterror))      # Number of entries to corrupt
    I = np.random.permutation(m)[:K]         # Random indices to corrupt
    y_n = np.copy(y)                         # Copy of the orginal message
    vec = np.random.rand(K) * np.mean(y)
    y_n[I] = vec                             # Corruption of selected inputs
    return y_n


############################ Functions to solve the MIIMIZATION PROBLEM ############################

# let's create some utility functions: to solve the minimization problem
def get_standard_format_matrix(A: np.ndarray) -> np.ndarray:
    m, p = A.shape
    I = np.eye(m)
    # let's prepare the 4 blocks
    ab1 = np.concatenate([-A, I, -np.eye(m), np.zeros((m, m)), np.zeros((m, p))], axis=1)
    ab2 = np.concatenate([A, I, np.zeros((m, m)), -np.eye(m), np.zeros((m, p))], axis=1)
    ab3 = np.concatenate([-np.eye(p), np.zeros((p, m)), np.zeros((p, m)), np.zeros((p, m)), -np.eye(p)], axis=1)
    
    Astandard = np.concatenate([ab1, ab2, ab3], axis=0)

    assert Astandard.shape == (2 * m + p, 2 * p + 3 * m), f"Expected {(2 * m + p, 2 * p + 3 * m)}. Found: {Astandard.shape}"

    return Astandard

def get_cost_coefficients(A: np.ndarray) -> np.ndarray:
    m, p = A.shape
    # let's build the cost coefficient 
    c = np.concatenate([np.zeros((1, p)), 
                        np.ones((1, m)),
                        np.zeros((1, m)),
                        np.zeros((1, m)),
                        np.zeros((1, p)),                        
                        ], 
                        axis=1).squeeze() 

    # c = [0, 0, 0, 0, 0...1, 1, 1,] (p 0s and m 1s)
    assert c.shape == (2 * p + 3 * m, ), f"Expected {(2 * p + 3 * m, )}. Found: {c.shape}"

    return c

def get_eq_constraints(A: np.ndarray, y_noise: np.ndarray) -> np.ndarray:
    m, p = A.shape
    y_2d = np.expand_dims(y_noise, axis=0)
    b_standard = np.concatenate([-y_2d, y_2d, -np.ones((1, p))], axis=1).squeeze()
    assert b_standard .shape == (2 * m + p,), f"Expected {(2 * m + p,)}. Found: {b_standard .shape}"
    return b_standard



def decode_secrete_message(A: np.ndarray, y_noise: np.ndarray):
    # according to the scipy.linprog documentation
    # the function accepts 5 arguments
    # c, Aub, bub, Aeq, b_eq, lb, ub

    # minimize: c @ x
    # such that
    # A_ub @ x <= b_ub
    # A_eq @ x == b_eq
    # lb <= x <= ub

    c = get_cost_coefficients(A)
    Astandard = get_standard_format_matrix(A, y_noise)
    b_standard = get_eq_constraints(A, y_noise)

    x_standard = opt.linprog(c, A_ub=None, b_ub=None, A_eq=Astandard, b_eq=b_standard, bounds=(0, None)).x
    return x_standard



############################ Functions to to determine whether the solution is a vertex ############################

def is_invertible(matrix: np.ndarray):
    # determining whether a matrix numerically is tricky...
    # using the following approach:
    # https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Make sure to pass an invertible matrix. found matrix with shape: {matrix.shape}")
    
    return np.linalg.matrix_rank(matrix) == matrix.shape[0]


def isVertex(A: np.ndarray, 
    solution: np.ndarray, 
    zero_thresh: float = 10 ** -16 
    ) -> bool:

    Astadard = get_standard_format_matrix(A)

    m, n = Astadard.shape

    # consider values less than a certain threshold as zero
    c_sol = np.clip(solution, zero_thresh, 1)
    c_sol = c_sol  * (c_sol >= zero_thresh) # any value less than 'zero_thresh' will be set to 0

    zero_entries_indices = [i for i, v in enumerate(solution) if v == 0]

    print(f"found {len(zero_entries_indices)} zero entries with n - m equals: {n - m}")

    # find all the combinations of set with (n - m) zero entries
    cs = list(combinations(zero_entries_indices, n - m))

    for null_entries in tqdm(cs, desc="iterating through possible combinations"):
        # convert the null_entries to a set
        null_entries = set(null_entries)
        # extract the basic_entries
        basic_entries = [i for i, _ in enumerate(solution) if i not in null_entries]

        # extract the submatrix
        basic_submatrix = Astadard[:, basic_entries]
        # make sure the shape is correct
        assert basic_submatrix.shape == (m, m)

        # check the invertibility of the basic submatrix 
        is_basic = is_invertible(basic_submatrix)

        # if we found an invertible, then our job is done here
        if is_basic:    
            return True
        
        # otherwise... move to the next candidate set of null entries.

    return False


if __name__ == '__main__':
    data = io.loadmat('messageFromAlice.mat')
    A = data['A']
    yprime = data['yprime'].T
    yprime = np.squeeze(yprime)
    d = data['d'][0][0]

    # save x_standard once instead of rerunning everytime
    if not os.path.exists(X_STANDARD_SOL_PATH):
        x_standard = decode_secrete_message(A, yprime)
        np.save(X_STANDARD_SOL_PATH, x_standard)
    else:
        x_standard = np.load(X_STANDARD_SOL_PATH)

    # since x_standard contains xprime, t and slack variables, we need to extract x' before proceeding
    xprime = x_standard[:A.shape[1]]

    # isVertex(A = A, solution=x_standard)

    Astandard = get_standard_format_matrix(A)

    x = np.asarray([[1, 2, 3], [10 ** -14, 1, 2], [1, 10 ** -16, 0.5]])

    x = x  * (x >= 10 ** -12).astype(float)

    print(x)

    # As_rank = np.linalg.matrix_rank(Astandard)  

    # print(As_rank)
    # print(Astandard.shape)
