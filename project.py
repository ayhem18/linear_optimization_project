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
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
X_STANDARD_SOL_PATH = os.path.join(SCRIPT_DIR, 'x_standard.npy')

MY_MESSAGE = "Test...test...Ayhem Speaking..."


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
# let's create some utility functions
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



def solve_relaxed(A: np.ndarray, y_noise: np.ndarray):
    # according to the scipy.linprog documentation
    # the function accepts 5 arguments
    # c, Aub, bub, Aeq, b_eq, lb, ub

    # minimize: c @ x
    # such that
    # A_ub @ x <= b_ub
    # A_eq @ x == b_eq
    # lb <= x <= ub

    c = get_cost_coefficients(A)
    Astandard = get_standard_format_matrix(A)
    b_standard = get_eq_constraints(A, y_noise)

    x_standard = opt.linprog(c, A_ub=None, b_ub=None, A_eq=Astandard, b_eq=b_standard, bounds=(0, None)).x
    return x_standard



def prepare_message(message: str, percent_error:float = 0.1) -> Tuple[int, np.ndarray]:
    # Message in binary form
    binary_vector, d = encoding_bin(message)
    x = binary_vector.astype(np.float32)

    # Length of the message
    size = x.shape
    n = size[0]

    # Length of the message which will be sent
    m = 4*n

    # Encoding matrix: we take a randomly generated matrix
    A = np.random.randn(m,n)

    # Message you wish to send
    y = A@x

    yprime = noisychannel(y, percent_error)    

    return A, yprime


def decode_relaxed_messages(message: str, percent_error: float = 0.1):
    A, yprime = prepare_message(message=message, percent_error=percent_error)

    m, n = A.shape

    file_name = os.path.join(SCRIPT_DIR, f'my_msg_{round(percent_error, 4)}.npy')

    if not os.path.exists(file_name): 
        x_standard = solve_relaxed(A, yprime)
        np.save(file_name, x_standard)
        return n, x_standard
    else:
        return n, np.load(file_name)  

def solve_integer(A: np.ndarray, y_noise: np.ndarray):
    # according to the scipy.linprog documentation
    # the function accepts 5 arguments
    # c, Aub, bub, Aeq, b_eq, lb, ub

    # minimize: c @ x
    # such that
    # A_ub @ x <= b_ub
    # A_eq @ x == b_eq
    # lb <= x <= ub

    c = get_cost_coefficients(A)
    Astandard = get_standard_format_matrix(A)
    b_standard = get_eq_constraints(A, y_noise)

    _, n = A.shape

    integr = np.zeros(len(c))
    for i in range(n):
        integr[i] = 1

    return n, opt.linprog(c, A_eq=Astandard, b_eq=b_standard, integrality=integr, method='highs', options={'maxiter': 10}).x


def decode_integer_messages(message: str, percent_error: float = 0.1):
    A, yprime = prepare_message(message=message, percent_error=percent_error)

    m, n = A.shape

    file_name = os.path.join(SCRIPT_DIR, f'integer_my_msg_{round(percent_error, 4)}.npy')

    if not os.path.exists(file_name): 
        n, x_standard = solve_integer(A, yprime)
        np.save(file_name, x_standard)
        return n, x_standard
    else:
        return n, np.load(file_name)   


if __name__ == '__main__':

    for pe in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n, x = decode_integer_messages(MY_MESSAGE, pe)
        xprime = x[:n]
        d = 8    # Number of bits per character
        message_decoded, binary_matrix = decoding_bin(xprime, d)
        print(f"The recovered message is with {pe} noise level:", message_decoded) 
