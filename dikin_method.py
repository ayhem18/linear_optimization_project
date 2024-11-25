"""
This script contains the implementation of the Dankin's method
"""

import random, os
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from typing import Optional
from tqdm import tqdm

from project import get_cost_coefficients, get_eq_constraints,get_standard_format_matrix, prepare_message, MY_MESSAGE, decoding_bin

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def dikin_initialization(A: np.ndarray, b:np.ndarray, epsilon: float) -> np.ndarray:
    # quick check: make sure the shapes match
    if A.ndim != 2:
        raise ValueError(f"The initialization function expects 'A' to be 2 dimensionsal. Found: {A.shape}")

    if b.ndim not in (1, 2):
        raise ValueError(f"The function expects the 'b' argument to be at most 2 dimensional. Found: {b.shape}")
    
    if b.ndim == 2 and 1 not in b.shape:
        raise ValueError(f"if the 'b' argument is 2 dimensional, then it has to have a singleton dimension. Found: {b.shape}")  

    if epsilon <= 0 or epsilon > 10 ** -3:
        raise ValueError("The value epsilon is expected in the range ]0, 0.001]")


    # the scipy function expects a 1 dimensional np.array
    b = np.squeeze(b)

    # the initial point can be found by solving the following problem:
    # min x 0 
    # such that A x = b
    # x >= epsilon 
    _, n = A.shape
    c = np.zeros((n,))
    x_0 = opt.linprog(c = c, A_ub = None, b_ub = None, A_eq = A, b_eq=b, bounds = (epsilon , None)).x

    if x_0 is None:
        return None

    assert np.allclose(A @ x_0, b) , "Ax = b issues"
    assert np.all(x_0 >= epsilon - 10 ** -5), "epislon issues" 

    return np.expand_dims(x_0, axis=-1)

def dikin_algorithm(A: np.ndarray, 
                     b: np.ndarray, 
                     c: np.ndarray, 
                     epsilon:float = 10 ** -6,
                     alpha: float = 0.9,
                     max_iterations: int = 100) -> Optional[np.ndarray]:
    # the problem is assumed to be on the standard form

    # step1: initialization
    x = dikin_initialization(A, b, epsilon=epsilon)


    if x is None:
        return None
    
    _, n = A.shape

    if c.ndim == 1:
        c = np.expand_dims(c, axis=-1) 

    assert c.shape == (n, 1), "make sure the vector of cost coefficients is of the correct shape"

    # define "e"
    e_vec = np.ones((1, n))
    
    for _ in tqdm(range(max_iterations)):

        X_k = np.diag(x.squeeze())
        # call x_k twice instead of (**2) to avoid numerical overflow
        w_k = np.linalg.inv(A @ X_k @ X_k @ A.T) @ A @ X_k @ X_k @ c

        r_k = c - A.T @ w_k

        # first stop condition
        if np.all(r_k >= 0) and np.all((e_vec @ X_k @ r_k).item() <= epsilon):
            return x
        
        d_y_k = - X_k @ r_k

        # check for unboundness
        if np.all(d_y_k > 0):
            return None
        # check for optimality
        if np.all(d_y_k == 0):
            return x

        # step computation
        d_y_k_neg = d_y_k[d_y_k < 0]

        if d_y_k_neg.size == 0:
            # return just in case
            return x

        alpha_k = alpha * np.min(1 / -d_y_k_neg)
        # new solution
        x = x + alpha_k * X_k @ d_y_k

    return x

def run_dikin_algo(percent_error: float):
    A, yprime = prepare_message(message=MY_MESSAGE, percent_error=percent_error)

    _, n = A.shape

    cost_vec = get_cost_coefficients(A)
    b_eq = get_eq_constraints(A, yprime)
    A_eq = get_standard_format_matrix(A)

    
    file_name = os.path.join(SCRIPT_DIR, f"dikin_my_msg_{percent_error}.npy") 
    
    if not os.path.exists(file_name):
        x_dikin = dikin_algorithm(A_eq, b_eq, cost_vec, 10 ** -3)        
        np.save(file_name, x_dikin)
    else:
        return n, np.load(file_name)
    
    return n, x_dikin


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    for pe in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"percentage error: {pe}")
        try:
            n, x = run_dikin_algo(pe)
            xprime = x[:n]
            d = 8    # Number of bits per character
            message_decoded, binary_matrix = decoding_bin(xprime, d)
            print(f"The recovered message is with {pe} noise level:", message_decoded) 
        except TypeError:
            continue