"""
This script contains the implementation of the Dankin's method
"""

import random, os
import numpy as np
import numpy.linalg as la
from typing import Optional
import scipy.optimize as opt

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
                     epsilon:float,
                     alpha: float = 5 * 10 ** -4,
                     max_iterations: int = 10 ** 3) -> Optional[np.ndarray]:
    # the problem is assumed to be on the standard form

    # step1: initialization
    x_k = dikin_initialization(A, b, epsilon=epsilon)


    if x_k is None:
        return None
    
    _, n = A.shape

    if c.ndim == 1:
        c = np.expand_dims(c, axis=-1) 

    assert c.shape == (n, 1), "make sure the vector of cost coefficients is of the correct shape"

    # define "e"
    e_vec = np.ones((1, n))
    
    iter_counter = 0

    while iter_counter <= max_iterations:
        # extract the diagonal matrix out of x_k
        # x_k = 
        x_dk = np.diag(x_k.squeeze())

        # step2: computation of dual estimates
        w_k = la.inv((A @ x_dk) @ (x_dk  @ A.T)) @ (A @ x_dk) @ x_dk @ c 

        # step3: computation of reduced costs
        r_k = c - A.T @ w_k

        # step4: check for optimality
        if np.all(r_k >= 0) and (e_vec @ x_dk @ r_k).item() <= epsilon:
            return x_k.squeeze()
        
        # step5: compute dky
        dy_k = -x_dk @ r_k

        # step6: 
        ## check for unboundedness
        if np.all(dy_k > 0):
            return None
        
        # check for optimal solution
        if np.all(dy_k == 0):
            return x_k.squeeze()
        
        
        # step7:
        # find the minimum value out of [- 1 / (dk_y (i))] out of i such that (dk_y (i) < 0) 
        step_k = np.min([-1 / (x) for x  in dy_k.squeeze() if x < 0]).item() * alpha

        x_k = x_k + step_k * (x_dk @ dy_k)

        # make sure to increment the counter
        iter_counter += 1

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
    for pe in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n, x = run_dikin_algo(pe)
        print(x)
        xprime = x[:n]
        d = 8    # Number of bits per character
        message_decoded, binary_matrix = decoding_bin(xprime, d)
        print(f"The recovered message is with {pe} noise level:", message_decoded) 
