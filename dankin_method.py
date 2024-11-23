"""
This script contains the implementation of the Dankin's method
"""

import random
import numpy as np
import numpy.linalg as la
from typing import Optional
import scipy.optimize as opt


def initialization(A: np.ndarray, b:np.ndarray, epsilon: float) -> np.ndarray:
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
    x_0 = opt.linprog(c = c, A_ub = None, b_ub = None, A_eq = A, b_eq=b, bounds = (epsilon , None)).result

    return x_0

def dinkin_algorithm(A: np.ndarray, 
                     b: np.ndarray, 
                     c: np.ndarray, 
                     epsilon:float,
                     alpha: float = 0.5) -> Optional[np.ndarray]:
    # the problem is assumed to be on the standard form

    # step1: initialization
    x_k = initialization(A, b)

    _, n = A.shape

    if c.ndim == 1:
        c = np.expand_dims(c, axis=-1) 

    assert c.shape == (1, n), "make sure the vector of cost coefficients is of the correct shape"

    # define "e"
    e_vec = np.ones((1, n))
    

    while True:
        # extract the diagonal matrix out of x_k
        x_k_diag = np.diag(np.diag(x_k))

        # step2: computation of dual estimates
        w_k = la.inv(A @ x_k_diag ** 2 @ A.T) @ A @ x_k_diag ** 2 @ c 

        # step3: computation of reduced costs
        r_k = c - A.T @ w_k

        # step4: check for optimality
        if np.all(r_k >= 0) and (e_vec @ x_k_diag @ r_k).item() <= epsilon:
            return x_k
        
        # step5: compute dky
        dy_k = -x_k_diag @ r_k

        # step6: 
        ## check for unboundedness
        if np.all(dy_k > 0):
            return None
        
        # check for optimal solution
        if np.all(dy_k == 0):
            return x_k
        
        
        # step7:
        # find the minimum value out of [- 1 / (dk_y (i))] out of i such that (dk_y (i) < 0) 
        step_k = np.min([-1 / (x) for x  in dy_k.squeeze() if x < 0]).item() * alpha

        x_k = x_k + step_k * (x_k_diag @ dy_k)


def few_tests():      
    m, n = random.randint(10, 25), random.randint(10, 25)

    A = np.random.randint(-10, 10, size=(m, n))
    b = np.random.randint(-10, 10, size=(n,))

    cost_vec = np.random.randint(-5, 5, size=(n,))

    # solve with scipy
    x_scipy = opt.linprog(c=cost_vec, A_eq=A, b_eq=b, bounds=(0, None))

    x_custom = dinkin_algorithm(A=A, b=b, c=cost_vec, epsilon=10 ** -5)

    cf_scipy = np.expand_dims(x_scipy, axis=0) @ np.expand_dims(cost_vec, axis=-1)
    
    cf_custom = np.expand_dims(x_custom, axis=0) @ np.expand_dims(cost_vec, axis=-1)

    print(f"cost function with scipy: {cf_scipy}")
    print(f"cost function with scipy: {cf_custom}")


