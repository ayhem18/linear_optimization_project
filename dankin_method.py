"""
This script contains the implementation of the Dankin's method
"""

import random, os
import numpy as np
import numpy.linalg as la
from typing import Optional
import scipy.optimize as opt


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


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
    x_0 = opt.linprog(c = c, A_ub = None, b_ub = None, A_eq = A, b_eq=b, bounds = (epsilon , None)).x

    if x_0 is None:
        return None

    assert np.allclose(A @ x_0, b) , "Ax = b issues"
    assert np.all(x_0 >= epsilon - 10 ** -5), "epislon issues" 

    return np.expand_dims(x_0, axis=-1)

def dankin_algorithm(A: np.ndarray, 
                     b: np.ndarray, 
                     c: np.ndarray, 
                     epsilon:float,
                     alpha: float = 5 * 10 ** -4) -> Optional[np.ndarray]:
    # the problem is assumed to be on the standard form

    # step1: initialization
    x_k = initialization(A, b, epsilon=epsilon)

    print("initialization: done")

    if x_k is None:
        return None
    
    _, n = A.shape

    if c.ndim == 1:
        c = np.expand_dims(c, axis=-1) 

    assert c.shape == (n, 1), "make sure the vector of cost coefficients is of the correct shape"

    # define "e"
    e_vec = np.ones((1, n))
    

    while True:
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



def few_tests():      
    random.seed(0)
    np.random.seed(0)

    for _ in range(50):
        m, n = random.randint(3, 12), random.randint(30, 50)

        A = np.random.randn(m, n)
        b = np.random.randn(m)

        cost_vec = np.random.randint(-3, 3, size=(n,))

        # solve with scipy
        x_scipy = opt.linprog(c=cost_vec, A_eq=A, b_eq=b, bounds=(0, None)).x

        x_custom = dankin_algorithm(A=A, b=b, c=cost_vec, epsilon=10 ** -3)

        assert (x_custom is None) == (x_scipy is None), "Either both null or none of them" 

        if x_custom is None:
            continue

        cf_scipy = np.expand_dims(x_scipy, axis=0) @ np.expand_dims(cost_vec, axis=-1)
        
        cf_custom = np.expand_dims(x_custom, axis=0) @ np.expand_dims(cost_vec, axis=-1)

        print(f"cost function with scipy: {cf_scipy}")
        print(f"cost function with scipy: {cf_custom}")


from project import get_cost_coefficients, get_eq_constraints,get_standard_format_matrix, decoding_bin, encoding_bin, noisychannel


if __name__ == '__main__':
    # few_tests()

    my_mess = "Hey ! Welcome "

    # Message in binary form
    binary_vector, d = encoding_bin(my_mess)
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

    # Noise added by the transmission channel
    # = normal N(0,sigma) for a % input of y
    percenterror = 0.05
    yprime = noisychannel(y, percenterror)


    cost_vec = get_cost_coefficients(A)
    b_eq = get_eq_constraints(A, yprime)
    A_eq = get_standard_format_matrix(A)

    print("Started decoding ...")
    x_dankin =dankin_algorithm(A_eq, b_eq, cost_vec, 10 ** -3)
    x_dankin_path = os.path.join(SCRIPT_DIR, 'x_dankin.npy')
    if not os.path.exists(x_dankin_path):
        np.save(x_dankin_path, x_dankin)
