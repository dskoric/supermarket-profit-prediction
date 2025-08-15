import numpy as np
import copy
import math

def compute_cost(x, y, w, b):
    """
    Compute cost for linear regression
    Args:
        x (ndarray): Shape (m,) Input data
        y (ndarray): Shape (m,) Target values
        w, b (scalar): Model parameters
    Returns:
        total_cost (float): Cost value
    """
    m = x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

def compute_gradient(x, y, w, b):
    """
    Compute gradient for linear regression
    Args:
        x (ndarray): Shape (m,) Input data
        y (ndarray): Shape (m,) Target values
        w, b (scalar): Model parameters
    Returns:
        dj_dw (scalar): Gradient for w
        dj_db (scalar): Gradient for b
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Perform gradient descent
    Args:
        x (ndarray): Shape (m,) Input data
        y (ndarray): Shape (m,) Target values
        w_in, b_in (scalar): Initial parameters
        cost_function: Function to compute cost
        gradient_function: Function to compute gradient
        alpha (float): Learning rate
        num_iters (int): Number of iterations
    Returns:
        w (scalar): Updated parameter w
        b (scalar): Updated parameter b
        J_history (list): Cost history
        w_history (list): Parameter history
    """
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
            
        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
        
    return w, b, J_history, w_history